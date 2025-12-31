"""claude llm agent for schema mapping with tool use."""

import os
from typing import List, Optional, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from schema_crush.orchestrator.agents.base_agent import AutonomousAgent, MappingProposal
from schema_crush.orchestrator.agents.tools import MAPPING_TOOLS


ENTITY_MATCHING_PROMPT = """you are an expert in biomedical schema mapping.

task: map a source entity (table/class) to a fhir resource R5 version type.

you have access to three matching tools:
1. biobert_match - biomedical semantic similarity
2. magneto_match - schema structure patterns (trained on gdc->fhir)
3. rule_match - knowledge base rules (htan/gdc expert mappings)

guidelines:
- use multiple tools to gather evidence
- consider semantic meaning (biobert) AND structural patterns (magneto) AND existing rules
- explain your reasoning clearly
- provide confidence score 0.0-1.0

examples:
- "case" -> Patient (high confidence, standard clinical concept)
- "biospecimen" -> Specimen (exact fhir resource match)
- "file" -> DocumentReference (established gdc pattern)
"""

FIELD_MATCHING_PROMPT = """you are a biomedical data expert with deep knowledge of FHIR, oncology, genomics, and clinical data standards.

task: map a source field to the appropriate FHIR R5 resource field path.

THINK FIRST - before using any tools:
1. look at the FIELD NAME - what does it mean semantically?
2. look at the SAMPLE VALUES if provided - what kind of data is this?
   - "G1, G2, G3" = tumor grading → relates to Condition staging/grading
   - "Pancreatic tumor" = tissue type → Specimen.type
   - "51.1, 6.9" months = survival time → Observation with time value
   - "GSM12345" = accession ID → identifier field
3. use YOUR DOMAIN KNOWLEDGE - you know FHIR, you know oncology, you know what these terms mean

then use tools to VALIDATE your hypothesis:
- search_fhir_fields: find candidate FHIR fields
- explore_fhir_resource: see all fields for a resource
- biobert_match, magneto_match, rule_match: score candidates (pass source + candidate list)

YOUR REASONING PROCESS:
1. "This field is called X and has values like Y"
2. "Based on my knowledge, this represents Z concept"
3. "In FHIR, the Z concept maps to Resource.field"
4. "Let me verify with tools..."
5. "Final answer: Resource.field with confidence N"

CRITICAL: you MUST provide a mapping. use your expertise to make the best choice.

provide your answer in this format:
CHOSEN TARGET: [Resource.field]
CONFIDENCE: [0.0-1.0]
REASONING: [your expert analysis of what the data means and why you chose this mapping]
"""

CONTENT_MATCHING_PROMPT = """you are an expert in biomedical terminology and coding standards.

task: map source data values to fhir coded values or standard terminologies.

you have access to three matching tools:
1. biobert_match - biomedical semantic similarity
2. magneto_match - schema structure patterns
3. rule_match - knowledge base rules with coding systems

guidelines:
- identify the appropriate coding system (snomed, loinc, icd-10, etc)
- use biobert for semantic matching of medical terms
- check rules for established value mappings
- map to observation codes when appropriate
- explain your reasoning clearly
- provide confidence score 0.0-1.0

examples:
- "Adenocarcinoma" -> SNOMED CT code (cancer diagnosis)
- "Fresh Frozen" -> specimen type code (tissue preservation)
- "Male" -> administrative-gender code (fhir value set)

note: this task is not fully implemented yet. focus on identifying the coding system and general approach.
"""


class ClaudeAgent(AutonomousAgent):
    """autonomous llm agent using claude with tool access for schema mapping.

    this agent can perform three types of mapping tasks:
    1. entity matching (source entity -> fhir resource)
    2. field matching (source field -> fhir field path)
    3. content matching (source values -> fhir coded values)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        task: str = "field",
        warmup: bool = True
    ):
        """initialize claude agent.

        args:
            model: claude model to use
            api_key: anthropic api key (uses ANTHROPIC_API_KEY env if not provided)
            task: mapping task type ("entity", "field", or "content")
            warmup: pre-load matchers on initialization (default: True)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.task = task
        self.name = f"claude_agent_{task}"

        # pre-load matchers to avoid first-call latency
        if warmup:
            from schema_crush.orchestrator.agents.tools import warmup_matchers
            warmup_matchers()

        # initialize llm with tools
        self.llm = ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            temperature=0
        ).bind_tools(MAPPING_TOOLS)

        # select task-specific prompt
        self.system_prompt = self._get_task_prompt(task)

    def _get_task_prompt(self, task: str) -> str:
        """get system prompt for specific task.

        args:
            task: "entity", "field", or "content"

        returns:
            task-specific system prompt
        """
        prompts = {
            "entity": ENTITY_MATCHING_PROMPT,
            "field": FIELD_MATCHING_PROMPT,
            "content": CONTENT_MATCHING_PROMPT
        }
        return prompts.get(task, FIELD_MATCHING_PROMPT)

    def propose_mappings(
        self,
        source_field: str,
        candidate_targets: Optional[List[str]] = None,
        context: dict = None
    ) -> List[MappingProposal]:
        """use llm with tools to propose mappings.

        tries RuleMatcher first - if exact match (1.0), skips LLM to save tokens.
        only calls Claude for uncertain cases.

        args:
            source_field: source field/entity/value to map
            candidate_targets: list of candidate targets
            context: additional context (ground_truth for learning, etc)

        returns:
            list of mapping proposals with llm reasoning
        """
        from langchain_core.messages import ToolMessage
        from schema_crush.orchestrator.agents.tools import (
            biobert_match, magneto_match, rule_match,
            explore_fhir_resource, search_fhir_fields,
            get_knowledge_base
        )

        context = context or {}
        kb = get_knowledge_base()

        # PRIORITY 1: exact lookup from knowledge base (O(1))
        rule_results = kb.db.lookup(source_field)
        if rule_results:
            _, dest = rule_results[0]
            if candidate_targets:
                dest_set = {d.destination.lower() for _, d in rule_results}
                for target in candidate_targets:
                    if target.lower() in dest_set:
                        return [MappingProposal(
                            source_field=source_field,
                            target_field=target,
                            confidence=1.0,
                            reasoning="exact match from knowledge base (skipped llm)",
                            supporting_evidence={"agent": self.name, "source": "knowledge_base_exact"}
                        )]
            else:
                return [MappingProposal(
                    source_field=source_field,
                    target_field=dest.destination,
                    confidence=1.0,
                    reasoning="exact match from knowledge base (skipped llm)",
                    supporting_evidence={"agent": self.name, "source": "knowledge_base_exact"}
                )]

        # PRIORITY 2: fuzzy text search on field names in KB
        # try original field name and common variations
        search_terms = [source_field]
        # add stemmed variations (e.g., "grading" -> "grade")
        if source_field.endswith("ing"):
            search_terms.append(source_field[:-3] + "e")  # grading -> grade
            search_terms.append(source_field[:-3])  # grading -> grad
        if source_field.endswith("s"):
            search_terms.append(source_field[:-1])  # stages -> stage

        fuzzy_results = []
        for term in search_terms:
            fuzzy_results.extend(kb.fuzzy_lookup(term, limit=5))

        # if good fuzzy match found, use it (threshold 0.30 to catch token matches)
        if fuzzy_results and fuzzy_results[0]["score"] >= 0.30:
            best = fuzzy_results[0]
            if candidate_targets:
                # check if target is in candidates
                for target in candidate_targets:
                    if target.lower() == best["target"].lower():
                        return [MappingProposal(
                            source_field=source_field,
                            target_field=target,
                            confidence=0.9,
                            reasoning=f"fuzzy match: '{source_field}' similar to '{best['source']}' ({best['schema']}) -> {best['target']}",
                            supporting_evidence={"agent": self.name, "source": "knowledge_base_fuzzy", "match": best}
                        )]
            else:
                return [MappingProposal(
                    source_field=source_field,
                    target_field=best["target"],
                    confidence=0.9,
                    reasoning=f"fuzzy match: '{source_field}' similar to '{best['source']}' ({best['schema']}) -> {best['target']}",
                    supporting_evidence={"agent": self.name, "source": "knowledge_base_fuzzy", "match": best}
                )]

        # PRIORITY 3: vector similarity for few-shot context (passed to LLM)
        similar = kb.find_similar(source_field, k=3)
        # also include fuzzy results in few-shot context
        if fuzzy_results:
            for fr in fuzzy_results[:3]:
                similar.append({"source": fr["source"], "target": fr["target"], "score": fr["score"]})
        few_shot_context = ""
        if similar:
            few_shot_context = "\n\nsimilar mappings from knowledge base:\n"
            for s in similar:
                few_shot_context += f"  {s['source']} -> {s['target']} (similarity: {s['score']:.2f})\n"

        # build source context string if available
        source_context_str = ""
        if context.get("source_entity"):
            source_context_str = f"\nsource entity/table: {context['source_entity']}"
        if context.get("sample_values"):
            vals = context["sample_values"][:5]  # limit to 5 samples
            source_context_str += f"\nsample values: {vals}"

        # construct query for llm based on mode (constrained vs generative)
        if candidate_targets:
            # constrained mode: ranking from pre-provided candidates
            user_message = f"""map this source to the best target:

source: {source_field}{source_context_str}

candidate targets:
{chr(10).join(f"  - {t}" for t in candidate_targets)}
{few_shot_context}
use the available tools to gather evidence, then provide your final recommendation in this exact format:

CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
"""
        else:
            # generative mode: discover and generate candidates
            user_message = f"""map this source field to the appropriate fhir field:

source: {source_field}{source_context_str}
{few_shot_context}
use the available tools to:
1. search for potential target fields using search_fhir_fields or explore_fhir_resource
2. score the discovered candidates using ALL THREE matchers (magneto_match, biobert_match, rule_match)
3. compare results from all matchers and choose the best match

REMEMBER: you MUST provide a mapping. never say "unknown" - always make your best guess.

provide your final recommendation in this exact format:

CHOSEN TARGET: [full fhir path like Resource.field]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation including what you discovered and why]
"""

        # add ground truth if available (for learning)
        if context.get("ground_truth"):
            user_message += f"\n\nground truth (for validation): {context['ground_truth']}"

        # invoke llm with tools (first call)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]

        response = self.llm.invoke(messages)
        messages.append(response)

        # execute tool calls in a loop (allow multiple rounds of tool calling)
        tool_map = {
            'biobert_match': biobert_match,
            'magneto_match': magneto_match,
            'rule_match': rule_match,
            'explore_fhir_resource': explore_fhir_resource,
            'search_fhir_fields': search_fhir_fields
        }

        max_iterations = 10  # increased for generative mode workflow
        iteration = 0

        # track tool results for fallback
        tool_results = {
            "candidates_discovered": [],
            "matcher_scores": {}  # {target: {"biobert": score, "magneto": score, ...}}
        }

        while hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations:
            iteration += 1

            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                if tool_name in tool_map:
                    # execute the tool
                    tool_func = tool_map[tool_name]
                    tool_result = tool_func.invoke(tool_args)

                    # track results for fallback
                    if tool_name in ['search_fhir_fields', 'explore_fhir_resource']:
                        # extract candidate paths from discovery tools
                        if isinstance(tool_result, list):
                            for item in tool_result[:10]:
                                if isinstance(item, dict) and 'path' in item:
                                    tool_results["candidates_discovered"].append(item['path'])
                        elif isinstance(tool_result, dict) and 'fields' in tool_result:
                            resource = tool_result.get('resource', '')
                            for field in tool_result['fields'][:10]:
                                tool_results["candidates_discovered"].append(f"{resource}.{field}")

                    elif tool_name in ['biobert_match', 'magneto_match', 'rule_match']:
                        # extract matcher scores
                        matcher = tool_name.replace('_match', '')
                        if isinstance(tool_result, list):
                            for item in tool_result:
                                if isinstance(item, dict):
                                    target = item.get('target', '')
                                    score = item.get('score', 0.0)
                                    if target not in tool_results["matcher_scores"]:
                                        tool_results["matcher_scores"][target] = {}
                                    tool_results["matcher_scores"][target][matcher] = score

                    # add tool result to messages
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call.get('id')
                    ))

            # invoke llm again with tool results
            response = self.llm.invoke(messages)
            messages.append(response)

        # parse llm response and tool calls
        proposals = self._parse_llm_response(
            response,
            source_field,
            candidate_targets,
            context,
            tool_results  # pass tool results for fallback
        )

        return proposals

    def _parse_llm_response(
        self,
        response: Any,
        source_field: str,
        candidate_targets: List[str],
        context: dict,
        tool_results: dict = None
    ) -> List[MappingProposal]:
        """parse llm response into mapping proposals.

        args:
            response: llm response with tool calls
            source_field: source field
            candidate_targets: candidate targets
            context: context dict
            tool_results: tracked results from tool calls for fallback

        returns:
            list of mapping proposals
        """
        import re

        tool_results = tool_results or {"candidates_discovered": [], "matcher_scores": {}}

        # extract content from response
        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            # extract text from content blocks
            content = ' '.join([
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in response.content
            ])
        else:
            content = str(response.content)

        # parse structured format
        chosen_target = None
        confidence = 0.5
        reasoning = content

        # try to extract CHOSEN TARGET
        target_match = re.search(r'CHOSEN TARGET:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if target_match:
            chosen_target = target_match.group(1).strip()
            # remove markdown formatting
            chosen_target = chosen_target.strip('*').strip()

        # try to extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5

        # try to extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # fallback: if no structured format, look for target in content
        if not chosen_target and candidate_targets:
            for target in candidate_targets:
                if target in content:
                    chosen_target = target
                    break

        # final fallback: use tool results if LLM didn't give structured answer
        if not chosen_target:
            if candidate_targets:
                chosen_target = candidate_targets[0]
            else:
                # try to extract resource.field pattern from content
                pattern_match = re.search(r'\b([A-Z][a-z]+)\.([a-z_]+(?:\.[a-z_]+)*)\b', content)
                if pattern_match:
                    chosen_target = pattern_match.group(0)

                # FALLBACK: use highest-scoring match from tool results
                elif tool_results["matcher_scores"]:
                    # find target with highest average score across matchers
                    best_target = None
                    best_score = 0.0

                    for target, scores in tool_results["matcher_scores"].items():
                        if scores:
                            avg_score = sum(scores.values()) / len(scores)
                            if avg_score > best_score:
                                best_score = avg_score
                                best_target = target

                    if best_target:
                        chosen_target = best_target
                        confidence = min(0.6, best_score)  # cap confidence for fallback
                        reasoning = f"Fallback from tool results (LLM didn't provide structured answer). Best match: {best_target} with avg score {best_score:.3f}"
                    else:
                        chosen_target = "unknown"
                else:
                    chosen_target = "unknown"

        # create proposal
        proposals = [MappingProposal(
            source_field=source_field,
            target_field=chosen_target,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence={
                "agent": self.name,
                "llm_response": content,
                "tool_results": tool_results
            }
        )]

        return proposals

    def explain_decision(self, proposal: MappingProposal) -> str:
        """provide detailed explanation for a mapping proposal.

        args:
            proposal: mapping proposal

        returns:
            detailed explanation
        """
        return f"{self.name} analysis:\n" \
               f"  task: {self.task} matching\n" \
               f"  source: {proposal.source_field}\n" \
               f"  target: {proposal.target_field}\n" \
               f"  confidence: {proposal.confidence:.3f}\n" \
               f"  reasoning: {proposal.reasoning}\n" \
               f"  evidence: {proposal.supporting_evidence}"