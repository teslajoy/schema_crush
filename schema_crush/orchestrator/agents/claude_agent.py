"""claude llm agent for schema mapping with tool use."""

import os
from typing import List, Optional, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from schema_crush.orchestrator.agents.base_agent import AutonomousAgent, MappingProposal
from schema_crush.orchestrator.agents.tools import MAPPING_TOOLS


ENTITY_MATCHING_PROMPT = """you are an expert in biomedical schema mapping.

task: map a source entity (table/class) to a fhir resource type.

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

FIELD_MATCHING_PROMPT = """you are an expert in biomedical schema field mapping.

task: map a source field to a fhir resource field path.

you have access to three matching tools:
1. biobert_match - biomedical semantic similarity
2. magneto_match - schema structure patterns (trained on gdc->fhir)
3. rule_match - knowledge base rules (htan/gdc expert mappings)

guidelines:
- prioritize magneto for field name matching (it's trained on this)
- use biobert for semantic validation
- check rules for established patterns
- consider field type compatibility (id vs identifier, date vs datetime)
- explain your reasoning clearly
- provide confidence score 0.0-1.0

examples:
- "case_id" -> Patient.id (structural identifier field)
- "primary_diagnosis" -> Condition.code (semantic clinical concept)
- "tissue_type" -> Specimen.type (domain-specific terminology)
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
        candidate_targets: List[str],
        context: dict
    ) -> List[MappingProposal]:
        """use llm with tools to propose mappings.

        args:
            source_field: source field/entity/value to map
            candidate_targets: list of candidate targets
            context: additional context (ground_truth for learning, etc)

        returns:
            list of mapping proposals with llm reasoning
        """
        from langchain_core.messages import ToolMessage
        from schema_crush.orchestrator.agents.tools import biobert_match, magneto_match, rule_match

        # construct query for llm
        user_message = f"""map this source to the best target:

source: {source_field}

candidate targets:
{chr(10).join(f"  - {t}" for t in candidate_targets)}

use the available tools to gather evidence, then provide your final recommendation in this exact format:

CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
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
            'rule_match': rule_match
        }

        max_iterations = 5  # prevent infinite loops
        iteration = 0

        while hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations:
            iteration += 1

            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                if tool_name in tool_map:
                    # execute the tool
                    tool_func = tool_map[tool_name]
                    tool_result = tool_func.invoke(tool_args)

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
            context
        )

        return proposals

    def _parse_llm_response(
        self,
        response: Any,
        source_field: str,
        candidate_targets: List[str],
        context: dict
    ) -> List[MappingProposal]:
        """parse llm response into mapping proposals.

        args:
            response: llm response with tool calls
            source_field: source field
            candidate_targets: candidate targets
            context: context dict

        returns:
            list of mapping proposals
        """
        import re

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
        if not chosen_target:
            for target in candidate_targets:
                if target in content:
                    chosen_target = target
                    break

        # final fallback
        if not chosen_target:
            chosen_target = candidate_targets[0] if candidate_targets else "unknown"

        # create proposal
        proposals = [MappingProposal(
            source_field=source_field,
            target_field=chosen_target,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence={
                "agent": self.name,
                "llm_response": content
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