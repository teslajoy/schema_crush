"""demonstrate multi-agent consensus using ClaudeAgent with different tool preferences.

this demo shows the modern architecture where:
- matchers (BioBERT, Magneto, Rule) are TOOLS, not agents
- ClaudeAgent is the only true autonomous agent (LLM with tool access)
- multiple ClaudeAgent instances with different configurations collaborate
- consensus is built from their proposals
"""

import os
from schema_crush.orchestrator.agents import ClaudeAgent, MappingProposal
from schema_crush.orchestrator.agents.tools import warmup_matchers


def create_specialized_agents():
    """create three ClaudeAgent instances with different tool preferences.

    we simulate different "agent personalities" by using custom prompts that
    emphasize different tools:
    - semantic agent: prefers biobert for biomedical concepts
    - structural agent: prefers magneto for schema patterns
    - knowledge agent: prefers rule-based matching from KB
    """

    # base field matching prompt
    base_prompt = """you are an expert in biomedical schema field mapping.

task: map a source field to a fhir resource field path.

you have access to five tools:
1. search_fhir_fields - search for fields across all fhir resources by keyword
2. explore_fhir_resource - get all fields for a specific fhir resource
3. biobert_match - biomedical semantic similarity (requires candidates list)
4. magneto_match - schema structure patterns (requires candidates list, trained on gdc->fhir)
5. rule_match - knowledge base rules (requires candidates list, htan/gdc expert mappings)
"""

    semantic_prompt = base_prompt + """
your specialty: SEMANTIC ANALYSIS
- prioritize biobert_match for understanding biomedical concepts
- use semantic similarity as primary signal
- validate with other tools but trust biomedical embeddings
- explain semantic relationships in your reasoning

provide your final recommendation in this exact format:
CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
"""

    structural_prompt = base_prompt + """
your specialty: STRUCTURAL PATTERN MATCHING
- prioritize magneto_match for schema structure analysis
- focus on field naming patterns and structural alignment
- magneto is trained on gdc->fhir, so it knows common patterns
- validate semantic fit but trust structural signals

provide your final recommendation in this exact format:
CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
"""

    knowledge_prompt = base_prompt + """
your specialty: KNOWLEDGE-BASED REASONING
- prioritize rule_match to leverage existing expert mappings
- look for established patterns in htan/gdc knowledge base
- when rules exist, they represent validated expert decisions
- fall back to other tools when no rules match

provide your final recommendation in this exact format:
CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
"""

    return {
        "semantic": (ClaudeAgent(task="field", warmup=False), semantic_prompt),
        "structural": (ClaudeAgent(task="field", warmup=False), structural_prompt),
        "knowledge": (ClaudeAgent(task="field", warmup=False), knowledge_prompt),
    }


def demo_multi_agent_proposals():
    """show how multiple specialized ClaudeAgents propose mappings with consensus."""

    print("\n" + "="*70)
    print("multi-agent consensus demo")
    print("="*70)
    print("\narchitecture:")
    print("  - matchers (biobert, magneto, rule) = tools (not agents)")
    print("  - ClaudeAgent = LLM with tool access (true autonomous agent)")
    print("  - multiple agents with different tool preferences collaborate")
    print("\n" + "="*70)

    # check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nerror: ANTHROPIC_API_KEY not found in environment")
        print("this demo requires a valid anthropic api key")
        print("set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    print("\ninitializing agents...")
    print("-"*70)

    # warmup matchers once (shared across all agents)
    warmup_matchers()

    # create specialized agents
    agents = create_specialized_agents()
    print(f"created {len(agents)} specialized claude agents:")
    for name in agents.keys():
        print(f"  - {name} agent")

    # test mapping task
    source_field = "case_id"
    candidate_targets = [
        "Patient.id",
        "Patient.identifier",
        "Condition.id",
        "Specimen.id",
        "Observation.subject"
    ]

    print(f"\nmapping task: {source_field}")
    print(f"candidates: {', '.join(candidate_targets)}")
    print("\n" + "="*70)

    # collect proposals from all agents
    all_proposals = []

    for agent_name, (agent, custom_prompt) in agents.items():
        print(f"\n{agent_name.upper()} AGENT:")
        print("-"*70)

        # temporarily override system prompt for this agent
        original_prompt = agent.system_prompt
        agent.system_prompt = custom_prompt

        try:
            proposals = agent.propose_mappings(source_field, candidate_targets, context={})

            if proposals:
                prop = proposals[0]  # get top proposal
                all_proposals.append((agent_name, prop))

                print(f"choice: {prop.target_field}")
                print(f"confidence: {prop.confidence:.3f}")
                print(f"reasoning summary: {prop.reasoning[:200]}...")
            else:
                print("no proposals generated")

        except Exception as e:
            print(f"error: {e}")
        finally:
            # restore original prompt
            agent.system_prompt = original_prompt

    # consensus analysis
    print("\n" + "="*70)
    print("CONSENSUS ANALYSIS:")
    print("-"*70)

    if not all_proposals:
        print("no proposals to analyze")
        return

    # count votes for each target
    target_votes = {}
    target_confidences = {}
    agent_votes = {}

    for agent_name, prop in all_proposals:
        target = prop.target_field
        target_votes[target] = target_votes.get(target, 0) + 1

        if target not in target_confidences:
            target_confidences[target] = []
        target_confidences[target].append(prop.confidence)

        if target not in agent_votes:
            agent_votes[target] = []
        agent_votes[target].append(agent_name)

    # sort by consensus
    consensus_targets = sorted(target_votes.items(), key=lambda x: x[1], reverse=True)

    for target, vote_count in consensus_targets[:3]:
        avg_conf = sum(target_confidences[target]) / len(target_confidences[target])
        voting_agents = ', '.join(agent_votes[target])

        print(f"\n{target}")
        print(f"  votes: {vote_count}/{len(agents)} agents")
        print(f"  voting agents: {voting_agents}")
        print(f"  avg confidence: {avg_conf:.3f}")
        print(f"  individual confidences: {', '.join(f'{c:.3f}' for c in target_confidences[target])}")

    print("\n" + "="*70)
    print(f"ground truth: Patient.id")

    if consensus_targets:
        winner = consensus_targets[0][0]
        vote_count = consensus_targets[0][1]
        print(f"consensus choice: {winner} ({vote_count}/{len(agents)} votes)")

        if winner == "Patient.id":
            print("result: CORRECT ✓")
        else:
            print("result: INCORRECT (but agents provided reasoning)")

    print("="*70)


def demo_tool_based_architecture():
    """demonstrate how matchers work as tools, not agents."""

    print("\n" + "="*70)
    print("tool-based architecture demo")
    print("="*70)
    print("\nshowing how matchers are tools that ClaudeAgent can call:")
    print()

    from schema_crush.orchestrator.agents.tools import biobert_match, magneto_match, rule_match
    from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher

    # example: direct tool usage
    source = "case_id"
    candidates = ["Patient.id", "Patient.identifier", "Condition.id"]

    print(f"source: {source}")
    print(f"candidates: {candidates}\n")

    # matchers as tools (what ClaudeAgent uses)
    print("1. TOOLS (langchain @tool wrappers):")
    print("-"*70)

    print("\nbiobert_match tool:")
    result = biobert_match.invoke({"source": source, "candidates": candidates})
    for r in result[:2]:
        print(f"  {r['target']}: {r['score']:.3f}")

    print("\nmagneto_match tool:")
    result = magneto_match.invoke({"source": source, "candidates": candidates})
    for r in result[:2]:
        print(f"  {r['target']}: {r['score']:.3f}")

    print("\nrule_match tool:")
    result = rule_match.invoke({"source": source, "candidates": candidates})
    for r in result[:2]:
        print(f"  {r['target']}: {r['score']:.3f}")

    print("\n2. UNDERLYING MATCHERS (scoring engines):")
    print("-"*70)

    # direct matcher usage (lower level)
    biobert = BioBERTMatcher()
    matches = biobert.match(source, candidates)
    print(f"\nBioBERTMatcher.match() -> {matches[0]}")

    magneto = MagnetoMatcher()
    matches = magneto.match(source, candidates)
    print(f"MagnetoMatcher.match() -> {matches[0]}")

    print("\n" + "="*70)
    print("\nkey insight:")
    print("  - matchers are SCORING FUNCTIONS (not agents)")
    print("  - tools are LANGCHAIN WRAPPERS around matchers")
    print("  - ClaudeAgent is the AUTONOMOUS AGENT with tool access")
    print("="*70)


if __name__ == '__main__':
    # demo 1: show architecture
    demo_tool_based_architecture()

    # demo 2: multi-agent consensus
    print("\n\n")
    demo_multi_agent_proposals()