"""demonstrate autonomous agents proposing mappings with reasoning."""

from schema_crush.orchestrator.agents import BioBERTAgent, MagnetoAgent, RuleAgent
from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher
from schema_crush.knowledge.mapping_rules import RuleDatabase, load_htan_rules, load_gdc_rules


def demo_multi_agent_proposals():
    """show how multiple agents propose mappings for the same source field."""

    print("initializing agents...")
    print("="*70)

    # initialize matchers
    biobert_matcher = BioBERTMatcher()
    magneto_matcher = MagnetoMatcher()

    db = RuleDatabase()
    db.add_rules(load_htan_rules())
    db.add_rules(load_gdc_rules())
    rule_matcher = RuleMatcher(db)

    # initialize agents
    biobert_agent = BioBERTAgent(biobert_matcher)
    magneto_agent = MagnetoAgent(magneto_matcher)
    rule_agent = RuleAgent(rule_matcher)

    print(f"agents initialized: {biobert_agent.name}, {magneto_agent.name}, {rule_agent.name}\n")

    # test mapping task
    source_field = "case_id"
    candidate_targets = [
        "Patient.id",
        "Patient.identifier",
        "Condition.id",
        "Specimen.id",
        "Observation.subject"
    ]
    context = {}

    print(f"mapping task: {source_field}")
    print(f"candidates: {', '.join(candidate_targets)}")
    print("\n" + "="*70)

    # collect proposals from all agents
    print("\n📊 BIOBERT AGENT PROPOSALS:")
    print("-"*70)
    biobert_proposals = biobert_agent.propose_mappings(source_field, candidate_targets, context)
    for i, prop in enumerate(biobert_proposals[:3], 1):
        print(f"{i}. {prop.target_field} (confidence: {prop.confidence:.3f})")
        print(f"   reasoning: {prop.reasoning}\n")

    print("\n📊 MAGNETO AGENT PROPOSALS:")
    print("-"*70)
    magneto_proposals = magneto_agent.propose_mappings(source_field, candidate_targets, context)
    for i, prop in enumerate(magneto_proposals[:3], 1):
        print(f"{i}. {prop.target_field} (confidence: {prop.confidence:.3f})")
        print(f"   reasoning: {prop.reasoning}\n")

    print("\n📊 RULE AGENT PROPOSALS:")
    print("-"*70)
    rule_proposals = rule_agent.propose_mappings(source_field, candidate_targets, context)
    for i, prop in enumerate(rule_proposals[:3], 1):
        print(f"{i}. {prop.target_field} (confidence: {prop.confidence:.3f})")
        print(f"   reasoning: {prop.reasoning}")
        if prop.supporting_evidence.get('matching_rules'):
            print(f"   rules: {len(prop.supporting_evidence['matching_rules'])} matched\n")
        else:
            print()

    # show consensus
    print("\n" + "="*70)
    print("CONSENSUS ANALYSIS:")
    print("-"*70)

    all_proposals = biobert_proposals + magneto_proposals + rule_proposals

    # find targets proposed by multiple agents
    target_counts = {}
    target_confidences = {}
    for prop in all_proposals:
        target = prop.target_field
        target_counts[target] = target_counts.get(target, 0) + 1
        if target not in target_confidences:
            target_confidences[target] = []
        target_confidences[target].append(prop.confidence)

    # sort by number of agents agreeing
    consensus_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)

    for target, count in consensus_targets[:3]:
        avg_conf = sum(target_confidences[target]) / len(target_confidences[target])
        print(f"• {target}")
        print(f"  agents agreeing: {count}/3")
        print(f"  average confidence: {avg_conf:.3f}")
        print(f"  confidences: {', '.join(f'{c:.3f}' for c in target_confidences[target])}\n")

    print("="*70)
    print("\n✅ ground truth: Patient.id")
    print(f"✅ top consensus: {consensus_targets[0][0]} ({consensus_targets[0][1]}/3 agents)")


if __name__ == '__main__':
    demo_multi_agent_proposals()