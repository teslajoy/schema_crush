"""interactive demo: pearl orchestrator with hitl review."""

import os
from pathlib import Path
from schema_crush.orchestrator.agents import ClaudeAgent

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# test fields with different confidence levels
TEST_FIELDS = [
    {
        'source': 'case_id',
        'candidates': ['Patient.id', 'Patient.identifier', 'ResearchStudy.id'],
        'ground_truth': 'Patient.id',
        'expected_confidence': 'high'  # should auto-accept
    },
    {
        'source': 'tissue_preservation_method',
        'candidates': ['Specimen.collection.method', 'Specimen.processing.method',
                      'Observation.method'],
        'ground_truth': 'Specimen.processing.method',
        'expected_confidence': 'medium'  # might need HITL
    },
    {
        'source': 'sample_quality_score',
        'candidates': ['Observation.valueQuantity', 'Specimen.condition',
                      'DiagnosticReport.result'],
        'ground_truth': 'Observation.valueQuantity',
        'expected_confidence': 'low'  # ambiguous
    }
]

def simulate_confidence_routing(confidence: float) -> str:
    """determine routing based on confidence thresholds."""
    if confidence >= 0.95:
        return "AUTO-ACCEPT"
    elif confidence >= 0.85:
        return "HITL REVIEW"
    else:
        return "REJECT"

def run_hitl_demo():
    """demonstrate confidence-based routing with simulated hitl."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("error: ANTHROPIC_API_KEY environment variable not set")
        return

    print("="*70)
    print("schema crush: confidence-based routing demo")
    print("="*70)
    print("\nconfidence thresholds:")
    print("  ≥0.95 → auto-accept (high confidence)")
    print("  0.85-0.95 → hitl review (medium confidence)")
    print("  <0.85 → reject (low confidence)")
    print("\n" + "="*70)

    agent = ClaudeAgent(task="field")

    auto_accepted = 0
    hitl_needed = 0
    rejected = 0

    for i, test in enumerate(TEST_FIELDS, 1):
        print(f"\n\n[mapping {i}/{len(TEST_FIELDS)}]")
        print("="*70)
        print(f"source field: {test['source']}")
        print(f"candidates: {', '.join(test['candidates'])}")
        print(f"ground truth: {test['ground_truth']}")
        print("-"*70)

        try:
            proposals = agent.propose_mappings(
                test['source'],
                test['candidates'],
                context={'ground_truth': test['ground_truth']}
            )

            if proposals:
                predicted = proposals[0].target_field
                confidence = proposals[0].confidence
                routing = simulate_confidence_routing(confidence)

                print(f"\nclaudeagent prediction:")
                print(f"  target: {predicted}")
                print(f"  confidence: {confidence:.3f}")
                print(f"  routing: {routing}")

                # simulate routing
                if routing == "AUTO-ACCEPT":
                    auto_accepted += 1
                    print(f"\n✓ auto-accepted (confidence {confidence:.3f} ≥ 0.95)")
                    correct = (predicted == test['ground_truth'])
                    print(f"  verification: {'CORRECT' if correct else 'ERROR - would need correction'}")

                elif routing == "HITL REVIEW":
                    hitl_needed += 1
                    print(f"\n⚠ sending to expert review (confidence {confidence:.3f} in [0.85, 0.95))")
                    print(f"  expert sees:")
                    print(f"    - source: {test['source']}")
                    print(f"    - proposed: {predicted}")
                    print(f"    - reasoning: {proposals[0].reasoning[:200]}...")
                    print(f"\n  [simulating expert decision...]")
                    correct = (predicted == test['ground_truth'])
                    if correct:
                        print(f"  expert: ✓ APPROVED")
                    else:
                        print(f"  expert: ✗ REJECTED, corrected to {test['ground_truth']}")

                else:  # REJECT
                    rejected += 1
                    print(f"\n✗ rejected (confidence {confidence:.3f} < 0.85)")
                    print(f"  reason: ambiguous mapping, insufficient evidence")
                    print(f"  action: flag for manual review or request more context")

        except Exception as e:
            print(f"\nerror: {e}")
            rejected += 1

    # summary
    print("\n\n" + "="*70)
    print("routing summary")
    print("="*70)
    total = len(TEST_FIELDS)
    print(f"\nauto-accepted: {auto_accepted}/{total} ({auto_accepted/total*100:.0f}%)")
    print(f"hitl review: {hitl_needed}/{total} ({hitl_needed/total*100:.0f}%)")
    print(f"rejected: {rejected}/{total} ({rejected/total*100:.0f}%)")

    print(f"\nexpert review burden: {hitl_needed}/{total} = {hitl_needed/total*100:.0f}%")
    print(f"(compared to 100% manual mapping)")

    print("\n" + "="*70)

if __name__ == '__main__':
    run_hitl_demo()