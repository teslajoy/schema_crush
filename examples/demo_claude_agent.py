"""demonstrate claude llm agent with tool use for schema mapping."""

import os
from pathlib import Path
from schema_crush.orchestrator.agents import ClaudeAgent

# try to load from .env file if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed

def demo_claude_field_matching():
    """demonstrate claude agent performing field matching task."""
    import time

    # check for api key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"debug: api_key = {api_key[:10] if api_key else 'None'}...")

    if not api_key:
        print("error: ANTHROPIC_API_KEY environment variable not set")
        print("set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    print("initializing claude agent for field matching...")
    print("="*70)

    # create field matching agent
    start = time.time()
    agent = ClaudeAgent(task="field")
    init_time = time.time() - start
    print(f"initialization complete ({init_time:.2f}s)\n")

    # test mapping task - harder example with more ambiguous field
    source_field = "tissue_preservation_method"
    candidate_targets = [
        "Specimen.type.coding.display",
        "Specimen.collection.method",
        "Specimen.processing.method",
        "Observation.method",
        "Procedure.code",
        "Specimen.container.type"
    ]
    context = {"ground_truth": "Specimen.processing.method"}

    print(f"\ntask: map '{source_field}' to fhir field")
    print(f"candidates: {', '.join(candidate_targets)}")
    print(f"ground truth: {context['ground_truth']}")
    print("\n" + "="*70)
    print("\nclaude agent reasoning with tools...")
    print("-"*70)

    try:
        # get proposals from claude
        proposals = agent.propose_mappings(source_field, candidate_targets, context)

        print(f"\nclaudeagent proposals ({len(proposals)}):\n")
        for i, prop in enumerate(proposals, 1):
            print(f"{i}. target: {prop.target_field}")
            print(f"   confidence: {prop.confidence:.3f}")
            print(f"\n   full reasoning:\n{prop.reasoning}\n")
            if prop.supporting_evidence.get('tool_calls'):
                print(f"   tools used: {', '.join(prop.supporting_evidence['tool_calls'])}")
            print()

        print("="*70)
        print(f"\nground truth: {context['ground_truth']}")
        print(f"claude chose: {proposals[0].target_field if proposals else 'none'}")
        print(f"correct: {proposals[0].target_field == context['ground_truth'] if proposals else False}")

    except Exception as e:
        print(f"\nerror during llm call: {e}")
        print("\nnote: this requires a valid anthropic api key")


def demo_claude_entity_matching():
    """demonstrate claude agent performing entity matching task."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY environment variable not set")
        return

    print("\ninitializing claude agent for entity matching...")
    print("="*70)

    # create entity matching agent
    agent = ClaudeAgent(task="entity")

    # test entity mapping
    source_entity = "biospecimen"
    candidate_resources = [
        "Patient",
        "Specimen",
        "Observation",
        "Condition",
        "DocumentReference"
    ]
    context = {"ground_truth": "Specimen"}

    print(f"\ntask: map entity '{source_entity}' to fhir resource")
    print(f"candidates: {', '.join(candidate_resources)}")
    print(f"ground truth: {context['ground_truth']}")
    print("\n" + "="*70)
    print("\nclaude agent reasoning with tools...")
    print("-"*70)

    try:
        proposals = agent.propose_mappings(source_entity, candidate_resources, context)

        print(f"\nclaudeagent proposals ({len(proposals)}):\n")
        for i, prop in enumerate(proposals, 1):
            print(f"{i}. target: {prop.target_field}")
            print(f"   confidence: {prop.confidence:.3f}")
            print(f"   reasoning: {prop.reasoning[:200]}...")
            print()

        print("="*70)
        print(f"\nground truth: {context['ground_truth']}")
        print(f"claude chose: {proposals[0].target_field if proposals else 'none'}")
        print(f"correct: {proposals[0].target_field == context['ground_truth'] if proposals else False}")

    except Exception as e:
        print(f"\nerror during llm call: {e}")


def demo_claude_generative_mode():
    """demonstrate claude agent discovering candidates itself (generative mode)."""
    import time

    # check for api key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("error: ANTHROPIC_API_KEY environment variable not set")
        return

    print("\n[generative mode] initializing claude agent...")
    print("="*70)

    start = time.time()
    agent = ClaudeAgent(task="field")
    init_time = time.time() - start
    print(f"initialization complete ({init_time:.2f}s)\n")

    # test mapping task - NO candidates provided
    source_field = "tissue_preservation_method"
    context = {"ground_truth": "Specimen.processing"}

    print(f"\ntask: map '{source_field}' to fhir field")
    print(f"mode: generative (no candidates provided - claude explores schema)")
    print(f"ground truth (from htan kb): {context['ground_truth']}")
    print("\n" + "="*70)
    print("\nclaude agent exploring schema and reasoning...")
    print("-"*70)

    try:
        # get proposals from claude - NO candidates
        proposals = agent.propose_mappings(source_field, candidate_targets=None, context=context)

        print(f"\nclaudeagent proposals ({len(proposals)}):\n")
        for i, prop in enumerate(proposals, 1):
            print(f"{i}. target: {prop.target_field}")
            print(f"   confidence: {prop.confidence:.3f}")
            print(f"\n   full reasoning:\n{prop.reasoning}\n")

        print("="*70)
        print(f"\nground truth: {context['ground_truth']}")
        print(f"claude chose: {proposals[0].target_field if proposals else 'none'}")
        print(f"match: {context['ground_truth'] in proposals[0].target_field if proposals else False}")

    except Exception as e:
        print(f"\nerror during llm call: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("claude llm agent demo\n")

    # demo constrained mode (with candidates)
    print("\n### mode 1: constrained (ranking from candidates) ###")
    demo_claude_field_matching()

    # demo generative mode (no candidates)
    print("\n\n### mode 2: generative (discovering candidates) ###")
    demo_claude_generative_mode()

    # demo entity matching
    # demo_claude_entity_matching()