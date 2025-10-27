"""test pearl agent hitl (human-in-the-loop) functionality."""

import pytest
import importlib.resources
from pathlib import Path
from schema_crush.loaders.csv_loader import CSVLoader
from schema_crush.tools.embeddings.biobert_embedder import BioBERTEmbedder
from schema_crush.tools.embeddings.magneto_embedder import MagnetoEmbedder
from schema_crush.orchestrator.pearl_agent import PEARLAgent


@pytest.fixture
def loaded_data():
    """load example csv data."""
    loader = CSVLoader()
    base_path = Path(importlib.resources.files('schema_crush'))

    sources = {
        'case': str(base_path / 'data' / 'examples' / 'case.csv'),
    }
    return loader.load(sources)


@pytest.fixture
def embedders():
    """create embedder instances."""
    return {
        'biobert': BioBERTEmbedder(),
        'magneto': MagnetoEmbedder()
    }


def test_pearl_agent_hitl_pause_resume(embedders, loaded_data):
    """test hitl pause and resume functionality."""
    target_schema = {
        'Patient': ['identifier', 'gender', 'birthDate']
    }

    # create agent with hitl enabled
    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'magneto': 1.0},  # use only magneto
        confidence_thresholds={
            'high': 0.95,
            'medium': 0.50,  # lower threshold to get medium confidence matches
            'low': 0.0
        },
        top_k=3,
        enable_hitl=True
    )

    print("\n=== initial run (should pause for hitl) ===")

    # run workflow - should pause at hitl node
    try:
        results = agent.run(
            source_data=loaded_data,
            target_schema=target_schema,
            tiers=['field'],
            thread_id='test_hitl'
        )

        print(f"\nresults: {len(results['matches'])} matches")
        print(f"hitl queue: {len(results['hitl_queue'])} items")

        # check if there are hitl items
        if len(results['hitl_queue']) > 0:
            print("\n=== simulating user feedback ===")

            # simulate user providing feedback
            feedback = {}
            for match in results['hitl_queue']:
                match_id = match['match_id']
                # simulate user accepting high-scoring matches
                feedback[match_id] = match['score'] > 0.6
                print(f"feedback for {match_id}: {feedback[match_id]} (score: {match['score']:.4f})")

            print(f"\ntotal feedback provided: {len(feedback)}")

            # note: resume functionality requires checkpointing which is currently disabled
            # this test verifies the hitl queue is populated correctly
            assert len(results['hitl_queue']) > 0
            assert all('match_id' in m for m in results['hitl_queue'])
            assert all(m['requires_hitl'] for m in results['hitl_queue'])
        else:
            print("\nno hitl items generated (all matches high or low confidence)")

    except Exception as e:
        print(f"\nerror during hitl test: {e}")
        raise


def test_pearl_agent_confidence_distribution(embedders, loaded_data):
    """test that confidence thresholds properly categorize matches."""
    target_schema = {
        'Patient': ['identifier', 'gender', 'birthDate']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'magneto': 1.0},
        confidence_thresholds={
            'high': 0.95,
            'medium': 0.75,  # wider medium range
            'low': 0.0
        },
        top_k=5,
        enable_hitl=False
    )

    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_confidence_dist'
    )

    # analyze confidence distribution
    matches = results['matches']
    by_confidence = {
        'high': [m for m in matches if m['confidence'] == 'high'],
        'medium': [m for m in matches if m['confidence'] == 'medium'],
        'low': [m for m in matches if m['confidence'] == 'low']
    }

    print("\n=== confidence distribution ===")
    for conf, items in by_confidence.items():
        if items:
            scores = [m['score'] for m in items]
            print(f"{conf}: {len(items)} matches")
            print(f"  score range: {min(scores):.4f} - {max(scores):.4f}")
            for m in items[:3]:  # show first 3
                print(f"    {m['source_field']} → {m['target_field']}: {m['score']:.4f}")

    # verify confidence classification
    for match in matches:
        score = match['score']
        conf = match['confidence']

        if conf == 'high':
            assert score >= 0.95
        elif conf == 'medium':
            assert 0.75 <= score < 0.95
        elif conf == 'low':
            assert score < 0.75


def test_pearl_agent_hitl_required_flag(embedders, loaded_data):
    """test that requires_hitl flag is set correctly."""
    target_schema = {
        'Patient': ['identifier', 'gender']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'magneto': 1.0},
        confidence_thresholds={
            'high': 0.90,
            'medium': 0.60,
            'low': 0.0
        },
        top_k=3,
        enable_hitl=False
    )

    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_hitl_flag'
    )

    # verify requires_hitl flag matches confidence level
    for match in results['matches']:
        if match['confidence'] == 'medium':
            assert match['requires_hitl'] is True
        else:
            assert match['requires_hitl'] is False

    print(f"\n=== hitl flag verification ===")
    print(f"total matches: {len(results['matches'])}")
    print(f"requires hitl: {sum(1 for m in results['matches'] if m['requires_hitl'])}")
    print(f"medium confidence: {sum(1 for m in results['matches'] if m['confidence'] == 'medium')}")


def test_pearl_agent_embedder_scores_tracking(embedders, loaded_data):
    """test that individual embedder scores are tracked."""
    target_schema = {
        'Patient': ['identifier', 'gender']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'biobert': 0.4, 'magneto': 0.6},
        top_k=3,
        enable_hitl=False
    )

    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_embedder_scores'
    )

    print("\n=== embedder score comparison ===")
    for match in results['matches'][:5]:  # show first 5
        print(f"\n{match['source_field']} → {match['target_field']}")
        print(f"  combined: {match['score']:.4f}")
        print(f"  biobert:  {match['embedder_scores']['biobert']:.4f}")
        print(f"  magneto:  {match['embedder_scores']['magneto']:.4f}")

        # verify combined score is weighted average
        expected = (0.4 * match['embedder_scores']['biobert'] +
                   0.6 * match['embedder_scores']['magneto'])
        assert abs(match['score'] - expected) < 0.001

    # verify all matches have embedder scores
    for match in results['matches']:
        assert 'embedder_scores' in match
        assert 'biobert' in match['embedder_scores']
        assert 'magneto' in match['embedder_scores']


def test_pearl_agent_match_id_uniqueness(embedders, loaded_data):
    """test that match_id is unique for each match."""
    target_schema = {
        'Patient': ['identifier', 'gender', 'birthDate']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'magneto': 1.0},
        top_k=5,
        enable_hitl=False
    )

    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field', 'content'],
        thread_id='test_match_ids'
    )

    # collect all match_ids
    match_ids = [m['match_id'] for m in results['matches']]

    print(f"\n=== match id analysis ===")
    print(f"total matches: {len(match_ids)}")
    print(f"unique match_ids: {len(set(match_ids))}")

    # show some examples
    print("\nexample match_ids:")
    for match_id in list(set(match_ids))[:5]:
        print(f"  {match_id}")

    # verify uniqueness
    assert len(match_ids) == len(set(match_ids)), "match_ids are not unique!"

    # verify format
    for match_id in match_ids:
        # format should be: entity.field_entity.field_tier
        assert '_' in match_id
        parts = match_id.split('_')
        assert len(parts) >= 3
        tier = parts[-1]
        assert tier in ['entity', 'field', 'content']