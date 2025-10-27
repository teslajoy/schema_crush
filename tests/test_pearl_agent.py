"""test pearl agent workflow."""

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


def test_pearl_agent_single_tier(embedders, loaded_data):
    """test pearl agent running single tier."""
    # create target fhir schema
    target_schema = {
        'Patient': ['id', 'identifier', 'birthDate', 'gender'],
        'Condition': ['code', 'id']
    }

    # initialize agent
    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'biobert': 0.3, 'magneto': 0.7},  # favor magneto for schema matching
        top_k=3,
        enable_hitl=False  # disable for testing
    )

    # run field tier only
    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_single_tier'
    )

    print("\n=== single tier results ===")
    print(f"matches: {len(results['matches'])}")
    print(f"hitl queue: {len(results['hitl_queue'])}")

    # verify results
    assert len(results['matches']) > 0
    assert 'field' in results['matches_by_tier']
    assert len(results['reasoning_log']) > 0


def test_pearl_agent_multi_tier(embedders, loaded_data):
    """test pearl agent running all tiers."""
    target_schema = {
        'Patient': ['id', 'identifier', 'birthDate', 'gender'],
        'Condition': ['code', 'id']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'biobert': 0.3, 'magneto': 0.7},
        top_k=3,
        enable_hitl=False
    )

    # run all tiers
    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['entity', 'field', 'content'],
        thread_id='test_multi_tier'
    )

    print("\n=== multi-tier results ===")
    for tier, matches in results['matches_by_tier'].items():
        print(f"{tier}: {len(matches)} matches")

    print(f"\ntotal matches (deduplicated): {len(results['matches'])}")
    print(f"hitl queue: {len(results['hitl_queue'])}")

    # verify results
    assert 'entity' in results['matches_by_tier']
    assert 'field' in results['matches_by_tier']
    assert 'content' in results['matches_by_tier']
    assert len(results['matches']) > 0


def test_pearl_agent_weighted_combination(embedders, loaded_data):
    """test that embedder weights are properly normalized and applied."""
    target_schema = {
        'Patient': ['identifier', 'gender']
    }

    # test with equal weights
    agent_equal = PEARLAgent(
        embedders=embedders,
        embedder_weights={'biobert': 1.0, 'magneto': 1.0},  # should normalize to 0.5 each
        top_k=3,
        enable_hitl=False
    )

    results_equal = agent_equal.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_equal_weights'
    )

    # test with magneto-heavy weights
    agent_magneto = PEARLAgent(
        embedders=embedders,
        embedder_weights={'biobert': 0.2, 'magneto': 0.8},
        top_k=3,
        enable_hitl=False
    )

    results_magneto = agent_magneto.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_magneto_heavy'
    )

    print("\n=== weight comparison ===")
    print(f"equal weights: {len(results_equal['matches'])} matches")
    print(f"magneto-heavy: {len(results_magneto['matches'])} matches")

    # verify both produce results
    assert len(results_equal['matches']) > 0
    assert len(results_magneto['matches']) > 0


def test_pearl_agent_confidence_thresholds(embedders, loaded_data):
    """test confidence level classification."""
    target_schema = {
        'Patient': ['identifier', 'gender', 'birthDate']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'magneto': 1.0},
        confidence_thresholds={
            'high': 0.95,
            'medium': 0.85,
            'low': 0.0
        },
        top_k=5,
        enable_hitl=False
    )

    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field'],
        thread_id='test_confidence'
    )

    # check confidence distribution
    matches = results['matches']
    high = [m for m in matches if m['confidence'] == 'high']
    medium = [m for m in matches if m['confidence'] == 'medium']
    low = [m for m in matches if m['confidence'] == 'low']

    print("\n=== confidence distribution ===")
    print(f"high: {len(high)}")
    print(f"medium: {len(medium)}")
    print(f"low: {len(low)}")

    # verify confidence levels are assigned
    assert len(matches) > 0
    assert all('confidence' in m for m in matches)


def test_pearl_agent_deduplication(embedders, loaded_data):
    """test that matches are deduplicated across tiers."""
    target_schema = {
        'Patient': ['identifier', 'gender']
    }

    agent = PEARLAgent(
        embedders=embedders,
        embedder_weights={'magneto': 1.0},
        top_k=5,
        enable_hitl=False
    )

    # run field and content tiers (should produce overlapping matches)
    results = agent.run(
        source_data=loaded_data,
        target_schema=target_schema,
        tiers=['field', 'content'],
        thread_id='test_dedup'
    )

    # count total matches across tiers
    total_before_dedup = sum(
        len(matches) for matches in results['matches_by_tier'].values()
    )
    total_after_dedup = len(results['matches'])

    print("\n=== deduplication ===")
    print(f"before dedup: {total_before_dedup}")
    print(f"after dedup: {total_after_dedup}")

    # verify deduplication occurred
    assert total_after_dedup <= total_before_dedup

    # verify no duplicate (source_field, target_field) pairs
    field_pairs = set()
    for match in results['matches']:
        pair = (match['source_entity'], match['source_field'],
                match['target_entity'], match['target_field'])
        assert pair not in field_pairs, f"duplicate match found: {pair}"
        field_pairs.add(pair)