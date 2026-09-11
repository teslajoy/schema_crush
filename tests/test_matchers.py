"""pytest tests for matcher infrastructure."""

import pytest
import numpy as np
from schema_crush.tools.matchers import BaseMatcher, RuleMatcher


@pytest.fixture(scope="module")
def rule_matcher():
    """rule matcher backed by the builtin flat mapping database.

    mirrors the production construction path (see schema_crush/mcp/server.py).
    """
    return RuleMatcher.from_builtin()


def test_rule_matcher_initialization(rule_matcher):
    """test rule matcher can be initialized."""
    assert rule_matcher.db is not None
    assert rule_matcher.db.stats()['counts']['sources'] > 1200


def test_rule_matcher_match(rule_matcher):
    """test rule matcher returns valid match results."""
    targets = ['Patient.id', 'Specimen.id', 'Observation.code']
    results = rule_matcher.match('case_id', targets)

    assert isinstance(results, list)
    assert len(results) == len(targets)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)
    assert all(0 <= r[1] <= 1 for r in results)


def test_rule_matcher_batch_similarity(rule_matcher):
    """test rule matcher returns valid similarity matrix."""
    sources = ['case_id', 'sample_id']
    targets = ['Patient.id', 'Specimen.id']
    matrix = rule_matcher.batch_similarity(sources, targets)

    assert matrix.shape == (2, 2)
    assert np.all(matrix >= 0) and np.all(matrix <= 1)