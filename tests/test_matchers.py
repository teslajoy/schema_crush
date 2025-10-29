"""pytest tests for matcher infrastructure."""

import pytest
import numpy as np
from schema_crush.tools.matchers import BaseMatcher, RuleMatcher
from schema_crush.knowledge.mapping_rules import RuleDatabase, load_htan_rules, load_gdc_rules


@pytest.fixture
def loaded_ruledb():
    """create database with htan and gdc rules loaded."""
    db = RuleDatabase()
    db.add_rules(load_htan_rules())
    db.add_rules(load_gdc_rules())
    return db


def test_rule_matcher_initialization(loaded_ruledb):
    """test rule matcher can be initialized."""
    matcher = RuleMatcher(loaded_ruledb)
    assert matcher.mapping_ruledb is not None
    assert matcher.mapping_ruledb.count() > 1200


def test_rule_matcher_match(loaded_ruledb):
    """test rule matcher returns valid match results."""
    matcher = RuleMatcher(loaded_ruledb)
    targets = ['Patient.id', 'Specimen.id', 'Observation.code']
    results = matcher.match('case_id', targets)

    assert isinstance(results, list)
    assert len(results) == len(targets)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)
    assert all(0 <= r[1] <= 1 for r in results)


def test_rule_matcher_batch_similarity(loaded_ruledb):
    """test rule matcher returns valid similarity matrix."""
    matcher = RuleMatcher(loaded_ruledb)
    sources = ['case_id', 'sample_id']
    targets = ['Patient.id', 'Specimen.id']
    matrix = matcher.batch_similarity(sources, targets)

    assert matrix.shape == (2, 2)
    assert np.all(matrix >= 0) and np.all(matrix <= 1)