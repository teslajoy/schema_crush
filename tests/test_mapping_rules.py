"""pytest tests for mapping rules infrastructure."""

import pytest
from schema_crush.knowledge.mapping_rules import (
    RuleDatabase,
    MappingRule,
    Reference,
    FieldMapping,
    load_htan_rules,
    load_gdc_rules
)


@pytest.fixture
def loaded_db():
    """create database with htan and gdc rules loaded."""
    db = RuleDatabase()
    db.add_rules(load_htan_rules())
    db.add_rules(load_gdc_rules())
    return db


def test_load_htan_rules():
    """test loading htan rules."""
    rules = load_htan_rules()
    assert len(rules) > 1000
    assert all(r.source == "htan" for r in rules)


def test_load_gdc_rules():
    """test loading gdc rules."""
    rules = load_gdc_rules()
    assert len(rules) > 200
    assert all(r.source == "gdc" for r in rules)


def test_database_find_by_target(loaded_db):
    """test finding rules by target resource."""
    rules = loaded_db.find_by_target("observation")
    assert len(rules) > 0
    assert all(r.target_resource.lower() == "observation" for r in rules)


def test_database_find_by_category(loaded_db):
    """test finding rules by category."""
    rules = loaded_db.find_by_category("specimen")
    assert len(rules) > 0


def test_database_filter(loaded_db):
    """test filtering with multiple criteria."""
    rules = loaded_db.filter(target_resource="observation", min_confidence=0.8)
    assert len(rules) > 0
    assert all(r.target_resource.lower() == "observation" for r in rules)


def test_database_statistics(loaded_db):
    """test database statistics."""
    stats = loaded_db.get_statistics()
    assert stats['total_rules'] > 1200
    assert 'htan' in stats['sources']
    assert 'gdc' in stats['sources']