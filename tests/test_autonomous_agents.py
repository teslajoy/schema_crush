"""pytest tests for autonomous agents."""

import pytest
from schema_crush.orchestrator.agents import (
    AutonomousAgent,
    MappingProposal,
    BioBERTAgent,
    MagnetoAgent,
    RuleAgent,
    get_agent
)
from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher
from schema_crush.knowledge.mapping_rules import RuleDatabase, load_htan_rules, load_gdc_rules


@pytest.fixture
def biobert_agent():
    """create biobert agent."""
    matcher = BioBERTMatcher()
    return BioBERTAgent(matcher)


@pytest.fixture
def magneto_agent():
    """create magneto agent."""
    matcher = MagnetoMatcher()
    return MagnetoAgent(matcher)


@pytest.fixture
def rule_agent():
    """create rule agent with loaded rules."""
    db = RuleDatabase()
    db.add_rules(load_htan_rules())
    db.add_rules(load_gdc_rules())
    matcher = RuleMatcher(db)
    return RuleAgent(matcher)


def test_biobert_agent_propose_mappings(biobert_agent):
    """test biobert agent returns valid proposals."""
    source = "case_id"
    targets = ["Patient.id", "Specimen.id", "Observation.code"]
    context = {}

    proposals = biobert_agent.propose_mappings(source, targets, context)

    assert isinstance(proposals, list)
    assert len(proposals) > 0
    assert all(isinstance(p, MappingProposal) for p in proposals)
    assert all(p.source_field == source for p in proposals)
    assert all(p.target_field in targets for p in proposals)
    assert all(0 <= p.confidence <= 1 for p in proposals)
    assert all(p.reasoning for p in proposals)
    assert all(p.supporting_evidence for p in proposals)


def test_magneto_agent_propose_mappings(magneto_agent):
    """test magneto agent returns valid proposals."""
    source = "sample_id"
    targets = ["Specimen.id", "Patient.id", "DocumentReference.id"]
    context = {}

    proposals = magneto_agent.propose_mappings(source, targets, context)

    assert isinstance(proposals, list)
    assert len(proposals) > 0
    assert all(isinstance(p, MappingProposal) for p in proposals)
    assert all(p.source_field == source for p in proposals)
    assert all(0 <= p.confidence <= 1 for p in proposals)


def test_rule_agent_propose_mappings(rule_agent):
    """test rule agent returns valid proposals."""
    source = "case_id"
    targets = ["Patient.id", "Specimen.id", "Condition.id"]
    context = {}

    proposals = rule_agent.propose_mappings(source, targets, context)

    assert isinstance(proposals, list)
    assert len(proposals) > 0
    assert all(isinstance(p, MappingProposal) for p in proposals)
    assert all(p.source_field == source for p in proposals)
    assert all('matching_rules' in p.supporting_evidence for p in proposals)


def test_agent_explain_decision(biobert_agent):
    """test agent can explain its decisions."""
    proposal = MappingProposal(
        source_field="case_id",
        target_field="Patient.id",
        confidence=0.85,
        reasoning="test reasoning",
        supporting_evidence={"test": "data"}
    )

    explanation = biobert_agent.explain_decision(proposal)

    assert isinstance(explanation, str)
    assert "case_id" in explanation
    assert "Patient.id" in explanation
    assert "0.85" in explanation or "0.850" in explanation


def test_agent_registry():
    """test agent registry can load agents."""
    # test registry contains expected agents
    from schema_crush.orchestrator.agents.registry import AGENT_REGISTRY
    assert "biobert" in AGENT_REGISTRY
    assert "magneto" in AGENT_REGISTRY
    assert "rule" in AGENT_REGISTRY

    # test get_agent raises for unknown agent
    with pytest.raises(KeyError):
        get_agent("unknown_agent", matcher=None)