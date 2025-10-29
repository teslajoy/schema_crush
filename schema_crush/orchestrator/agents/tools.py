"""tool definitions for claude agent to use matchers."""

from typing import List, Dict, Any
from langchain_core.tools import tool
from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher


# initialize matchers (singleton pattern)
_biobert_matcher = None
_magneto_matcher = None
_rule_matcher = None


def get_biobert_matcher():
    """get singleton biobert matcher instance."""
    global _biobert_matcher
    if _biobert_matcher is None:
        _biobert_matcher = BioBERTMatcher()
    return _biobert_matcher


def get_magneto_matcher():
    """get singleton magneto matcher instance."""
    global _magneto_matcher
    if _magneto_matcher is None:
        _magneto_matcher = MagnetoMatcher()
    return _magneto_matcher


def get_rule_matcher():
    """get singleton rule matcher instance."""
    global _rule_matcher
    if _rule_matcher is None:
        from schema_crush.knowledge.mapping_rules import RuleDatabase, load_htan_rules, load_gdc_rules
        db = RuleDatabase()
        db.add_rules(load_htan_rules())
        db.add_rules(load_gdc_rules())
        _rule_matcher = RuleMatcher(db)
    return _rule_matcher


@tool
def biobert_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use biobert embeddings for biomedical semantic similarity matching.

    best for: matching biomedical concepts and terminology.

    args:
        source: source field or term to match
        candidates: list of candidate target fields/terms

    returns:
        list of matches with scores, sorted by confidence
    """
    matcher = get_biobert_matcher()
    results = matcher.match(source, candidates)
    return [{"target": tgt, "score": float(score)} for tgt, score in results[:5]]


@tool
def magneto_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use magneto embeddings for schema structure matching.

    best for: matching field names and schema structures (trained on gdc->fhir).

    args:
        source: source field name to match
        candidates: list of candidate target field names

    returns:
        list of matches with scores, sorted by confidence
    """
    matcher = get_magneto_matcher()
    results = matcher.match(source, candidates)
    return [{"target": tgt, "score": float(score)} for tgt, score in results[:5]]


@tool
def rule_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use knowledge base rules for mapping lookup.

    best for: leveraging existing expert mappings from htan/gdc knowledge base.

    args:
        source: source field or entity name
        candidates: list of candidate target fields/resources

    returns:
        list of matches with scores and supporting rules
    """
    matcher = get_rule_matcher()
    results = matcher.match(source, candidates)

    # add rule evidence
    enriched_results = []
    for tgt, score in results[:5]:
        rules = matcher.mapping_ruledb.find_by_source(source)
        rule_count = len(rules)
        enriched_results.append({
            "target": tgt,
            "score": float(score),
            "rule_count": rule_count,
            "sources": list(set(r.source for r in rules[:3]))
        })

    return enriched_results


def warmup_matchers():
    """pre-load all matchers to avoid first-call latency.

    call this once at startup to load models into memory.
    subsequent tool calls will be much faster.
    """
    print("warming up matchers (loading models)...")
    get_biobert_matcher()
    get_magneto_matcher()
    get_rule_matcher()
    print("matchers ready")


# export tools for agent
MAPPING_TOOLS = [biobert_match, magneto_match, rule_match]