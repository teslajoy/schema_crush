"""tool definitions for claude agent to use matchers."""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from schema_crush.tools.matchers import BioBERTMatcher, MagnetoMatcher, RuleMatcher


# singleton knowledge base
_knowledge_base = None


def get_knowledge_base():
    """get singleton knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        from schema_crush.learning.knowledge_base import KnowledgeBase
        _knowledge_base = KnowledgeBase.load()
        print("loaded knowledge base (db, calibrators, vectors)")
    return _knowledge_base


def get_biobert_matcher():
    """get biobert matcher from knowledge base."""
    kb = get_knowledge_base()
    if not hasattr(kb, '_biobert_matcher') or kb._biobert_matcher is None:
        kb._biobert_matcher = BioBERTMatcher(use_expert_embeddings=True)
    return kb._biobert_matcher


def get_magneto_matcher():
    """get magneto matcher from knowledge base."""
    kb = get_knowledge_base()
    if not hasattr(kb, '_magneto_matcher') or kb._magneto_matcher is None:
        kb._magneto_matcher = MagnetoMatcher(use_expert_embeddings=True)
    return kb._magneto_matcher


def get_rule_matcher():
    """get rule matcher from knowledge base."""
    kb = get_knowledge_base()
    if not hasattr(kb, '_rule_matcher') or kb._rule_matcher is None:
        kb._rule_matcher = RuleMatcher(kb.db)
    return kb._rule_matcher


def apply_calibration(matcher_name: str, raw_score: float, tier: str = None) -> float:
    """apply calibration to raw matcher score via knowledge base.

    args:
        matcher_name: name of matcher ("biobert", "magneto", "rule")
        raw_score: raw confidence score from matcher
        tier: optional tier for tier-specific calibration ("entity", "field", "content")

    returns:
        calibrated score (or raw score if no calibration available)
    """
    kb = get_knowledge_base()
    return kb.calibrate(matcher_name, raw_score, tier)


@tool
def biobert_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use biobert embeddings for biomedical semantic similarity matching.

    best for: matching biomedical concepts and terminology.

    note: scores are calibrated based on historical accuracy (96.9% accuracy when confident).

    args:
        source: source field or term to match
        candidates: list of candidate target fields/terms

    returns:
        list of matches with calibrated scores, sorted by confidence
    """
    matcher = get_biobert_matcher()
    results = matcher.match(source, candidates)

    # apply calibration to scores
    calibrated_results = []
    for tgt, raw_score in results[:5]:
        calibrated_score = apply_calibration("biobert", raw_score)
        calibrated_results.append({
            "target": tgt,
            "score": float(calibrated_score)
        })

    return calibrated_results


@tool
def magneto_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use magneto embeddings for schema structure matching.

    best for: matching field names and schema structures (trained on gdc->fhir).

    note: scores are calibrated based on historical accuracy (96.9% accuracy when confident).

    args:
        source: source field name to match
        candidates: list of candidate target field names

    returns:
        list of matches with calibrated scores, sorted by confidence
    """
    matcher = get_magneto_matcher()
    results = matcher.match(source, candidates)

    # apply calibration to scores
    calibrated_results = []
    for tgt, raw_score in results[:5]:
        calibrated_score = apply_calibration("magneto", raw_score)
        calibrated_results.append({
            "target": tgt,
            "score": float(calibrated_score)
        })

    return calibrated_results


@tool
def rule_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use knowledge base rules for mapping lookup.

    best for: leveraging existing expert mappings from htan/gdc knowledge base.

    note: scores are calibrated based on historical accuracy.

    args:
        source: source field or entity name
        candidates: list of candidate target fields/resources

    returns:
        list of matches with calibrated scores
    """
    matcher = get_rule_matcher()
    results = matcher.match(source, candidates)

    # apply calibration to scores
    calibrated_results = []
    for tgt, raw_score in results[:5]:
        calibrated_score = apply_calibration("rule", raw_score)
        calibrated_results.append({
            "target": tgt,
            "score": float(calibrated_score),
        })

    return calibrated_results


@tool
def explore_fhir_resource(resource_type: str) -> Dict[str, Any]:
    """explore a fhir resource to see its available fields.

    use this when you need to discover what fields are available for a fhir resource
    before making a mapping decision.

    args:
        resource_type: fhir resource name (e.g., "patient", "specimen", "observation")

    returns:
        dict with resource fields and description
    """
    from schema_crush.tools.fhir_schema_tool import get_schema_explorer

    print(f"[linkml query] explore_fhir_resource('{resource_type}')")

    explorer = get_schema_explorer()
    fields = explorer.get_resource_fields(resource_type)
    description = explorer.get_resource_description(resource_type)

    if not fields:
        # try to find similar resources
        all_resources = explorer.get_all_resources()
        similar = [r for r in all_resources if resource_type.lower() in r.lower()]
        result = {
            "error": f"resource '{resource_type}' not found",
            "similar_resources": similar[:5],
            "total_resources": len(all_resources)
        }
        print(f"[linkml result] resource not found, suggested {len(similar)} similar")
        return result

    result = {
        "resource": resource_type,
        "fields": fields[:30],  # limit to first 30 fields
        "total_fields": len(fields),
        "description": description[:200] if description else "no description available"
    }
    print(f"[linkml result] found {len(fields)} fields for {resource_type}")
    return result


@tool
def search_fhir_fields(search_term: str, resource_filter: Optional[str] = None) -> List[Dict[str, str]]:
    """search for fhir fields matching a term across all resources.

    use this to find potential target fields when you're not sure which
    resource or field path to use.

    args:
        search_term: keyword to search for (e.g., "method", "date", "identifier")
        resource_filter: optional resource to filter results (e.g., "specimen")

    returns:
        list of matching field paths with their resources
    """
    from schema_crush.tools.fhir_schema_tool import get_schema_explorer

    filter_msg = f" in {resource_filter}" if resource_filter else " across all resources"
    print(f"[linkml query] search_fhir_fields('{search_term}'{filter_msg})")

    explorer = get_schema_explorer()
    results = explorer.search_fields(search_term, resource_filter)

    # limit results and format
    limited_results = results[:20]
    print(f"[linkml result] found {len(results)} matches, returning top {len(limited_results)}")
    return limited_results


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
MAPPING_TOOLS = [biobert_match, magneto_match, rule_match, explore_fhir_resource, search_fhir_fields]