"""tool definitions for claude agent to use matchers."""

from typing import List, Dict, Any, Optional
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