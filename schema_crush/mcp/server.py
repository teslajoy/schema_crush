"""MCP server for schema_crush - exposes mapping tools to Claude Desktop/Code."""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from schema_crush import __version__


# create server instance
server = Server("schema-crush")

# lazy-loaded singletons
_knowledge_base = None
_matchers = {}


def get_knowledge_base():
    """get singleton knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        from schema_crush.learning.knowledge_base import KnowledgeBase
        _knowledge_base = KnowledgeBase.load()
    return _knowledge_base


def get_matcher(name: str):
    """get cached matcher instance."""
    global _matchers
    if name not in _matchers:
        kb = get_knowledge_base()
        if name == "biobert":
            from schema_crush.tools.matchers import BioBERTMatcher
            _matchers[name] = BioBERTMatcher(use_expert_embeddings=True)
        elif name == "magneto":
            from schema_crush.tools.matchers import MagnetoMatcher
            if MagnetoMatcher is None:
                raise RuntimeError(
                    "MagnetoMatcher is unavailable. magneto is not on pypi and must be "
                    "installed from source:\n"
                    "  pip install 'magneto @ git+https://github.com/VIDA-NYU/"
                    "magneto-matcher.git@main#subdirectory=algorithms/magneto'\n"
                    "biobert_match and rule_match work without it."
                )
            _matchers[name] = MagnetoMatcher(use_expert_embeddings=True)
        elif name == "rule":
            from schema_crush.tools.matchers import RuleMatcher
            _matchers[name] = RuleMatcher(kb.db)
    return _matchers.get(name)


def apply_calibration(matcher_name: str, raw_score: float, tier: str = None) -> float:
    """apply calibration to raw matcher score."""
    kb = get_knowledge_base()
    return kb.calibrate(matcher_name, raw_score, tier)


# ============================================================================
# tool definitions
# ============================================================================

TOOLS = [
    types.Tool(
        name="biobert_match",
        description="Use BioBERT embeddings for biomedical semantic similarity matching. Best for matching biomedical concepts and terminology. Returns calibrated confidence scores.",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source field or term to match"
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate target fields/terms"
                },
                "tier": {
                    "type": "string",
                    "enum": ["entity", "field", "content"],
                    "description": "Optional tier for tier-specific calibration"
                }
            },
            "required": ["source", "candidates"]
        }
    ),
    types.Tool(
        name="magneto_match",
        description="Use Magneto embeddings for schema structure matching. Best for matching field names (trained on GDC->FHIR). Returns calibrated confidence scores.",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source field name to match"
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate target field names"
                },
                "tier": {
                    "type": "string",
                    "enum": ["entity", "field", "content"],
                    "description": "Optional tier for tier-specific calibration"
                }
            },
            "required": ["source", "candidates"]
        }
    ),
    types.Tool(
        name="rule_match",
        description="Use knowledge base rules for mapping lookup. Best for leveraging existing expert mappings from HTAN/GDC. 94.7% accuracy on field matching, 99.6% on content.",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source field or entity name"
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate target fields/resources"
                },
                "tier": {
                    "type": "string",
                    "enum": ["entity", "field", "content"],
                    "description": "Optional tier for tier-specific calibration"
                }
            },
            "required": ["source", "candidates"]
        }
    ),
    types.Tool(
        name="lookup_mapping",
        description="Direct O(1) lookup in the flat mapping database. Returns ALL matches from curated GDC/HTAN mappings.",
        inputSchema={
            "type": "object",
            "properties": {
                "source_term": {
                    "type": "string",
                    "description": "Source field name (e.g., 'sample_id', 'diagnosis')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context for disambiguation (e.g., 'demographic', 'sample')"
                },
                "schema": {
                    "type": "string",
                    "description": "Optional schema filter (e.g., 'gdc', 'htan')"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Include content value mappings (ex, 'G1' -> SNOMED code)",
                    "default": False
                }
            },
            "required": ["source_term"]
        }
    ),
    types.Tool(
        name="find_similar",
        description="Semantic search in vector store for similar mappings. Good for finding examples when exact match doesn't exist.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query term to find similar mappings for"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="explore_fhir_resource",
        description="Explore a FHIR resource to see its available fields. Use when you need to discover what fields are available before mapping.",
        inputSchema={
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "description": "FHIR resource name (e.g., 'Patient', 'Specimen', 'Observation')"
                }
            },
            "required": ["resource_type"]
        }
    ),
    types.Tool(
        name="search_fhir_fields",
        description="Search for FHIR fields matching a term across all resources.",
        inputSchema={
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Keyword to search for (e.g., 'method', 'date', 'identifier')"
                },
                "resource_filter": {
                    "type": "string",
                    "description": "Optional resource to filter results (e.g., 'Specimen')"
                }
            },
            "required": ["search_term"]
        }
    ),
    types.Tool(
        name="search_snomed",
        description="Search SNOMED CT codes. Best for diagnosis codes, clinical findings, procedures, body structures.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (e.g., 'adenocarcinoma', 'diabetes')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="search_loinc",
        description="Search LOINC codes. Best for lab tests, observation codes, clinical measurements.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (e.g., 'glucose', 'hemoglobin A1c')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="search_ontology",
        description="Search biomedical ontologies via EBI OLS4 (HPO, MONDO, NCIt, UBERON, GO, etc.).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term"
                },
                "ontology": {
                    "type": "string",
                    "description": "Optional ontology filter (e.g., 'mondo', 'ncit', 'hpo', 'uberon')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="record_feedback",
        description="Record a HITL feedback decision for the learning loop. Use after reviewing a mapping.",
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source term that was mapped"
                },
                "proposed_target": {
                    "type": "string",
                    "description": "The mapping that was proposed"
                },
                "decision": {
                    "type": "string",
                    "enum": ["accept", "reject", "correct"],
                    "description": "Your decision on the mapping"
                },
                "ground_truth": {
                    "type": "string",
                    "description": "Correct target if decision is 'correct'"
                },
                "matcher": {
                    "type": "string",
                    "description": "Which matcher proposed this (biobert, magneto, rule)"
                },
                "tier": {
                    "type": "string",
                    "enum": ["entity", "field", "content"],
                    "description": "Mapping tier"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score at time of decision"
                }
            },
            "required": ["source", "proposed_target", "decision", "matcher", "tier", "confidence"]
        }
    ),
    types.Tool(
        name="get_feedback_stats",
        description="Get statistics about collected feedback for monitoring the learning loop.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="get_transformation_rules",
        description="Get FHIR transformation rules consolidated from GDC, CDA, HTAN, ICGC transformers. Use this to understand correct FHIR paths for biomedical data fields like staging, demographics, specimens.",
        inputSchema={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Optional section filter: 'patient', 'condition', 'staging', 'snomed', 'grade', 'observation', 'specimen', 'document', 'medication', 'codes', 'entities', or 'all'",
                    "default": "all"
                }
            },
            "required": []
        }
    ),
    types.Tool(
        name="profile_csv",
        description="Analyze a CSV to understand its use-case, column relationships, and get mapping recommendations. Use BEFORE mapping to get context about cryptic column names like 'death_event_1death_0censor'. Returns use-case (survival_analysis, variant_tracking, etc.), column groups, content types, and FHIR mapping recommendations.",
        inputSchema={
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of column names from the CSV"
                },
                "sample_data": {
                    "type": "object",
                    "description": "Dict mapping column names to list of sample values (3-5 values per column)",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "use_llm": {
                    "type": "boolean",
                    "description": "Use LLM for enhanced profiling (interprets cryptic names). Default true.",
                    "default": True
                }
            },
            "required": ["columns", "sample_data"]
        }
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """list available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """execute a tool and return results."""
    try:
        result = await _execute_tool(name, arguments)
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def _execute_tool(name: str, args: dict) -> Any:
    """execute tool by name."""

    # matcher tools
    if name == "biobert_match":
        matcher = get_matcher("biobert")
        results = matcher.match(args["source"], args["candidates"])
        tier = args.get("tier")
        return [
            {"target": tgt, "score": round(apply_calibration("biobert", score, tier), 3)}
            for tgt, score in results[:5]
        ]

    elif name == "magneto_match":
        matcher = get_matcher("magneto")
        results = matcher.match(args["source"], args["candidates"])
        tier = args.get("tier")
        return [
            {"target": tgt, "score": round(apply_calibration("magneto", score, tier), 3)}
            for tgt, score in results[:5]
        ]

    elif name == "rule_match":
        matcher = get_matcher("rule")
        results = matcher.match(args["source"], args["candidates"])
        tier = args.get("tier")
        return [
            {"target": tgt, "score": round(apply_calibration("rule", score, tier), 3)}
            for tgt, score in results[:5]
        ]

    # lookup tools
    elif name == "lookup_mapping":
        kb = get_knowledge_base()
        mappings = kb.db.lookup(
            args["source_term"],
            context=args.get("context"),
            schema=args.get("schema")
        )
        results = [
            {
                "source": src.source,
                "target": dest.destination,
                "schema": src.source_schema,
                "tier": src.tier.value,
                "context": src.source_context or ""
            }
            for src, dest in mappings  # no limit - return ALL
        ]

        # optionally include content value mappings
        if args.get("include_content"):
            content_mappings = kb.db.lookup_content(args["source_term"])
            for cv, targets in content_mappings:
                for cft in targets:
                    results.append({
                        "source": cv.source_value,
                        "target": cft.fhir_path,
                        "schema": cv.source_category,
                        "tier": "content",
                        "context": cft.context or "",
                        "code": cv.code,
                        "system": cv.system,
                        "display": cv.display
                    })

        return results

    elif name == "find_similar":
        kb = get_knowledge_base()
        if kb.vector_store:
            return kb.vector_store.find_similar(args["query"], k=args.get("k", 5))
        return []

    # FHIR exploration
    elif name == "explore_fhir_resource":
        from schema_crush.tools.fhir_schema_tool import get_schema_explorer
        explorer = get_schema_explorer()
        fields = explorer.get_resource_fields(args["resource_type"])
        description = explorer.get_resource_description(args["resource_type"])
        if not fields:
            all_resources = explorer.get_all_resources()
            similar = [r for r in all_resources if args["resource_type"].lower() in r.lower()]
            return {"error": f"resource not found", "similar": similar[:5]}
        return {
            "resource": args["resource_type"],
            "fields": fields[:30],
            "total_fields": len(fields),
            "description": description[:200] if description else None
        }

    elif name == "search_fhir_fields":
        from schema_crush.tools.fhir_schema_tool import get_schema_explorer
        explorer = get_schema_explorer()
        return explorer.search_fields(args["search_term"], args.get("resource_filter"))[:20]

    # terminology search
    elif name == "search_snomed":
        import requests
        url = "https://tx.fhir.org/r4/ValueSet/$expand"
        params = {
            "url": "http://snomed.info/sct?fhir_vs",
            "filter": args["query"],
            "count": args.get("limit", 10)
        }
        response = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=15)
        response.raise_for_status()
        data = response.json()
        return [
            {"code": item.get("code"), "display": item.get("display"), "system": "http://snomed.info/sct"}
            for item in data.get("expansion", {}).get("contains", [])
        ]

    elif name == "search_loinc":
        import requests
        url = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"
        params = {"terms": args["query"], "type": "question", "df": "text,LOINC_NUM", "count": args.get("limit", 10)}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        codes = data[1] if len(data) > 1 else []
        displays = data[3] if len(data) > 3 else []
        return [
            {"code": displays[i][1] if i < len(displays) else codes[i], "display": displays[i][0] if i < len(displays) else "", "system": "http://loinc.org"}
            for i, code in enumerate(codes)
        ]

    elif name == "search_ontology":
        import requests
        url = "https://www.ebi.ac.uk/ols4/api/search"
        params = {"q": args["query"], "rows": args.get("limit", 10), "format": "json"}
        if args.get("ontology"):
            params["ontology"] = args["ontology"].lower()
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return [
            {"code": doc.get("obo_id", doc.get("short_form")), "display": doc.get("label"), "ontology": doc.get("ontology_name")}
            for doc in data.get("response", {}).get("docs", [])
        ]

    # feedback tools
    elif name == "record_feedback":
        from schema_crush.learning import FeedbackStore, Decision
        fb = FeedbackStore()
        fb_id = fb.record(
            source=args["source"],
            proposed_target=args["proposed_target"],
            decision=Decision(args["decision"]),
            matcher=args["matcher"],
            tier=args["tier"],
            confidence=args["confidence"],
            ground_truth=args.get("ground_truth"),
        )
        return {"status": "recorded", "feedback_id": fb_id}

    elif name == "get_feedback_stats":
        from schema_crush.learning import FeedbackStore
        fb = FeedbackStore()
        return fb.stats()

    elif name == "get_transformation_rules":
        # import from claude_agent (single source of truth)
        from schema_crush.orchestrator.agents.claude_agent import TRANSFORMATION_RULES
        section = args.get("section", "all").lower()

        # section filters - map to markdown headers
        sections = {
            "patient": "### Patient Demographics",
            "condition": "### Condition (Diagnosis)",
            "staging": "### Stage/Grade Hierarchy",
            "hierarchy": "### Stage/Grade Hierarchy",
            "snomed": "### Staging SNOMED Codes",
            "grade": "### Grade Value SNOMED Codes",
            "observation": "### Observation Patterns",
            "specimen": "### Specimen Hierarchy",
            "document": "### DocumentReference",
            "medication": "### MedicationAdministration",
            "codes": "### Key Code Systems",
            "entities": "### Entity → Resource Mapping",
        }

        if section == "all":
            return TRANSFORMATION_RULES
        elif section in sections:
            # extract just that section
            start_marker = sections[section]
            lines = TRANSFORMATION_RULES.split("\n")
            result = []
            in_section = False
            for line in lines:
                if line.startswith(start_marker):
                    in_section = True
                elif line.startswith("### ") and in_section:
                    break
                if in_section:
                    result.append(line)
            return "\n".join(result) if result else f"Section '{section}' not found. Available: {', '.join(sections.keys())}, all"
        else:
            return f"Unknown section: {section}. Available: {', '.join(sections.keys())}, all"

    elif name == "profile_csv":
        from schema_crush.orchestrator.agents.csv_profiler import CSVProfiler
        use_llm = args.get("use_llm", True)
        profiler = CSVProfiler(use_llm=use_llm)
        profile = profiler.profile(args["columns"], args["sample_data"])

        # convert to serializable dict
        content_types = {}
        for col, ct in profile.content_types.items():
            content_types[col] = {
                "dtype": ct.dtype,
                "pattern": ct.pattern,
                "vocabulary": ct.vocabulary,
                "nullable": ct.nullable
            }

        return {
            "use_case": profile.use_case,
            "analysis_purpose": profile.analysis_purpose,
            "data_provenance": profile.data_provenance,
            "primary_entity": profile.primary_entity,
            "column_groups": profile.column_groups,
            "content_types": content_types,
            "column_relationships": profile.column_relationships,
            "recommendations": profile.recommendations
        }

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """run the MCP server."""
    from mcp.server.lowlevel.server import NotificationOptions

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="schema-crush",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )


def main_sync():
    """console_script entry point.

    main() is a coroutine, so it cannot be used as a console_script target
    directly: setuptools would call it and discard the un-awaited coroutine
    without ever starting the server.
    """
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()