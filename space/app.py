"""schema crush as a hosted gradio mcp server.

wraps the same tool dispatcher used by the stdio mcp server
(schema_crush.mcp.server._execute_tool) so there is one implementation of each
tool. gradio derives the mcp schema from the type hints and docstrings below,
and serves it at /gradio_api/mcp/.

run locally:
    pip install -e ".[space]"
    python space/app.py

the mcp endpoint is then http://127.0.0.1:7860/gradio_api/mcp/
"""

import asyncio
import json
import os
from typing import Optional

import gradio as gr

from schema_crush import __version__
from schema_crush.mcp.server import TOOLS, _execute_tool

# record_feedback mutates a shared store. on a hosted multi-user space that is
# writable by every visitor, so it is opt-in via env var and off by default.
ENABLE_FEEDBACK_WRITES = os.getenv("ENABLE_FEEDBACK_WRITES", "").lower() in ("1", "true", "yes")

# profile_csv can call an llm, which would spend the space owner's api key on
# behalf of anonymous visitors. hosted default is off.
ALLOW_LLM_PROFILING = os.getenv("ALLOW_LLM_PROFILING", "").lower() in ("1", "true", "yes")

DOI = "10.5281/zenodo.22713617"
REPO = "https://github.com/teslajoy/schema_crush"


def _run(tool: str, **kwargs) -> str:
    """execute a tool through the shared dispatcher and json-encode the result."""
    cleaned = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    try:
        result = asyncio.run(_execute_tool(tool, cleaned))
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    try:
        return json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        return str(result)


def _split(text: str) -> list[str]:
    """parse a comma or newline separated list into a list of strings."""
    if not text:
        return []
    parts = [p.strip() for chunk in text.split("\n") for p in chunk.split(",")]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# mapping cascade
#
# quality comes from the ordering, not from any single matcher. curated
# knowledge answers most real columns exactly; embeddings are a last resort
# because raw similarity between short field names barely separates correct
# from incorrect targets.
# ---------------------------------------------------------------------------

# fuzzy_lookup scores single-token overlap as 0.6 * overlap / max(tokens), so one
# shared token out of two lands on exactly 0.30 - and its own filter is
# `if score < 0.3: continue`, so the weakest possible match passes by a hair.
# That is how PatientNum -> Group.identifier and OS_days -> Patient.birthDate got
# emitted, both at a confident-looking 0.300.
#
# 0.55 admits only near-complete matches: exact (1.0), substring, stem (0.65-0.70),
# and full token overlap (0.60). Partial-token noise is excluded.
FUZZY_ACCEPT = 0.55

# calibrated scores estimate empirical accuracy, so this floor reads directly:
# below it, the suggestion is worse than a coin flip and is reported as
# unmapped rather than dressed up as a mapping. biobert on bare field names
# routinely lands near 0.06, and at that level it returns the SAME target for
# unrelated columns, so a low floor manufactures confident-looking noise.
EMBED_ACCEPT = 0.50


def _candidate_targets(kb, limit: int = 400) -> list[str]:
    """distinct fhir destinations in the knowledge base, as embedding candidates."""
    global _CANDIDATES
    if _CANDIDATES is None:
        try:
            conn = kb.db.conn if hasattr(kb.db, "conn") else None
            if conn is None:
                import sqlite3
                from schema_crush import get_package_path
                conn = sqlite3.connect(get_package_path("data/db/flat_mappings.db"))
            cur = conn.cursor()
            cur.execute(
                "SELECT destination, COUNT(*) n FROM destinations "
                "WHERE destination LIKE '%.%' GROUP BY destination ORDER BY n DESC LIMIT ?",
                (limit,),
            )
            _CANDIDATES = [r[0] for r in cur.fetchall()]
        except Exception:
            _CANDIDATES = []
    return _CANDIDATES


_CANDIDATES = None


def _map_one(kb, column: str, samples: list) -> dict:
    """map a single column through the cascade, returning the first stage that answers."""
    base = {"source": column, "target": None, "confidence": 0.0,
            "stage": "unmapped", "note": ""}

    # stage 1: exact curated lookup
    try:
        hits = kb.lookup(column)
    except Exception:
        hits = []
    if hits:
        _, dest = hits[0]
        alts = sorted({d.destination for _, d in hits})
        return {**base, "target": dest.destination, "confidence": 1.0,
                "stage": "knowledge_base",
                "note": "exact curated mapping"
                        + (f"; {len(alts)} distinct targets for this term" if len(alts) > 1 else "")}

    # stage 2: fuzzy over curated field names
    try:
        fuzzy = kb.fuzzy_lookup(column, limit=5)
    except Exception:
        fuzzy = []
    # prefer an actual field path over a bare resource name: "Observation" alone
    # is not a mapping anyone can act on, "Observation.valueQuantity" is.
    usable = [f for f in fuzzy
              if f.get("score", 0) >= FUZZY_ACCEPT and "." in str(f.get("target", ""))]
    if usable:
        best = usable[0]
        return {**base, "target": best.get("target"),
                "confidence": round(float(best.get("score", 0)), 3),
                "stage": "fuzzy",
                "note": f"name similar to curated term '{best.get('source')}'; "
                        "score is string similarity, not calibrated accuracy"}

    # stage 3: calibrated embedding similarity
    try:
        from schema_crush.tools.matchers import BioBERTMatcher
        global _EMBEDDER
        if _EMBEDDER is None:
            _EMBEDDER = BioBERTMatcher(use_expert_embeddings=True)
        cands = _candidate_targets(kb)
        if cands:
            scored = _EMBEDDER.match(column, cands)
            if scored:
                tgt, raw = scored[0]
                cal = kb.calibrate("biobert", float(raw), "field")
                if cal >= EMBED_ACCEPT:
                    return {**base, "target": tgt, "confidence": round(float(cal), 3),
                            "stage": "embedding",
                            "note": "calibrated biobert; review before use"}
                return {**base, "stage": "unmapped",
                        "note": f"best embedding guess {tgt} calibrated to {cal:.3f}, "
                                "below the accuracy floor"}
    except Exception as exc:
        return {**base, "stage": "unmapped", "note": f"embedding stage failed: {exc}"}

    return {**base, "note": "no curated, fuzzy, or embedding match; needs human review"}


_EMBEDDER = None


def map_with_agent(file_obj, api_key: str, pasted_columns: str = "", max_cols: int = 40):
    """map columns with ClaudeAgent, using a key the visitor supplies.

    the key is used for this request only: it is passed to the agent, never
    written to disk, never logged, and never kept in module state.
    """
    key = (api_key or "").strip()
    if not key:
        return [], "", "Paste an Anthropic API key to run the agent. It is used for this request only."
    if not key.startswith("sk-ant-"):
        return [], "", "That does not look like an Anthropic API key (they start with `sk-ant-`)."

    headers, samples = [], {}
    if file_obj is not None:
        import csv as _csv
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
        try:
            with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
                reader = _csv.reader(fh)
                headers = [h.strip() for h in next(reader) if h and h.strip()]
                samples = {h: [] for h in headers}
                for i, row in enumerate(reader):
                    if i >= 200:
                        break
                    for j, val in enumerate(row):
                        if j < len(headers) and val and len(samples[headers[j]]) < 3:
                            samples[headers[j]].append(val.strip()[:60])
        except Exception as exc:
            return [], "", f"could not parse that csv: {exc}"
    else:
        headers = _split(pasted_columns)

    if not headers:
        return [], "", "Upload a csv or paste column names."
    if len(headers) > max_cols:
        return [], "", f"That csv has {len(headers)} columns; this demo maps at most {max_cols}."

    try:
        from schema_crush.orchestrator.agents import ClaudeAgent
        agent = ClaudeAgent(task="field", api_key=key)
    except Exception as exc:
        return [], "", f"could not start the agent: {type(exc).__name__}: {exc}"

    rows = []
    for col in headers:
        try:
            proposals = agent.propose_mappings(
                col,
                candidate_targets=None,
                context={"sample_values": samples.get(col, [])},
            )
            if proposals:
                p = proposals[0]
                rows.append({
                    "source": col,
                    "sample_values": ", ".join(samples.get(col, [])[:3]),
                    "target": p.target_field,
                    "confidence": round(float(p.confidence), 3),
                    "reasoning": (p.reasoning or "")[:300],
                })
            else:
                rows.append({"source": col, "sample_values": "", "target": None,
                             "confidence": 0.0, "reasoning": "no proposal"})
        except Exception as exc:
            rows.append({"source": col, "sample_values": "", "target": None,
                         "confidence": 0.0, "reasoning": f"{type(exc).__name__}: {exc}"})

    del agent, key

    table = [[r["source"], r["sample_values"], r["target"] or "—",
              f"{r['confidence']:.3f}" if r["confidence"] else "—", r["reasoning"]]
             for r in rows]
    mapped = sum(1 for r in rows if r["target"])
    summary = (f"**{mapped} of {len(rows)} columns mapped ({100*mapped/len(rows):.0f}%)** "
               "by the agent. Key discarded.")
    return table, json.dumps(rows, indent=2, default=str), summary


def map_uploaded_csv(file_obj, max_rows_scanned: int = 200):
    """read an uploaded csv, map every column, return (table, json, summary).

    only the header row and a few sample values are read. no row data is stored
    or logged.
    """
    empty = [], "", ""
    if file_obj is None:
        return [], "", "upload a csv to begin."

    import csv as _csv
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        return [], "", "could not read the uploaded file."

    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = _csv.reader(fh)
            try:
                headers = next(reader)
            except StopIteration:
                return [], "", "that csv appears to be empty."
            headers = [h.strip() for h in headers if h and h.strip()]
            if not headers:
                return [], "", "no column headers found in that csv."
            samples = {h: [] for h in headers}
            for i, row in enumerate(reader):
                if i >= max_rows_scanned:
                    break
                for j, val in enumerate(row):
                    if j < len(headers) and val and len(samples[headers[j]]) < 3:
                        samples[headers[j]].append(val.strip()[:60])
    except Exception as exc:
        return [], "", f"could not parse that csv: {exc}"

    rows = _map_columns(headers, samples)

    table = [[
        r["source"],
        r.get("sample_values", ""),
        r["target"] or "—",
        f"{r['confidence']:.3f}" if r["confidence"] else "—",
        r["stage"],
        r["note"],
    ] for r in rows]

    by_stage: dict = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    mapped = sum(1 for r in rows if r["target"])
    pct = (100 * mapped / len(rows)) if rows else 0
    summary = (
        f"**{mapped} of {len(rows)} columns mapped ({pct:.0f}%)**  ·  "
        + "  ·  ".join(f"{k}: {v}" for k, v in sorted(by_stage.items()))
        + "\n\nColumns marked `unmapped` had no curated match and scored below the "
        "calibrated accuracy floor. They are left blank on purpose rather than "
        "filled with a low-confidence guess."
    )

    payload = json.dumps(
        {"columns": len(rows), "mapped": mapped, "coverage_by_stage": by_stage,
         "mappings": rows},
        indent=2, default=str,
    )
    return table, payload, summary


def _map_columns(columns: list[str], samples: dict | None = None) -> list[dict]:
    """run the cascade over every column."""
    from schema_crush.orchestrator.agents.tools import get_knowledge_base
    kb = get_knowledge_base()
    samples = samples or {}
    out = []
    for col in columns:
        row = _map_one(kb, col, samples.get(col, []))
        vals = samples.get(col, [])
        if vals:
            row["sample_values"] = ", ".join(str(v) for v in vals[:3])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# matcher tools
# ---------------------------------------------------------------------------

@gr.mcp.tool()
def biobert_match(source: str, candidates: str, tier: Optional[str] = None) -> str:
    """Match a biomedical term against candidate FHIR targets using BioBERT embeddings.

    Best for matching biomedical concepts and terminology where the source term
    carries clinical meaning. Returns calibrated confidence scores, not raw
    cosine similarity: scores are mapped through an isotonic calibrator fitted
    on 1,280 expert mapping rules, so 0.8 means roughly 80% empirical accuracy.

    Args:
        source: Source field or term to match, for example "primary_diagnosis".
        candidates: Candidate target fields, comma or newline separated,
            for example "Condition.code, Patient.id, Observation.value".
        tier: Optional tier for tier-specific calibration. One of
            "entity", "field", or "content".

    Returns:
        JSON list of candidates with calibrated confidence scores, best first.
    """
    return _run("biobert_match", source=source, candidates=_split(candidates), tier=tier)


@gr.mcp.tool()
def magneto_match(source: str, candidates: str, tier: Optional[str] = None) -> str:
    """Match a schema field name against candidates using Magneto structural embeddings.

    Best for matching field names by structure rather than clinical meaning.
    Trained on GDC to FHIR mappings. Returns calibrated confidence scores.

    Requires the optional magneto package, which is not on PyPI. If it is not
    installed this returns an error explaining how to install it; biobert_match
    and rule_match work without it.

    Args:
        source: Source field name to match, for example "case_id".
        candidates: Candidate target field names, comma or newline separated.
        tier: Optional tier for tier-specific calibration. One of
            "entity", "field", or "content".

    Returns:
        JSON list of candidates with calibrated confidence scores, best first.
    """
    return _run("magneto_match", source=source, candidates=_split(candidates), tier=tier)


@gr.mcp.tool()
def rule_match(source: str, candidates: str, tier: Optional[str] = None) -> str:
    """Match a term against candidates using the curated expert knowledge base.

    Highest precision of the three matchers because it draws on expert-curated
    HTAN and GDC mappings rather than inference: 94.7% accuracy on field
    matching and 99.6% on content. Try this before the embedding matchers.

    Args:
        source: Source field or entity name, for example "tumor_grade".
        candidates: Candidate target fields or resources, comma or newline separated.
        tier: Optional tier for tier-specific calibration. One of
            "entity", "field", or "content".

    Returns:
        JSON list of candidates with calibrated confidence scores, best first.
    """
    return _run("rule_match", source=source, candidates=_split(candidates), tier=tier)


# ---------------------------------------------------------------------------
# knowledge base lookup
# ---------------------------------------------------------------------------

@gr.mcp.tool()
def lookup_mapping(
    source_term: str,
    context: Optional[str] = None,
    schema: Optional[str] = None,
    include_content: bool = False,
) -> str:
    """Look up every curated mapping for a source term. Exact match, O(1).

    Searches the flat mapping database of expert-curated GDC and HTAN mappings
    and returns all matches rather than a ranked guess. Use this first: if the
    term is already curated, no inference is needed.

    Args:
        source_term: Source field name, for example "sample_id" or "diagnosis".
        context: Optional context for disambiguation, for example "demographic".
        schema: Optional schema filter, for example "gdc" or "htan".
        include_content: Also return content value mappings, for example the
            code that "G1" maps to in SNOMED.

    Returns:
        JSON list of every curated mapping for the term.
    """
    return _run(
        "lookup_mapping",
        source_term=source_term,
        context=context,
        schema=schema,
        include_content=include_content,
    )


@gr.mcp.tool()
def find_similar(query: str, k: int = 5) -> str:
    """Find semantically similar mappings by vector search over the knowledge base.

    Use when lookup_mapping returns nothing and you want precedent: the nearest
    curated mappings serve as few-shot examples for an uncertain term.

    Args:
        query: Term to find similar curated mappings for.
        k: Number of results to return.

    Returns:
        JSON list of similar mappings with similarity scores.
    """
    return _run("find_similar", query=query, k=k)


@gr.mcp.tool()
def get_transformation_rules(section: str = "all") -> str:
    """Get FHIR transformation rules consolidated from GDC, CDA, HTAN, and ICGC.

    Explains the correct FHIR paths for common biomedical data patterns such as
    cancer staging, demographics, and specimen handling. Read these before
    proposing a mapping for anything non-obvious.

    Args:
        section: Section filter. One of "patient", "condition", "staging",
            "snomed", "grade", "observation", "specimen", "document",
            "medication", "codes", "entities", or "all".

    Returns:
        The transformation rules for the requested section, as text.
    """
    return _run("get_transformation_rules", section=section)


# ---------------------------------------------------------------------------
# fhir schema exploration
# ---------------------------------------------------------------------------

@gr.mcp.tool()
def explore_fhir_resource(resource_type: str) -> str:
    """List the available fields on a FHIR R5 resource.

    Use to discover what a resource actually offers before mapping onto it,
    rather than guessing a field name that may not exist.

    Args:
        resource_type: FHIR resource name, for example "Patient", "Specimen",
            "Observation", or "Condition".

    Returns:
        JSON describing the resource and its fields.
    """
    return _run("explore_fhir_resource", resource_type=resource_type)


@gr.mcp.tool()
def search_fhir_fields(search_term: str, resource_filter: Optional[str] = None) -> str:
    """Search for FHIR fields matching a keyword across all resources.

    Use when you know what a field means but not which resource holds it.

    Args:
        search_term: Keyword to search for, for example "method", "date",
            or "identifier".
        resource_filter: Optional resource to restrict results to, for
            example "Specimen".

    Returns:
        JSON list of matching FHIR field paths.
    """
    return _run("search_fhir_fields", search_term=search_term, resource_filter=resource_filter)


# ---------------------------------------------------------------------------
# terminology
# ---------------------------------------------------------------------------

@gr.mcp.tool()
def search_snomed(query: str, limit: int = 10) -> str:
    """Search SNOMED CT for clinical concept codes.

    Best for diagnoses, clinical findings, procedures, and body structures.

    Args:
        query: Search term, for example "adenocarcinoma" or "diabetes".
        limit: Maximum number of results.

    Returns:
        JSON list of SNOMED codes with display names.
    """
    return _run("search_snomed", query=query, limit=limit)


@gr.mcp.tool()
def search_loinc(query: str, limit: int = 10) -> str:
    """Search LOINC for laboratory and observation codes.

    Best for lab tests, observation codes, and clinical measurements.

    Args:
        query: Search term, for example "glucose" or "hemoglobin A1c".
        limit: Maximum number of results.

    Returns:
        JSON list of LOINC codes with display names.
    """
    return _run("search_loinc", query=query, limit=limit)


@gr.mcp.tool()
def search_ontology(query: str, ontology: Optional[str] = None, limit: int = 10) -> str:
    """Search biomedical ontologies via EBI OLS4.

    Covers HPO, MONDO, NCIt, UBERON, GO and others. Use for phenotypes,
    diseases, anatomy, and cancer terminology that is not in SNOMED or LOINC.

    Args:
        query: Search term.
        ontology: Optional ontology filter, for example "mondo", "ncit",
            "hpo", or "uberon".
        limit: Maximum number of results.

    Returns:
        JSON list of ontology terms with identifiers and definitions.
    """
    return _run("search_ontology", query=query, ontology=ontology, limit=limit)


# ---------------------------------------------------------------------------
# profiling and feedback
# ---------------------------------------------------------------------------

@gr.mcp.tool()
def profile_csv(columns: str, sample_data: str = "", use_llm: bool = False) -> str:
    """Analyze CSV columns to infer use case, column groups, and mapping recommendations.

    Run this before mapping a new dataset. It interprets cryptic column names
    such as "death_event_1death_0censor" and reports the likely use case
    (survival analysis, variant tracking, and so on) plus per-column content
    types and FHIR recommendations.

    Args:
        columns: Column names, comma or newline separated.
        sample_data: Optional JSON object mapping each column name to a list of
            3 to 5 sample values, for example
            {"grade": ["G1", "G2"], "age": ["45", "67"]}.
        use_llm: Use an LLM for richer interpretation of cryptic names. On the
            hosted Space this is disabled unless the operator enables it.

    Returns:
        JSON profile with use case, column groups, content types, and
        recommendations.
    """
    parsed: dict = {}
    if sample_data:
        try:
            parsed = json.loads(sample_data)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"sample_data is not valid JSON: {exc}"}, indent=2)

    if use_llm and not ALLOW_LLM_PROFILING:
        return json.dumps(
            {
                "error": "llm profiling is disabled on this deployment",
                "hint": "call with use_llm=false, or set ALLOW_LLM_PROFILING=1 and "
                        "ANTHROPIC_API_KEY if you are the operator",
            },
            indent=2,
        )

    return _run("profile_csv", columns=_split(columns), sample_data=parsed, use_llm=use_llm)


@gr.mcp.tool()
def map_csv_columns(columns: str, sample_values: str = "") -> str:
    """Map a whole set of CSV column names to FHIR R5 paths in one call.

    Runs the full matching cascade per column and stops at the first stage that
    answers, which is what makes the result good:

      1. knowledge base exact lookup, confidence 1.0
      2. fuzzy match over curated field names
      3. calibrated BioBERT embedding similarity
      4. unmapped, flagged for human review

    Prefer this over calling biobert_match yourself. A single matcher used alone
    performs badly on field names: raw embedding scores for unrelated FHIR paths
    cluster within a few points of each other, so the wrong target often ranks
    first. The cascade avoids that by trying curated knowledge first, which
    resolves the large majority of real-world columns exactly.

    Uses no LLM and needs no API key.

    Args:
        columns: Column names, comma or newline separated.
        sample_values: Optional JSON object mapping column name to a list of
            example values, for example {"grade": ["G1", "G2"]}. Improves
            content-tier hints.

    Returns:
        JSON list of {source, target, confidence, stage, note}, one per column,
        plus a summary of coverage by stage.
    """
    cols = _split(columns)
    if not cols:
        return json.dumps({"error": "no column names provided"}, indent=2)

    samples: dict = {}
    if sample_values:
        try:
            samples = json.loads(sample_values)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"sample_values is not valid JSON: {exc}"}, indent=2)

    rows = _map_columns(cols, samples)
    summary: dict = {}
    for r in rows:
        summary[r["stage"]] = summary.get(r["stage"], 0) + 1
    return json.dumps({"mappings": rows, "coverage_by_stage": summary}, indent=2, default=str)


@gr.mcp.tool()
def get_feedback_stats() -> str:
    """Get statistics on collected human-in-the-loop feedback.

    Reports accept, reject, and correct counts so you can monitor whether the
    learning loop is receiving review signal.

    Returns:
        JSON summary of feedback counts and rates.
    """
    return _run("get_feedback_stats")


@gr.mcp.tool()
def record_feedback(
    source: str,
    proposed_target: str,
    decision: str,
    matcher: str,
    tier: str,
    confidence: float,
    ground_truth: Optional[str] = None,
) -> str:
    """Record a human review decision on a proposed mapping.

    Feeds the learning loop that recalibrates matcher confidence. Writes are
    disabled on the public Space because the store is shared across all
    visitors; run your own instance with ENABLE_FEEDBACK_WRITES=1 to collect
    feedback.

    Args:
        source: Source term that was mapped.
        proposed_target: The mapping that was proposed.
        decision: Your decision. One of "accept", "reject", or "correct".
        matcher: Which matcher proposed it: "biobert", "magneto", or "rule".
        tier: Mapping tier: "entity", "field", or "content".
        confidence: Confidence score at the time of the decision.
        ground_truth: The correct target, required when decision is "correct".

    Returns:
        JSON confirmation, or an error if writes are disabled.
    """
    if not ENABLE_FEEDBACK_WRITES:
        return json.dumps(
            {
                "error": "feedback writes are disabled on this deployment",
                "hint": "the feedback store is shared across all visitors. run your "
                        "own instance with ENABLE_FEEDBACK_WRITES=1 to record feedback.",
            },
            indent=2,
        )
    return _run(
        "record_feedback",
        source=source,
        proposed_target=proposed_target,
        decision=decision,
        matcher=matcher,
        tier=tier,
        confidence=confidence,
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------

INTRO = f"""
# schema crush

**Agentic MCP server for mapping heterogeneous biomedical schemas to HL7 FHIR R5.**

Calibrated matchers over expert-curated GDC and HTAN mappings. Confidence
scores are isotonic-calibrated on 1,280 expert rules, so a 0.8 means roughly
80% empirical accuracy rather than an uninterpretable similarity number.

**Connect an MCP client to:** `/gradio_api/mcp/`
&nbsp;&nbsp;·&nbsp;&nbsp; [source]({REPO}) &nbsp;·&nbsp; [archive](https://doi.org/{DOI}) &nbsp;·&nbsp; v{__version__}

Research use only. Not for clinical decision-making.
"""


# theme belongs in Blocks() on gradio 5.x. gradio 6 moves it to launch(), which
# is what the DeprecationWarning in the space logs is announcing; launch() does
# not accept it yet on 5.50, so passing it there raises TypeError.
with gr.Blocks(title="schema crush", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tabs():
        with gr.Tab("map a csv"):
            gr.Markdown(
                "**Upload a CSV and get FHIR R5 mappings for every column.**\n\n"
                "Only the header row and up to three example values per column are "
                "read. Row data is never stored or logged. Do not upload PHI.\n\n"
                "Each column runs through the cascade and stops at the first stage "
                "that answers: curated knowledge base, then fuzzy name match, then "
                "calibrated embeddings. Columns that no stage can answer are left "
                "blank rather than filled with a guess."
            )
            csv_in = gr.File(label="csv file", file_types=[".csv", ".tsv", ".txt"])
            map_btn = gr.Button("map columns", variant="primary")
            csv_summary = gr.Markdown()
            csv_table = gr.Dataframe(
                headers=["column", "sample values", "FHIR target", "confidence", "stage", "note"],
                datatype=["str"] * 6,
                wrap=True,
                label="mappings",
            )
            csv_json = gr.Code(label="json", language="json")
            map_btn.click(
                map_uploaded_csv,
                inputs=csv_in,
                outputs=[csv_table, csv_json, csv_summary],
            )
            gr.Markdown(
                "---\n**No file handy?** Paste column names instead. This is also the "
                "`map_csv_columns` tool that an MCP client calls."
            )
            paste_cols = gr.Textbox(
                label="column names (comma or newline separated)",
                value="sample_id, tumor_subtype, survival_months, gender, primary_diagnosis",
            )
            paste_samples = gr.Textbox(
                label="sample values (optional JSON object)",
                value='{"tumor_subtype": ["basal", "classical"]}',
            )
            paste_out = gr.Code(label="json", language="json")
            gr.Button("map these columns").click(
                map_csv_columns, [paste_cols, paste_samples], paste_out
            )

            gr.Markdown(
                "---\n"
                "**Want the cryptic columns resolved too?** Names like "
                "`death_event_1death_0censor` need a model to interpret them. "
                "Connect this Space to Claude Desktop as an MCP server and your own "
                "Claude does that reasoning, using these 14 tools. See the About "
                "section below for the config."
            )

        with gr.Tab("map with agent"):
            gr.Markdown(
                "**The full pipeline, including the reasoning stage.**\n\n"
                "The free tab above answers from curated knowledge only, so columns "
                "it has never seen come back blank. This tab runs the agent over the "
                "same 14 tools, which is what resolves names like `OS_days`, "
                "`CA19_9_UperML`, or `death_event_1death_0censor`.\n\n"
                "It needs an Anthropic API key, which **you** supply. The key is used "
                "for this request only: never written to disk, never logged, never "
                "kept after the response. If you would rather not paste a key, connect "
                "this Space to Claude Desktop instead and your own Claude does the "
                "same reasoning (config in the About section)."
            )
            ag_key = gr.Textbox(
                label="your Anthropic API key (sk-ant-…)",
                type="password",
                placeholder="sk-ant-…",
            )
            ag_file = gr.File(label="csv file", file_types=[".csv", ".tsv", ".txt"])
            ag_cols = gr.Textbox(
                label="or paste column names",
                placeholder="OS_days, CA19_9_UperML, NeoadjuvantYN",
            )
            ag_btn = gr.Button("map with agent", variant="primary")
            ag_summary = gr.Markdown()
            ag_table = gr.Dataframe(
                headers=["column", "sample values", "FHIR target", "confidence", "reasoning"],
                datatype=["str"] * 5,
                wrap=True,
                label="agent mappings",
            )
            ag_json = gr.Code(label="json", language="json")
            # show_api=False keeps this out of the MCP schema: an mcp tool that
            # takes an api key would invite clients to pass one over the wire.
            ag_btn.click(
                map_with_agent,
                inputs=[ag_file, ag_key, ag_cols],
                outputs=[ag_table, ag_json, ag_summary],
                show_api=False,
            )

        with gr.Tab("lookup"):
            gr.Markdown("*Look up every curated mapping for a source term. Start here.*")
            lu_term = gr.Textbox(label="source term", value="primary_diagnosis")
            lu_ctx = gr.Textbox(label="context (optional)")
            lu_schema = gr.Textbox(label="schema filter (optional)", placeholder="gdc or htan")
            lu_content = gr.Checkbox(label="include content value mappings")
            lu_out = gr.Code(label="result", language="json")
            gr.Button("look up", variant="primary").click(
                lookup_mapping, [lu_term, lu_ctx, lu_schema, lu_content], lu_out
            )
            gr.Examples(
                examples=[["primary_diagnosis", "", "", False],
                          ["tumor_grade", "", "gdc", True],
                          ["sample_id", "sample", "", False]],
                inputs=[lu_term, lu_ctx, lu_schema, lu_content],
            )

            gr.Markdown("---\n*No exact match? Find the nearest curated mappings instead.*")
            sim_k = gr.Slider(1, 20, value=5, step=1, label="how many similar mappings")
            gr.Button("find_similar").click(find_similar, [lu_term, sim_k], lu_out)

        with gr.Tab("match"):
            gr.Markdown("*Score candidate FHIR targets with a calibrated matcher.*")
            m_source = gr.Textbox(label="source term", value="tumor_grade")
            m_cands = gr.Textbox(
                label="candidate targets (comma separated)",
                value="Observation.valueCodeableConcept, Condition.stage, Patient.id",
            )
            m_tier = gr.Dropdown(["", "entity", "field", "content"], label="tier", value="")
            m_out = gr.Code(label="result", language="json")
            with gr.Row():
                gr.Button("rule_match", variant="primary").click(
                    rule_match, [m_source, m_cands, m_tier], m_out
                )
                gr.Button("biobert_match").click(
                    biobert_match, [m_source, m_cands, m_tier], m_out
                )
                gr.Button("magneto_match").click(
                    magneto_match, [m_source, m_cands, m_tier], m_out
                )

        with gr.Tab("explore fhir"):
            gr.Markdown("*Discover what a FHIR resource offers, or find a field by keyword.*")
            e_res = gr.Textbox(label="resource type", value="Specimen")
            e_out = gr.Code(label="result", language="json")
            gr.Button("explore resource", variant="primary").click(
                explore_fhir_resource, e_res, e_out
            )
            s_term = gr.Textbox(label="field keyword", value="identifier")
            s_filter = gr.Textbox(label="restrict to resource (optional)")
            gr.Button("search fields").click(
                search_fhir_fields, [s_term, s_filter], e_out
            )

        with gr.Tab("terminology"):
            gr.Markdown("*Search SNOMED CT, LOINC, and the OLS4 ontologies.*")
            t_query = gr.Textbox(label="query", value="adenocarcinoma")
            t_limit = gr.Slider(1, 25, value=10, step=1, label="max results")
            t_onto = gr.Textbox(label="ontology filter (ontology search only)",
                                placeholder="mondo, ncit, hpo, uberon")
            t_out = gr.Code(label="result", language="json")
            with gr.Row():
                gr.Button("search_snomed", variant="primary").click(
                    search_snomed, [t_query, t_limit], t_out
                )
                gr.Button("search_loinc").click(search_loinc, [t_query, t_limit], t_out)
                gr.Button("search_ontology").click(
                    search_ontology, [t_query, t_onto, t_limit], t_out
                )

        with gr.Tab("profile csv"):
            gr.Markdown("*Infer use case and mapping recommendations from column names.*")
            p_cols = gr.Textbox(
                label="column names (comma separated)",
                value="case_id, primary_diagnosis, tumor_grade, death_event_1death_0censor",
            )
            p_samples = gr.Textbox(
                label="sample values (optional JSON object)",
                value='{"tumor_grade": ["G1", "G2", "G3"]}',
            )
            p_out = gr.Code(label="result", language="json")
            gr.Button("profile", variant="primary").click(
                profile_csv, [p_cols, p_samples], p_out
            )

        with gr.Tab("rules"):
            gr.Markdown("*FHIR transformation rules consolidated from GDC, CDA, HTAN, ICGC.*")
            r_section = gr.Dropdown(
                ["all", "patient", "condition", "staging", "snomed", "grade",
                 "observation", "specimen", "document", "medication", "codes", "entities"],
                label="section", value="staging",
            )
            r_out = gr.Code(label="result")
            gr.Button("get rules", variant="primary").click(
                get_transformation_rules, r_section, r_out
            )

        with gr.Tab("feedback"):
            gr.Markdown(
                "*The human-in-the-loop learning loop. Stats are readable here; "
                "writes are disabled on the public deployment because the store is "
                "shared across all visitors.*"
            )
            fb_out = gr.Code(label="result", language="json")
            gr.Button("get_feedback_stats", variant="primary").click(
                get_feedback_stats, None, fb_out
            )

            gr.Markdown("---\n*Record a review decision (requires a self-hosted instance).*")
            with gr.Row():
                fb_source = gr.Textbox(label="source term", value="tumor_grade")
                fb_target = gr.Textbox(
                    label="proposed target", value="Observation.valueCodeableConcept"
                )
            with gr.Row():
                fb_decision = gr.Dropdown(
                    ["accept", "reject", "correct"], label="decision", value="accept"
                )
                fb_matcher = gr.Dropdown(
                    ["rule", "biobert", "magneto"], label="matcher", value="rule"
                )
                fb_tier = gr.Dropdown(
                    ["entity", "field", "content"], label="tier", value="field"
                )
            with gr.Row():
                fb_conf = gr.Slider(0.0, 1.0, value=0.85, step=0.01, label="confidence")
                fb_truth = gr.Textbox(label="ground truth (if decision is 'correct')")
            gr.Button("record_feedback").click(
                record_feedback,
                [fb_source, fb_target, fb_decision, fb_matcher, fb_tier, fb_conf, fb_truth],
                fb_out,
            )

    gr.Markdown(
        f"`{len(TOOLS)}` tools exposed over MCP. "
        "Feedback writes and LLM profiling are disabled on this deployment."
    )


if __name__ == "__main__":
    # ssr_mode=False: gradio 5's server-side rendering runs a node sidecar that
    # adds a failure mode and buys nothing here, since every view is rendered
    # from a tool call rather than prefetched.
    demo.launch(mcp_server=True, ssr_mode=False)