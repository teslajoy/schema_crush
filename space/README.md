---
title: Schema Crush
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: MCP server mapping biomedical schemas to HL7 FHIR R5
tags:
  - mcp-server
  - fhir
  - biomedical-data
  - schema-matching
  - bioinformatics
---

# Schema Crush

**Agentic MCP server for mapping heterogeneous biomedical schemas to HL7 FHIR R5.**

Maps messy source schemas (GDC, HTAN, GEO, arbitrary CSVs) onto FHIR R5 using
calibrated matchers over an expert-curated knowledge base of 2,346 mappings.

## Connect an MCP client

```
https://<this-space>.hf.space/gradio_api/mcp/
```

For Claude Desktop, add to your MCP config:

```json
{
  "mcpServers": {
    "schema-crush": {
      "url": "https://<this-space>.hf.space/gradio_api/mcp/"
    }
  }
}
```

## Why calibrated confidence

Most matchers return a raw similarity score, and a 0.8 from a cosine
similarity tells you nothing about whether the match is actually right. Schema
Crush passes every score through an isotonic calibrator fitted on 1,280 expert
mapping rules, so a reported 0.8 corresponds to roughly 80% empirical accuracy.
That makes the score usable as a decision threshold: route low-confidence
mappings to human review instead of guessing.

## The 14 tools

**Knowledge base**
- `lookup_mapping` exact O(1) lookup in curated GDC/HTAN mappings, start here
- `find_similar` vector search for precedent when no exact match exists
- `get_transformation_rules` FHIR transformation rules from GDC, CDA, HTAN, ICGC

**Matchers** (all return calibrated confidence)
- `rule_match` expert knowledge base, 94.7% field / 99.6% content accuracy
- `biobert_match` biomedical semantic similarity
- `magneto_match` schema structure similarity, trained on GDC to FHIR

**FHIR schema**
- `explore_fhir_resource` list fields on a FHIR R5 resource
- `search_fhir_fields` find a field by keyword across all resources

**Terminology**
- `search_snomed` diagnoses, findings, procedures, body structures
- `search_loinc` lab tests and clinical measurements
- `search_ontology` HPO, MONDO, NCIt, UBERON, GO via EBI OLS4

**Profiling and feedback**
- `profile_csv` infer use case and mapping recommendations from column names
- `get_feedback_stats` monitor the human-in-the-loop learning loop
- `record_feedback` record a review decision

## Deployment notes

Two tools are restricted on this public deployment:

- `record_feedback` writes to a store shared by every visitor, so writes are
  disabled. Self-host with `ENABLE_FEEDBACK_WRITES=1` to collect feedback.
- `profile_csv` can call an LLM, which would spend the operator's API key on
  behalf of anonymous visitors. Set `ALLOW_LLM_PROFILING=1` and
  `ANTHROPIC_API_KEY` to enable it.

`magneto_match` requires a package that is not on PyPI. It returns an
explanatory error unless magneto is installed; the other 13 tools are
unaffected.

First request after a cold start is slow, because BioBERT loads on demand.

## Research use only

Not for clinical decision-making.

## Source and citation

- Source: https://github.com/teslajoy/schema_crush
- Archive: https://doi.org/10.5281/zenodo.22713617
- `pip install schema-crush`

```bibtex
@software{sanati_schema_crush,
  author    = {Sanati, Nasim},
  title     = {Schema Crush: calibrated semantic schema matching for
               biomedical data to FHIR},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22713617},
  url       = {https://github.com/teslajoy/schema_crush}
}
```