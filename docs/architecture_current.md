# schema_crush Architecture (Jan 2026)

## Current Status Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐        ┌────────────────┐        ┌────────────────┐   │
│   │   MCP Server   │        │  ClaudeAgent   │        │   map_csv.py   │   │
│   │   (14 tools)   │        │  (LangChain)   │        │    (CLI)       │   │
│   │                │        │                │        │                │   │
│   │  ACTIVE        │        │  ACTIVE        │        │  ACTIVE        │   │
│   │  Claude Desktop│        │  Programmatic  │        │  Quick maps    │   │
│   └───────┬────────┘        └───────┬────────┘        └───────┬────────┘   │
│           │                         │                         │             │
│           │    ┌────────────────────┴─────────────────┐      │             │
│           │    │                                      │      │             │
│           ▼    ▼                                      ▼      ▼             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      KnowledgeBase                               │      │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │      │
│   │  │     db      │  │ calibrators │  │      vectors        │     │      │
│   │  │ FlatMapping │  │ tier-aware  │  │ MappingVectorStore  │     │      │
│   │  │  Database   │  │ rule/bio/mag│  │    (ChromaDB)       │     │      │
│   │  └─────────────┘  └─────────────┘  └─────────────────────┘     │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              MATCHERS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐             │
│   │  RuleMatcher   │   │ BioBERTMatcher │   │ MagnetoMatcher │             │
│   │                │   │                │   │                │             │
│   │  RECOMMENDED   │   │  semantic      │   │  schema-trained│             │
│   │  97.6% entity  │   │  5.9% entity   │   │  13.4% entity  │             │
│   │  92.0% field   │   │  13.3% field   │   │  23.3% field   │             │
│   │  99.2% content │   │  68.0% content │   │  72.4% content │             │
│   │                │   │                │   │                │             │
│   │  exact + fuzzy │   │  BioBERT emb   │   │  MPNet emb     │             │
│   └────────────────┘   └────────────────┘   └────────────────┘             │
│                                                                              │
│   Strategy: Rule first (fast path) → embedders for unknowns → LLM fallback │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    flat_mappings.db (SQLite)                     │      │
│   │                                                                  │      │
│   │  sources (2,346)          destinations (2,563)                  │      │
│   │  ├─ id                    ├─ source_id                          │      │
│   │  ├─ source                ├─ destination                        │      │
│   │  ├─ source_schema         ├─ entity         ← NEW               │      │
│   │  ├─ tier                  ├─ dest_system                        │      │
│   │  └─ source_context        ├─ dest_code                          │      │
│   │                           ├─ subject_ref    ← POPULATED         │      │
│   │  content_values (1,557)   ├─ focus_ref      ← POPULATED         │      │
│   │  content_fhir_targets     └─ specimen_ref   ← POPULATED         │      │
│   │                                                                  │      │
│   │  Schemas: gdc, htan, cda, icgc, gtex, 1000genome, cellosaurus  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    calibrators/pkl/ (Pickle)                     │      │
│   │                                                                  │      │
│   │  rule_flat_calibrator.pkl    ← tier-aware (entity/field/content)│      │
│   │  biobert_flat_calibrator.pkl                                    │      │
│   │  magneto_flat_calibrator.pkl                                    │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         NOT YET IMPLEMENTED                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐             │
│   │ PEARL Agent    │   │  HITL Queue    │   │   Feedback     │             │
│   │                │   │                │   │   Store        │             │
│   │  DEFINED       │   │  NOT STARTED   │   │  **ACTIVE**    │             │
│   │  not wired     │   │  needs UI      │   │  sqlite-backed │             │
│   │                │   │                │   │                │             │
│   │  langgraph     │   │  web/cli       │   │  user_id +     │             │
│   │  state machine │   │  review queue  │   │  timestamp     │             │
│   └────────────────┘   └────────────────┘   └────────────────┘             │
│                                                                              │
│   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐             │
│   │  Clustering    │   │   REST API     │   │  Multi-Agent   │             │
│   │                │   │                │   │                │             │
│   │  NOT STARTED   │   │  NOT STARTED   │   │  NOT STARTED   │             │
│   │  batch review  │   │  FastAPI wrap  │   │  specialists   │             │
│   │                │   │                │   │                │             │
│   │  group similar │   │  /map, /lookup │   │  entity/field/ │             │
│   │  fields        │   │  /feedback     │   │  content agents│             │
│   └────────────────┘   └────────────────┘   └────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Status

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| MCP Server | **ACTIVE** | `schema_crush/mcp/server.py` | 14 tools for Claude Desktop/Code |
| ClaudeAgent | **ACTIVE** | `orchestrator/agents/claude_agent.py` | LangChain-based, loads FHIR template rules |
| CSVProfiler | **ACTIVE** | `orchestrator/agents/csv_profiler.py` | Use-case detection, cryptic column interpretation |
| map_csv.py | **ACTIVE** | `examples/map_csv.py` | CLI entry point |
| KnowledgeBase | **ACTIVE** | `learning/knowledge_base.py` | Loads db + calibrators + vectors |
| FlatMappingDatabase | **ACTIVE** | `mappings/flat.py` | 2,346 sources, 2,563 destinations |
| Calibrators | **ACTIVE** | `calibrators/pkl/` | Tier-aware (entity/field/content) |
| RuleMatcher | **ACTIVE** | `tools/matchers/rule_matcher.py` | 97.6% entity accuracy |
| BioBERTMatcher | **ACTIVE** | `tools/matchers/biobert_matcher.py` | Expert embeddings |
| MagnetoMatcher | **ACTIVE** | `tools/matchers/magneto_matcher.py` | Expert embeddings |
| Feedback Store | **ACTIVE** | `learning/feedback_store.py` | SQLite with full traceability |
| Transformation Rules | **ACTIVE** | `knowledge/rules/fhir_transformation_rules.md` | 700+ lines, MCP-exposed |
| Template Rules | **ACTIVE** | `knowledge/rules/fhir_template_rules.md` | FHIR patterns from jinja templates |
| Vector Store | **ACTIVE** | `learning/mapping_vector_store.py` | ChromaDB, rebuildable from SQLite |
| PEARL Agent | HISTORICAL | `orchestrator/pearl_agent.py` | LangGraph state machine, not active |
| HITL Queue | NOT STARTED | - | Needs UI design |
| Clustering | NOT STARTED | - | For batch HITL review |
| REST API | NOT STARTED | - | FastAPI wrapper |
| Multi-Agent | NOT STARTED | - | Specialist agents |

## Data Flow

```
Source Schema (CSV/JSON)
        │
        ▼
┌───────────────────┐
│  Tier Detection   │  ← entity / field / content
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────┐
│   RuleMatcher     │────▶│  score > 0.95?  │───yes──▶ ACCEPT (fast path)
└─────────┬─────────┘     └────────┬────────┘
          │                        │ no
          ▼                        ▼
┌───────────────────┐     ┌─────────────────┐
│ BioBERT + Magneto │────▶│  ensemble score │
└─────────┬─────────┘     └────────┬────────┘
          │                        │
          ▼                        ▼
┌───────────────────┐     ┌─────────────────┐
│    Calibrate      │────▶│  score > 0.85?  │───yes──▶ ACCEPT
└─────────┬─────────┘     └────────┬────────┘
          │                        │ no
          ▼                        ▼
┌───────────────────┐     ┌─────────────────┐
│   LLM Fallback    │────▶│    HITL Queue   │  (not implemented)
│  (ClaudeAgent)    │     │   for review    │
└───────────────────┘     └─────────────────┘
```

## Calibration Strategy

| Tier | Samples | Rule Acc | BioBERT Acc | Magneto Acc |
|------|---------|----------|-------------|-------------|
| Entity | 984 | 97.6% | 5.9% | 13.4% |
| Field | 941 | 92.0% | 13.3% | 23.3% |
| Content | 421 | 99.2% | 68.0% | 72.4% |

**Key insight**: Rule matcher handles 95%+ of cases. Use embedders only for unknowns.

## Knowledge Layer

### Transformation Rules

FHIR transformation rules consolidated from GDC, CDA, HTAN, ICGC transformers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRANSFORMATION RULES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  knowledge/rules/fhir_transformation_rules.md (700+ lines)      │      │
│   │  ├─ Patient Demographics (race, ethnicity, birthsex extensions) │      │
│   │  ├─ Condition/Diagnosis (staging, grade, body site)            │      │
│   │  ├─ Observation patterns (survey, biospecimen, exposure)       │      │
│   │  ├─ Specimen hierarchy (sample → portion → analyte → aliquot)  │      │
│   │  ├─ DocumentReference (files, DRS URIs)                        │      │
│   │  ├─ MedicationAdministration (treatments)                      │      │
│   │  ├─ ID minting (deterministic UUIDs)                           │      │
│   │  └─ SNOMED/LOINC/ICD-10 code systems                          │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  knowledge/rules/fhir_template_rules.md (jinja patterns)        │      │
│   │  ├─ Patient, Condition, Specimen field mappings                 │      │
│   │  ├─ Observation variants (staging, grade, IHC, FISH, CNV, SNV) │      │
│   │  └─ Common patterns (survival, identifiers, coded values)      │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                          │                                                  │
│                          ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  ClaudeAgent loads rules into system prompt                     │      │
│   │  MCP: get_transformation_rules(section=staging|snomed|...)      │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vector Store (ChromaDB)

ChromaDB is used as a **derived semantic retrieval index** - NOT a learning or feedback store.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VECTOR STORE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│   Location: schema_crush/data/db/chroma/                                    │
│                                                                              │
│   Purpose:                                                                   │
│   ├─ find_similar() lookups                                                 │
│   └─ few-shot context for LLM-assisted review                               │
│                                                                              │
│   Key property: FULLY REBUILDABLE from flat_mappings.db                     │
│   Canonical truth always lives in SQLite.                                   │
│                                                                              │
│   Methods:                                                                   │
│   ├─ find_similar(query) → top-k similar mappings                          │
│   ├─ add_mapping()       → incremental update                               │
│   └─ reindex(db)         → full rebuild from SQLite                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feedback Store

SQLite-backed store for HITL decisions with full audit trail.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEEDBACK STORE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│   Location: schema_crush/data/db/feedback.db                                │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  feedback table                                                  │      │
│   │  ├─ id (autoincrement)                                          │      │
│   │  ├─ source, source_context      ← what was mapped               │      │
│   │  ├─ proposed_target             ← system proposal               │      │
│   │  ├─ ground_truth                ← user correction (if any)      │      │
│   │  ├─ decision                    ← accept/reject/correct         │      │
│   │  ├─ matcher, tier, confidence   ← context                       │      │
│   │  ├─ timestamp                   ← ISO UTC                       │      │
│   │  ├─ user_id                     ← who made decision             │      │
│   │  └─ notes                       ← optional comments             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   Exports:                                                                   │
│   ├─ export_for_calibration() → retrain calibrators                        │
│   └─ export_new_mappings()    → add corrections to knowledge base          │
│                                                                              │
│   MCP Tools: record_feedback, get_feedback_stats                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Store Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CANONICAL vs DERIVED STORES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CANONICAL (source of truth):                                              │
│   ├─ flat_mappings.db    → expert-curated mappings                         │
│   ├─ feedback.db         → user decisions                                   │
│   └─ calibrators/*.pkl   → trained confidence curves                        │
│                                                                              │
│   DERIVED (rebuildable):                                                     │
│   └─ chroma/             → semantic index, rebuilt from flat_mappings.db    │
│                                                                              │
│   Rule: If losing ChromaDB loses knowledge, you're storing wrong thing.     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Learning Loop (Current)

```
User Feedback → FeedbackStore → export_for_calibration() → Recalibrate
                             → export_new_mappings()    → Update flat_mappings.db
```

Retrain with feedback:
```bash
python calibrators/calibrate_flat_mappings.py --feedback
```

**Current limitations**: Feedback is incorporated in a one-way manner. Planned safeguards:
- Holdout splits (last 20% by time, never trains)
- Minimum sample thresholds before retraining
- ECE/Brier metric regression checks
- Versioned retraining (kb_version, calibrator_version, etc.)