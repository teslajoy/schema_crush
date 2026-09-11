# schema crush

<p align="">
  <img src="./img/schema_crush.png" alt="mapping" width="150"/>
</p>

a modular framework for semantic schema matching that aligns heterogeneous biomedical schemas to fhir standards using calibrated matchers, curated knowledge, and optional llm-assisted review.

built on expert-curated fhir aggregator mappings (gdc, htan) with human-in-the-loop feedback and guarded calibration.

![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22713617.svg)](https://doi.org/10.5281/zenodo.22713617)

## data statement

this repository ships source code, expert-curated mapping tables, pre-trained
calibrators, and documentation. it contains **no patient-level data**. the
htan, tcga, and geo exports used during development are excluded by
`.gitignore` and must be obtained from their original sources. see
[examples/fetch_geo.py](examples/fetch_geo.py) for a reproducible download of
the public geo datasets.

## install

```bash
pip install -r requirements.txt
```
or
```bash
pip install -e .
```

## usage

map a csv file to fhir:
```bash
python examples/map_csv.py data.csv --entity patient -o mappings.json
```

entity types:
- `patient` -> Patient, Condition, Observation
- `sample` -> Specimen, Observation
- `file` -> DocumentReference, Observation

output:
```json
[
  {"source": "patient_id", "target": "Patient.identifier", "confidence": 1.0},
  {"source": "diagnosis", "target": "Condition.code", "confidence": 0.85}
]
```

options:
```bash
python examples/map_csv.py data.csv -e patient              # with csv profiler (default)
python examples/map_csv.py data.csv -e patient --no-profile # skip profiling
python examples/map_csv.py data.csv -e patient --no-llm-profile  # heuristics only
python examples/map_csv.py data.csv -e sample -o out.json   # save to file
python examples/map_csv.py data.csv -e file -n 5            # limit to first 5 columns
```

run demos:
```bash
python examples/demo_claude_agent.py
```

## geo data workflow

download public GEO datasets and map them to FHIR.

### 1. fetch geo data

```bash
# list available datasets
python examples/fetch_geo.py --list

# download one dataset (default: GSE71729)
python examples/fetch_geo.py GSE71729

# download all pancreatic cancer datasets
python examples/fetch_geo.py --all
```

this downloads to `data/geo/`:
- `GSE*_metadata.csv` - full sample metadata
- `GSE*_clinical.csv` - parsed clinical characteristics
- `GSE*_expression.csv` - gene expression matrix (if available)

curated pancreatic datasets: GSE62452, GSE28735, GSE15471, GSE21501, GSE57495, GSE71729

### 2. map to fhir

```bash
# map clinical columns to FHIR fields
python examples/map_csv.py data/geo/GSE71729_clinical.csv -e sample -o mappings.json
```

### 3. review mappings

output shows source -> target with confidence:
```json
[
  {"source": "sample_id", "target": "Specimen.identifier", "confidence": 1.0},
  {"source": "tumor_subtype", "target": "Observation.valueCodeableConcept", "confidence": 0.85},
  {"source": "survival_months", "target": "Observation.valueQuantity", "confidence": 0.85},
  {"source": "death_event_1death_0censor", "target": "Patient.deceasedBoolean", "confidence": 0.95}
]
```

### example: full pipeline

```bash
# 1. fetch GEO data
python examples/fetch_geo.py GSE71729

# 2. map with profiling (interprets cryptic column names)
python examples/map_csv.py data/geo/GSE71729_clinical.csv -e sample

# 3. save mappings to file
python examples/map_csv.py data/geo/GSE71729_clinical.csv -e sample -o mappings.json
```

## csv profiling

the csv profiler analyzes your data before mapping to understand:
- **use case**: survival analysis, variant tracking, lab results, staging/grading, etc.
- **column relationships**: survival_time + death_event pairs, tnm groupings
- **cryptic column names**: `death_event_1death_0censor` -> "binary survival indicator (1=death, 0=censored)"

this context helps the llm make better mapping decisions for ambiguous fields.

example profile output:
```json
{
  "use_case": "survival_analysis",
  "analysis_purpose": "clinical survival data from geo for pancreatic cancer cohort",
  "column_groups": {
    "survival": ["survival_months", "death_event_1death_0censor"],
    "staging": ["stage", "t_stage", "n_stage"]
  },
  "recommendations": {
    "death_event_1death_0censor": "binary survival indicator -> Patient.deceasedBoolean (1=true, 0=false)"
  }
}
```

## core architecture

| layer | component | description |
|-------|-----------|-------------|
| data | `mappings/flat.py` | flatmappingdatabase with source/destination/contentvalue structures |
| knowledge | `knowledge/mapping_rules/` | curated mapping rules (gdc, htan -> fhir) |
| embedders | `tools/embeddings/` | biobert (biomedical) + magneto (schema-trained) |
| matchers | `tools/matchers/` | biobertmatcher, magnetomatcher, rulematcher |
| calibration | `orchestrator/calibration.py` | tier-aware confidence calibration |
| agent | `orchestrator/agents/` | claudeagent with llm tool-calling (optional) |
| feedback | `learning/feedback_store.py` | hitl decisions for evaluation and guarded retraining |
| retrieval | `learning/mapping_vector_store.py` | chromadb semantic index (rebuildable from sqlite) |

## curated knowledge sources

| source | description |
|--------|-------------|
| gdc | genomic data commons -> fhir mappings |
| htan | human tumor atlas network -> fhir mappings |
| others | additional fhir aggregator sources |

these datasets provide expert-curated source -> fhir mappings. schema crush does not fine-tune models on this data; it uses it to build a canonical mapping database and to calibrate matcher confidence.

## fhir aggregator data (used for calibration)

| source | patients | specimens | observations | documents |
|--------|----------|-----------|--------------|-----------|
| gdc | 44,736 | 593,840 | 807,834 | 1,121,816 |
| cda | 159,047 | 742,505 | 833,168 | - |
| htan | 2,080 | 7,532 | 214,853 | 196,158 |
| gtex | 980 | 43,559 | 1 | 49 |
| icgc | 400 | 813 | 401 | 15,063 |
| 1000genome | 3,500 | 3,500 | 1 | 48 |
| cellosaurus | 1,677 | 1,717 | 1 | - |

## 3-tier matching strategy

| tier | task | example |
|------|------|---------|
| entity | table -> fhir resource | `case` -> `Patient` |
| field | column -> fhir path | `participant_id` -> `Patient.identifier` |
| content | value -> fhir code | `"Adenocarcinoma"` -> snomed ct 35917007 |

## implementation status

| component | status | notes |
|-----------|--------|-------|
| flatmappingdatabase | done | o(1) lookup, sqlite cache |
| biobert embedder | done | dmis-lab/biobert-v1.1 |
| magneto embedder | done | schema-trained on gdc |
| rulematcher | done | 96.9% accuracy |
| biobert matcher | done | overconfident (needs calibration) |
| magneto matcher | done | best for field matching |
| claudeagent | done | llm with tool-calling + fast path |
| calibration system | done | context-aware backoff calibration |
| knowledgebase | done | unified loader for all knowledge |
| vector store | done | chromadb for similarity search |
| pearl orchestrator | partial | state machine defined, needs updates |
| feedback store | done | sqlite-backed with traceability |
| hitl interface | missing | web/cli ui for review |
| adaptive weights | missing | auto-adjust tool weights |
| cli integration | missing | `schema_crush match source.csv` |

## matcher performance (legacy - pre-calibration)

| matcher | precision@1 | recall@5 | accuracy | notes |
|---------|-------------|----------|----------|-------|
| rulematcher | 97% | 97% | 96.9% | best - uses knowledge base |
| magneto | 25% | 65% | 13.3% | good for field names |
| biobert | 5% | 15% | 8.2% | overconfident, needs calibration |

> see calibration system section below for current tier-aware results

## calibration system

### what is calibrated confidence?

a classifier is **calibrated** when its predicted confidence equals its actual accuracy: if a model says "90% confident", it should be correct 90% of the time. most ml models are **overconfident** — they output high scores even when wrong.

**calibration** learns a mapping from raw scores to true probabilities using held-out data:

```
P(correct | confidence = c) ≈ c   (perfectly calibrated)
```

we measure calibration quality using **expected calibration error (ECE)**:

```
ECE = Σ (|Bₘ|/n) · |acc(Bₘ) - conf(Bₘ)|
```

where `Bₘ` are confidence bins, `acc()` is accuracy within bin, `conf()` is mean confidence. lower ECE = better calibrated. a perfectly calibrated model has ECE = 0.

**reliability diagrams** visualize this: plot accuracy vs confidence per bin — a calibrated model follows the diagonal.

references: platt scaling (platt, 1999), temperature scaling (guo et al., 2017), histogram binning (zadrozny & elkan, 2001).

### how confidence is computed

confidence scores come from different sources depending on the matching path:

| path | confidence source | calibrated? |
|------|-------------------|-------------|
| **exact match** | `1.0` hardcoded — found verbatim in knowledge base | n/a (always correct) |
| **fuzzy match** | fuzzywuzzy string similarity ratio (0-1) | no (empirically reliable >0.85) |
| **embedding match** | cosine similarity from biobert/magneto | **yes** — mapped via calibration curves |
| **llm fallback** | model's self-reported confidence | no (uncalibrated) |

example:
```
sample_id -> 1.0    # exact lookup in flat_mappings.db
tissue -> 0.9       # fuzz.ratio("tissue", "TumorTissueType") ≈ 90%
tumor_grade -> 0.87 # magneto cosine sim → calibrated to 87% true accuracy
survival -> 0.85    # llm stated "85% confident"
```

### calibration implementation

calibration maps raw embedding matcher scores to actual accuracy. uses backoff-key scheme to handle sparse contexts.

### how it works

1. **binning**: divide confidence [0,1] into 10 bins, track accuracy per bin
2. **backoff keys**: `matcher:tier:schema:context` → fallback to more general keys when data is sparse
3. **eligibility**: a key needs 300+ samples and 100+ positives to be used (prevents overfitting)

key structure:
```
matcher : tier   : schema : context
   │        │        │        │
magneto : field  :  gdc   : specimen
```

backoff chain (most specific → global fallback):
```
magneto:field:gdc:specimen  ->  magneto:field:gdc:*  ->  magneto:field:*:*  ->  magneto:*:*:*
```

### calibration plots

generated in `examples/` and `calibrators/calibration_logs/`:

| plot | shows |
|------|-------|
| `calibration_reliability.png` | predicted vs actual accuracy per matcher (diagonal = perfect) |
| `calibration_*_contexts.png` | global vs tier-specific curves + key data distribution |
| `calibration_key_stats.png` | sample counts per key, which keys are eligible |
| `*_flat_calibration.png` | per-tier breakdown (entity, field, content) |

### current results (jan 2026)

| matcher | tier | accuracy | ece | status |
|---------|------|----------|-----|--------|
| **rule** | entity | **97.6%** | 0.005 | excellent |
| **rule** | field | **92.0%** | 0.08 | well-calibrated |
| **rule** | content | **99.2%** | 0.008 | near-perfect |
| biobert | entity | 5.9% | 0.79 | overconfident |
| biobert | field | 13.3% | 0.74 | overconfident |
| biobert | content | 68.0% | 0.26 | moderate |
| magneto | entity | 13.4% | 0.14 | moderate |
| magneto | field | 23.3% | 0.27 | moderate |
| magneto | content | **72.4%** | 0.04 | well-calibrated |

### calibration dataset

| metric | value |
|--------|-------|
| total sources | 2,346 |
| total destinations | 2,563 |
| entity tier samples | 984 |
| field tier samples | 941 |
| content tier samples | 421 |
| content values (coded) | 1,557 |

**schemas included**: gdc, htan, cda, icgc, gtex, 1000genome, cellosaurus

**entity candidates** (10 FHIR resources):
```
Condition, Device, DocumentReference, Medication, MedicationAdministration,
Observation, Patient, ResearchStudy, ServiceRequest, Specimen
```

**why rule matcher excels**: uses exact/fuzzy matching against curated mappings from gdc/htan transformers. entity tier now matches source → FHIR resource (not full path), improving from 52% to 97.6%.

**reference field coverage**: subject_ref 92.3%, focus_ref 18.2%, specimen_ref 9.8%

### retrain calibrators

```bash
python calibrators/calibrate_flat_mappings.py --embeddings
python examples/plot_calibration.py
```

## data flow

```
source schema (csv/json)
        |
        v
+------------------+
|  flatmappingdb   |<-- expert rules (gdc, htan, chembl)
+--------+---------+
         |
   +-----+-----+
   v     v     v
 rule  biobert magneto   <-- matchers
   |     |     |
   +-----+-----+
         |
         v
+------------------+
|   calibrators    |<-- tier-aware confidence
+--------+---------+
         |
         v (only for uncertain cases)
+------------------+
|   claudeagent    |<-- llm tool-calling
+--------+---------+
         |
         v
  fhir mapping output
```

## key optimizations

| feature | benefit |
|---------|---------|
| fast path | skip llm for exact rule matches (~93% of fields) |
| expert embeddings | pre-computed vectors for known sources |
| calibrated scores | confidence reflects true accuracy |
| few-shot context | similar mappings passed to llm |

## pearl agent workflow (historical)

this workflow describes an earlier theoretical experimental orchestration model and is not currently active in the production system. current execution uses deterministic matcher + calibration pipelines with optional llm assistance.

```
perceive -> reason -> act -> hitl -> learn -> (next tier or end)
```

- perceive: extract schema metadata
- reason: generate matches using weighted embedder combination
- act: filter and rank matches by confidence
- hitl: queue medium confidence (85-95%) matches for human review
- learn: collect metrics and update from feedback

## confidence levels (historical)

- high: score >= 95% (auto-accept)
- medium: score 85-95% (requires hitl)
- low: score < 85% (filtered out)

> current system uses calibrated confidence from backoff-key calibration

## file structure

```
schema_crush/
├── tools/
│   ├── embeddings/           # biobert, magneto wrappers
│   ├── matchers/             # basematcher interface + implementations
│   └── fhir_schema_tool.py   # linkml fhir schema explorer
├── orchestrator/
│   ├── agents/
│   │   ├── claude_agent.py   # llm with tool-calling + fhir rules
│   │   ├── csv_profiler.py   # use-case detection + recommendations
│   │   └── tools.py          # tool definitions for agent
│   ├── pearl_agent.py        # pearl workflow (langgraph)
│   └── calibration.py        # confidence calibration
├── mappings/                 # flatmappingdatabase, loaders
├── knowledge/
│   ├── mapping_rules/        # gdc, htan parsers
│   └── rules/                # fhir transformation + template rules (md)
├── learning/                 # knowledgebase, vectorstore
├── mcp/                      # mcp server (14 tools)
└── data/resources/           # gdc/htan json mappings, linkml schema
```

## loaders

| loader | input | output | use case |
|--------|-------|--------|----------|
| `flat_loader` | sqlite db | `FlatMappingDatabase` | primary loader - o(1) lookup for all mappings |
| `curated_loader` | `data/resources/gdc_mapping/*.json` | sources + destinations | expert-curated gdc/htan->fhir rules |
| `fhir_aggregator_loader` | fhir ndjson files | `SourceDestinationPair` list | extract patterns from real fhir data |
| `project_loader` | project configs | schema metadata | load source schemas for mapping |

data flow: `curated_loader` + `fhir_aggregator_loader` → `normalizer` -> `flat_loader` (sqlite)

## mcp server

expose schema_crush as an mcp (model context protocol) server for claude desktop/code.

### run server

```bash
python -m schema_crush.mcp.server
```

or add to claude desktop config (`~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "schema-crush": {
      "command": "python",
      "args": ["-m", "schema_crush.mcp.server"],
      "cwd": "/path/to/schema_crush"
    }
  }
}
```

### tools (14)

| tool | description | location |
|------|-------------|----------|
| **profiling** | | |
| `profile_csv` | analyze csv use-case, interpret cryptic column names, get mapping recommendations | `orchestrator/agents/csv_profiler.py` |
| **matchers** | | |
| `biobert_match` | semantic matching using biobert embeddings | `tools/matchers/biobert_matcher.py` |
| `magneto_match` | schema-trained matching using magneto | `tools/matchers/magneto_matcher.py` |
| `rule_match` | exact/fuzzy matching against curated rules | `tools/matchers/rule_matcher.py` |
| **lookups** | | |
| `lookup_mapping` | find all mappings for a source term | `learning/knowledge_base.py` |
| `find_similar` | find similar source terms in knowledge base | `learning/mapping_vector_store.py` |
| **fhir schema** | | |
| `explore_fhir_resource` | get fields and structure of a fhir r5 resource | `tools/fhir_schema_tool.py` |
| `search_fhir_fields` | search for fhir fields by keyword | `tools/fhir_schema_tool.py` |
| **terminology** | | |
| `search_snomed` | search snomed ct codes by term | tx.fhir.org API |
| `search_loinc` | search loinc codes by term | clinicaltables.nlm.nih.gov API |
| `search_ontology` | search ontologies (nci thesaurus, etc.) | ebi.ac.uk/ols4 API |
| **feedback** | | |
| `record_feedback` | record user corrections to mappings | `learning/feedback_store.py` |
| `get_feedback_stats` | get feedback statistics | `learning/feedback_store.py` |
| **reference** | | |
| `get_transformation_rules` | get fhir transformation rules | `orchestrator/agents/claude_agent.py` |

### example queries

```
"where does tumor_grade map to in FHIR?"
-> calls lookup_mapping, returns Observation.valueCodeableConcept.text + SNOMED codes

"what fields does Patient have?"
-> calls explore_fhir_resource

"find SNOMED code for adenocarcinoma"
-> calls search_snomed, returns 35917007

"how do I map staging to FHIR?"
-> calls get_transformation_rules with section=staging
```

## knowledge layer

### transformation rules

fhir transformation rules consolidated from gdc, cda, htan, icgc transformers.

| document | location | purpose |
|----------|----------|---------|
| transformation rules | `knowledge/rules/fhir_transformation_rules.md` | full documentation (700+ lines) |
| template rules | `knowledge/rules/fhir_template_rules.md` | fhir patterns from jinja templates |
| mcp operational | inline in `mcp/server.py` | quick lookup via `get_transformation_rules` |

sections covered: patient demographics, condition/diagnosis, staging hierarchy, snomed codes, observation patterns, specimen hierarchy, documentreference, medicationadministration, id minting, extensions.

### vector store (chromadb)

chromadb is used as a **derived semantic retrieval index** to support:
- `find_similar` lookups
- few-shot context for llm-assisted review

chromadb is fully rebuildable from `flat_mappings.db` and is **not a learning or feedback store**. canonical truth always lives in sqlite.

location: `schema_crush/data/db/chroma/`

### feedback store

sqlite-backed store for hitl decisions with full traceability.

| field | purpose |
|-------|---------|
| `user_id` | who made the decision |
| `timestamp` | when (iso utc) |
| `source` + `source_context` | what was being mapped |
| `proposed_target` | system proposal |
| `ground_truth` | user correction (if any) |
| `decision` | accept/reject/correct |
| `matcher` + `tier` + `confidence` | matcher context |

location: `schema_crush/data/db/feedback.db`

exports:
- `export_for_calibration()` -> retrain calibrators with user decisions
- `export_new_mappings()` -> add corrections to knowledge base

retrain with feedback:
```bash
python calibrators/calibrate_flat_mappings.py --feedback
```

> **current limitations**: feedback is incorporated into calibration in a one-way manner. planned safeguards include holdout splits, minimum sample thresholds, metric regression checks (ece/brier), and versioned retraining.

## what's missing

| planned | status | notes |
|---------|--------|-------|
| mapping workbench | design done | drag-drop ui for hitl review, see docs/MAPPING_WORKBENCH.md |
| fhir aggregator training | not started | learn from real fhir data (patterns, codes, references) |
| rest api | not started | fastapi wrapper around mcp tools |
| chromadb persistence | done | rebuildable index stored on disk |
| synthia integration | not started | synthetic fhir data generation |
| fine-tuning pipeline | not started | triplet loss training |
| hitl clustered feedback | not started | group similar matches for bulk review |
| ontology mining | not started | auto-discover from snomed/loinc |
| federation | not started | external fhir server validation |
| context layout refactor | not started | move agent identity, pattern recognition, and transformation rules out of inline python strings in `claude_agent.py` into `knowledge/identity/`, `knowledge/rules/` (split per source: gdc, cda, htan, icgc), and `knowledge/templates/`. lets `mcp/server.py:get_transformation_rules` use directory listing instead of header-splitting a python string |
| hosted bioscience connectors | not started | this project predates the claude bioscience connectors (chembl, uniprot, and related life-science resources). several bulk downloads and local caches here exist only because those lookups had to be done offline. revisit the ontology and reference-resource paths to query hosted connectors on demand instead of shipping or downloading bulk data |

## documentation

| document | contents |
|----------|----------|
| [docs/schema_crush_overview.md](docs/schema_crush_overview.md) | technical overview and abstract, start here |
| [docs/architecture_current.md](docs/architecture_current.md) | current architecture, entry points, and data flow |
| [docs/PROMPT_ENGINEERING_ARCHITECTURE.md](docs/PROMPT_ENGINEERING_ARCHITECTURE.md) | multi-agent prompt design |
| [docs/MAPPING_WORKBENCH.md](docs/MAPPING_WORKBENCH.md) | design proposal for a hitl review ui, not implemented |
| [demo/schema_crush_tutorial.ipynb](demo/schema_crush_tutorial.ipynb) | executable tutorial with stored outputs |
| [demo/WALKTHROUGH.md](demo/WALKTHROUGH.md) | 20 to 30 minute technical walkthrough |

the tutorial is jupytext-paired: `demo/schema_crush_tutorial.py` is the source
of record and `demo/schema_crush_tutorial.ipynb` is generated from it.

## citation

if you use schema crush in your work, please cite it:

> Sanati, N. (2026). *Schema Crush: calibrated semantic schema matching for
> biomedical data to FHIR* (Version 1.2.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.22713617

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

`10.5281/zenodo.22713617` is the concept doi and always resolves to the latest
release. to cite this exact version instead, use `10.5281/zenodo.22713618`.

structured metadata lives in [CITATION.cff](CITATION.cff), which github uses to
render the "cite this repository" button.

## license

mit, see [LICENSE](LICENSE).

## output files (legacy)

> current cli outputs json to stdout or file via `-o`

- `*_mappings.json`: final source to target mappings
- `*_scores.json`: detailed scores with embedder contributions
- `*_reasoning.txt`: agent decision log

## testing

```bash
pytest tests/
```