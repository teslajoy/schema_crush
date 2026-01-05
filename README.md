# schema crush

<p align="">
  <img src="./img/schema_crush.png" alt="mapping" width="150"/>
</p>

a modular framework for semantic schema matching that aligns heterogeneous biomedical schemas to fhir standards using multi-agent ai. trained on fhir aggregator data (ex. htan, gdc), and others.

![Status](https://img.shields.io/badge/Status-Build%20Passing-lgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
python map_csv.py data.csv --entity patient -o mappings.json
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
```
python map_csv.py data.csv -e patient              # print to stdout
python map_csv.py data.csv -e sample -o out.json   # save to file
python map_csv.py data.csv -e file -n 5            # limit to first 5 columns
```

run demos:
```bash
python examples/demo_claude_agent.py
```

## core architecture

| layer | component | description |
|-------|-----------|-------------|
| data | `mappings/flat.py` | flatmappingdatabase with source/destination/contentvalue structures |
| knowledge | `knowledge/mapping_rules/` | curated mapping rules (gdc, htan -> fhir) |
| embedders | `tools/embeddings/` | biobert (biomedical) + magneto (schema-trained) |
| matchers | `tools/matchers/` | biobertmatcher, magnetomatcher, rulematcher |
| agent | `orchestrator/agents/` | claudeagent with llm tool-calling |
| orchestrator | `orchestrator/pearl_agent.py` | pearl workflow (perceive, reason, act, hitl, learn) |
| learning | `learning/` | knowledgebase, mappingvectorstore (chromadb) |

## training data sources

| source | description |
|--------|-------------|
| gdc | genomic data commons -> fhir mappings |
| htan | human tumor atlas network -> fhir mappings |
| others | additional fhir aggregator sources |

## fhir aggregator data (available for training)

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
| hitl interface | missing | web/cli ui for review |
| feedback store | missing | persist hitl decisions |
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

calibration maps raw matcher confidence to actual accuracy. uses backoff-key scheme to handle sparse contexts.

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

### current results (dec 2025)

| matcher | tier | accuracy | ece | notes |
|---------|------|----------|-----|-------|
| **rule** | entity | 52.1% | 0.48 | one-to-many mappings (source → multiple valid targets) |
| **rule** | field | **94.7%** | 0.05 | excellent |
| **rule** | content | **99.6%** | 0.004 | near-perfect (terminology lookups) |
| biobert | entity | 1.3% | 0.85 | overconfident, use calibrated score |
| biobert | field | 14.7% | 0.74 | overconfident |
| biobert | content | 69.8% | 0.24 | better on terminology |
| magneto | entity | 3.8% | 0.29 | low confidence but honest |
| magneto | field | 26.7% | 0.25 | moderate |
| magneto | content | 70.0% | 0.05 | well-calibrated on terminology |

**why "global" accuracy differs from "content"**: global = weighted average of entity + field + content. entity matching is harder (52% for rule, 1-4% for embedders), which pulls down the global average. content matching (terminology lookups) is easier since it's often exact matches.

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

## pearl agent workflow (legacy - not active)

```
perceive -> reason -> act -> hitl -> learn -> (next tier or end)
```

- perceive: extract schema metadata
- reason: generate matches using weighted embedder combination
- act: filter and rank matches by confidence
- hitl: queue medium confidence (85-95%) matches for human review
- learn: collect metrics and update from feedback

## confidence levels (legacy)

- high: score >= 95% (auto-accept)
- medium: score 85-95% (requires hitl)
- low: score < 85% (filtered out)

> current system uses calibrated confidence from backoff-key calibration

## file structure

```
schema_crush/
├── tools/
│   ├── embeddings/         # biobert, magneto wrappers
│   ├── matchers/           # basematcher interface + implementations
│   └── fhir_schema_tool.py # linkml fhir schema explorer
├── orchestrator/
│   ├── agents/             # claudeagent + tool definitions
│   ├── pearl_agent.py      # pearl workflow (langgraph)
│   └── calibration.py      # confidence calibration
├── mappings/               # flatmappingdatabase, loaders
├── knowledge/              # mapping rules, parsers
├── learning/               # knowledgebase, vectorstore
├── loaders/                # csv loader (limited)
└── data/resources/         # gdc/htan json mappings, linkml schema
```

## loaders

| loader | input | output | use case |
|--------|-------|--------|----------|
| `flat_loader` | sqlite db | `FlatMappingDatabase` | primary loader - o(1) lookup for all mappings |
| `curated_loader` | `data/resources/gdc_mapping/*.json` | sources + destinations | expert-curated gdc/htan->fhir rules |
| `fhir_aggregator_loader` | fhir ndjson files | `SourceDestinationPair` list | extract patterns from real fhir data |
| `project_loader` | project configs | schema metadata | load source schemas for mapping |

data flow: `curated_loader` + `fhir_aggregator_loader` → `normalizer` -> `flat_loader` (sqlite)

## what's missing

| planned | status | notes |
|---------|--------|-------|
| mapping workbench | design done | drag-drop ui for hitl review, see docs/MAPPING_WORKBENCH.md |
| fhir aggregator training | not started | learn from real fhir data (patterns, codes, references) |
| mcp server | not started | expose as model context protocol server for claude desktop/code |
| chromadb persistence | in-memory only | not persisted to disk |
| synthia integration | not started | synthetic fhir data generation |
| fine-tuning pipeline | not started | triplet loss training |
| hitl clustered feedback | not started | group similar matches for bulk review |
| ontology mining | not started | auto-discover from snomed/loinc |
| federation | not started | external fhir server validation |

## output files (legacy)

> current cli outputs json to stdout or file via `-o`

- `*_mappings.json`: final source to target mappings
- `*_scores.json`: detailed scores with embedder contributions
- `*_reasoning.txt`: agent decision log

## testing

```bash
pytest tests/
```