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

run the claude agent demo:

```bash
python examples/demo_claude_agent.py
```

run the gdc embedder evaluation:

```bash
python examples/evaluate_gdc_embedders.py
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
| calibration system | done | tier-aware confidence calibration |
| knowledgebase | done | unified loader for all knowledge |
| vector store | done | chromadb for similarity search |
| pearl orchestrator | partial | state machine defined, needs updates |
| hitl interface | missing | web/cli ui for review |
| feedback store | missing | persist hitl decisions |
| adaptive weights | missing | auto-adjust tool weights |
| cli integration | missing | `schema_crush match source.csv` |

## matcher performance (gdc evaluation)

| matcher | precision@1 | recall@5 | accuracy | notes |
|---------|-------------|----------|----------|-------|
| rulematcher | 97% | 97% | 96.9% | best - uses knowledge base |
| magneto | 25% | 65% | 13.3% | good for field names |
| biobert | 5% | 15% | 8.2% | overconfident, needs calibration |

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

## pearl agent workflow

```
perceive -> reason -> act -> hitl -> learn -> (next tier or end)
```

- perceive: extract schema metadata
- reason: generate matches using weighted embedder combination
- act: filter and rank matches by confidence
- hitl: queue medium confidence (85-95%) matches for human review
- learn: collect metrics and update from feedback

## confidence levels

- high: score >= 95% (auto-accept)
- medium: score 85-95% (requires hitl)
- low: score < 85% (filtered out)

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

## output files

all results saved to `results/` directory:

- `*_mappings.json`: final source to target mappings
- `*_scores.json`: detailed scores with embedder contributions
- `*_reasoning.txt`: agent decision log
- `*_metadata.json`: run configuration and metrics
- `*_hitl_queue.json`: matches requiring human review
- `*_summary.txt`: human-readable summary

## testing

```bash
pytest tests/
```

## components

- `loaders/`: data loaders (csv, json)
- `tools/embeddings/`: biobert and magneto embedders
- `tools/matchers/`: matcher implementations
- `orchestrator/`: pearl agent workflow
- `learning/`: knowledge base and vector store
- `data/`: example data and ground truth mappings