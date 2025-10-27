# schema_crush architecture

## overview

schema mapping system using multi-agent ai with langgraph for orchestration.

## directory structure

```
schema_crush/                    # repo root
├── requirements.txt             # python dependencies
├── ARCHITECTURE.md              # this file
└── schema_crush/                # python package
    ├── __init__.py
    ├── cli.py                   # command line interface
    ├── loaders/                 # universal data and mapping loaders
    │   ├── base.py              # base loader interface
    │   ├── data_loader.py       # csv/excel/json data loader
    │   ├── mapping_loader.py    # universal mapping format loader
    │   └── __init__.py
    ├── agents/                  # three core agents
    │   ├── base.py              # base agent interface
    │   ├── data_profiler.py     # rule-based data profiling agent
    │   ├── claude_de.py         # anthropic claude data engineer agent
    │   ├── data_engineer.md     # claude agent system prompt
    │   ├── biomni.py            # biomedical knowledge graph agent
    │   ├── magneto.py           # pattern matching + meta-learning agent
    │   └── __init__.py
    ├── orchestrator/            # langgraph pearl loop orchestrator
    │   ├── graph.py             # langgraph state machine
    │   ├── state.py             # shared state definitions
    │   └── __init__.py
    ├── config/                  # configuration files
    │   └── settings.yaml        # agent weights and thresholds
    └── data/                    # data directory
        ├── examples/            # example csv files
        │   ├── case.csv
        │   ├── biospecimen.csv
        │   └── file.csv
        ├── resources/           # existing mappings
        │   └── mappings.json
        └── vector_store/        # chromadb embeddings (to be created)
```

## 1. universal loader architecture

### data loader (`loaders/data_loader.py`)

purpose: load ANY source data format with full column vectors (todo: future - deterministic loaders for now)

**supported formats:**
- csv files (current: case.csv, biospecimen.csv)
- excel files (.xlsx, .xls)
- json files (gdc format, htan format)
- parquet files
- database connections 

**key features:**
- loads full column data as vectors for agent training
- extracts schema metadata (types, nulls, samples, unique counts)
- detects identifier fields (id, uuid, code patterns)
- detects biomedical fields (diagnosis, specimen, anatomy terms)
- returns predicted schema mapping to standardized FHIR/Iceberg schema 

**output format:**
```python
{
  "entities": {
    "case": {
      "fields": {
        "participant_id": {
          "type": "object",
          "vector": ["HTA-001", "HTA-002", ...],  # full column data
          "samples": ["HTA-001", "HTA-002", "HTA-003"],
          "null_count": 0,
          "unique_count": 5,
          "is_identifier": True,
          "is_biomedical": False
        },
        "primary_diagnosis": {
          "type": "object",
          "vector": ["Adenocarcinoma", "Squamous Cell Carcinoma", ...],
          "samples": ["Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell Carcinoma"],
          "null_count": 0,
          "unique_count": 4,
          "is_identifier": False,
          "is_biomedical": True
        }
      },
      "row_count": 5,
      "source": "data/examples/case.csv"
    }
  }
}
```

### mapping loader (`loaders/mapping_loader.py`)

purpose: load existing expert mappings from multiple formats for learning

**supported formats:**
1. **crdc linkml** (yaml format)
   - location: external crdc repos
   - structure: linkml schema with class/slot definitions
   - example: patient class with identifier slot

2. **htan mappings** (json format)
   - location: existing htan work (user has these)
   - structure: custom json with entity -> field mappings
   - example: case -> Patient, participant_id -> Patient.identifier

3. **gdc and HTAN to fhir** (json format)
   - location: existing gdc mappings
   - structure: json with node/resourceType/fieldMapping
   - example: currently in data/resources/mappings.json (10k+ lines)

4. **schema_crush output** (json format)
   - location: previous pipeline results
   - structure: clean_mappings with entity_mappings + field_mappings
   - example: schema_crush_mappings_20251027_103731.json

**key features:**
- auto-detects format from file structure
- converts all formats to universal internal representation
- extracts confidence scores when available
- stores in vector database for similarity search
- supports incremental learning from new mappings

**universal mapping format:**
```python
{
  "source": "htan_biospecimen_mappings_v1",
  "format": "htan_json",
  "timestamp": "2025-10-27T10:00:00",
  "mappings": {
    "entity_level": {
      "biospecimen": {
        "target": "Specimen",
        "confidence": 0.98,
        "source_format": "htan",
        "validated": True
      }
    },
    "field_level": {
      "biospecimen.specimen_id": {
        "target": "Specimen.identifier",
        "confidence": 0.95,
        "transform": None,
        "source_format": "htan",
        "validated": True
      },
      "biospecimen.tissue_type": {
        "target": "Specimen.type.coding.display",
        "confidence": 0.92,
        "transform": "terminology_lookup",
        "source_format": "htan",
        "validated": True
      }
    },
    "content_level": {
      "tissue_type": {
        "source_values": ["Fresh Frozen", "FFPE", "Slide"],
        "target_system": "http://hl7.org/fhir/ValueSet/specimen-type",
        "mappings": {
          "Fresh Frozen": "frozen-specimen",
          "FFPE": "ffpe-specimen",
          "Slide": "slide-specimen"
        }
      }
    }
  }
}
```

## 2. three-tier mapping architecture

### tier 1: entity mapping
**task:** map source entity to target resource type
**example:** case -> Patient, biospecimen -> Specimen, file -> DocumentReference
**agents used:** all three (biomni for biomedical knowledge, magneto for patterns, claude_de for profiling)

### tier 2: field mapping
**task:** map source fields to target paths
**example:** participant_id -> Patient.identifier, primary_diagnosis -> Condition.code
**agents used:** all three (biomni for embeddings, magneto for jaccard + patterns, claude_de for data types)

### tier 3: content mapping
**task:** map field values to target terminologies
**example:** "Adenocarcinoma" -> SNOMED CT code 35917007
**agents used:** biomni (primary), magneto (pattern validation), claude_de (data validation)

## 3. agent architecture

### biomni agent (https://github.com/snap-stanford/Biomni)
- **model:** sentence-transformers (bert-based) TODO: THIS IS NOT BERT repo above
- **knowledge:** biomedical ontologies + terminology graphs
- **method:** embedding similarity for semantic matching
- **output:** similarity scores 0.0-1.0

### magneto agent (https://github.com/VIDA-NYU/magneto-matcher)
- **model:** meta-learning from past successful mappings TODO: THIS IS NOT implemented correctly not using this agent
- **knowledge:** pattern library + historical mapping vectors
- **method:** jaccard similarity + string patterns + learned weights
- **output:** composite scores 0.0-1.0

### claude_de agent (https://github.com/wshobson/agents) TODO: pull in from accurate resource
- **model:** data profiling + statistical analysis
- **knowledge:** data type patterns + validation rules
- **method:** profile analysis + compatibility scoring
- **output:** compatibility scores 0.0-1.0

## 4. langgraph pearl orchestrator

### pearl loop stages

**perceive:**
- load source data with full column vectors
- load existing expert mappings
- extract schema metadata
- prepare agent inputs

**reason:**
- apply domain rules (3-tier: entity -> field -> content)
- check existing mappings in vector store
- determine which agents to invoke
- set confidence thresholds

**act:**
- invoke agents in parallel via langgraph
- collect similarity scores from each agent
- apply bayesian weights (context-aware per domain + field type)
- compute composite scores

**learn:**
- store successful mappings (≥0.95 auto, 0.85-0.95 after hitl validation)
- update agent weights based on accuracy
- enrich vector database
- log failures for analysis

### langgraph state machine

```python
# simplified state flow
StateGraph:
  nodes:
    - perceive_node: load data + mappings
    - reason_node: apply rules + check history
    - act_node: parallel agent invocation
    - learn_node: store results + update weights
    - hitl_node: human validation (0.85-0.95 range)

  edges:
    perceive -> reason
    reason -> act
    act -> conditional:
      if score >= 0.95: learn
      if 0.85 <= score < 0.95: hitl -> learn
      if score < 0.85: reject
```

### confidence routing

- **≥0.95:** auto-approve, proceed to learn
- **0.85-0.95:** route to human-in-the-loop validation
- **<0.85:** reject, log for review

## 5. vector database architecture

### chromadb vector store

**purpose:** store successful mapping embeddings for similarity search

**what gets stored:**
- entity mapping pairs with embeddings
- field mapping pairs with embeddings
- content mapping pairs with embeddings
- confidence scores + validation status
- source format + timestamp

**query pattern:**
```python
# when new mapping needed
query_embedding = embed("case.participant_id")
similar_mappings = chroma_collection.query(query_embeddings=[query_embedding], n_results=5)

# returns top 5 similar past mappings with scores
# example: [
#   ("patient.identifier", 0.94),
#   ("patient.id", 0.89),
#   ("subject.identifier", 0.87)
# ]
```

## 6. migration strategy from schema_crush1

### phase 1: loaders (current)
1. create universal data loader supporting csv initially
2. create universal mapping loader supporting existing formats
3. test with case.csv + biospecimen.csv + mappings.json
4. verify schema extraction matches schema_crush1 output

### phase 2: agents (one at a time)
1. port biomni agent (sentence-transformers + biomedical knowledge)
2. port magneto agent (pattern matching + jaccard)
3. port claude_de agent (data profiling)
4. test each agent individually with known inputs
5. verify scores match schema_crush1 results

### phase 3: orchestrator (langgraph)
1. define state schema for pearl loop
2. create perceive node (data + mapping loading)
3. create reason node (rule application)
4. create act node (parallel agent invocation)
5. create learn node (vector store + weight updates)
6. create hitl node (human validation queue)
7. test full pipeline with example data

### phase 4: validation
1. run schema_crush1 pipeline, capture results
2. run schema_crush pipeline with same inputs
3. compare entity mappings (should match)
4. compare field mappings (should match)
5. compare confidence scores (should be similar)
6. verify 80% field mapping success rate maintained

## 7. configuration

### settings.yaml structure
```yaml
agents:
  biomni:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    weight: 0.4
  magneto:
    weight: 0.35
  claude_de:
    weight: 0.25

thresholds:
  auto_approve: 0.95
  hitl_required: 0.85
  reject: 0.85

vector_store:
  backend: chromadb
  dimension: 384
  collection_name: "schema_mappings"

mappings:
  formats:
    - crdc_linkml
    - htan_json
    - gdc_json
    - schema_crush_json
```

## 8. success metrics

- **entity mapping accuracy:** target ≥95%
- **field mapping accuracy:** target ≥80% (current proven rate)
- **confidence calibration:** ≥0.95 should be 98%+ correct
- **hitl efficiency:** 0.85-0.95 range should be 90%+ correct after validation
- **learning improvement:** accuracy should increase over time as vector store grows

## notes

- start with 100% human-in-the-loop initially
- learn from existing expert mappings (crdc, htan, gdc)
- use domain rules to boost scores (proven to work)
- maintain backward compatibility with schema_crush1 (original implementation)

## progress / learning vs initial design

### implemented (current state)

**loaders:**
- `csv_loader.py` - loads csv with full column vectors, schema detection
- focusing on csv first before expanding to other formats

**embedders (tools/embeddings/):**
- `biobert_embedder.py` - biomedical embeddings using dmis-lab/biobert-v1.1
- `magneto_embedder.py` - schema matching embeddings using magneto library
- both support 3-tier matching: embed_entity(), embed_field(), embed_content()

**orchestrator:**
- `pearl_agent.py` - langgraph workflow with perceive -> reason -> act -> hitl -> learn
- multi-tier execution in single run (entity -> field -> content)
- weighted embedder combination (auto-normalized)
- confidence classification and hitl queueing
- match deduplication across tiers
- reasoning transparency via logs

**reporting:**
- `mapping_report.py` - saves mappings, scores, reasoning, metadata, hitl queue, summary
- all results timestamped and saved to results/ directory

**evaluation:**
- tested on gdc -> fhir mappings (20 fields)
- magneto: 25% precision@1, 65% recall@5
- biobert: 5% precision@1, 15% recall@5
- magneto outperforms 5x on schema matching (trained on gdc benchmark)

### design changes from initial plan

| planned | actual | reason |
|---------|--------|--------|
| agents/biomni.py | tools/embeddings/biobert_embedder.py | biomni repo doesn't provide embedder, used biobert for biomedical text |
| agents/magneto.py | tools/embeddings/magneto_embedder.py | wrapper around magneto library's EmbeddingMatcher |
| agents/claude_de.py | agents/data_profiler.py | rule-based profiler implemented, llm agent deferred |
| orchestrator/graph.py + state.py | orchestrator/pearl_agent.py | combined into single file with langgraph StateGraph |
| config/settings.yaml | agent init params | hardcoded for simplicity, config file deferred |
| loaders/mapping_loader.py | not implemented | using gdc mappings directly for evaluation |
| vector_store/chromadb | not implemented | in-memory matching sufficient for now |

### key learnings

1. **magneto trained on gdc performs best** - 5x better than general biomedical embeddings for schema matching
2. **tier 2 (field names) scores highest** - name-only matching outperforms name+values for field identification
3. **separate tier scores important** - orchestrator needs individual tier scores, not combined
4. **embedder api differences** - biobert and magneto have different embed_content signatures, needed try/except handling
5. **langgraph checkpointing limitations** - pandas dataframes not serializable, checkpointing disabled for now
6. **deduplication needed** - running multiple tiers produces overlapping matches, keep highest priority

### not yet implemented

1. **mapping_loader** - load existing crdc/htan/gdc mappings into vector store for learning
2. **chromadb vector store** - store successful mappings for similarity search
3. **claude llm agent** - anthropic agent for data profiling (data_engineer.md exists as reference)
4. **settings.yaml** - configuration file for weights/thresholds
5. **content-level value mapping** - terminology mapping (tier 3 does name+values, not value -> code)
6. **hitl interface** - web/cli ui for reviewing medium confidence matches
7. **cli integration** - end-to-end command line workflow

### next priorities

1. **real data and training on existing mappings**
   - load gdc/htan mappings from data/resources/ into training pipeline
   - create mapping_loader to parse existing mapping formats
   - evaluate: would synthetic fhir data from synthia help? (could generate diverse examples)
   - use existing mappings as ground truth for embedder fine-tuning
   - implement triplet loss training (anchor, positive, negative) from magneto paper

2. **scalable hitl interface** for reviewing medium confidence (85-95%) matches
   - **problem**: hitl doesn't scale for millions of columns (can't review 1:1 field pairs)
   - **solution**: aggregate feedback strategies

   **feedback aggregation levels:**
   - entity-level: "all Patient <-> Demographics fields approved"
   - prefix-level: "all gene_* fields correct"
   - tier-level: "only review field/content tiers"
   - confidence bucket: "auto-accept 0.90-0.95 if >80% historically correct"

   **clustering similar columns:**
   - group candidate matches using embedding similarity
   - review one representative per cluster (10 clusters instead of 1000 matches)
   - propagate feedback to similar matches

   **semi-automated propagation:**
   - train lightweight classifier on accepted vs rejected scores
   - apply to unreviewed matches for auto-labeling
   - keep only uncertain samples in future hitl rounds

   **hierarchical ui (collapsible tree):**
   ```
   Entity: Patient
    ├── id -> identifier         auto
    ├── age -> birthDate         manual
    └── sex -> gender            auto
   Entity: Specimen
    ├── sample_id -> id          auto
    └── collection_date -> issued  manual
   ```

   **progressive hitl rounds:**
   - round 1: collect feedback on 50 random medium-confidence pairs
   - round 2: retrain weights, present new uncertain set only
   - after 2-3 rounds, majority maps automatically

   **persistent feedback store:**
   - save validated pairs (positive + negative) in knowledge base
   - schema: source_field, target_field, approved, context, timestamp
   - future runs preload and skip re-asking
   - store in chromadb or sqlite for fast lookup

3. **cli integration** for end-to-end workflow
   - load source csv -> run matching -> generate report -> review hitl -> save results
   - single command: `schema_crush match source.csv --target fhir --output results/`

4. **chromadb vector store** for similarity search
   - store successful mappings with embeddings
   - query similar past mappings before running embedders
   - learn from historical matches

5. **embedder fine-tuning** on domain-specific data
   - fine-tune magneto/biobert on gdc->fhir mappings
   - use synthetic data generation (llm-based) from magneto paper approach
   - evaluate: does synthia-generated fhir data improve generalization?