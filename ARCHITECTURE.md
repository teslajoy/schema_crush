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

1. **mapping_rules infrastructure** - structured composite mapping rules from htan/gdc
2. **chromadb vector store** - semantic retrieval layer for fuzzy matching
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

## 9. hybrid mapping rules architecture

### conceptual alignment

the system uses a hybrid approach combining structured rules with vector retrieval:

layer | component | purpose
------|-----------|--------
structured knowledge | mapping_rules/ | authoritative composite mappings (entity + field + reference + content)
semantic retrieval | retrieval/ | convert rules -> text -> vector for fuzzy search
reasoning integration | pearl orchestrator | retrieve + lookup + combine scores

### implementation flow

step | component | description
-----|-----------|-------------
1. rule foundation | mapping_rules/ | build mappingrule, reference, fieldmapping dataclasses and parsers
2. vector layer | retrieval/ | embed each mappingrule -> chromadb for similarity retrieval
3. hybrid reasoning | pearl orchestrator | query vector store -> candidate ids -> fetch from ruledatabase -> combine scores

### directory structure addition

```
schema_crush/
├── mapping_rules/              # structured composite mapping rules
│   ├── __init__.py
│   ├── base_rule.py           # mappingrule, reference, fieldmapping dataclasses
│   ├── rule_database.py       # store/query rules (sqlite/json)
│   ├── htan_rule_parser.py    # parse htan mappings.json -> mappingrule[]
│   └── gdc_rule_parser.py     # parse gdc case.json/file.json -> mappingrule[]
└── retrieval/                  # semantic retrieval layer (optional next)
    ├── __init__.py
    ├── vector_store.py         # chromadb wrapper
    ├── rule_embedder.py        # embed mappingrule -> text -> vector
    └── semantic_search.py      # query interface
```

### mapping rule structure

composite rules capture multi-layered fhir logic:

```python
@dataclass
class mappingrule:
    """composite mapping rule with context."""

    # tier 1: entity/concept level
    source_node: str                    # "bts:urinebiospcimentype"
    target_resource: str                # "observation"
    subclass_of: optional[str]          # "bts:biospecimen"

    # tier 2: relationship context
    references: list[reference]         # observation.focus -> specimen
    category: list[str]                 # ["specimen", "laboratory"]

    # tier 3: field mappings
    field_mappings: list[fieldmapping]  # component[valuestring]

    # metadata
    confidence: float = 1.0             # from human-curated source
    source: str = "htan"                # provenance
    rule_id: str                        # unique identifier
```

### example htan composite mapping

from htan mappings.json:

```json
{
  "node": "bts:urinebiospcimentype",
  "fhir:resourcetype": "observation",
  "fhir:reference": [{"fhir:resourcetype": "specimen", "fhir:field": "focus"}],
  "fhir:fieldmapping": [{
    "fhir:field": "component",
    "fhir:system": "https://humantumoratlas.org/urinebiospcimentype",
    "fhir:type": "valuestring",
    "fhir:code": "urinebiospcimentype",
    "fhir:category": "laboratory"
  }],
  "rdfs:subclassof": "bts:biospecimen"
}
```

parses to:

```python
mappingrule(
    source_node="bts:urinebiospcimentype",
    target_resource="observation",
    subclass_of="bts:biospecimen",
    references=[reference(resource="specimen", field="focus")],
    category=["specimen"],
    field_mappings=[
        fieldmapping(
            field="component",
            type="valuestring",
            system="https://humantumoratlas.org/urinebiospcimentype",
            code="urinebiospcimentype",
            category="laboratory"
        )
    ],
    confidence=1.0,
    source="htan",
    rule_id="htan_urinebiospcimentype"
)
```

### how pearl uses hybrid retrieval

```python
reason (tier 1 - entity):
  1. vector search (fuzzy):
     query: "biospecimen urine sample"
     -> retrieve top-5 similar htan nodes
     -> [bts:urinebiospcimentype, bts:biospecimen, ...]

  2. rule database (structured):
     lookup: mappingrule(source_node="bts:urinebiospcimentype")
     -> get full composite rule with references + field_mappings

  3. combine:
     embedder_score: 0.87 (biobert similarity)
     rule_match_score: 1.0 (exact match in database)
     vector_similarity: 0.92 (chromadb retrieval)
     final_score: 0.87 * 0.3 + 1.0 * 0.5 + 0.92 * 0.2 = 0.95

reason (tier 2 - field):
  1. vector search:
     query: "specimen collection date"
     -> retrieve: specimen.collection.collectedDateTime

  2. rule database:
     filter: rules where target_resource="specimen"
             and field_mappings contains "collection"
     -> get structured field mapping with type, system, code

  3. combine scores and validate type constraints

reason (tier 3 - content):
  1. use embedders on field values
  2. validate against rule field_mappings (type constraints)
     ex., if rule says "valuestring", reject numeric match
```

### phased implementation

phase | tasks | deliverable
------|-------|-------------
1. structured rules | create base_rule.py, htan_rule_parser.py, gdc_rule_parser.py, rule_database.py | 1,253 structured rules loaded and queryable
2. vector retrieval | create vector_store.py, rule_embedder.py, semantic_search.py | fuzzy search -> structured composite rules
3. pearl integration | update reasonagent to use hybrid retrieval, combine scores | working pearl with hybrid matching

### data sources

source | location | count | structure
-------|----------|-------|----------
htan mappings | data/resources/htan_mapping/mappings.json | 1023 | composite rules with resource + references + field_mappings
gdc case mappings | data/resources/gdc_mapping/case.json | 122 | entity -> fhir patient with demographic extensions
gdc file mappings | data/resources/gdc_mapping/file.json | 68 | entity -> fhir documentreference
gdc project mappings | data/resources/gdc_mapping/project.json | 40 | entity -> fhir researchstudy

total: 1,253 curated composite mapping rules

### performance comparison (gdc evaluation, 20 test fields)

matcher | precision@1 | recall@5 | f1 score | confidence range | notes
--------|-------------|----------|----------|------------------|-------
biobert | 5% (1/20) | 15% (3/20) | 0.0750 | 0.79-0.93 | overconfident - high scores but poor accuracy. general biomedical embeddings don't understand schema structure.
magneto | 25% (5/20) | 65% (13/20) | 0.3611 | 0.25-0.72 | best performer - trained on gdc schema matching. calibrated confidence matches actual accuracy.
rulematcher | 15% (3/20) | 30% (6/20) | 0.2000 | 0.90 (uniform) | knowledge-based matching using composite rules. needs scoring tuning - too generous with tier 1 matches.

**Key insight**: high confidence ≠ high accuracy. biobert's 0.90 confidence maps to ~5% accuracy, while magneto's 0.53 confidence maps to actual correctness. phase 3 learning will calibrate confidence scores based on historical performance.

## 10. multi-agent architecture (phase 2 - implemented)

### distinction: matchers vs autonomous agents

component | role | responsibility | example
----------|------|----------------|--------
**matchers** | scoring tools | compute similarity between source/target pairs, return scores 0-1 | BioBERTMatcher, MagnetoMatcher, RuleMatcher
**autonomous agents** | reasoning entities | analyze context, propose mappings with explanations, generate supporting evidence | BioBERTAgent, MagnetoAgent, RuleAgent

### architecture layers

```
BioBERTEmbedder (tools/embeddings/) -> wraps library
BioBERTMatcher (tools/matchers/) -> implements BaseMatcher interface
BioBERTAgent (orchestrator/agents/) -> uses matcher as tool, adds reasoning
Multi-Agent System (orchestrator/) -> collects proposals, arbitrates consensus
```

### autonomous agent interface

**file**: `orchestrator/agents/base_agent.py`

```python
@dataclass
class MappingProposal:
    """agent's proposed mapping with reasoning."""
    source_field: str
    target_field: str
    confidence: float
    reasoning: str
    supporting_evidence: Optional[dict] = None

class AutonomousAgent(ABC):
    """autonomous agent that proposes mappings with reasoning."""

    @abstractmethod
    def propose_mappings(
        self, source_field: str, candidate_targets: List[str], context: dict
    ) -> List[MappingProposal]:
        """propose mappings with reasoning and confidence scores."""
        pass

    @abstractmethod
    def explain_decision(self, proposal: MappingProposal) -> str:
        """provide detailed explanation for a mapping proposal."""
        pass
```

### implemented agents

agent | matcher used | reasoning approach | performance on case_id
------|--------------|-------------------|----------------------
BioBERTAgent | BioBERTMatcher | semantic similarity in biomedical embedding space | top-1: Patient.id (0.861), avg consensus: 0.764
MagnetoAgent | MagnetoMatcher | schema patterns learned from gdc training data | top-1: Patient.id (0.532), calibrated confidence
RuleAgent | RuleMatcher | knowledge base rules with tier-based scoring | top-1: Patient.id (0.900), needs calibration tuning

### multi-agent consensus example

**task**: map `case_id` to fhir resource

**agent proposals**:
- BioBERTAgent: Patient.id (0.861) - "biobert semantic similarity: 0.861. 'case_id' and 'Patient.id' share biomedical context."
- MagnetoAgent: Patient.id (0.532) - "magneto schema matching: 0.532. trained on gdc->fhir mappings, recognizes structural patterns."
- RuleAgent: Patient.id (0.900) - "no direct rule match for case_id -> Patient.id (inferred score: 0.900)"

**consensus**: 3/3 agents agree on Patient.id, average confidence: 0.764

**ground truth**: Patient.id (correct)

### agent registry

dynamic loading from config:

```python
from schema_crush.orchestrator.agents import get_agent

agent = get_agent("biobert", matcher=biobert_matcher)
agent = get_agent("magneto", matcher=magneto_matcher)
agent = get_agent("rule", matcher=rule_matcher)
```

### phase 2 deliverables (completed)

- AutonomousAgent base interface with MappingProposal dataclass
- BioBERTAgent, MagnetoAgent, RuleAgent implementations
- BioBERTMatcher, MagnetoMatcher wrapping embedders
- Agent registry for dynamic loading (AGENT_REGISTRY)
- Multi-agent consensus demonstration (3/3 agreement on Patient.id)
- All tests passing (5/5 in test_autonomous_agents.py)

### phase 3 - llm autonomous agent (completed)

**critical refactor**: phase 2 agents were not true autonomous agents - they were wrappers around matchers with templated reasoning strings. phase 3 replaces them with a single LLM-based agent that has access to matcher tools.

#### new architecture: tools vs agents

component | type | responsibility
----------|------|---------------
BioBERTMatcher, MagnetoMatcher, RuleMatcher | tools | scoring functions that return similarity scores
biobert_match, magneto_match, rule_match | langchain tools | @tool decorated functions for Claude to call
ClaudeAgent | autonomous agent | LLM with tool access, makes reasoned decisions

#### ClaudeAgent implementation

**file**: `orchestrator/agents/claude_agent.py`

```python
class ClaudeAgent(AutonomousAgent):
    """autonomous llm agent using claude with tool access for schema mapping.

    this agent can perform three types of mapping tasks:
    1. entity matching (source entity -> fhir resource)
    2. field matching (source field -> fhir field path)
    3. content matching (source values -> fhir coded values)
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: Optional[str] = None, task: str = "field"):
        self.llm = ChatAnthropic(model=model, api_key=api_key, temperature=0).bind_tools(MAPPING_TOOLS)
        self.system_prompt = self._get_task_prompt(task)  # task-specific prompts
```

#### task-specific prompts

task | prompt | guidelines
-----|--------|------------
entity | ENTITY_MATCHING_PROMPT | map source entity/table -> fhir resource type
field | FIELD_MATCHING_PROMPT | map source field -> fhir field path (prioritize magneto for structural matching)
content | CONTENT_MATCHING_PROMPT | map source values -> fhir coded values with appropriate coding system

#### agentic loop (multi-turn tool calling)

```python
# 1. initial llm call
response = self.llm.invoke([SystemMessage(prompt), HumanMessage(query)])

# 2. execute tools iteratively (max 5 iterations)
while hasattr(response, 'tool_calls') and response.tool_calls:
    for tool_call in response.tool_calls:
        tool_result = tool_map[tool_name].invoke(tool_args)
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call['id']))

    # 3. invoke llm again with tool results
    response = self.llm.invoke(messages)

# 4. parse structured output
CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
```

#### example: tissue_preservation_method mapping

**query**: map "tissue_preservation_method" to fhir field

**candidates**: Specimen.type.coding.display, Specimen.collection.method, Specimen.processing.method, Observation.method, Procedure.code, Specimen.container.type

**agentic reasoning process**:

iteration | tool called | top result | score
----------|-------------|------------|------
1 | magneto_match | Specimen.processing.method | 0.60
1 | magneto_match | Specimen.collection.method | 0.55
2 | biobert_match | Specimen.processing.method | 0.82
2 | biobert_match | Specimen.collection.method | 0.82
3 | rule_match | (no matches) | 0.0

**claude's reasoning**:
```
CHOSEN TARGET: Specimen.processing.method
CONFIDENCE: 0.75
REASONING:

"Tissue preservation method" refers to techniques used to maintain tissue
integrity after collection (e.g., formalin fixation, freezing, paraffin
embedding). This is fundamentally a processing step that occurs after the
initial collection/procurement of the specimen.

Evidence:
1. Magneto structural matching (trained on GDC->FHIR): Ranks
   Specimen.processing.method highest (0.60) followed by
   Specimen.collection.method (0.55)

2. BioBERT semantic matching: Shows high semantic similarity for both
   processing.method (0.82) and collection.method (0.82)

3. Domain knowledge: In biomedical workflows, preservation is typically
   considered a processing step in the specimen lifecycle, occurring
   post-collection to maintain specimen quality for analysis.
```

**ground truth**: Specimen.collection.method (different from Claude's choice, but Claude's reasoning is defensible - preservation could be either collection or processing)

#### key advantages

1. **true autonomy**: Claude decides which tools to call and when, not hardcoded logic
2. **multi-turn reasoning**: can call multiple tools in sequence to gather evidence
3. **explainable decisions**: provides detailed reasoning based on tool results
4. **task flexibility**: same agent handles entity/field/content matching via different prompts
5. **tool composability**: easy to add new matcher tools without changing agent code

#### tools interface

**file**: `orchestrator/agents/tools.py`

```python
@tool
def biobert_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use biobert embeddings for biomedical semantic similarity matching."""
    matcher = get_biobert_matcher()
    results = matcher.match(source, candidates)
    return [{"target": tgt, "score": float(score)} for tgt, score in results[:5]]

@tool
def magneto_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use magneto embeddings for schema structure matching."""
    matcher = get_magneto_matcher()
    results = matcher.match(source, candidates)
    return [{"target": tgt, "score": float(score)} for tgt, score in results[:5]]

@tool
def rule_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use knowledge base rules for mapping lookup."""
    matcher = get_rule_matcher()
    results = matcher.match(source, candidates)
    return [{"target": tgt, "score": float(score), ...} for tgt, score in results[:5]]

MAPPING_TOOLS = [biobert_match, magneto_match, rule_match]
```

#### deleted files (fake agents)

- orchestrator/agents/biobert_agent.py (replaced by tools)
- orchestrator/agents/magneto_agent.py (replaced by tools)
- orchestrator/agents/rule_agent.py (replaced by tools)
- orchestrator/agents/registry.py (no longer needed)

#### phase 3 deliverables (completed)

- ClaudeAgent with multi-turn agentic loop
- Three task-specific prompts (entity, field, content)
- LangChain @tool wrappers for matchers
- Structured output parsing (CHOSEN TARGET, CONFIDENCE, REASONING)
- Demo script (examples/demo_claude_agent.py)
- Proper tool execution with ToolMessage feedback

#### next: phase 4 - consensus and learning

planned components:
- Multi-agent orchestrator (collect proposals from multiple ClaudeAgent instances)
- AdaptiveWeightManager (adjust tool weights based on accuracy)
- ConfidenceCalibrator (calibrate overconfident tools like biobert)
- ChromaDB integration (store proposals and outcomes for learning)

#### performance optimization options

current performance (single mapping):
- initialization: 4-5s (model loading with warmup)
- claude api: 25-28s (3-4 iterations with tool calling)
- total: ~32s

further optimization options (if needed):
1. **async tool calling**: call tools in parallel instead of sequentially
2. **smaller models**: use distilled/quantized versions of biobert/magneto
3. **rule caching**: pre-compute rule embeddings and cache to disk
4. **streaming responses**: show claude's reasoning in real-time
5. **batch mode**: process multiple mappings in one session

#### improving ground truth alignment

when claude disagrees with ground truth (ex, preservation -> processing vs collection):
1. **accept claude's reasoning**: it's making valid domain-based decisions with evidence
2. **adjust the prompt**: add more specific guidance about your data model's definitions
3. **use this for learning**: feed back corrections to train confidence calibration
4. **add more training data**: include examples where preservation -> collection.method in knowledge base
5. **domain-specific rules**: add explicit rules for edge cases in RuleDatabase 
