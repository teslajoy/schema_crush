# schema_crush deep dive walkthrough
## 20-30 minute technical presentation

**audience**: computational biologists (ML/embeddings background), senior engineers, mixed graph experience

**emphasis**: embeddings math, agent learning loop, thresholds, step-by-step structure

---

## table of contents

1. [architecture diagram](#1-architecture-diagram)
2. [module dependency graph](#2-module-dependency-graph)
3. [initialization order](#3-initialization-order)
4. [data structures](#4-data-structures)
5. [matchers deep dive](#5-matchers-deep-dive)
6. [embeddings math deep dive](#6-embeddings-math-deep-dive) ★ for comp bio
7. [calibration and thresholds](#7-calibration-and-thresholds) ★ learning math
8. [knowledge base](#8-knowledge-base)
9. [what is an agent](#9-what-is-an-agent) ★ agent fundamentals
10. [langchain tools](#10-langchain-tools)
11. [claude agent step-by-step](#11-claude-agent-step-by-step)
12. [langgraph explained](#12-langgraph-explained) ★ graph basics first
13. [pearl agent workflow](#13-pearl-agent-workflow)
14. [glossary of terms](#14-glossary-of-terms)

---

## 1. architecture diagram

```
                            SCHEMA_CRUSH ARCHITECTURE
====================================================================================

                         +---------------------------+
                         |   Source Schema           |
                         |   (GDC / HTAN / custom)   |
                         +---------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                              DATA LAYER                                           |
|  +------------------+    +-------------------+    +--------------------+           |
|  | flat.py:15-89    |    | flat_loader.py    |    | content_loader.py  |           |
|  |                  |    | :load_flat_       |    | :load_content_     |           |
|  | Source           |<---| mappings():159    |<---| mappings():45      |           |
|  | Destination      |    |                   |    |                    |           |
|  | FlatMappingDB    |    | JSON -> DB        |    | YAML -> Content    |           |
|  +------------------+    +-------------------+    +--------------------+           |
+-----------------------------------------------------------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
                    v                 v                 v
+-----------------------------------------------------------------------------------+
|                            MATCHING LAYER                                         |
|  +----------------+    +------------------+    +------------------+                |
|  | rule_matcher   |    | biobert_matcher  |    | magneto_matcher  |                |
|  | .py:18-156     |    | .py:12-98        |    | .py:12-95        |                |
|  |                |    |                  |    |                  |                |
|  | O(1) lookup    |    | BioBERT model    |    | Magneto model    |                |
|  | exact match    |    | semantic sim     |    | structure sim    |                |
|  | score: 0/0.5/1 |    | cosine distance  |    | cosine distance  |                |
|  +----------------+    +------------------+    +------------------+                |
|         |                      |                       |                          |
|         +----------------------+-----------------------+                          |
|                                |                                                  |
|                                v                                                  |
|                    +----------------------+                                       |
|                    | calibration.py:45-180|                                       |
|                    | ConfidenceCalibrator |                                       |
|                    | raw -> calibrated    |                                       |
|                    +----------------------+                                       |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                          KNOWLEDGE LAYER                                          |
|  +------------------------+         +------------------------+                    |
|  | knowledge_base.py:8-85 |         | vector_store.py:7-69   |                    |
|  | KnowledgeBase          |-------->| MappingVectorStore     |                    |
|  |  .db                   |         | ChromaDB index         |                    |
|  |  .calibrators          |         | find_similar(k=5)      |                    |
|  |  .embeddings           |         +------------------------+                    |
|  |  .vectors              |                                                       |
|  +------------------------+                                                       |
+-----------------------------------------------------------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
+-----------------------------------------------------------------------------------+
|                           AGENT LAYER                                             |
|  +----------------------------------+    +----------------------------------+     |
|  | tools.py:61-235                  |    | claude_agent.py:85-394           |     |
|  |                                  |    |                                  |     |
|  | @tool biobert_match()            |    | ClaudeAgent                      |     |
|  | @tool magneto_match()            |--->| .propose_mappings()              |     |
|  | @tool rule_match()               |    | .fast_path (kb lookup)           |     |
|  | @tool explore_fhir_resource()    |    | .tool_calling_loop               |     |
|  | @tool search_fhir_fields()       |    | .parse_response                  |     |
|  +----------------------------------+    +----------------------------------+     |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        ORCHESTRATION LAYER                                        |
|  +------------------------------------------------------------------------+       |
|  | pearl_agent.py:1-700                                                   |       |
|  |                                                                        |       |
|  |  StateGraph (langgraph)                                                |       |
|  |  +----------+    +--------+    +-----+    +------+    +-------+        |       |
|  |  | perceive |--->| reason |--->| act |--->| hitl |--->| learn |        |       |
|  |  +----------+    +--------+    +-----+    +------+    +-------+        |       |
|  |       ^                                                   |            |       |
|  |       |              +----------+                         |            |       |
|  |       +--------------| bump_tier|<------------------------+            |       |
|  |                      +----------+                                      |       |
|  +------------------------------------------------------------------------+       |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
                         +---------------------------+
                         |   FHIR Mapping Output     |
                         |   MappingProposal[]       |
                         +---------------------------+
```

---

## 2. module dependency graph

```
IMPORT DEPENDENCIES (who imports what)
======================================

Level 0: No dependencies (pure data structures)
-----------------------------------------------
mappings/flat.py
  - Source, Destination, Tier, FlatMappingDatabase
  - ContentValue, ContentFhirTarget
  - imports: dataclasses, typing, enum (stdlib only)

orchestrator/agents/base_agent.py
  - AutonomousAgent (ABC), MappingProposal
  - imports: abc, dataclasses (stdlib only)


Level 1: Depends on Level 0
---------------------------
mappings/flat_loader.py
  └── imports: flat.py (Source, Destination, Tier, FlatMappingDatabase)

mappings/content_loader.py
  └── imports: flat.py (FlatMappingDatabase, ContentValue, ContentFhirTarget)

tools/embeddings/biobert_embedder.py
  └── imports: sentence_transformers, torch (external only)

tools/embeddings/magneto_embedder.py
  └── imports: sentence_transformers, torch (external only)


Level 2: Depends on Level 1
---------------------------
mappings/expert_embeddings.py
  └── imports: flat_loader.py (load_flat_mappings)

tools/matchers/rule_matcher.py
  ├── imports: flat.py (FlatMappingDatabase, Tier)
  └── imports: flat_loader.py (load_flat_mappings)

tools/matchers/biobert_matcher.py
  ├── imports: biobert_embedder.py (BioBERTEmbedder)
  └── imports: expert_embeddings.py (load_expert_embeddings)

tools/matchers/magneto_matcher.py
  ├── imports: magneto_embedder.py (MagnetoEmbedder)
  └── imports: expert_embeddings.py (load_expert_embeddings)


Level 3: Depends on Level 2
---------------------------
orchestrator/calibration.py
  └── imports: (none from schema_crush - standalone)

tools/fhir_schema_tool.py
  └── imports: (none from schema_crush - uses linkml yaml)

learning/vector_store.py
  └── imports: (none from schema_crush - uses chromadb)


Level 4: Depends on Level 3
---------------------------
learning/knowledge_base.py
  ├── imports: flat_loader.py (load_flat_mappings)
  ├── imports: calibration.py (load_calibrators)
  ├── imports: expert_embeddings.py (load_expert_embeddings)
  └── imports: vector_store.py (MappingVectorStore)


Level 5: Depends on Level 4
---------------------------
orchestrator/agents/tools.py
  ├── imports: matchers/* (BioBERTMatcher, MagnetoMatcher, RuleMatcher)
  ├── imports: knowledge_base.py (KnowledgeBase)
  └── imports: fhir_schema_tool.py (get_schema_explorer)


Level 6: Depends on Level 5
---------------------------
orchestrator/agents/claude_agent.py
  ├── imports: base_agent.py (AutonomousAgent, MappingProposal)
  └── imports: tools.py (MAPPING_TOOLS, warmup_matchers, get_knowledge_base)


Level 7: Depends on Level 6
---------------------------
orchestrator/pearl_agent.py
  ├── imports: embedders (BioBERTEmbedder, MagnetoEmbedder)
  └── imports: langgraph (StateGraph, END, MemorySaver)
```

---

## 3. initialization order

```
STARTUP SEQUENCE (what loads when)
==================================

1. FlatMappingDatabase created
   file: mappings/flat.py:45-89

   db = FlatMappingDatabase()
   db.sources = {}           # dict[str, Source]
   db.destinations = {}      # dict[int, Destination]
   db._source_index = {}     # dict[str, list[int]] for O(1) lookup
   db.content_values = []    # list[ContentValue]
   db.content_fhir_targets = [] # list[ContentFhirTarget]


2. load_flat_mappings() builds database
   file: mappings/flat_loader.py:159-220

   def load_flat_mappings():
       # check sqlite cache first
       if cache_exists and not stale:
           return db.load_sqlite(cache_path)

       # build from json
       db = FlatMappingDatabase()
       _load_gdc(db)      # GDC entity/field mappings
       _load_htan(db)     # HTAN entity/field mappings
       load_content_mappings(db)  # content tier

       db.save_sqlite(cache_path)
       return db


3. Matchers initialized (on demand)
   file: tools.py:22-43

   def get_biobert_matcher():
       kb = get_knowledge_base()
       if not hasattr(kb, '_biobert_matcher'):
           kb._biobert_matcher = BioBERTMatcher(use_expert_embeddings=True)
           # loads: dmis-lab/biobert-base-cased-v1.1 (~400MB)
       return kb._biobert_matcher


4. KnowledgeBase.load() assembles all
   file: learning/knowledge_base.py:25-55

   @classmethod
   def load(cls, include_vectors=True):
       kb = cls()
       kb.db = load_flat_mappings()           # level 1
       kb.calibrators = load_calibrators()     # level 3
       kb.embeddings = load_expert_embeddings() # level 2
       if include_vectors:
           kb.vectors = MappingVectorStore()   # level 3
           kb.vectors.index(kb.db)             # build chromadb index
       return kb


5. ClaudeAgent warmup (optional)
   file: claude_agent.py:114-117

   if warmup:
       warmup_matchers()  # pre-load biobert, magneto, rule
       # avoids 10-30 second delay on first call


LAZY LOADING PATTERN
====================
- Models load ONLY when first matcher.match() called
- Expert embeddings load ONLY if use_expert_embeddings=True
- Vector index builds ONLY if include_vectors=True
- LLM skips entirely if fast_path finds exact match
```

---

## 4. data structures

### 4a. source (entity/field being mapped)

```python
# file: mappings/flat.py:15-25

@dataclass
class Source:
    id: int                    # unique identifier
    source: str                # "case", "case_id", "biospecimen"
    source_context: str        # parent entity: "demographic" for "gender"
    tier: Tier                 # ENTITY, FIELD, or CONTENT

# example instances:
Source(id=1, source="case", source_context="", tier=Tier.ENTITY)
Source(id=15, source="case_id", source_context="case", tier=Tier.FIELD)
Source(id=42, source="gender", source_context="demographic", tier=Tier.FIELD)
```

### 4b. destination (fhir target)

```python
# file: mappings/flat.py:28-38

@dataclass
class Destination:
    id: int                    # unique identifier
    destination: str           # "Patient", "Patient.id", "Specimen.type"
    tier: Tier                 # ENTITY, FIELD, or CONTENT
    source_id: int             # foreign key to Source.id

# example instances:
Destination(id=1, destination="Patient", tier=Tier.ENTITY, source_id=1)
Destination(id=15, destination="Patient.id", tier=Tier.FIELD, source_id=15)
```

### 4c. content value (source data values)

```python
# file: mappings/flat.py:41-48

@dataclass
class ContentValue:
    id: int                    # unique identifier
    source_value: str          # "Male", "Adenocarcinoma", "Fresh Frozen"
    source_field_id: int       # foreign key to Source.id (which field)

# example:
ContentValue(id=1, source_value="Male", source_field_id=42)  # gender field
```

### 4d. content fhir target (coded values)

```python
# file: mappings/flat.py:51-59

@dataclass
class ContentFhirTarget:
    id: int                    # unique identifier
    fhir_path: str             # "Patient.gender"
    fhir_value: str            # "male" (fhir code)
    coding_system: str         # "http://hl7.org/fhir/administrative-gender"
    content_value_id: int      # foreign key to ContentValue.id

# example:
ContentFhirTarget(id=1, fhir_path="Patient.gender", fhir_value="male",
                  coding_system="administrative-gender", content_value_id=1)
```

### 4e. flatmappingdatabase (main storage)

```python
# file: mappings/flat.py:62-89

@dataclass
class FlatMappingDatabase:
    sources: Dict[int, Source] = field(default_factory=dict)
    destinations: Dict[int, Destination] = field(default_factory=dict)
    content_values: List[ContentValue] = field(default_factory=list)
    content_fhir_targets: List[ContentFhirTarget] = field(default_factory=list)

    # internal indexes for O(1) lookup
    _source_index: Dict[str, List[int]]      # source_name -> [source_ids]
    _dest_by_source: Dict[int, List[Destination]]  # source_id -> [destinations]

    def lookup(self, source: str, context: str = None) -> List[Tuple[Source, Destination]]:
        """O(1) lookup by source name."""
        # uses _source_index for instant retrieval
```

### 4f. mapping proposal (agent output)

```python
# file: orchestrator/agents/base_agent.py:8-18

@dataclass
class MappingProposal:
    source_field: str          # what we're mapping from
    target_field: str          # what we're mapping to
    confidence: float          # 0.0 - 1.0 calibrated score
    reasoning: str             # why this mapping was chosen
    supporting_evidence: dict  # tool results, scores, etc.

# example:
MappingProposal(
    source_field="tumor_stage",
    target_field="Condition.stage",
    confidence=0.92,
    reasoning="magneto and biobert agree on Condition.stage...",
    supporting_evidence={"biobert": 0.89, "magneto": 0.94, "rule": 0.0}
)
```

---

## 5. matchers deep dive

### 5a. base matcher interface

```python
# file: tools/matchers/base.py:5-25

class BaseMatcher(ABC):
    """abstract base class for all matchers."""

    @abstractmethod
    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        """
        match source to best targets.

        args:
            source: source field name ("case_id", "tumor_stage")
            targets: candidate fhir targets ["Patient.id", "Condition.stage"]

        returns:
            list of (target, score) sorted by score descending
            score is 0.0 - 1.0
        """
        pass
```

### 5b. rule matcher (o(1) lookup)

```python
# file: tools/matchers/rule_matcher.py:18-80

class RuleMatcher(BaseMatcher):
    """knowledge base lookup matcher."""

    def __init__(self, db: FlatMappingDatabase = None):
        self.db = db or load_flat_mappings()

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        # O(1) lookup in database
        results = self.db.lookup(source)

        scores = []
        for target in targets:
            score = 0.0
            for src, dest in results:
                if dest.destination.lower() == target.lower():
                    score = 1.0  # exact match
                    break
                elif dest.destination.split('.')[0].lower() == target.split('.')[0].lower():
                    score = 0.5  # resource match only
            scores.append((target, score))

        return sorted(scores, key=lambda x: -x[1])

# scoring:
#   1.0 = exact destination match (case -> Patient, case_id -> Patient.id)
#   0.5 = resource match only (case_id -> Patient when dest is Patient.identifier)
#   0.0 = no match in knowledge base
```

### 5c. biobert matcher (semantic)

```python
# file: tools/matchers/biobert_matcher.py:12-60

class BioBERTMatcher(BaseMatcher):
    """biomedical semantic similarity using biobert."""

    def __init__(self, use_expert_embeddings: bool = False):
        self.model = BioBERTEmbedder()  # loads dmis-lab/biobert-base-cased-v1.1
        self.use_expert = use_expert_embeddings
        self.expert_embs = None

        if use_expert_embeddings:
            try:
                self.expert_embs = load_expert_embeddings()
            except FileNotFoundError:
                pass

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        # check expert embeddings first (pre-computed for known mappings)
        if self.expert_embs and source in self.expert_embs:
            source_emb = self.expert_embs[source]
        else:
            source_emb = self.model.encode([source])[0]

        target_embs = self.model.encode(targets)

        # cosine similarity
        scores = cosine_similarity([source_emb], target_embs)[0]

        results = list(zip(targets, scores))
        return sorted(results, key=lambda x: -x[1])

# biobert strength: biomedical terminology
#   "adenocarcinoma" <-> "malignant neoplasm" = high similarity
#   "specimen" <-> "biospecimen" = high similarity
```

### 5d. magneto matcher (structure)

```python
# file: tools/matchers/magneto_matcher.py:12-58

class MagnetoMatcher(BaseMatcher):
    """schema structure matching using magneto embeddings."""

    def __init__(self, use_expert_embeddings: bool = False):
        self.model = MagnetoEmbedder()  # trained on schema matching
        # ... similar to biobert but different model

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        # similar flow to biobert
        # ...

# magneto strength: field name patterns
#   "case_id" <-> "Patient.id" = high similarity (id patterns)
#   "primary_diagnosis" <-> "Condition.code" = high similarity (trained on gdc->fhir)
```

### 5e. matcher comparison

```
MATCHER COMPARISON TABLE
========================

| Matcher  | Speed    | Strength                    | Weakness              |
|----------|----------|-----------------------------|-----------------------|
| Rule     | O(1)     | exact matches, 100% precise | no generalization     |
| BioBERT  | O(n)     | biomedical semantics        | not trained on schema |
| Magneto  | O(n)     | schema structure patterns   | domain-specific       |

WHEN TO USE EACH:
- Rule: known mappings, fast path, high confidence
- BioBERT: medical terminology, content values
- Magneto: field names, entity matching, schema patterns
```

---

## 6. embeddings math deep dive

★ **for computational biologists** - you know this, but here's how we use it

### 6a. what is an embedding?

```
TEXT TO VECTOR
==============

input: "adenocarcinoma"
           |
           v
    +-------------+
    | transformer |  <- BioBERT (12 layers, 768 hidden dim)
    | encoder     |     trained on PubMed abstracts
    +-------------+
           |
           v
output: [0.023, -0.156, 0.891, ..., 0.042]  <- 768-dimensional vector
        ^                                ^
        |                                |
    encodes "cancer"              encodes "glandular"
    semantics                     tissue origin


WHY IT WORKS:
- transformer attention learns contextual relationships
- similar concepts -> similar vectors
- trained on 18B tokens of biomedical text (BioBERT)
```

### 6b. cosine similarity explained

```
COSINE SIMILARITY
=================

formula: cos(θ) = (A · B) / (||A|| × ||B||)

         A · B = dot product = Σ(a_i × b_i)
         ||A|| = magnitude = √(Σa_i²)

geometric interpretation:

         ^  B
        /|
       / |
      /  |
     /θ  |
    +----+---> A

    cos(0°) = 1.0   <- vectors point same direction (identical meaning)
    cos(90°) = 0.0  <- orthogonal (unrelated concepts)
    cos(180°) = -1.0 <- opposite (antonyms, rare in practice)


EXAMPLE:

    "specimen"  -> [0.8, 0.2, 0.1, ...]
    "biospecimen" -> [0.78, 0.22, 0.09, ...]

    cos(θ) = 0.97  <- very similar!

    "specimen"  -> [0.8, 0.2, 0.1, ...]
    "diagnosis" -> [0.1, 0.9, 0.3, ...]

    cos(θ) = 0.42  <- less similar
```

### 6c. why cosine over euclidean?

```
COSINE VS EUCLIDEAN
===================

euclidean distance: d = √(Σ(a_i - b_i)²)

problem with euclidean:

    "specimen" (long document) -> [0.8, 0.2, 0.1, ...] × 10 = large magnitude
    "specimen" (short text)    -> [0.08, 0.02, 0.01, ...]   = small magnitude

    euclidean sees these as DIFFERENT (different magnitudes)
    cosine sees these as SAME (same direction)

cosine is magnitude-invariant:
    - normalizes vectors to unit length
    - compares DIRECTION not LENGTH
    - better for text where document length varies
```

### 6d. expert embeddings (pre-computed)

```
EXPERT EMBEDDINGS OPTIMIZATION
==============================

problem:
    BioBERT encodes "tumor_stage" at runtime = 50-100ms
    for 1000 fields = 50-100 seconds!

solution: pre-compute embeddings for known sources

    FlatMappingDatabase
         |
         v
    for each source in db.sources:
        emb = biobert.encode(source.source)
        expert_embs[source.source] = emb
         |
         v
    save to pickle file (~5MB)

runtime lookup:
    if source in expert_embs:
        return expert_embs[source]  # O(1) dict lookup, instant
    else:
        return biobert.encode(source)  # fallback for unknown

speedup: ~1000x for known mappings
```

### 6e. embedding space visualization

```
UMAP PROJECTION
===============

problem: 768 dimensions is hard to visualize
solution: dimensionality reduction

    768D embedding space
         |
         v
    +--------+
    | UMAP   |  <- Uniform Manifold Approximation and Projection
    +--------+     preserves local structure (neighbors stay neighbors)
         |
         v
    2D plot

    case ●         ● patient
              ●              <- cluster: clinical entities
         diagnosis

                        ● specimen
                  ● biospecimen   <- cluster: sample entities
              ● tissue

interpretation:
    - clusters = semantic similarity
    - distance ≈ cosine distance (roughly)
    - useful for debugging matcher behavior
```

---

## 7. calibration and thresholds

★ **learning math** - how the system learns from feedback

### 7a. why calibration?

```
RAW SCORES ARE NOT CONFIDENCE
=============================

Problem:
  BioBERT says "case" -> "Patient" has score 0.72
  But historically, 0.72 from BioBERT is correct 95% of the time!

  Magneto says "case_id" -> "Patient.id" has score 0.85
  But historically, 0.85 from Magneto is correct 89% of the time!

Solution: Calibration
  Map raw scores to actual historical accuracy
  0.72 (biobert raw) -> 0.95 (calibrated confidence)
  0.85 (magneto raw) -> 0.89 (calibrated confidence)
```

### 6b. calibration implementation

```python
# file: orchestrator/calibration.py:45-120

@dataclass
class CalibrationBin:
    lower: float           # bin lower bound (e.g., 0.7)
    upper: float           # bin upper bound (e.g., 0.8)
    accuracy: float        # historical accuracy in this bin
    count: int             # number of samples in bin

class ConfidenceCalibrator:
    """isotonic regression-style calibration."""

    def __init__(self):
        self.calibration_curves: Dict[str, List[CalibrationBin]] = {}

    def fit(self, matcher_name: str, predictions: List[float], actuals: List[bool]):
        """
        fit calibration curve from historical data.

        args:
            matcher_name: "biobert", "magneto", "rule"
            predictions: raw scores [0.72, 0.85, 0.91, ...]
            actuals: was prediction correct? [True, True, False, ...]
        """
        # bin predictions by score range
        # calculate accuracy per bin
        # store as calibration curve

    def calibrate_score(self, matcher_name: str, raw_score: float) -> float:
        """convert raw score to calibrated confidence."""
        bins = self.calibration_curves.get(matcher_name, [])
        for bin in bins:
            if bin.lower <= raw_score < bin.upper:
                return bin.accuracy
        return raw_score  # fallback
```

### 6c. tier-aware calibration

```
CALIBRATION BY TIER
===================

Different tiers have different accuracy patterns:

Entity tier (case -> Patient):
  - Usually high accuracy (limited options)
  - BioBERT: 0.7 raw -> 0.98 calibrated

Field tier (case_id -> Patient.id):
  - Medium accuracy (many field options)
  - BioBERT: 0.7 raw -> 0.92 calibrated

Content tier (Male -> gender/male):
  - Variable accuracy (terminology mapping)
  - BioBERT: 0.7 raw -> 0.85 calibrated

Implementation:
  calibrators["biobert:entity"] = entity_curve
  calibrators["biobert:field"] = field_curve
  calibrators["biobert:content"] = content_curve
```

---

## 7. knowledge base

### 7a. unified access pattern

```python
# file: learning/knowledge_base.py:8-85

@dataclass
class KnowledgeBase:
    """single entry point for all knowledge components."""

    db: FlatMappingDatabase = None      # rule-based knowledge
    calibrators: Dict = None             # score calibration
    embeddings: Dict = None              # pre-computed expert embeddings
    vectors: MappingVectorStore = None   # semantic search index

    @classmethod
    def load(cls, include_vectors: bool = True) -> "KnowledgeBase":
        """factory method to load all components."""
        kb = cls()
        kb.db = load_flat_mappings()
        kb.calibrators = load_calibrators()

        try:
            kb.embeddings = load_expert_embeddings()
        except FileNotFoundError:
            kb.embeddings = None

        if include_vectors:
            kb.vectors = MappingVectorStore()
            kb.vectors.index(kb.db)

        return kb
```

### 7b. vector store (semantic search)

```python
# file: learning/vector_store.py:7-69

class MappingVectorStore:
    """chromadb vector store for semantic retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("mappings")
        self.model = SentenceTransformer(model_name)
        self._indexed = False

    def index(self, db: FlatMappingDatabase):
        """index all source->destination pairs from database."""
        sources, targets, ids = [], [], []

        # entity + field tier
        for src in db.sources.values():
            for dest in db._dest_by_source.get(src.id, []):
                sources.append(src.source)
                targets.append(dest.destination)
                ids.append(f"{src.id}_{dest.id}")

        # content tier
        for cv in db.content_values:
            for ct in db.content_fhir_targets:
                if ct.content_value_id == cv.id:
                    sources.append(cv.source_value)
                    targets.append(ct.fhir_path)
                    ids.append(f"cv_{cv.id}_{ct.id}")

        embeddings = self.model.encode(sources)
        self.collection.add(ids=ids, embeddings=embeddings,
                           metadatas=[{"source": s, "target": t}
                                     for s, t in zip(sources, targets)])
        self._indexed = True

    def find_similar(self, query: str, k: int = 5) -> List[Dict]:
        """find k most similar mappings for few-shot context."""
        query_emb = self.model.encode([query])
        results = self.collection.query(query_embeddings=query_emb, n_results=k)

        return [{"source": m["source"], "target": m["target"],
                 "score": 1 - d}  # distance to similarity
                for m, d in zip(results["metadatas"][0], results["distances"][0])]
```

### 7c. knowledge base methods

```python
# unified interface methods

kb = KnowledgeBase.load()

# O(1) rule lookup
results = kb.lookup("case")  # [(Source, Destination), ...]

# calibrated scoring
calibrated = kb.calibrate("biobert", 0.72, tier="field")  # 0.92

# semantic search for few-shot
similar = kb.find_similar("tumor_stage", k=3)
# [{"source": "tumor_stage", "target": "Condition.stage", "score": 0.95},
#  {"source": "clinical_stage", "target": "Condition.stage", "score": 0.89}, ...]
```

---

## 8. langchain tools

### 8a. @tool decorator explained

```python
# file: orchestrator/agents/tools.py:61-88

from langchain_core.tools import tool

@tool
def biobert_match(source: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """use biobert embeddings for biomedical semantic similarity matching.

    best for: matching biomedical concepts and terminology.

    args:
        source: source field or term to match
        candidates: list of candidate target fields/terms

    returns:
        list of matches with calibrated scores, sorted by confidence
    """
    matcher = get_biobert_matcher()
    results = matcher.match(source, candidates)

    calibrated_results = []
    for tgt, raw_score in results[:5]:
        calibrated_score = apply_calibration("biobert", raw_score)
        calibrated_results.append({"target": tgt, "score": float(calibrated_score)})

    return calibrated_results

# WHAT @tool DOES:
# 1. Parses function signature -> JSON schema for LLM
#    {"source": {"type": "string"}, "candidates": {"type": "array", "items": {"type": "string"}}}
#
# 2. Uses docstring -> tool description for LLM
#    "use biobert embeddings for biomedical semantic similarity matching..."
#
# 3. Wraps function for langchain tool protocol
#    tool.invoke({"source": "case", "candidates": ["Patient", "Specimen"]})
```

### 8b. all available tools

```python
# file: orchestrator/agents/tools.py:235

MAPPING_TOOLS = [
    biobert_match,        # semantic biomedical matching
    magneto_match,        # schema structure matching
    rule_match,           # knowledge base lookup
    explore_fhir_resource, # list fields for a fhir resource
    search_fhir_fields,   # search fields across all resources
]

# tool signatures:
biobert_match(source: str, candidates: List[str]) -> List[Dict]
magneto_match(source: str, candidates: List[str]) -> List[Dict]
rule_match(source: str, candidates: List[str]) -> List[Dict]
explore_fhir_resource(resource_type: str) -> Dict
search_fhir_fields(search_term: str, resource_filter: str = None) -> List[Dict]
```

### 8c. singleton pattern for tools

```python
# file: orchestrator/agents/tools.py:8-44

_knowledge_base = None  # module-level singleton

def get_knowledge_base():
    """get singleton knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase.load()
        print("loaded knowledge base (db, calibrators, vectors)")
    return _knowledge_base

def get_biobert_matcher():
    """get biobert matcher from knowledge base."""
    kb = get_knowledge_base()
    if not hasattr(kb, '_biobert_matcher') or kb._biobert_matcher is None:
        kb._biobert_matcher = BioBERTMatcher(use_expert_embeddings=True)
    return kb._biobert_matcher

# WHY SINGLETON?
# - BioBERT model is ~400MB, takes 10-30 seconds to load
# - Load once, reuse for all tool calls
# - Matchers attached to KnowledgeBase instance
```

---

## 9. claude agent

### 9a. agent initialization

```python
# file: orchestrator/agents/claude_agent.py:94-128

class ClaudeAgent(AutonomousAgent):

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str = None, task: str = "field", warmup: bool = True):

        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.task = task

        # pre-load matchers to avoid first-call latency
        if warmup:
            warmup_matchers()  # loads biobert, magneto, rule

        # initialize llm with tools
        self.llm = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0)
        self.llm = self.llm.bind_tools(MAPPING_TOOLS)  # attach tools

        # select task-specific prompt
        self.system_prompt = {
            "entity": ENTITY_MATCHING_PROMPT,
            "field": FIELD_MATCHING_PROMPT,
            "content": CONTENT_MATCHING_PROMPT
        }[task]
```

### 9b. fast path (skip llm)

```python
# file: orchestrator/agents/claude_agent.py:174-188

def propose_mappings(self, source_field: str, candidate_targets: List[str] = None,
                     context: dict = None) -> List[MappingProposal]:

    kb = get_knowledge_base()

    # FAST PATH: check knowledge base first, skip LLM if exact match
    if candidate_targets:
        rule_results = kb.db.lookup(source_field)
        if rule_results:
            dest_set = {d.destination.lower() for _, d in rule_results}
            for target in candidate_targets:
                if target.lower() in dest_set:
                    return [MappingProposal(
                        source_field=source_field,
                        target_field=target,
                        confidence=1.0,
                        reasoning="exact match from knowledge base (skipped llm)",
                        supporting_evidence={"agent": self.name, "source": "knowledge_base"}
                    )]

    # SLOW PATH: continue to LLM...

# FAST PATH BENEFIT:
# - ~93% of known mappings skip LLM entirely
# - Saves ~$0.01-0.03 per mapping
# - Instant response vs 2-5 second LLM call
```

### 9c. few-shot context

```python
# file: orchestrator/agents/claude_agent.py:190-196

    # get similar mappings for few-shot context
    similar = kb.find_similar(source_field, k=3)
    few_shot_context = ""
    if similar:
        few_shot_context = "\n\nsimilar mappings from knowledge base:\n"
        for s in similar:
            few_shot_context += f"  {s['source']} -> {s['target']} (similarity: {s['score']:.2f})\n"

# EXAMPLE FEW-SHOT CONTEXT:
# similar mappings from knowledge base:
#   tumor_stage -> Condition.stage (similarity: 0.95)
#   clinical_stage -> Condition.stage (similarity: 0.89)
#   pathologic_stage -> Observation.value (similarity: 0.82)
```

### 9d. tool calling loop

```python
# file: orchestrator/agents/claude_agent.py:236-277

    # invoke llm with tools (first call)
    messages = [
        SystemMessage(content=self.system_prompt),
        HumanMessage(content=user_message)
    ]

    response = self.llm.invoke(messages)
    messages.append(response)

    # execute tool calls in a loop
    tool_map = {
        'biobert_match': biobert_match,
        'magneto_match': magneto_match,
        'rule_match': rule_match,
        'explore_fhir_resource': explore_fhir_resource,
        'search_fhir_fields': search_fhir_fields
    }

    max_iterations = 10
    iteration = 0

    while hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations:
        iteration += 1

        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})

            if tool_name in tool_map:
                tool_func = tool_map[tool_name]
                tool_result = tool_func.invoke(tool_args)

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get('id')
                ))

        # invoke llm again with tool results
        response = self.llm.invoke(messages)
        messages.append(response)
```

### 9e. response parsing

```python
# file: orchestrator/agents/claude_agent.py:289-375

def _parse_llm_response(self, response, source_field, candidate_targets, context):
    """parse llm response into mapping proposals."""

    content = response.content  # extract text

    # parse structured format
    # CHOSEN TARGET: Patient.id
    # CONFIDENCE: 0.95
    # REASONING: BioBERT and Magneto both strongly agree...

    target_match = re.search(r'CHOSEN TARGET:\s*(.+?)(?:\n|$)', content)
    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
    reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.DOTALL)

    chosen_target = target_match.group(1).strip() if target_match else candidates[0]
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    reasoning = reasoning_match.group(1).strip() if reasoning_match else content

    return [MappingProposal(
        source_field=source_field,
        target_field=chosen_target,
        confidence=confidence,
        reasoning=reasoning,
        supporting_evidence={"agent": self.name, "llm_response": content}
    )]
```

---

## 10. langgraph pearl agent

### 10a. state definition

```python
# file: orchestrator/pearl_agent.py:55-87

class PEARLState(TypedDict):
    """state for pearl agent workflow."""

    # input
    source_data: Dict[str, Any]           # entity -> dataframe dict
    target_schema: Dict[str, List[str]]   # entity -> field list

    # perception
    source_entities: List[str]            # extracted from source_data
    target_entities: List[str]            # extracted from target_schema
    source_fields: Dict[str, List[str]]   # entity -> fields
    target_fields: Dict[str, List[str]]   # entity -> fields

    # reasoning
    current_tier: str                     # "entity", "field", "content"
    matches_by_tier: Dict[str, List[Dict]]
    embedder_scores: Dict[str, Any]
    reasoning_log: Annotated[List[str], operator.add]  # accumulator

    # action
    filtered_matches: Annotated[List[Dict], operator.add]
    hitl_queue: Annotated[List[Dict], operator.add]

    # learning
    user_feedback: Dict[str, bool]        # match_id -> correct?
    embedder_weights: Dict[str, float]    # dynamic weighting
    metrics: Dict[str, Any]

    # control flow
    next_action: str
    iteration: int
    tiers_to_run: List[str]
    awaiting_hitl: bool
```

### 10b. workflow construction

```python
# file: orchestrator/pearl_agent.py:580-628

def _build_workflow(self) -> StateGraph:
    """build langgraph workflow with conditional tier progression."""

    workflow = StateGraph(PEARLState)

    # add nodes (each is a callable that transforms state)
    workflow.add_node("perceive", self.perceive_agent)   # extract schema info
    workflow.add_node("reason", self.reason_agent)       # generate matches
    workflow.add_node("act", self.act_agent)             # filter/rank
    workflow.add_node("hitl", self.hitl_agent)           # human review
    workflow.add_node("learn", self.learn_agent)         # update from feedback
    workflow.add_node("bump_tier", bump_tier)            # move to next tier

    # set entry point
    workflow.set_entry_point("perceive")

    # add edges (transitions)
    workflow.add_edge("perceive", "reason")
    workflow.add_edge("reason", "act")
    workflow.add_edge("act", "hitl")

    # conditional edge: next tier or finish?
    workflow.add_conditional_edges(
        "learn",
        maybe_next_tier,  # function returns "bump_tier" or "finish"
        {"bump_tier": "bump_tier", "finish": END}
    )

    workflow.add_edge("bump_tier", "reason")

    # compile with hitl interrupt
    return workflow.compile(interrupt_before=["hitl"])
```

### 10c. workflow visualization

```
PEARL WORKFLOW DIAGRAM
======================

    +----------+
    | perceive |  <- extract source/target schema metadata
    +----------+
         |
         v
    +--------+
    | reason |  <- run matchers (biobert, magneto), score candidates
    +--------+
         |
         v
    +-----+
    | act |  <- filter low scores, rank by confidence
    +-----+
         |
         v
    +------+
    | hitl |  <- INTERRUPT: human reviews medium-confidence matches
    +------+
         |
         v
    +-------+
    | learn |  <- update weights from feedback
    +-------+
         |
         v
    [more tiers?]
         |
    +----+----+
    |         |
    v         v
+----------+  END
| bump_tier|
+----------+
    |
    +---> back to "reason" for next tier
```

---

## 11. glossary of terms

### mathematical terms

| Term | Definition | Used In |
|------|------------|---------|
| **cosine similarity** | dot product of unit vectors; measures angle between embeddings; range [-1, 1], usually [0, 1] for positive embeddings | biobert_matcher, magneto_matcher |
| **embedding** | dense vector representation of text; captures semantic meaning in n-dimensional space | all matchers |
| **isotonic regression** | monotonic function fitting; ensures calibration curve is non-decreasing | calibration.py |
| **O(1) lookup** | constant time retrieval regardless of data size; achieved via hash table indexing | rule_matcher, flat.py |
| **confidence calibration** | mapping raw scores to actual accuracy probabilities | calibration.py |

### computer science terms

| Term | Definition | Used In |
|------|------------|---------|
| **singleton pattern** | ensure only one instance of a class exists; used for expensive resources | tools.py (_knowledge_base) |
| **factory method** | class method that creates instances; `KnowledgeBase.load()` | knowledge_base.py |
| **abstract base class (ABC)** | interface definition; cannot be instantiated directly | base.py, base_agent.py |
| **dataclass** | python decorator for auto-generating `__init__`, `__repr__`, etc. | flat.py, base_agent.py |
| **lazy loading** | defer initialization until first use; saves memory/time | all matchers |
| **foreign key** | reference to another table's primary key | Destination.source_id |

### langchain/langgraph terms

| Term | Definition | Used In |
|------|------------|---------|
| **@tool decorator** | wraps function as LLM-callable tool with schema | tools.py |
| **bind_tools()** | attaches tool list to LLM for function calling | claude_agent.py |
| **StateGraph** | directed graph with typed state dict; nodes transform state | pearl_agent.py |
| **add_node()** | register callable as graph node | pearl_agent.py |
| **add_edge()** | define transition between nodes | pearl_agent.py |
| **add_conditional_edges()** | routing based on function output | pearl_agent.py |
| **interrupt_before** | pause workflow before node for human input | pearl_agent.py |
| **MemorySaver** | checkpointer for saving/resuming workflow state | pearl_agent.py |

### domain terms

| Term | Definition | Used In |
|------|------------|---------|
| **FHIR** | Fast Healthcare Interoperability Resources; healthcare data standard | throughout |
| **GDC** | Genomic Data Commons; cancer genomics data model | flat_loader.py |
| **HTAN** | Human Tumor Atlas Network; cancer data model | flat_loader.py |
| **LinkML** | Linked Data Modeling Language; schema definition | fhir_schema_tool.py |
| **tier** | mapping granularity: entity (table), field (column), content (value) | throughout |
| **SNOMED CT** | clinical terminology system for coded values | content tier |

### agent terms

| Term | Definition | Used In |
|------|------------|---------|
| **PEARL** | Perceive-Reason-Act-Learn loop; agent architecture | pearl_agent.py |
| **HITL** | Human-in-the-loop; human review for uncertain cases | pearl_agent.py |
| **few-shot learning** | provide examples to guide model behavior | claude_agent.py |
| **tool calling** | LLM invokes functions to gather information | claude_agent.py |
| **multi-turn** | multiple LLM invocations in conversation | claude_agent.py |

---

## file reference index

```
QUICK REFERENCE: FILE -> LINE NUMBERS
=====================================

DATA LAYER
  mappings/flat.py
    Source: 15-25
    Destination: 28-38
    ContentValue: 41-48
    ContentFhirTarget: 51-59
    FlatMappingDatabase: 62-89

  mappings/flat_loader.py
    load_flat_mappings(): 159-220
    _load_gdc(): 45-90
    _load_htan(): 93-140

MATCHERS
  tools/matchers/base.py
    BaseMatcher: 5-25

  tools/matchers/rule_matcher.py
    RuleMatcher: 18-156

  tools/matchers/biobert_matcher.py
    BioBERTMatcher: 12-98

  tools/matchers/magneto_matcher.py
    MagnetoMatcher: 12-95

CALIBRATION
  orchestrator/calibration.py
    CalibrationBin: 12-18
    ConfidenceCalibrator: 45-180
    load_calibrators(): 183-210

KNOWLEDGE
  learning/knowledge_base.py
    KnowledgeBase: 8-85

  learning/vector_store.py
    MappingVectorStore: 7-69

AGENTS
  orchestrator/agents/tools.py
    get_knowledge_base(): 12-19
    get_biobert_matcher(): 22-27
    biobert_match(): 61-88
    MAPPING_TOOLS: 235

  orchestrator/agents/claude_agent.py
    ClaudeAgent: 85-394
    propose_mappings(): 145-287
    _parse_llm_response(): 289-377

  orchestrator/pearl_agent.py
    PEARLState: 55-87
    PEARLAgent: 524-700
    _build_workflow(): 580-628
```

---

## presentation flow (20-30 min)

```
SUGGESTED PRESENTATION ORDER
============================

1. INTRO (2 min)
   - Show architecture diagram
   - "source schema -> FHIR output"

2. DATA LAYER (5 min)
   - Show FlatMappingDatabase tables
   - Run tutorial cells 3a-3e
   - Explain O(1) lookup

3. MATCHERS (5 min)
   - Show matcher comparison table
   - Run tutorial cell 4
   - Explain when to use each

4. CALIBRATION (3 min)
   - Explain why raw scores != confidence
   - Show calibration transformation

5. KNOWLEDGE BASE (3 min)
   - Show unified access pattern
   - Demonstrate find_similar()

6. LANGCHAIN TOOLS (5 min)
   - Explain @tool decorator
   - Show how LLM sees tools
   - Demonstrate tool invocation

7. CLAUDE AGENT (5 min)
   - Show fast path optimization
   - Walk through tool calling loop
   - Show response parsing

8. LANGGRAPH (5 min)
   - Show StateGraph construction
   - Display workflow diagram
   - Explain HITL interrupt

9. Q&A (remaining time)
```