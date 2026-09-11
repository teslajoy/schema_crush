# mapping workbench design

> **status: design proposal, not implemented.** this document describes a
> planned UI for mapping review. it is included for provenance and to document
> intended direction. no code in this repository implements it. for what is
> built today, see [architecture_current.md](architecture_current.md).

## overview

visual ui for mapping messy sources (geo, tcga, etc.) to fhir schema with drag-and-drop, relationship handling, and learning from feedback.

## workflow per source

```
1. load source schema (csv headers, json keys, etc.)
2. show major entities first (case, sample, file, etc.)
3. for each entity:
   - auto-suggest fhir resource (Patient, Specimen, DocumentReference)
   - show all matcher scores (rule, biobert, magneto)
   - user confirms or picks different resource
4. for each field in entity:
   - show dropdown of valid fhir paths (from fhir schema)
   - pre-select best match with confidence
   - show all scores so user can override if overconfident
5. handle relationships:
   - observation.focus -> specimen
   - specimen.parent -> specimen
   - condition.subject -> patient
6. save mappings to db for o(1) lookup next time
7. update calibrators from user corrections
```

## ui components

```
+------------------------------------------------------------------+
|  source: GEO GSE12345                          [save] [export]   |
+------------------------------------------------------------------+
|                                                                   |
|  source schema              |  fhir target                       |
|  ========================   |  ============================       |
|                             |                                     |
|  [case] -----------------> [Patient v]  (rule: 98%, bio: 85%)    |
|    |                        |                                     |
|    +- case_id -----------> [Patient.identifier v]  (99%)         |
|    +- age_at_diagnosis --> [Patient.birthDate v]   (72%)  [!]    |
|    +- primary_diagnosis -> [Condition.code v]      (91%)         |
|                             |   \-> subject: Patient              |
|                             |                                     |
|  [sample] ---------------> [Specimen v]  (rule: 97%)             |
|    |                        |                                     |
|    +- sample_id ---------> [Specimen.identifier v] (98%)         |
|    +- tissue_type -------> [Specimen.type v]       (87%)  [?]    |
|    +- parent_sample -----> [Specimen.parent v]     (ref)         |
|                             |   \-> parent: Specimen              |
|                             |                                     |
|  [file] -----------------> [DocumentReference v]                 |
|    ...                      |                                     |
|                             |                                     |
+------------------------------------------------------------------+
|  legend: [!] low confidence  [?] medium (needs review)           |
|          [v] dropdown        (ref) reference relationship        |
+------------------------------------------------------------------+
```

## key features

### 1. fhir-aware dropdowns
- populated from linkml schema (already have fhir_schema_tool.py)
- grouped by resource type
- show field descriptions on hover

### 2. relationship handling
- detect when target is a reference type
- show linked resource picker
- validate reference targets exist in mapping

### 3. confidence display
| score range | display | action |
|-------------|---------|--------|
| >= 95% | green checkmark | auto-selected |
| 85-95% | yellow [?] | pre-selected, needs confirm |
| < 85% | red [!] | show but don't select |

### 4. all scores visible
```
tissue_type -> Specimen.type
  rule:    0.87  [====----]
  biobert: 0.92  [=====---]  <- overconfident!
  magneto: 0.65  [===-----]

  calibrated: 0.87
```

### 5. drag and drop
- drag source field
- drop on fhir path
- or drop on "new resource" to create reference

### 6. mapping persistence
```
user confirms mapping
        |
        v
+------------------+
| flatmappingdb    | <- o(1) lookup next time
+------------------+
        |
        v
+------------------+
| calibrators      | <- update confidence curves
+------------------+
```

### 7. smart clustering (batch review)
group similar uncertain fields to reduce review fatigue:
```
[Batch Review: 12 date fields]
created_datetime, updated_datetime, diagnosis_date, birth_date...
-> All map to [*.dateTime]?  [Accept All] [Review Each]

[Batch Review: 8 identifier fields]
case_id, sample_id, patient_id, submitter_id...
-> All map to [*.identifier]?  [Accept All] [Review Each]
```

### 8. smart ordering
show fields in priority order, not alphabetical:
1. uncertain (yellow [?]) - needs human decision
2. low confidence (red [!]) - likely wrong
3. high confidence (green) - just confirm

### 9. inline context preview
show sample values without switching views:
```
age_at_diagnosis -> [Patient.birthDate v]  (72%) [!]
  samples: [45, 67, 32, 58, ...]
  type: integer (likely age in years, not date)
```

## data model additions

### project mappings table
```python
@dataclass
class ProjectMapping:
    project_id: str           # "geo_gse12345"
    source_entity: str        # "case"
    source_field: str         # "age_at_diagnosis"
    target_resource: str      # "Patient"
    target_path: str          # "Patient.birthDate"
    confidence: float         # calibrated score
    user_confirmed: bool      # was this reviewed?
    user_corrected: bool      # did user change it?
    original_prediction: str  # what was auto-suggested
    timestamp: datetime
```

### reference mappings
```python
@dataclass
class ReferenceMapping:
    source_resource: str      # "Observation"
    source_field: str         # "focus"
    target_resource: str      # "Specimen"
    cardinality: str          # "0..1" or "0..*"
```

## tech stack options

| option | pros | cons |
|--------|------|------|
| streamlit | fast to build, python-native | limited drag-drop |
| gradio | python-native, good for ml | limited customization |
| react + fastapi | full control, proper drag-drop | more code |
| panel/holoviz | python-native, flexible | learning curve |

recommendation: **streamlit first** for prototype, migrate to react if needed

## mcp server functions

functions needed for MCP server (used by both Claude agents and React UI):

### core mapping
```python
load_source_schema(path: str, entity_type: str) -> SourceSchema
    # load csv/json, extract fields + sample values
    # returns: {fields: [{name, samples, inferred_type}], entity}

suggest_mapping(source_field: str, entity_type: str, samples: list) -> Suggestion
    # run all matchers, return best with scores
    # returns: {target, confidence, reasoning, scores: {rule, biobert, magneto}}

suggest_all_mappings(schema: SourceSchema) -> list[Suggestion]
    # batch suggest for all fields in schema

get_fhir_paths(resource: str) -> list[str]
    # get valid paths for dropdown (from linkml schema)
    # e.g., get_fhir_paths("Patient") -> ["Patient.identifier", "Patient.gender", ...]

search_fhir_fields(query: str) -> list[dict]
    # search across all resources
```

### hitl review
```python
get_review_queue(threshold: float = 0.90) -> list[ReviewItem]
    # prioritized list: uncertain first, then low confidence
    # returns: [{source, target, confidence, samples, reasoning}]

confirm_mapping(source: str, target: str) -> bool
    # mark as human-confirmed

reject_mapping(source: str, target: str) -> bool
    # mark as rejected, remove from suggestions

correct_mapping(source: str, old_target: str, new_target: str) -> bool
    # record correction for learning

batch_confirm(field_pattern: str, target_pattern: str) -> int
    # e.g., batch_confirm("*_id", "*.identifier") -> 12 confirmed
```

### calibration & learning
```python
get_calibration_stats() -> CalibrationStats
    # returns: {ece, brier, per_bin: [{range, predicted, actual, n}]}

get_matcher_performance() -> dict
    # returns: {rule: {accuracy, notes}, biobert: {...}, magneto: {...}}

get_correction_patterns() -> list[CorrectionPattern]
    # common human overrides
    # returns: [{pattern, wrong_target, correct_target, count}]

update_calibrators() -> bool
    # retrain from accumulated feedback

save_to_knowledge_base() -> bool
    # persist confirmed mappings to flatmappingdb
```

### feedback loop (learning from corrections)
```python
record_feedback(source: str, predicted: str, actual: str, matcher_scores: dict) -> bool
    # log: what we predicted vs what human chose
    # used for recalibration + pattern detection

apply_correction_to_calibrators(source: str, predicted: str, actual: str,
                               matcher_scores: dict, calibrated_conf: float) -> None
    # immediate update: add data point to EACH matcher's calibration bins
    # matcher_scores: {rule: 0.9, biobert: 0.85, magneto: 0.6}
    # updates per-matcher calibration (fixes "biobert overconfident at 80-90%")
    # predicted != actual -> was_correct=False
    # predicted == actual -> was_correct=True

detect_systematic_errors() -> list[SystematicError]
    # analyze corrections to find patterns
    # e.g., "age_* fields often wrongly mapped to birthDate"
    # returns: [{pattern, error_rate, suggested_fix}]

add_learned_rule(source_pattern: str, target: str, confidence: float) -> bool
    # when human corrects multiple times, add as new rule
    # e.g., "survival_months" -> Observation.valueQuantity (learned from 5 corrections)

get_learning_summary() -> LearningSummary
    # returns: {
    #   new_rules_added: 12,
    #   calibration_improved: {old_ece: 0.08, new_ece: 0.05},
    #   patterns_detected: ["age_* -> birthDate is wrong 80% of time"],
    #   total_feedback: 234
    # }
```

### feedback data model
```python
@dataclass
class FeedbackRecord:
    timestamp: datetime
    source_field: str
    predicted_target: str
    actual_target: str           # what human chose
    was_correction: bool         # predicted != actual
    matcher_scores: dict         # {rule: 0.9, biobert: 0.7, magneto: 0.5}
    calibrated_confidence: float
    context: dict                # {entity, schema, samples}
```

### acceptable feedback (in scope)
```
ACCEPT:
- confirm mapping        "yes, this is correct"
- reject mapping         "no, this is wrong"
- correct mapping        "no, it should be X instead"
- skip/unsure            "I don't know, skip for now"
- batch confirm          "accept all *_id -> *.identifier"
- ask for explanation    "why did you suggest this?"
- ask for alternatives   "what else could this map to?"
- request FHIR info      "show me Specimen fields"

REJECT (politely):
- general questions      "what is FHIR?" -> "This tool is for mapping fields. For FHIR docs, see hl7.org/fhir"
- unrelated requests     "write me code" -> "I can only help with schema mappings here."
- data modification      "change my CSV" -> "I can map fields but not modify source data."
- export to unsupported  "export to XML" -> "Supported formats: JSON, LinkML, CSV"
```

### polite rejection responses
```python
REJECTION_RESPONSES = {
    "general_question": "I'm focused on helping you map fields to FHIR. For general FHIR questions, check hl7.org/fhir",
    "off_topic": "I can only help with schema mapping in this tool. Is there a field you'd like to map?",
    "modify_source": "I can map your fields to FHIR but can't modify your source data.",
    "unsupported_export": "I support JSON, LinkML, and CSV exports. Which would you like?",
    "unclear_intent": "I'm not sure what you mean. You can: confirm, reject, correct a mapping, or ask why I suggested it.",
}
```

### vector store (learning from inputs)

uses chromadb to store embeddings and find similar mappings:

```
new source field                   vector store (chromadb)
      |                            +---------------------------+
      v                            | embeddings:               |
 embed(source)  ----search---->    |   "patient_id" -> [0.2,..] |
      |                            |   "sample_type" -> [0.4,..]|
      v                            |   "diagnosis" -> [0.1,..]  |
 similar mappings                  +---------------------------+
      |                            | metadata:                 |
      v                            |   source, target, schema  |
 few-shot context for LLM          |   confirmed, corrected    |
                                   +---------------------------+
```

```python
### vector store functions
index_mapping(source: str, target: str, metadata: dict) -> None
    # add confirmed mapping to vector store
    # metadata: {schema, entity, confidence, user_confirmed}

find_similar(query: str, k: int = 5) -> list[SimilarMapping]
    # find k most similar past mappings
    # returns: [{source, target, similarity, metadata}]

index_feedback(source: str, target: str, was_correct: bool) -> None
    # update vector with feedback signal
    # boost/demote based on human corrections

rebuild_index() -> None
    # reindex all mappings after bulk updates

get_index_stats() -> dict
    # returns: {total_vectors, sources_indexed, schemas_covered}
```

### learning flow
```
1. user uploads new CSV
         |
         v
2. for each field, find_similar() from vector store
         |
         v
3. similar mappings -> few-shot context for LLM
         |
         v
4. LLM + matchers -> suggestion with confidence
         |
         v
5. user confirms/corrects
         |
         v
6. index_mapping() -> add to vector store
   index_feedback() -> update weights
   apply_correction_to_calibrators() -> update confidence curves
         |
         v
7. next similar field benefits from this learning
```

### what gets stored in vector store
```python
@dataclass
class IndexedMapping:
    id: str                      # unique id
    source_text: str             # "tissue_preservation_method"
    source_embedding: list       # [0.23, 0.45, ...] from biobert/magneto
    target: str                  # "Specimen.processing.method"
    metadata: dict               # {
                                 #   schema: "gdc",
                                 #   entity: "sample",
                                 #   confidence: 0.92,
                                 #   user_confirmed: True,
                                 #   times_suggested: 5,
                                 #   times_accepted: 4,
                                 #   times_corrected: 1,
                                 # }
```

### chat/explain
```python
explain_mapping(source: str, target: str) -> str
    # why was this mapping suggested?

get_alternatives(source: str, top_k: int = 5) -> list[Alternative]
    # what other targets could this map to?

ask_about_mapping(source: str, question: str) -> str
    # chat interface: "why not Patient.deceased?"
```

### export & stats
```python
export_mappings(format: str) -> str
    # format: "json" | "linkml" | "csv"

get_session_stats() -> SessionStats
    # returns: {auto_accepted, human_reviewed, corrected, total, avg_review_time}
```

## api endpoints (rest wrapper for mcp)

```
GET  /projects                    # list all mapping projects
POST /projects                    # create new project
GET  /projects/{id}/schema        # get source schema
GET  /projects/{id}/mappings      # get current mappings
POST /projects/{id}/mappings      # save mapping
GET  /fhir/resources              # list fhir resources
GET  /fhir/resources/{type}       # get fields for resource
POST /match                       # run matchers on source field
POST /feedback                    # record user correction
```

## phases

### phase 1: core ui
- [ ] streamlit app with source/target columns
- [ ] fhir dropdown populated from schema
- [ ] show matcher scores
- [ ] save to flatmappingdb

### phase 2: relationships
- [ ] detect reference fields
- [ ] link to other mapped resources
- [ ] validate reference integrity

### phase 3: learning
- [ ] track user corrections
- [ ] update calibrators from feedback
- [ ] improve suggestions over time

### phase 4: batch + scale
- [ ] upload csv/json source
- [ ] process all fields
- [ ] cluster uncertain matches
- [ ] bulk approve/reject

## known gaps & mitigations

| gap | risk | mitigation |
|-----|------|------------|
| no versioning | retrain calibrators, it gets worse, can't rollback | add `rollback_calibrators(version_id)`, save snapshots before updates |
| chromadb collection strategy | one big collection = cross-contamination; per-project = no cross-learning | **per-schema collection** (gdc, htan, geo) with project metadata filter |
| concurrency | multiple users mapping same schema, conflicting feedback | optimistic locking + merge strategy, or queue feedback for batch apply |
| cold start | new schema type, vector store has nothing similar | fallback chain: vector search -> fuzzy KB -> LLM generative mode |

### versioning functions
```python
snapshot_calibrators(tag: str) -> str
    # save current state before risky update
    # returns: version_id

rollback_calibrators(version_id: str) -> bool
    # restore previous calibrator state

list_calibrator_versions() -> list[dict]
    # returns: [{version_id, tag, timestamp, ece_before, ece_after}]
```

### collection strategy
```
chromadb collections:
  - schema_gdc          # all gdc mappings
  - schema_htan         # all htan mappings
  - schema_geo          # all geo mappings
  - schema_custom       # user-uploaded schemas

query with metadata filter:
  find_similar(query, k=5, filter={"project": "geo_gse12345"})
  find_similar(query, k=5)  # cross-schema search
```

### cold start fallback
```
1. vector search -> no results (new schema)
         |
         v
2. fuzzy KB search -> partial matches from other schemas
         |
         v
3. LLM generative mode -> explore FHIR schema directly
         |
         v
4. human review (lower auto-accept threshold for new schemas)
```

## integration with existing code

```
existing                          new
--------                          ---
fhir_schema_tool.py      ->    dropdown population
rulematcher              ->    confidence scores
biobert_matcher          ->    confidence scores
magneto_matcher          ->    confidence scores
flatmappingdb            ->    persistence layer
calibration.py           ->    score adjustment + learning
```