# Schema Crush: Technical Overview

## Abstract

Schema Crush is a semantic schema matching framework designed to align heterogeneous biomedical data schemas to HL7 FHIR (Fast Healthcare Interoperability Resources) standards. The system combines rule-based matching, embedding-based semantic similarity, and calibrated confidence estimation to achieve high-accuracy mappings with minimal human intervention. Built on expert-curated mappings from FHIR Aggregator sources (GDC, HTAN, CDA), Schema Crush provides a practical solution for biomedical data harmonization.

---

## 1. Problem Statement

### 1.1 The Schema Heterogeneity Challenge

Biomedical research generates vast amounts of data across institutions, each using proprietary schemas. A single concept like "patient identifier" may appear as:

- `participant_id` (GDC)
- `HTAN_Participant_ID` (HTAN)
- `subject_id` (CDA)
- `sample_id` (GEO)
- `patient_id` (institutional systems)

This heterogeneity creates significant barriers to data integration, cross-study analysis, and reproducible research.

### 1.2 Why FHIR?

FHIR R5 provides a standardized data model for healthcare and research data. By mapping source schemas to FHIR, we enable:

- **Interoperability**: Data can flow between systems
- **Semantic clarity**: Each field has a well-defined meaning
- **Extensibility**: FHIR supports extensions for domain-specific needs
- **Tooling**: Rich ecosystem of FHIR-aware tools and validators

### 1.3 The Mapping Problem

Schema matching requires solving three distinct sub-problems:

| Tier | Task | Example |
|------|------|---------|
| **Entity** | Source table → FHIR Resource | `case` → `Patient` |
| **Field** | Source column → FHIR path | `tumor_grade` → `Observation.valueCodeableConcept.text` |
| **Content** | Source value → FHIR code | `"Adenocarcinoma"` → SNOMED CT 35917007 |

---

## 2. Core Concepts

### 2.1 Calibrated Confidence

**Definition**: A classifier is *calibrated* when its predicted confidence equals its empirical accuracy. If a model outputs 90% confidence, it should be correct 90% of the time.

**Problem**: Most machine learning models are *overconfident* — they output high scores even when predictions are incorrect. This is particularly problematic for schema matching, where users must decide which mappings to trust.

**Solution**: Schema Crush learns a calibration function that maps raw similarity scores to true probabilities:

```
P(correct | raw_score = s) ≈ calibrated_score
```

**Measurement**: Calibration quality is measured using Expected Calibration Error (ECE):

```
ECE = Σ (|Bₘ|/n) · |accuracy(Bₘ) - confidence(Bₘ)|
```

Where `Bₘ` are confidence bins. A perfectly calibrated model has ECE = 0.

### 2.2 Three-Tier Matching

Schema Crush decomposes the mapping problem into three tiers, each with different candidate sets and evaluation criteria:

**Entity Tier**: Maps source tables/contexts to FHIR Resources (10 candidates: Patient, Specimen, Observation, Condition, etc.)

**Field Tier**: Maps source columns to FHIR element paths (hundreds of candidates per resource)

**Content Tier**: Maps categorical values to terminology codes (SNOMED CT, LOINC, ICD-10)

### 2.3 Fast Path vs. LLM Fallback

The system prioritizes efficiency through a cascading matcher architecture:

1. **Exact match**: Direct lookup in curated knowledge base (confidence = 1.0)
2. **Fuzzy match**: String similarity against known mappings (confidence = similarity ratio)
3. **Embedding match**: Semantic similarity via BioBERT/Magneto (confidence = calibrated)
4. **LLM fallback**: Claude-based reasoning for unknown fields (confidence = model-stated)

In practice, ~93% of mappings are resolved via the fast path (exact + fuzzy), avoiding expensive LLM calls.

### 2.4 Backoff Calibration

Calibration curves are context-specific, but some contexts have sparse data. Schema Crush uses a backoff scheme:

```
magneto:field:gdc:specimen  →  magneto:field:gdc:*  →  magneto:field:*:*  →  magneto:*:*:*
```

Each key requires minimum sample thresholds (300+ samples, 100+ positives) to be eligible, preventing overfitting to sparse contexts.

---

## 3. Architecture

### 3.1 Data Stores

Schema Crush distinguishes between canonical and derived stores:

**Canonical (source of truth)**:
- `flat_mappings.db` — SQLite database with 2,346 curated source→FHIR mappings
- `feedback.db` — User decisions for evaluation and retraining
- `calibrators/*.pkl` — Trained confidence calibration curves

**Derived (rebuildable)**:
- `chroma/` — ChromaDB vector index for semantic similarity search

**Design principle**: If losing ChromaDB loses knowledge, you're storing the wrong thing in ChromaDB. Canonical truth always lives in SQLite.

### 3.2 Matchers

| Matcher | Method | Strengths | Calibrated Accuracy |
|---------|--------|-----------|---------------------|
| **RuleMatcher** | Exact + fuzzy string matching | Fast, interpretable | 97.6% (entity), 92.0% (field), 99.2% (content) |
| **BioBERTMatcher** | Foundation model embeddings (BioBERT) | Biomedical domain knowledge | 5.9% - 68.0% (varies by tier) |
| **MagnetoMatcher** | Foundation model + column encoding (MPNet) | Schema-aware representations | 13.4% - 72.4% (varies by tier) |

#### Foundation Models Used

| Embedder | Base Model | Pre-training | Our Usage |
|----------|------------|--------------|-----------|
| **BioBERT** | `dmis-lab/biobert-base-cased-v1.1` | BERT pre-trained on 18B words of PubMed + PMC biomedical literature | Used as-is via sentence-transformers |
| **Magneto** | `all-mpnet-base-v2` (MPNet) | General-purpose sentence transformer | Wrapped with Magneto library's column encoding strategies (header + sample values) |

**Note**: We do not fine-tune these foundation models. BioBERT provides biomedical domain knowledge from its pre-training. Magneto adds schema-aware column encoding on top of MPNet's general-purpose embeddings.

**Key insight**: The rule matcher handles 95%+ of cases. Embedding matchers are used only for unknown fields not in the knowledge base.

### 3.3 Knowledge Base

The knowledge base aggregates mappings from multiple FHIR Aggregator sources:

| Source | Description | Records |
|--------|-------------|---------|
| GDC | Genomic Data Commons | ~1,200 mappings |
| HTAN | Human Tumor Atlas Network | ~400 mappings |
| CDA | Cancer Data Aggregator | ~300 mappings |
| ICGC | International Cancer Genome Consortium | ~200 mappings |
| GTEx | Genotype-Tissue Expression | ~100 mappings |
| Others | 1000Genomes, Cellosaurus | ~150 mappings |

Total: 2,346 unique source terms → 2,563 destination mappings

### 3.4 Entry Points

**MCP Server** (13 tools): Integration with Claude Desktop/Code for interactive mapping assistance

**ClaudeAgent**: LangChain-based programmatic interface for automated pipelines

**CLI** (`map_csv.py`): Command-line tool for batch mapping of CSV files

---

## 4. Use Cases

### 4.1 Interactive Schema Mapping

A researcher has a new dataset with unfamiliar column names. Using Claude Desktop with the Schema Crush MCP:

```
User: "Where does tumor_grade map to in FHIR?"

Claude: [calls lookup_mapping tool]

Response: tumor_grade maps to Observation.valueCodeableConcept.text
- SNOMED codes: 1222599008 (grade finding)
- Also related: tumor_stage, histologic_grade
- Confidence: 0.97 (from GDC curated mappings)
```

### 4.2 Batch CSV Mapping

A data engineer needs to map a GEO clinical dataset:

```bash
python examples/map_csv.py GSE62452_clinical.csv --entity patient -o mappings.json
```

Output:
```json
[
  {"source": "sample_id", "target": "Specimen.identifier", "confidence": 1.0},
  {"source": "tissue", "target": "Observation", "confidence": 0.9},
  {"source": "grading", "target": "Observation.valueCodeableConcept.text", "confidence": 0.9},
  {"source": "stage", "target": "Condition.stage.type.coding", "confidence": 0.9},
  {"source": "survival_months", "target": "Observation.value_quantity", "confidence": 0.85},
  {"source": "survival_status", "target": "Patient.deceasedBoolean", "confidence": 0.9}
]
```

### 4.3 Terminology Lookup

A clinician needs the correct SNOMED code for a diagnosis:

```
User: "Find SNOMED code for pancreatic adenocarcinoma"

Claude: [calls search_snomed tool → tx.fhir.org API]

Response:
- 254626006: Adenocarcinoma of pancreas
- 372003004: Ductal adenocarcinoma of pancreas
- 722524005: Primary adenocarcinoma of pancreas
```

### 4.4 FHIR Schema Exploration

A developer needs to understand FHIR Condition staging:

```
User: "What fields does Condition have for staging?"

Claude: [calls explore_fhir_resource tool → LinkML schema]

Response:
- Condition.stage (0..*) - Stage/grade, usually assessed formally
- Condition.stage.summary - Simple summary (disease progression)
- Condition.stage.type - Kind of staging (clinical, pathological)
- Condition.stage.assessment - Reference to formal assessment
```

---

## 5. Confidence Scoring

### 5.1 Score Sources

| Matching Path | Confidence Source | Calibrated? |
|---------------|-------------------|-------------|
| Exact match | 1.0 (hardcoded) | N/A — always correct by definition |
| Fuzzy match | FuzzyWuzzy ratio (0-1) | No — empirically reliable above 0.85 |
| Embedding match | Cosine similarity | Yes — mapped via calibration curves |
| LLM fallback | Model self-report | No — uncalibrated |

### 5.2 Interpreting Confidence

- **≥ 0.95**: High confidence, suitable for automated acceptance
- **0.85 - 0.95**: Medium confidence, benefits from human review
- **< 0.85**: Low confidence, requires manual mapping

### 5.3 Current Calibration Results (January 2026)

| Matcher | Tier | Accuracy | ECE | Status |
|---------|------|----------|-----|--------|
| Rule | Entity | 97.6% | 0.005 | Excellent |
| Rule | Field | 92.0% | 0.08 | Well-calibrated |
| Rule | Content | 99.2% | 0.008 | Near-perfect |
| BioBERT | Content | 68.0% | 0.26 | Moderate |
| Magneto | Content | 72.4% | 0.04 | Well-calibrated |

---

## 6. Learning Loop

### 6.1 Feedback Collection

User decisions are stored in the FeedbackStore with full traceability:

- `source`, `source_context` — what was being mapped
- `proposed_target` — system proposal
- `ground_truth` — user correction (if any)
- `decision` — accept / reject / correct
- `matcher`, `tier`, `confidence` — context
- `timestamp`, `user_id` — audit trail

### 6.2 Calibrator Retraining

Feedback can be incorporated into calibration:

```bash
python calibrators/calibrate_flat_mappings.py --feedback
```

**Current limitations**: Feedback is incorporated in a one-way manner. Planned safeguards include:
- Holdout splits (last 20% by time never trains)
- Minimum sample thresholds before retraining
- ECE regression checks (only retrain if metrics improve)
- Versioned retraining for reproducibility

---

## 7. API Reference

### 7.1 MCP Tools

| Tool | Description |
|------|-------------|
| `lookup_mapping` | Find all mappings for a source term |
| `rule_match` | Exact/fuzzy matching against curated rules |
| `biobert_match` | Semantic matching using BioBERT embeddings |
| `magneto_match` | Schema-trained matching using Magneto |
| `find_similar` | Find similar source terms via vector search |
| `explore_fhir_resource` | Get FHIR R5 resource structure |
| `search_fhir_fields` | Search FHIR fields by keyword |
| `search_snomed` | Search SNOMED CT codes |
| `search_loinc` | Search LOINC codes |
| `search_ontology` | Search ontologies (NCI Thesaurus, etc.) |
| `record_feedback` | Record user mapping decisions |
| `get_feedback_stats` | Get feedback statistics |
| `get_transformation_rules` | Get FHIR transformation rules |

### 7.2 CLI Usage

```bash
# Map a CSV file
python examples/map_csv.py data.csv --entity patient -o mappings.json

# Entity types
--entity patient  # → Patient, Condition, Observation
--entity sample   # → Specimen, Observation
--entity file     # → DocumentReference, Observation

# Options
-n 5              # Limit to first 5 columns
-o output.json    # Save to file (default: stdout)
```

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

- **No HITL UI**: Feedback collection requires programmatic calls
- **Limited to known schemas**: Best performance on GDC/HTAN-like data
- **English only**: Source terms assumed to be English
- **FHIR R5**: Does not support older FHIR versions

### 8.2 Planned Improvements

| Component | Status | Description |
|-----------|--------|-------------|
| HITL Queue UI | Design done | Web/CLI interface for human review |
| REST API | Not started | FastAPI wrapper for programmatic access |
| Calibration safeguards | Partial | Holdout validation, metric regression checks |
| Clustering | Not started | Group similar fields for batch review |
| Fine-tuning pipeline | Not started | Triplet loss training for embedders |

---

## 9. References

### Calibration Methods
- Platt, J. (1999). Probabilistic outputs for support vector machines.
- Guo, C., et al. (2017). On calibration of modern neural networks. ICML.
- Zadrozny, B. & Elkan, C. (2001). Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers. ICML.

### Schema Matching
- Rahm, E. & Bernstein, P.A. (2001). A survey of approaches to automatic schema matching. VLDB Journal.
- Madhavan, J., et al. (2001). Generic schema matching with Cupid. VLDB.

### FHIR
- HL7 FHIR R5 Specification: https://hl7.org/fhir/R5/
- FHIR Aggregator: https://github.com/FHIR/fhir-aggregator

### Embeddings
- Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model. Bioinformatics.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP.

---

## 10. Quick Start

```bash
# Install
pip install -e .

# Test mapping
python examples/map_csv.py your_data.csv --entity patient -o mappings.json

# Run MCP server (for Claude Desktop)
python -m schema_crush.mcp.server

# Retrain calibrators
python calibrators/calibrate_flat_mappings.py --embeddings
```

---

*Schema Crush is developed as part of the Schematrix project for biomedical data harmonization.*