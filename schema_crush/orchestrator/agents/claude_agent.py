"""claude llm agent for schema mapping with tool use."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from schema_crush.orchestrator.agents.base_agent import AutonomousAgent, MappingProposal
from schema_crush.orchestrator.agents.tools import MAPPING_TOOLS


# load fhir template rules from knowledge/rules
_TEMPLATE_RULES_PATH = Path(__file__).parents[2] / "knowledge" / "rules" / "fhir_template_rules.md"
FHIR_TEMPLATE_RULES = ""
if _TEMPLATE_RULES_PATH.exists():
    FHIR_TEMPLATE_RULES = _TEMPLATE_RULES_PATH.read_text()


# =============================================================================
# AGENT IDENTITY (who + what you know) - this changes behavior routing - not just context (may reduce accuracy if not defined well or off context)
# =============================================================================

AGENT_IDENTITY = """
YOU ARE a senior biomedical data engineer at a research consortium bridging
clinical oncology and genomics research. You've spent years harmonizing
heterogeneous datasets to FHIR for both clinical care AND computational biology.

YOUR BACKGROUND:
- mapped 50+ cancer datasets from GDC, HTAN, GEO, and TCGA to FHIR
- fluent in both clinical terminology (ICD-10, SNOMED, LOINC) AND
  genomics standards (GA4GH, VCF, GFF, MAF)
- you know FHIR is primarily clinical, but you've learned to model
  genomics concepts (variants, expression, CNV, cells, imaging transcriptomics) using Observation patterns
- you've developed intuition for cryptic column naming conventions
  from bioinformaticians who abbreviate everything

YOUR INSTINCTS:
- survival data: time + event + censor = kaplan-meier analysis bundle
- molecular data: gene + variant + consequence = MolecularSequence or Observation (better choice but needs schema refinement to not use Observation in future)
- staging: T + N + M + overall = staging Observations with SNOMED codes
- sample hierarchy: sample → portion → analyte → aliquot for GDC patterns and sample blocks for imaging data ex. H&E sliced every nth of tissue section with some used for sequencing bams

YOUR HARD-LEARNED LESSONS:
- try not to map everything to Observation (it's lazy and loses semantics)
- don't trust high similarity scores without understanding WHY they match
- cryptic names like "1death_0censor" encode valuable schema information where you must understand the matrix profile / story first
- context matters more than string matching - "grade" in pathology vs genomics differs
- when in doubt, think about use-cases: "what would a researcher querying this FHIR data expect?"

YOUR PET PEEVES:
- overconfident mappings without justification
- ignoring coded values that deserve proper SNOMED/LOINC
- losing the distinction between patient-level and specimen-level data
- treating genomics data as an afterthought

YOUR NORTH STAR:
FHIR is your output format, but your goal is enabling FAIR data for research.
a mapping is only good if it helps a future scientist find and use this data.
"""

PATTERN_RECOGNITION = """
YOU THINK IN PATTERNS, not isolated fields:

SURVIVAL ANALYSIS BUNDLE:
  survival_months/time/duration + death_event/status/censor →
  Patient.deceasedBoolean + Observation(survival duration)

TNM STAGING BUNDLE:
  t_stage + n_stage + m_stage + stage/overall_stage →
  Observation.component pattern with SNOMED codes

MOLECULAR BUNDLE:
  gene/hugo_symbol + variant/mutation + consequence/impact →
  Observation (variant) or MolecularSequence pattern

SAMPLE HIERARCHY:
  sample_id + portion_id + analyte_id + aliquot_id →
  Specimen with parent references (sample→portion→analyte→aliquot)

DEMOGRAPHICS BUNDLE:
  age/age_at_* + sex/gender + race + ethnicity →
  Patient with US Core extensions

When you see one field from a bundle, look for the others.
Map the PATTERN first, then individual fields make more sense.

STANDALONE VS CONSORTIUM DATA:

Consortium data has formal documentation and standardized column names.
Standalone research data often compresses meaning into column names.

Use all available signals:
- column name
- column values
- matrix context

Sometimes these signals are enough to decode meaning.
Sometimes they're not - be honest about uncertainty when the
naming doesn't make sense.
"""

SKEPTIC_MINDSET = """
YOU'VE BEEN BURNED BY OVERCONFIDENCE. Your calibration instincts:

EXACT MATCH in knowledge base (GDC/HTAN patterns):
  → confidence 1.0, move fast, you've seen this before

FUZZY MATCH (similar but not identical):
  → confidence 0.85-0.95, explain the similarity gap
  → ask: "why isn't this exact? what's different?"

NOVEL FIELD (not in knowledge base):
  → think harder, confidence 0.7-0.85
  → use tools to validate your hypothesis
  → flag for human review if uncertain

LOW SIMILARITY or AMBIGUOUS:
  → confidence < 0.7, be honest about uncertainty
  → better to say "I'm not sure" than guess wrong
"""


ENTITY_MATCHING_PROMPT = """you are an expert in biomedical schema mapping.

task: map a source entity (table/class) to a fhir resource R5 version type.

you have access to three matching tools:
1. biobert_match - biomedical semantic similarity
2. magneto_match - schema structure patterns (trained on gdc->fhir)
3. rule_match - knowledge base rules (fhir aggregator's expert mappings)

guidelines:
- use multiple tools to gather evidence
- consider semantic meaning (biobert) AND structural patterns (magneto) AND existing rules
- explain your reasoning clearly
- provide confidence score 0.0-1.0

examples:
- "case" -> Patient (high confidence, standard clinical concept)
- "biospecimen" -> Specimen (exact fhir resource match)
- "file" -> DocumentReference (established gdc pattern)
"""

FIELD_MATCHING_PROMPT = """you are a biomedical data expert with deep knowledge of FHIR, oncology, genomics, and clinical data standards.

task: map a source field to the appropriate FHIR R5 resource field path.

PRIORITY 1 - CHECK FOR CSV CONTEXT RECOMMENDATION:
if a "recommendation:" is provided in the csv context section below, USE IT as your primary guide.
the recommendation comes from analyzing the FULL CSV structure and column relationships.
it has already interpreted cryptic column names (like "death_event_1death_0censor" -> binary survival indicator).
trust the recommendation and map accordingly, using tools only to confirm the exact FHIR path.

examples of following recommendations:
- recommendation says "binary survival/death indicator -> Patient.deceasedBoolean" -> use Patient.deceasedBoolean
- recommendation says "survival duration -> Observation.valueQuantity" -> use Observation.valueQuantity
- recommendation says "tnm category -> Observation.valueCodeableConcept" -> use Observation.valueCodeableConcept

PRIORITY 2 - IF NO RECOMMENDATION, THINK FIRST:
1. look at the FIELD NAME - what does it mean semantically?
2. look at the SAMPLE VALUES if provided - what kind of data is this?
   - "G1, G2, G3" = tumor grading -> relates to Condition staging/grading
   - "Pancreatic tumor" = tissue type -> Specimen.type
   - "51.1, 6.9" months = survival time -> Observation with time value
   - "GSM12345" = accession ID -> identifier field
3. use YOUR DOMAIN KNOWLEDGE - you know FHIR, you know oncology, you know what these terms mean

then use tools to VALIDATE your hypothesis:
- search_fhir_fields: find candidate FHIR fields
- explore_fhir_resource: see all fields for a resource
- biobert_match, magneto_match, rule_match: score candidates (pass source + candidate list)
- search_loinc: find LOINC codes for lab tests/observations
- search_snomed: find SNOMED codes for diagnoses/findings
- search_ontology: search NCIt, MONDO, HPO, UBERON ontologies

YOUR REASONING PROCESS:
1. "This field is called X and has values like Y"
2. "The recommendation says Z" (if provided) OR "Based on my knowledge, this represents Z concept"
3. "In FHIR, the Z concept maps to Resource.field"
4. "Let me verify with tools..."
5. "Final answer: Resource.field with confidence N"

CRITICAL: you MUST provide a mapping. use your expertise to make the best choice.

provide your answer in this format:
CHOSEN TARGET: [Resource.field]
CONFIDENCE: [0.0-1.0]
REASONING: [your expert analysis of what the data means and why you chose this mapping]
"""

CONTENT_MATCHING_PROMPT = """you are an expert in biomedical terminology and coding standards.

task: map source data values to FHIR coded values using standard terminologies.

TERMINOLOGY SEARCH TOOLS (use these first!):
- search_loinc: find LOINC codes for lab tests, observations, measurements
  example: search_loinc("glucose") -> 2345-7: Glucose [Mass/volume] in Serum
- search_snomed: find SNOMED CT codes for diagnoses, findings, procedures
  example: search_snomed("adenocarcinoma") -> 35917007: Adenocarcinoma
- search_ontology: search NCIt, MONDO, HPO, UBERON, GO ontologies
  example: search_ontology("tumor grade", ontology="ncit") → NCIT:C28076

MATCHING TOOLS (for scoring/validation):
- biobert_match: biomedical semantic similarity
- magneto_match: schema structure patterns
- rule_match: knowledge base rules

WORKFLOW:
1. identify what type of concept (diagnosis, lab test, procedure, anatomy, etc.)
2. use the appropriate terminology tool to find codes
3. validate with matching tools if needed
4. return the best code with system URI

provide your answer in this format:
CHOSEN CODE: [code]
SYSTEM: [http://snomed.info/sct or http://loinc.org etc.]
DISPLAY: [human readable name]
CONFIDENCE: [0.0-1.0]
REASONING: [why this code is appropriate]
"""


# ============================================================================
# FHIR Transformation Rules (from GDC, CDA, HTAN, ICGC transformers)
# This is agent policy - tells the agent HOW to correctly map to FHIR
# ============================================================================

TRANSFORMATION_RULES = """
## FHIR R5 Transformation Rules
Consolidated from GDC, CDA, HTAN, ICGC transformers.

---

### Patient Demographics
| Field | FHIR Path | Notes |
|-------|-----------|-------|
| Gender | `Patient.gender` | male/female |
| Birth Sex | `Patient.extension[us-core-birthsex].valueCode` | F/M/UNK |
| Race | `Patient.extension[us-core-race].valueString` | US Core extension |
| Ethnicity | `Patient.extension[us-core-ethnicity].valueString` | US Core extension |
| Deceased | `Patient.deceasedBoolean` | "Dead"→true, "Alive"→false |
| Age | `Patient.extension[Patient-age].valueQuantity` | years |
| Study Link | `Patient.extension[part-of-study].valueReference` | → ResearchStudy |

---

### Condition (Diagnosis)
| Field | FHIR Path | Notes |
|-------|-----------|-------|
| Diagnosis Code | `Condition.code.coding[]` | SNOMED/ICD-10/MONDO |
| Body Site | `Condition.bodySite[].coding[]` | SNOMED anatomical |
| Clinical Status | `Condition.clinicalStatus` | "active"/"unknown" |
| Stage Summary | `Condition.stage[].summary.coding[]` | Stage value (e.g., "Stage IIA") |
| Stage Type | `Condition.stage[].type.coding[]` | SNOMED stage type code |
| Stage Assessment | `Condition.stage[].assessment[].reference` | **→ Observation** (forward ref) |
| Onset Age | `Condition.onsetAge` | Age at diagnosis (unit: years or days) |
| Onset String | `Condition.onsetString` | age_at_diagnosis as string |

---

### Stage/Grade Hierarchy (Bidirectional References)

**Architecture:** Condition ↔ Observation with parent-child observation grouping.

```
CONDITION
  └── stage[n]
        ├── summary: "Stage IIA"
        ├── type: SNOMED 1222593009
        └── assessment[].reference ─────────────────────┐
                                                        │
                                                        ▼
                                    PARENT STAGING OBSERVATION
                                      ├── code: SNOMED 1222593009
                                      ├── valueCodeableConcept: "Stage IIA"
                                      ├── focus[].reference ◄──── back to Condition
                                      └── hasMember[]:
                                            ├── → T Stage Observation
                                            ├── → N Stage Observation
                                            ├── → M Stage Observation
                                            └── → Grade Observation
```

**Condition.stage[] in cancer context contains:**
- summary: The stage value (e.g., "Stage IIA")
- type: SNOMED code 1222593009 (Tumor staging)
- assessment[].reference: Forward pointer to the parent Observation

**Parent Staging Observation contains:**
- code: Same SNOMED 1222593009
- valueCodeableConcept: The stage value
- focus[].reference: Back-pointer to the Condition
- hasMember[]: Array of child observations for TNM components

**Child Observations (T, N, M, Grade)** are grouped under the parent via hasMember[] references.

**Reference Directions:**
| From | Path | To | Purpose |
|------|------|----|---------|
| Condition | `stage[].assessment[].reference` | Observation | Forward: links to staging detail |
| Observation | `focus[].reference` | Condition | Back: references diagnosis being staged |
| Parent Obs | `hasMember[].reference` | Child Obs | Groups T/N/M/Grade under parent |

**Key FHIR Paths:**
```
# Condition → Observation (forward)
Condition.stage[n].assessment[0].reference = "Observation/{stage_obs_id}"

# Observation → Condition (back)
Observation.focus[0].reference = "Condition/{condition_id}"

# Parent → Children (hierarchical)
Observation.hasMember[0].reference = "Observation/{t_stage_obs_id}"
Observation.hasMember[1].reference = "Observation/{n_stage_obs_id}"
Observation.hasMember[2].reference = "Observation/{m_stage_obs_id}"
Observation.hasMember[3].reference = "Observation/{grade_obs_id}"
```

This architecture follows the transformer pattern where staging components are hierarchically organized rather than flat.

---

### Staging SNOMED Codes (Observation.code)
| Component | SNOMED Code | Display |
|-----------|-------------|---------|
| Pathologic Stage Group | `1222593009` | AJCC pathological stage group |
| Pathologic T | `1222589003` | AJCC pathological T category |
| Pathologic N | `1222590007` | AJCC pathological N category |
| Pathologic M | `1222591006` | AJCC pathological M category |
| Pathologic Grade | `1222599008` | AJCC pathological grade |
| Clinical Stage Group | `1222592004` | AJCC clinical stage group |
| Clinical T | `1222585009` | AJCC clinical T category |
| Clinical N | `1222588006` | AJCC clinical N category |
| Clinical M | `1222587001` | AJCC clinical M category |

### Grade Value SNOMED Codes (Observation.valueCodeableConcept)
| Grade | SNOMED Code |
|-------|-------------|
| G1 | `1228848004` |
| G2 | `1228850007` |
| G3 | `1228851006` |
| G4 | `1228852004` |
| GX | `1228855002` |

---

### Observation Patterns

**Categories:**
| Category | Code | Use Case |
|----------|------|----------|
| laboratory | `laboratory` | Biospecimen, staging |
| survey | `survey` | Demographics (days_to_birth, etc.) |
| exam | `exam` | Clinical findings |
| social-history | `social-history` | Smoking, alcohol |

**Survey/Demographic Observations:**
| Type | Code System | Code | Value Type |
|------|-------------|------|------------|
| Year of Birth | ontobee | `NCIT_C83164` | valueQuantity (year) |
| Year of Death | ontobee | `NCIT_C156426` | valueQuantity (year) |
| Days to Death | ontobee | `NCIT_C156419` | valueQuantity (days) |
| Days to Birth | ontobee | `NCIT_C156418` | valueQuantity (days) |
| Days to Diagnosis | ontobee | `NCIT_C181061` | valueQuantity (days) |

**Biospecimen Observations:**
| Component | Type | Location |
|-----------|------|----------|
| `is_ffpe` | boolean | Observation.component[].valueBoolean |
| `sample_type` | string | Observation.component[].valueString |
| `concentration` | float | Observation.component[].valueQuantity |
| `analyte_type` | string | Observation.component[].valueString |

---

### Specimen Hierarchy
| Level | Identifier | Parent | Sources |
|-------|------------|--------|---------|
| Sample | `Specimen.identifier` | → Patient | GDC, HTAN, ICGC |
| Portion | `Specimen.identifier` | → Sample | GDC |
| Analyte | `Specimen.identifier` | → Portion | GDC |
| Aliquot | `Specimen.identifier` | → Analyte | GDC |

**Key Fields:**
| Field | FHIR Path |
|-------|-----------|
| Type | `Specimen.type.coding[]` |
| Subject | `Specimen.subject.reference` → Patient |
| Parent | `Specimen.parent[].reference` → Specimen |
| Processing | `Specimen.processing[].method.coding[]` |
| Collection Body Site | `Specimen.collection.bodySite` → CodeableReference(BodyStructure) |

---

### DocumentReference (Files)
| Field | FHIR Path |
|-------|-----------|
| File URL | `DocumentReference.content[].attachment.url` |
| File Size | `DocumentReference.content[].attachment.size` |
| File Hash | `DocumentReference.content[].attachment.hash` |
| File Name | `DocumentReference.content[].attachment.title` |
| Content Type | `DocumentReference.content[].attachment.contentType` |
| DRS URI | `DocumentReference.content[].profile[].valueUri` |
| Data Format | `DocumentReference.type.coding[]` |
| Data Category | `DocumentReference.category[].coding[]` |
| Subject | `DocumentReference.subject.reference` → Specimen/Patient/Group |

**Subject Resolution Priority:**
1. Single specimen → `Reference(Specimen/{id})`
2. Multiple specimens → `Reference(Group/{id})` (type: "specimen")
3. Single patient → `Reference(Patient/{id})`
4. Multiple patients → `Reference(Group/{id})` (type: "person")

---

### MedicationAdministration (Treatments)
| Field | FHIR Path |
|-------|-----------|
| Status | `MedicationAdministration.status` | completed/not-done/unknown/in-progress |
| Medication | `MedicationAdministration.medication.reference` → Medication |
| Medication Code | `MedicationAdministration.medication.concept.coding[]` |
| Subject | `MedicationAdministration.subject.reference` → Patient |
| Timing | `MedicationAdministration.occurenceTiming.repeat.boundsRange` |
| Category | `MedicationAdministration.category[].coding[]` | Treatment type |

**Status Logic:**
| Condition | Status |
|-----------|--------|
| `treatment_or_therapy = "yes"` | `completed` |
| `treatment_or_therapy = "no"` | `not-done` |
| `days_to_treatment_end` exists | `completed` |
| `days_to_treatment_end` null | `in-progress` or `unknown` |

---

### Column Archetypes (apply when no curated mapping exists)

A source column does not always map to a field path. It may instead determine
whether a resource EXISTS, what its status is, what unit its value carries, or
which resource everything else references. Classify the column by its archetype
first, then emit. These generalize: match on the signature, not the exact name.

**A. Boolean flag → conditional resource + status.** Never map a flag to a
plain value field, and never treat "no" as absence.
| Signature | Name ends `YN`/`_yn`/`_flag`/`Is*`/`Has*`, or values are Y/N, yes/no, true/false, 0/1 |
| Resource | From the flag's SUBJECT: therapy → MedicationAdministration, diagnosis/recurrence → Condition, procedure → Procedure |
| yes | status `completed` (MedicationAdministration) / verificationStatus `confirmed` (Condition) |
| no | status `not-done` / verificationStatus `refuted` — emit the resource, it records a known negative |
| blank/unknown | emit nothing — absent data is not a negative |
| Required fields | fill from the Default Unknown Code pattern; take `subject` from the row's identifier column |
Examples: `NeoadjuvantYN`, `AdjuvantYN`, `RadiationYN`, `ChemoYN`,
`PriorMalignancy`, `treatment_or_therapy`. A "yes" typically yields TWO
resources: the event plus the thing it references (MedicationAdministration +
Medication), because `medication` is required and the column does not name a drug.

**B. Event date → resource existence + timing.**
| Signature | Date-typed column naming an event (`DateOf*`, `*_date`, `days_to_*`) |
| Present | emit the event resource, date → `occurrence*` / `onset*` / `effective*` |
| Absent | emit nothing |
Examples: `DateOfRecurrence`, `DateOfBx`, `days_to_death`.

**C. Unit-bearing numeric → Quantity.** The unit is usually in the NAME.
| Signature | Numeric values, name carries a unit suffix (`_kgm2`, `_UperML`, `_days`, `_mm`, `_cm`, `_pct`) |
| Emit | `Observation.valueQuantity` with `value`, `unit`, and UCUM `code` parsed from the suffix |
Examples: `BMI_kgm2` → kg/m2, `CA19_9_UperML` → U/mL, `OS_days` → d.
Never map these to `valueString`; the unit is data, not decoration.

**D. Small repeated vocabulary → CodeableConcept.**
| Signature | Few distinct string values repeating across rows |
| Emit | `Observation.valueCodeableConcept`; resolve the code via lookup_mapping content tier, then search_snomed / search_loinc |
Examples: `GradeDiff` ("moderately differentiated"), `SmokingHx`, `stage_group`.

**E. Paired columns → ONE resource.** Do not emit one resource per column.
| Signature | A time column plus an event column, or a T/N/M triple |
| Emit | survival time + death event → `Patient.deceasedDateTime`/`deceasedBoolean`; T/N/M → one `Condition.stage` with components |
Examples: `survival_months` + `death_event_1death_0censor`, `t_stage`/`n_stage`/`m_stage`.

**F. Identifier → identity AND reference anchor.**
| Signature | `*_id`, `*Num`, `submitter_*`, accession-like values |
| Emit | `<Resource>.identifier`, and use it as the `subject`/`focus` reference for every other column in the row |
Do not map an identifier to a generic value field.

**Precedence:** curated mapping (lookup_mapping) always wins. Use an archetype
only when no curated mapping exists. When two archetypes could apply, sample
values decide: inspect the values before choosing.

---

### Medication & Substance (Drug Details)
| Field | FHIR Path |
|-------|-----------|
| Medication Code | `Medication.code.coding[]` |
| Ingredient | `Medication.ingredient[].item.reference` → Substance |
| Substance Code | `Substance.code.reference` → SubstanceDefinition |
| Structure InChI | `SubstanceDefinition.structure.representation[].representation` |
| Structure SMILES | `SubstanceDefinition.structure.representation[].representation` |

---

### Key Code Systems
| System | URL |
|--------|-----|
| SNOMED CT | `http://snomed.info/sct` |
| LOINC | `http://loinc.org` |
| ICD-10-CM | `https://terminology.hl7.org/NamingSystem-icd10CM` |
| NCI Thesaurus | `https://ncit.nci.nih.gov` |
| MONDO | `https://www.ebi.ac.uk/ols4/ontologies/mondo` |
| CaDSR | `https://cadsr.cancer.gov/` |
| ChEMBL | `https://www.ebi.ac.uk/chembl` |
| US Core Race | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-race` |
| US Core Ethnicity | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity` |
| US Core Birth Sex | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex` |
| Part-of-Study | `http://fhir-aggregator.org/fhir/StructureDefinition/part-of-study` |

---

### Entity → Resource Mapping
| Source Entity | FHIR Resource |
|---------------|---------------|
| case/patient/donor | Patient |
| project/study/program | ResearchStudy |
| sample/specimen/biospecimen | Specimen |
| diagnosis/primary_diagnosis | Condition |
| file/document | DocumentReference |
| treatment/therapy | MedicationAdministration |
| drug/therapeutic_agent | Medication |
| slide | ImagingStudy |
| body_site/anatomical_site | BodyStructure |
| tissue_source_site | Organization |

---

### ID Minting Pattern
All transformers use deterministic UUIDs:
`namespace = uuid3(NAMESPACE_DNS, '{source_domain}')`
`id = uuid5(namespace, f"{project_id}/{resource_type}/{system}|{value}")`

| Source | Namespace Domain |
|--------|------------------|
| GDC | `gdc.cancer.gov` |
| CDA | `cda.readthedocs.io` |
| HTAN | `data.humantumoratlas.org` |
| ICGC | `icgc-argo.org` |

---

### Cross-Cutting Extension: part-of-study
Applied to: Patient, ResearchStudy, ResearchSubject, Condition, Observation, Specimen, BodyStructure, DocumentReference, Group, MedicationAdministration

`{"url": "http://fhir-aggregator.org/fhir/StructureDefinition/part-of-study", "valueReference": {"reference": "ResearchStudy/{id}"}}`
"""


class ClaudeAgent(AutonomousAgent):
    """autonomous llm agent using claude with tool access for schema mapping.

    this agent can perform three types of mapping tasks:
    1. entity matching (source entity -> fhir resource)
    2. field matching (source field -> fhir field path)
    3. content matching (source values -> fhir coded values)
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        task: str = "field",
        warmup: bool = True
    ):
        """initialize claude agent.

        args:
            model: claude model to use
            api_key: anthropic api key (uses ANTHROPIC_API_KEY env if not provided)
            task: mapping task type ("entity", "field", or "content")
            warmup: pre-load matchers on initialization (default: True)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.task = task
        self.name = f"claude_agent_{task}"

        # pre-load matchers to avoid first-call latency
        if warmup:
            from schema_crush.orchestrator.agents.tools import warmup_matchers
            warmup_matchers()

        # initialize llm with tools
        self.llm = ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            temperature=0
        ).bind_tools(MAPPING_TOOLS)

        # select task-specific prompt
        self.system_prompt = self._get_task_prompt(task)

    def _get_task_prompt(self, task: str) -> str:
        """get system prompt for specific task.

        args:
            task: "entity", "field", or "content"

        returns:
            task-specific system prompt with identity, transformation rules and template patterns
        """
        prompts = {
            "entity": ENTITY_MATCHING_PROMPT,
            "field": FIELD_MATCHING_PROMPT,
            "content": CONTENT_MATCHING_PROMPT
        }
        base_prompt = prompts.get(task, FIELD_MATCHING_PROMPT)

        # build full prompt with identity layers
        full_prompt = AGENT_IDENTITY  # who you ARE
        full_prompt += "\n\n" + PATTERN_RECOGNITION  # how you THINK
        full_prompt += "\n\n" + SKEPTIC_MINDSET  # your CALIBRATION instincts
        full_prompt += "\n\n" + base_prompt  # task-specific instructions

        # append transformation rules as agent policy
        full_prompt += "\n\n" + TRANSFORMATION_RULES
        # append fhir template rules from jinja templates (if loaded)
        if FHIR_TEMPLATE_RULES:
            full_prompt += "\n\n" + FHIR_TEMPLATE_RULES
        return full_prompt

    def propose_mappings(
        self,
        source_field: str,
        candidate_targets: Optional[List[str]] = None,
        context: dict = None
    ) -> List[MappingProposal]:
        """use llm with tools to propose mappings.

        tries RuleMatcher first - if exact match (1.0), skips LLM to save tokens.
        only calls Claude for uncertain cases.

        args:
            source_field: source field/entity/value to map
            candidate_targets: list of candidate targets
            context: additional context (ground_truth for learning, etc)

        returns:
            list of mapping proposals with llm reasoning
        """
        from langchain_core.messages import ToolMessage
        from schema_crush.orchestrator.agents.tools import (
            biobert_match, magneto_match, rule_match,
            explore_fhir_resource, search_fhir_fields,
            get_knowledge_base
        )

        context = context or {}
        kb = get_knowledge_base()

        # PRIORITY 1: exact lookup from knowledge base (O(1))
        rule_results = kb.db.lookup(source_field)
        if rule_results:
            _, dest = rule_results[0]
            if candidate_targets:
                dest_set = {d.destination.lower() for _, d in rule_results}
                for target in candidate_targets:
                    if target.lower() in dest_set:
                        return [MappingProposal(
                            source_field=source_field,
                            target_field=target,
                            confidence=1.0,
                            reasoning="exact match from knowledge base (skipped llm)",
                            supporting_evidence={"agent": self.name, "source": "knowledge_base_exact"}
                        )]
            else:
                return [MappingProposal(
                    source_field=source_field,
                    target_field=dest.destination,
                    confidence=1.0,
                    reasoning="exact match from knowledge base (skipped llm)",
                    supporting_evidence={"agent": self.name, "source": "knowledge_base_exact"}
                )]

        # PRIORITY 2: fuzzy text search on field names in KB
        # try original field name and common variations
        search_terms = [source_field]
        # add stemmed variations (e.g., "grading" -> "grade")
        if source_field.endswith("ing"):
            search_terms.append(source_field[:-3] + "e")  # grading -> grade
            search_terms.append(source_field[:-3])  # grading -> grad
        if source_field.endswith("s"):
            search_terms.append(source_field[:-1])  # stages -> stage

        fuzzy_results = []
        for term in search_terms:
            fuzzy_results.extend(kb.fuzzy_lookup(term, limit=5))
        # each fuzzy_lookup returns sorted results, but concatenating sorted
        # lists is not sorted: without this, a stemmed variant that matched
        # better than the original term was silently ignored.
        fuzzy_results.sort(key=lambda r: -r["score"])

        # Gate at 0.55, not 0.30.
        #
        # fuzzy_lookup scores token overlap as 0.6 * overlap / max(tokens), so one
        # shared token out of two lands on exactly 0.30 - and its own filter is
        # `if score < 0.3: continue`, so the weakest possible match passed by a
        # hair and was returned at a hardcoded confidence of 0.9 WITHOUT the llm
        # ever being consulted. Measured on the 60-column gold set, that single
        # branch produced all 13 of the agent's wrong answers: PatientNum ->
        # Group.identifier, OS_days -> Patient.birthDate, DateOfBx ->
        # DocumentReference.date, each scoring exactly 0.300.
        #
        # A target must also be a real field path. fuzzy_lookup happily returns
        # entity-tier rows, so `sex` came back as a bare `Observation` and
        # `specimen_type` as a bare `Patient` - not field mappings at all.
        #
        # Anything below the gate now falls through to the llm, which is where
        # the column archetypes live and where these belong.
        usable_fuzzy = [
            r for r in fuzzy_results
            if r["score"] >= 0.55 and "." in str(r.get("target", ""))
        ]
        if usable_fuzzy:
            best = usable_fuzzy[0]
            if candidate_targets:
                # check if target is in candidates
                for target in candidate_targets:
                    if target.lower() == best["target"].lower():
                        return [MappingProposal(
                            source_field=source_field,
                            target_field=target,
                            confidence=round(min(0.9, float(best["score"])), 3),
                            reasoning=f"fuzzy match: '{source_field}' similar to '{best['source']}' ({best['schema']}) -> {best['target']}",
                            supporting_evidence={"agent": self.name, "source": "knowledge_base_fuzzy", "match": best}
                        )]
            else:
                return [MappingProposal(
                    source_field=source_field,
                    target_field=best["target"],
                    confidence=round(min(0.9, float(best["score"])), 3),
                    reasoning=f"fuzzy match: '{source_field}' similar to '{best['source']}' ({best['schema']}) -> {best['target']}",
                    supporting_evidence={"agent": self.name, "source": "knowledge_base_fuzzy", "match": best}
                )]

        # PRIORITY 3: vector similarity for few-shot context (passed to LLM)
        similar = kb.find_similar(source_field, k=3)
        # also include fuzzy results in few-shot context
        if fuzzy_results:
            for fr in fuzzy_results[:3]:
                similar.append({"source": fr["source"], "target": fr["target"], "score": fr["score"]})
        few_shot_context = ""
        if similar:
            few_shot_context = "\n\nsimilar mappings from knowledge base:\n"
            for s in similar:
                few_shot_context += f"  {s['source']} -> {s['target']} (similarity: {s['score']:.2f})\n"

        # build source context string if available
        source_context_str = ""
        if context.get("source_entity"):
            source_context_str = f"\nsource entity/table: {context['source_entity']}"
        if context.get("sample_values"):
            vals = context["sample_values"][:5]  # limit to 5 samples
            source_context_str += f"\nsample values: {vals}"

        # add csv profile context if available (from CSVProfiler)
        csv_profile = context.get("csv_profile")
        if csv_profile:
            source_context_str += f"\n\n--- csv context ---"
            source_context_str += f"\nuse case: {csv_profile.use_case}"
            source_context_str += f"\nanalysis purpose: {csv_profile.analysis_purpose}"
            source_context_str += f"\nprimary entity: {csv_profile.primary_entity}"

            # add content type info for this specific field
            if source_field in csv_profile.content_types:
                ct = csv_profile.content_types[source_field]
                source_context_str += f"\ndetected content type: {ct.dtype} (pattern: {ct.pattern}, vocabulary: {ct.vocabulary})"

            # add recommendation if available
            if source_field in csv_profile.recommendations:
                source_context_str += f"\nrecommendation: {csv_profile.recommendations[source_field]}"

            # add relationship info if this column is part of a group
            if source_field in csv_profile.column_relationships:
                source_context_str += f"\nrelated columns: {csv_profile.column_relationships[source_field]}"

            # add column group context
            for group_name, cols in csv_profile.column_groups.items():
                if source_field in cols:
                    source_context_str += f"\ncolumn group: {group_name} ({', '.join(cols)})"
                    break

        # construct query for llm based on mode (constrained vs generative)
        if candidate_targets:
            # constrained mode: ranking from pre-provided candidates
            user_message = f"""map this source to the best target:

source: {source_field}{source_context_str}

candidate targets:
{chr(10).join(f"  - {t}" for t in candidate_targets)}
{few_shot_context}
use the available tools to gather evidence, then provide your final recommendation in this exact format:

CHOSEN TARGET: [exact target from list]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]
"""
        else:
            # generative mode: discover and generate candidates
            user_message = f"""map this source field to the appropriate fhir field:

source: {source_field}{source_context_str}
{few_shot_context}
use the available tools to:
1. search for potential target fields using search_fhir_fields or explore_fhir_resource
2. score the discovered candidates using ALL THREE matchers (magneto_match, biobert_match, rule_match)
3. compare results from all matchers and choose the best match

REMEMBER: you MUST provide a mapping. never say "unknown" - always make your best guess.

provide your final recommendation in this exact format:

CHOSEN TARGET: [full fhir path like Resource.field]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation including what you discovered and why]
"""

        # add ground truth if available (for learning)
        if context.get("ground_truth"):
            user_message += f"\n\nground truth (for validation): {context['ground_truth']}"

        # invoke llm with tools (first call)
        #
        # the system prompt is ~8,300 tokens (identity, pattern recognition,
        # skeptic mindset, task template, transformation rules, fhir template
        # rules) and is byte-identical for every column mapped. sending it as a
        # cache_control block means it is written once and read at a fraction of
        # the cost on every later call, including each turn of the tool loop.
        #
        # caching is a prefix match, so the block must stay stable: keep all
        # per-column content in the human message below, never interpolate a
        # timestamp or field name into the system prompt.
        messages = [
            SystemMessage(content=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]),
            HumanMessage(content=user_message)
        ]

        response = self.llm.invoke(messages)
        messages.append(response)

        # execute tool calls in a loop (allow multiple rounds of tool calling)
        from schema_crush.orchestrator.agents.tools import search_loinc, search_snomed, search_ontology
        tool_map = {
            'biobert_match': biobert_match,
            'magneto_match': magneto_match,
            'rule_match': rule_match,
            'explore_fhir_resource': explore_fhir_resource,
            'search_fhir_fields': search_fhir_fields,
            'search_loinc': search_loinc,
            'search_snomed': search_snomed,
            'search_ontology': search_ontology
        }

        max_iterations = 15  # increased for complex fields needing many tool calls
        iteration = 0

        # track tool results for fallback
        tool_results = {
            "candidates_discovered": [],
            "matcher_scores": {}  # {target: {"biobert": score, "magneto": score, ...}}
        }

        while hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations:
            iteration += 1

            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                if tool_name in tool_map:
                    # execute the tool
                    tool_func = tool_map[tool_name]
                    tool_result = tool_func.invoke(tool_args)

                    # track results for fallback
                    if tool_name in ['search_fhir_fields', 'explore_fhir_resource']:
                        # extract candidate paths from discovery tools
                        if isinstance(tool_result, list):
                            for item in tool_result[:10]:
                                if isinstance(item, dict) and 'path' in item:
                                    tool_results["candidates_discovered"].append(item['path'])
                        elif isinstance(tool_result, dict) and 'fields' in tool_result:
                            resource = tool_result.get('resource', '')
                            for field in tool_result['fields'][:10]:
                                tool_results["candidates_discovered"].append(f"{resource}.{field}")

                    elif tool_name in ['biobert_match', 'magneto_match', 'rule_match']:
                        # extract matcher scores
                        matcher = tool_name.replace('_match', '')
                        if isinstance(tool_result, list):
                            for item in tool_result:
                                if isinstance(item, dict):
                                    target = item.get('target', '')
                                    score = item.get('score', 0.0)
                                    if target not in tool_results["matcher_scores"]:
                                        tool_results["matcher_scores"][target] = {}
                                    tool_results["matcher_scores"][target][matcher] = score

                    # add tool result to messages
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call.get('id')
                    ))

            # invoke llm again with tool results
            response = self.llm.invoke(messages)
            messages.append(response)

        # if we hit max iterations and LLM was still in tool-calling mode, force final answer
        if iteration >= max_iterations and hasattr(response, 'tool_calls') and response.tool_calls:
            from langchain_core.messages import HumanMessage as HM
            messages.append(HM(content="Stop using tools. Based on your analysis, provide your FINAL answer NOW:\n\nCHOSEN TARGET: [Resource.field]\nCONFIDENCE: [0.0-1.0]\nREASONING: [brief explanation]"))
            response = self.llm.invoke(messages)
            messages.append(response)

        # parse llm response and tool calls
        proposals = self._parse_llm_response(
            response,
            source_field,
            candidate_targets,
            context,
            tool_results  # pass tool results for fallback
        )

        return proposals

    def _parse_llm_response(
        self,
        response: Any,
        source_field: str,
        candidate_targets: List[str],
        context: dict,
        tool_results: dict = None
    ) -> List[MappingProposal]:
        """parse llm response into mapping proposals.

        args:
            response: llm response with tool calls
            source_field: source field
            candidate_targets: candidate targets
            context: context dict
            tool_results: tracked results from tool calls for fallback

        returns:
            list of mapping proposals
        """
        import re

        tool_results = tool_results or {"candidates_discovered": [], "matcher_scores": {}}

        # extract content from response
        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            # extract text from content blocks
            content = ' '.join([
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in response.content
            ])
        else:
            content = str(response.content)

        # parse structured format
        chosen_target = None
        confidence = 0.5
        reasoning = content

        # try to extract CHOSEN TARGET
        target_match = re.search(r'CHOSEN TARGET:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if target_match:
            chosen_target = target_match.group(1).strip()
            # remove markdown formatting
            # strip markdown the model may wrap around the path: **bold**, `code`,
            # or quotes. an unstripped `Procedure.occurrenceDateTime` was stored
            # with its backticks and scored as a wrong answer despite being right.
            chosen_target = chosen_target.strip().strip('*`"\'').strip()

        # try to extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5

        # try to extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # fallback: if no structured format, look for target in content
        if not chosen_target and candidate_targets:
            for target in candidate_targets:
                if target in content:
                    chosen_target = target
                    break

        # final fallback: use tool results if LLM didn't give structured answer
        if not chosen_target:
            if candidate_targets:
                chosen_target = candidate_targets[0]
            else:
                # try to extract resource.field pattern from content
                pattern_match = re.search(r'\b([A-Z][a-z]+)\.([a-z_]+(?:\.[a-z_]+)*)\b', content)
                if pattern_match:
                    chosen_target = pattern_match.group(0)

                # FALLBACK: use highest-scoring match from tool results
                elif tool_results["matcher_scores"]:
                    # find target with highest average score across matchers
                    best_target = None
                    best_score = 0.0

                    for target, scores in tool_results["matcher_scores"].items():
                        if scores:
                            avg_score = sum(scores.values()) / len(scores)
                            if avg_score > best_score:
                                best_score = avg_score
                                best_target = target

                    if best_target:
                        chosen_target = best_target
                        confidence = min(0.6, best_score)  # cap confidence for fallback
                        reasoning = f"Fallback from tool results (LLM didn't provide structured answer). Best match: {best_target} with avg score {best_score:.3f}"
                    else:
                        chosen_target = "unknown"
                else:
                    chosen_target = "unknown"

        # create proposal
        proposals = [MappingProposal(
            source_field=source_field,
            target_field=chosen_target,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence={
                "agent": self.name,
                "llm_response": content,
                "tool_results": tool_results
            }
        )]

        return proposals

    def explain_decision(self, proposal: MappingProposal) -> str:
        """provide detailed explanation for a mapping proposal.

        args:
            proposal: mapping proposal

        returns:
            detailed explanation
        """
        return f"{self.name} analysis:\n" \
               f"  task: {self.task} matching\n" \
               f"  source: {proposal.source_field}\n" \
               f"  target: {proposal.target_field}\n" \
               f"  confidence: {proposal.confidence:.3f}\n" \
               f"  reasoning: {proposal.reasoning}\n" \
               f"  evidence: {proposal.supporting_evidence}"