# FHIR Template Transformation Rules

extracted from jinja templates in `/templates/` folder.
use these patterns when mapping source fields to fhir resources.

---

## Patient

**template:** `Patient.yaml.jinja`

```yaml
resourceType: Patient
active: true
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| patient_id, participant_id, subject_id | Patient.identifier | identifier |
| gender, sex | Patient.gender | code (male/female/other/unknown) |
| race | Patient.extension[us-core-race] | extension |
| ethnicity | Patient.extension[us-core-ethnicity] | extension |
| birth_sex | Patient.extension[us-core-birthsex] | code (F/M/UNK) |
| deceased, vital_status, death_event | Patient.deceasedBoolean | boolean (dead=true, alive=false, 1=true, 0=false) |
| age, age_at_diagnosis | Patient.extension[age] | quantity (years) |

---

## Condition (Diagnosis)

**template:** `Condition.yaml.jinja`

```yaml
resourceType: Condition
clinicalStatus:
  coding:
  - system: http://terminology.hl7.org/CodeSystem/condition-clinical
    code: active
category:
- coding:
  - system: http://terminology.hl7.org/CodeSystem/condition-category
    code: encounter-diagnosis
  - system: http://snomed.info/sct
    code: '439401001'
    display: Diagnosis
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| diagnosis, disease, condition | Condition.code.coding[] | CodeableConcept (SNOMED/ICD-10/MONDO) |
| body_site, anatomic_site | Condition.bodySite[].coding[] | CodeableConcept (SNOMED anatomical) |
| age_at_diagnosis | Condition.onsetAge | Age (value + unit) |
| stage | Condition.stage[].summary | CodeableConcept |

---

## Specimen

**template:** `Specimen.yaml.jinja`

```yaml
resourceType: Specimen
subject: # reference to Patient
collection:
  collector:
    reference: {{ sample_source_site }}  # -> Organization
  bodySite:
    concept: {{ metastasis_site }}  # CodeableConcept
processing:
  - method: {{ sample_preservation_method }}
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| sample_id, specimen_id, bems_id | Specimen.identifier | identifier |
| sample_type, specimen_type | Specimen.type | CodeableConcept |
| tissue_type, tissue | Specimen.type | CodeableConcept |
| sample_source_site | Specimen.collection.collector.reference | Reference(Organization) |
| metastasis_site | Specimen.collection.bodySite.concept | CodeableConcept |
| preservation_method, sample_preservation | Specimen.processing[].method | CodeableConcept |
| parent_sample_id | Specimen.parent[].reference | Reference(Specimen) |

---

## Observation - Generic

**template:** `Observation.yaml.jinja`

```yaml
resourceType: Observation
status: final
category:
- coding:
  - system: http://terminology.hl7.org/CodeSystem/observation-category
    code: laboratory
code:
  coding:
  - system: https://loinc.org
    code: 68992-7
    display: Specimen-related information panel
subject: null  # -> Patient or Specimen
focus: null    # -> Condition (for staging observations)
```

**value types by data pattern:**
| data pattern | fhir path | example |
|--------------|-----------|---------|
| numeric continuous | Observation.valueQuantity | survival_months: 51.1 |
| numeric with unit | Observation.valueQuantity | concentration: 10.5 ng/mL |
| categorical/coded | Observation.valueCodeableConcept | grade: G2 |
| free text | Observation.valueString | histology: "adenocarcinoma" |
| boolean | Observation.valueBoolean | is_ffpe: true |
| integer | Observation.valueInteger | copy_number: 3 |

---

## Observation - Cancer Staging (mCODE)

**templates:** `Observation-stage_group.yaml.jinja`, `Observation-stage_t.yaml.jinja`, `Observation-stage_n.yaml.jinja`, `Observation-stage_m.yaml.jinja`

**references:**
- https://build.fhir.org/ig/HL7/fhir-mCODE-ig/StructureDefinition-mcode-cancer-stage.html
- https://build.fhir.org/ig/hl7-eu/pcsp/StructureDefinition-mcode-cancer-stage-group.html

```yaml
resourceType: Observation
status: final
category:
- coding:
  - system: http://terminology.hl7.org/CodeSystem/observation-category
    code: laboratory
effectiveDateTime: {{ observation_effective_date }}
valueCodeableConcept: {{ stage_concept }}
hasMember: null  # references to T/N/M child observations
```

**field mappings:**
| source field | fhir path | snomed code |
|--------------|-----------|-------------|
| stage, stage_group, ajcc_stage | Observation.valueCodeableConcept | code: 1222593009 (pathologic) or 1222592004 (clinical) |
| t_stage, pathologic_t | Observation.valueCodeableConcept | code: 1222589003 (pathologic) or 1222585009 (clinical) |
| n_stage, pathologic_n | Observation.valueCodeableConcept | code: 1222590007 (pathologic) or 1222588006 (clinical) |
| m_stage, pathologic_m | Observation.valueCodeableConcept | code: 1222591006 (pathologic) or 1222587001 (clinical) |

**stage hierarchy pattern:**
```
Parent Stage Observation (stage_group)
  hasMember[]:
    -> T Stage Observation
    -> N Stage Observation
    -> M Stage Observation
    -> Grade Observation
```

---

## Observation - Tumor Grade

**template:** `Observation-grade.yaml.jinja`

```yaml
resourceType: Observation
status: final
valueCodeableConcept: {{ grade_concept }}
```

**field mappings:**
| source field | fhir path | snomed codes for values |
|--------------|-----------|------------------------|
| grade, grading, tumor_grade | Observation.valueCodeableConcept | G1: 1228848004, G2: 1228850007, G3: 1228851006, G4: 1228852004, GX: 1228855002 |

**observation code:** SNOMED 1222599008 (AJCC pathological grade)

---

## Observation - Sequence Variant (SNV/Mutation)

**template:** `Observation-seqvar.yaml.jinja`

```yaml
resourceType: Observation
code:
  coding:
  - system: https://loinc.org
    code: 55208-3
    display: DNA analysis discrete sequence variation panel
```

**field mappings (as components):**
| source field | fhir path | value type |
|--------------|-----------|------------|
| gene | Observation.component[].valueString | string |
| chromosome | Observation.component[].valueString | string (chr1-22, X, Y) |
| position_start, genomic_start | Observation.component[].valueInteger | integer |
| position_end, genomic_stop | Observation.component[].valueInteger | integer |
| hgvs_cdna, hgvs_genotype | Observation.component[].valueString | string (HGVS notation) |
| variant_type | Observation.component[].valueCodeableConcept | coded |
| variant_frequency, vaf | Observation.component[].valueQuantity | decimal (0-1) |
| clinical_significance | Observation.component[].valueCodeableConcept | coded (pathogenic, likely_pathogenic, vus, etc.) |

---

## Observation - Copy Number Variation (CNV)

**template:** `Observation-cnv.yaml.jinja`

```yaml
resourceType: Observation
code:
  coding:
  - system: https://loinc.org
    code: 101397-8
    display: Copy number variation analysis in Blood or Tissue by Sequencing
```

**field mappings (as components):**
| source field | fhir path | value type |
|--------------|-----------|------------|
| gene | Observation.component[].valueString | string |
| copy_number, cn | Observation.component[].valueInteger | integer |
| copy_number_result | Observation.component[].valueCodeableConcept | coded (gain, loss, amplification) |

---

## Observation - Immunohistochemistry (IHC)

**template:** `Observation-ihc.yaml.jinja`

```yaml
resourceType: Observation
code:
  coding:
  - system: http://www.ebi.ac.uk/
    code: NCIT_C23020
    display: Immunohistochemistry (IHC)
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| ihc_result, marker_status | Observation.valueCodeableConcept | coded (positive, negative, equivocal) |
| marker_name, biomarker | Observation.component[].valueString | string (ER, PR, HER2, etc.) |
| percent_positive | Observation.component[].valueQuantity | percentage |

---

## Observation - FISH

**template:** `Observation-fish.yaml.jinja`

```yaml
resourceType: Observation
code:
  coding:
  - system: http://www.ebi.ac.uk/
    code: NCIT_C17563
    display: Fluorescence in Situ Hybridization
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| fish_result | Observation.valueCodeableConcept | coded (positive, negative, amplified) |
| gene_tested | Observation.component[].valueString | string |

---

## Observation - Histology

**template:** `Observation-histology.yaml.jinja`

```yaml
resourceType: Observation
valueString: {{ histology }}
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| histology, morphology, histologic_type | Observation.valueString | string |

---

## Observation - Percent Tumor

**template:** `Observation-percent_tumor.yaml.jinja`

```yaml
resourceType: Observation
valueString: {{ percent_tumor }}
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| percent_tumor, tumor_percentage | Observation.valueQuantity | percentage (0-100) |

---

## Observation - Metastasis Site

**template:** `Observation-metastasis_site.yaml.jinja`

```yaml
resourceType: Observation
valueString: {{ metastasis_site }}
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| metastasis_site, met_site | Observation.valueCodeableConcept | CodeableConcept (SNOMED anatomical) |

---

## Procedure

**template:** `Procedure.yaml.jinja`

```yaml
resourceType: Procedure
status: completed
occurrenceAge: {{ procedure_occurrence_age }}
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| procedure, surgery, treatment_type | Procedure.code | CodeableConcept (SNOMED procedure) |
| age_at_procedure | Procedure.occurrenceAge | Age (value + unit) |
| procedure_date | Procedure.occurrenceDateTime | dateTime |

---

## ResearchStudy

**template:** `ResearchStudy.yaml.jinja`

```yaml
resourceType: ResearchStudy
status: active
name: {{ study_name }}
description: {{ study_description }}
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| study_id, project_id | ResearchStudy.identifier | identifier |
| study_name, project_name | ResearchStudy.name | string |
| description | ResearchStudy.description | string |

---

## Organization

**template:** `Organization.yaml.jinja`

```yaml
resourceType: Organization
type:
- coding:
  - system: http://terminology.hl7.org/CodeSystem/organization-type
    code: prov
    display: Healthcare Provider
- coding:
  - system: http://terminology.hl7.org/CodeSystem/organization-type
    code: edu
    display: Educational Institute
```

**field mappings:**
| source field | fhir path | value type |
|--------------|-----------|------------|
| organization_id, site_id | Organization.identifier | identifier |
| organization_name, site_name | Organization.name | string |
| tissue_source_site | Organization.identifier | identifier |

---

## Common Patterns

### survival data
| source field pattern | fhir mapping | notes |
|---------------------|--------------|-------|
| survival_months, os_time, survival_time | Observation.valueQuantity | unit: months or days |
| death_event, os_event, vital_status, deceased | Patient.deceasedBoolean | 1/death/dead -> true, 0/censored/alive -> false |
| days_to_death | Observation.valueQuantity | unit: days |

### identifiers
| source field pattern | fhir mapping |
|---------------------|--------------|
| *_id, *_identifier, accession | [Resource].identifier |
| sample_id, specimen_id | Specimen.identifier |
| patient_id, participant_id, subject_id | Patient.identifier |

### coded values
| data type | fhir value type |
|-----------|----------------|
| categorical with few values | valueCodeableConcept |
| free text descriptions | valueString |
| numeric measurements | valueQuantity |
| true/false, 0/1 | valueBoolean |

### observation subject rules
| observation type | subject | focus |
|-----------------|---------|-------|
| patient-level (demographics, survival) | Patient | - |
| specimen-level (molecular, assay) | Specimen | - |
| condition-related (staging, grade) | Patient | Condition |