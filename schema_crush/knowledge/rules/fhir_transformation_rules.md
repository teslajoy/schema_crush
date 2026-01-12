# FHIR Transformation Rules Reference

> Consolidated mapping rules extracted from GDC, CDA, HTAN, and ICGC transformers

---

## Table of Contents

1. [Source Files Overview](#source-files-overview)
2. [ID Minting Conventions](#id-minting-conventions)
3. [Patient](#1-patient)
4. [ResearchStudy](#2-researchstudy)
5. [ResearchSubject](#3-researchsubject)
6. [Condition](#4-condition)
7. [Observation](#5-observation)
8. [Specimen](#6-specimen)
9. [BodyStructure](#7-bodystructure)
10. [ImagingStudy](#8-imagingstudy)
11. [DocumentReference](#9-documentreference)
12. [Group](#10-group)
13. [Organization](#11-organization)
14. [Practitioner](#12-practitioner)
15. [MedicationAdministration](#13-medicationadministration)
16. [Medication](#14-medication)
17. [Substance & SubstanceDefinition](#15-substance--substancedefinition)
18. [Encounter](#16-encounter)
19. [Cross-Cutting Patterns](#17-cross-cutting-patterns)
20. [Validation & Error Handling](#18-validation--error-handling)

---

## Source Files Overview

| Abbreviation | Source File | Data Origin | Namespace |
|--------------|-------------|-------------|-----------|
| **GDC** | `entity2fhir.py` | Genomic Data Commons | `gdc.cancer.gov` |
| **CDA** | `transformer.py` | Cancer Data Aggregator (GDC+PDC+IDC+ICDC) | `cda.readthedocs.io` |
| **HTAN** | `htan2fhir.py` | Human Tumor Atlas Network | `data.humantumoratlas.org` |
| **ICGC** | `icgc2fhir.py` | International Cancer Genome Consortium | `icgc-argo.org` |
| **CELLO** | `entity2fhir.py` | Cellosaurus cell lines | `cellosaurus.org` |

---

## ID Minting Conventions

All transformers use deterministic UUID generation for reproducible IDs.

| Source | Namespace Creation | ID Pattern |
|--------|-------------------|------------|
| GDC | `uuid3(NAMESPACE_DNS, 'gdc.cancer.gov')` | `uuid5(ns, "{resource_type}/{system}\|{value}")` |
| CDA | `uuid3(NAMESPACE_DNS, 'cda.readthedocs.io')` | `uuid5(ns, "{project_id}/{resource_type}/{system}\|{value}")` |
| HTAN | `uuid3(NAMESPACE_DNS, 'data.humantumoratlas.org')` | `uuid5(ns, "{project_id}/{resource_type}/{system}\|{value}")` |
| ICGC | `uuid3(NAMESPACE_DNS, 'icgc-argo.org')` | `uuid5(ns, "{resource_type}/{system}\|{value}")` |
| CELLO | `uuid3(NAMESPACE_DNS, 'cellosaurus.org')` | Same as GDC pattern |

**Core Minting Function:**
```python
def mint_id(identifier: Identifier, resource_type: str, project_id: str, namespace: UUID) -> str:
    identifier_string = f"{resource_type}/{identifier.system}|{identifier.value}"
    return str(uuid5(namespace, f"{project_id}/{identifier_string}"))
```

---

## 1. Patient

**Purpose:** Represents a research subject/case with demographic information.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| Primary ID | `Patient.id` | ✓ Minted from case_id | ✓ Minted from subject_id | ✓ Minted from HTAN Participant ID | ✓ Minted from icgc_donor_id |
| Official Identifier | `Patient.identifier[use=official]` | `case_id` | `subject_id` | `HTAN Participant ID` | `icgc_donor_id` |
| Secondary Identifiers | `Patient.identifier[use=secondary]` | `case_submitter_id` | Multi-system (GDC/PDC/IDC/ICDC) | — | — |
| Gender | `Patient.gender` | Lookup table → `male`/`female` | — | — | Direct from `donor_sex` |
| Deceased Boolean | `Patient.deceasedBoolean` | `"Dead"`→`true`, `"Alive"`→`false` | Same logic | Same logic | — |
| Birth Sex | `Patient.extension[us-core-birthsex]` | `valueCode`: `F`/`M` | `valueCode`: `F`/`M`/`UNK` | — | `valueCode`: `F`/`M` |
| Race | `Patient.extension[us-core-race]` | `valueString` mapped | `valueString` mapped | `valueString` direct | — |
| Ethnicity | `Patient.extension[us-core-ethnicity]` | `valueString` mapped | `valueString` normalized | `valueString` direct | — |
| Age | `Patient.extension[Patient-age]` | `valueQuantity.value` | — | — | — |
| Address | `Patient.address[].country` | — | — | Country of Residence | — |
| Study Link | `Patient.extension[part-of-study]` | — | ✓ `valueReference` to ResearchStudy | — | — |

### Extension URLs

```
http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex
http://hl7.org/fhir/us/core/StructureDefinition/us-core-race
http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity
http://hl7.org/fhir/SearchParameter/patient-extensions-Patient-age
http://fhir-aggregator.org/fhir/StructureDefinition/part-of-study
```

### Identifier System URLs by Source

| Source | System URL Pattern |
|--------|-------------------|
| GDC | `https://gdc.cancer.gov/{field_name}` |
| CDA-GDC | `https://gdc.cancer.gov/{field_name}` |
| CDA-PDC | `https://proteomic.datacommons.cancer.gov/pdc/{field_name}` |
| CDA-IDC | `https://portal.imaging.datacommons.cancer.gov/{field_name}` |
| CDA-ICDC | `https://caninecommons.cancer.gov/{field_name}` |
| HTAN | `https://data.humantumoratlas.org` |
| ICGC | `https://platform.icgc-argo.org/donor_id` |
| CELLO | `https://www.cellosaurus.org/cell-line-primary-accession` |

### Validation Rules

| Rule | Regex/Logic | Source |
|------|-------------|--------|
| Submitter ID format | `^[A-Za-z0-9\-.]+$` | GDC |
| Gender mapping | Lookup table with fallback | GDC, CDA |

---

## 2. ResearchStudy

**Purpose:** Represents a research project/program in the data commons hierarchy.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| ID | `ResearchStudy.id` | Minted from project_id | Minted from associated_project | Minted from Atlas Name | Minted from project_code |
| Identifier | `ResearchStudy.identifier[]` | project, dbGaP accession | associated_project | Atlas Name | project_code |
| Name/Title | `ResearchStudy.name` | Project name | Project name | Atlas name (minus "HTAN " prefix) | Project description |
| Status | `ResearchStudy.status` | `"active"` | `"active"` | `"open"` | `"active"` |
| Condition | `ResearchStudy.condition[]` | Disease types → SNOMED | primary_diagnosis_condition | — | Cancer type SNOMED |
| Hierarchy | `ResearchStudy.partOf[]` | Parent project reference | — | HTAN program reference | ICGC program reference |
| Study Link | `ResearchStudy.extension[]` | — | `part-of-study` → CRDC | — | — |

### Hierarchy Patterns

| Source | Hierarchy |
|--------|-----------|
| GDC | Program → Project (2 levels max) |
| CDA | All → CRDC parent |
| HTAN | HTAN Program → Atlas Project |
| ICGC | ICGC Program → Project → Study |

### Condition SNOMED Codes (ICGC Examples)

| Project | SNOMED Code | Display |
|---------|-------------|---------|
| ESAD-UK, ESCA-CN | `276803003` | Adenocarcinoma of esophagus |
| LUSC-KR, LUSC-CN | `118286007` | Squamous Cell Neoplasms |

---

## 3. ResearchSubject

**Purpose:** Links Patient to ResearchStudy (enrollment relationship).

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| ID | `ResearchSubject.id` | Minted from patient identifier | Minted from researchsubject_id | Minted from patient identifier | Minted from patient identifier |
| Status | `ResearchSubject.status` | `"active"` | `"active"` | `"active"` | `"active"` |
| Subject | `ResearchSubject.subject` | `Reference(Patient/{id})` | Same | Same | Same |
| Study | `ResearchSubject.study` | `Reference(ResearchStudy/{id})` | Same | Same | Same |

---

## 4. Condition

**Purpose:** Represents a diagnosis with staging information.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| ID | `Condition.id` | Minted from diagnosis_id | Minted from diagnosis_id | Minted from patient+diagnosis | Minted from patient+project |
| Clinical Status | `Condition.clinicalStatus` | `"unknown"` | `"active"` | `"active"` | Context-dependent |
| Category | `Condition.category[]` | `encounter-diagnosis` + SNOMED 439401001 | Same | Same | — |
| Code | `Condition.code` | SNOMED lookup or raw | SNOMED or raw | Raw diagnosis string | ICD-10 |
| ICD-10 | `Condition.code.coding[]` | Validated via `icd10.find()` | — | — | Direct |
| MONDO | `Condition.code.coding[]` | NCIt → MONDO mapping | — | — | — |
| Body Site | `Condition.bodySite[]` | Primary site → SNOMED | — | Direct from mapping | SNOMED anatomical |
| Onset Age | `Condition.onsetAge` | Days as Age (unit: `d`) | — | Years as Age (unit: `a`) | String |
| Onset String | `Condition.onsetString` | — | age_at_diagnosis | — | donor_age_at_diagnosis |
| Recorded Date | `Condition.recordedDate` | — | — | Year of Diagnosis → DateTime | — |
| Stage | `Condition.stage[]` | Multiple ConditionStage entries | Single stage | Multiple stages | — |

### Clinical Status Mapping (ICGC)

| Source Value | FHIR Code | Display |
|--------------|-----------|---------|
| relapse_type in defined list | `relapse` | Relapse |
| `no evidence of disease`, `stable` | `remission` | Remission |
| default | `active` | Active |

### Staging Information

#### Stage Type SNOMED Codes

| Stage Component | SNOMED Code | Sources |
|-----------------|-------------|---------|
| Pathologic Stage Group | `1222593009` | GDC, CDA, HTAN |
| Pathologic T | `1222589003` | GDC, CDA |
| Pathologic N | `1222590007` | GDC, CDA |
| Pathologic M | `1222591006` | GDC, CDA |
| Pathologic Grade | `1222599008` | CDA |
| Clinical Stage Group | `1222592004` | CDA |
| Clinical T | `1222585009` | CDA |
| Clinical N | `1222588006` | CDA |
| Clinical M | `1222587001` | CDA |

#### Stage Priority Order (CDA)

1. `pathologic_stage`
2. `pathologic_stage_t`
3. `pathologic_stage_n`
4. `pathologic_stage_m`
5. `grade`
6. `clinical_stage`
7. `clinical_stage_t`
8. `clinical_stage_n`
9. `clinical_stage_m`

---

## 5. Observation

**Purpose:** Captures measurements, staging details, biospecimen properties, survey data, and exposure history.

### Observation Categories

| Category | Code | Display | Sources |
|----------|------|---------|---------|
| Laboratory | `laboratory` | Laboratory | GDC, CDA, HTAN, ICGC |
| Survey | `survey` | Survey | GDC |
| Exam | `exam` | Exam | GDC, HTAN, ICGC |
| Social History | `social-history` | Social History | GDC, ICGC |

### 5a. Staging Observations

| Field | FHIR Path | Description |
|-------|-----------|-------------|
| Code | `Observation.code.coding[]` | SNOMED stage type code |
| Value | `Observation.valueCodeableConcept` | Stage value with SNOMED coding |
| Focus | `Observation.focus[]` | `Reference(Condition/{id})` |
| Subject | `Observation.subject` | `Reference(Patient/{id})` |
| Has Member | `Observation.hasMember[]` | References to child T/N/M/grade observations (GDC) |

### 5b. Survey/Demographic Observations

| Observation Type | Code System | Code | Value Type | Sources |
|------------------|-------------|------|------------|---------|
| Year of Birth | ontobee | `NCIT_C83164` | `valueQuantity` (year) | GDC |
| Year of Death | ontobee | `NCIT_C156426` | `valueQuantity` (year) | GDC |
| Days to Death | ontobee | `NCIT_C156419` | `valueQuantity` (days) | GDC, CDA |
| Days to Birth | ontobee | `NCIT_C156418` | `valueQuantity` (days) | GDC, CDA |
| Days to Last Follow-up | ontobee | `NCIT_C181065` | `valueQuantity` (days) | GDC |
| Days to Last Known Disease Status | ontobee | `NCIT_C181066` | `valueQuantity` (days) | GDC |
| Days to Diagnosis | ontobee | `NCIT_C181061` | `valueQuantity` (days) | GDC |
| Cause of Death | LOINC | `69453-9` | `valueString` | CDA |
| Method of Diagnosis | ontobee | `NCIT_C177576` | `valueString` | CDA |
| Patient Information | LOINC | `52460-3` | Components | HTAN |

### 5c. Biospecimen Observations

**Pattern:** Component-based observations linked to Specimen.

| Component | Type | Specimen Level | Sources |
|-----------|------|----------------|---------|
| `composition` | string | Sample | GDC |
| `is_ffpe` | boolean | Sample, Portion | GDC |
| `sample_type` | string | All levels | GDC, CDA |
| `weight` | float | Portion | GDC |
| `analyte_type` | string | Analyte, Aliquot | GDC |
| `concentration` | float | Analyte, Aliquot | GDC |
| `rna_integrity_number` | float | Analyte | GDC |
| `aliquot_quantity` | float | Aliquot | GDC |
| `aliquot_volume` | float | Aliquot | GDC |
| `updated_datetime` | dateTime | All levels | GDC |
| `days_to_collection` | int | Specimen | CDA |
| `specimen_type` | string | Specimen | CDA |
| `primary_disease_type` | string | Specimen | CDA |
| `percentage_cellularity` | string | Sample | ICGC |
| `level_of_cellularity` | float | Sample | ICGC |
| `analyzed_sample_interval` | float | Sample | ICGC |
| `specimen_storage` | string | Specimen | ICGC |

### 5d. Exposure Observations

| Observation | Code System | Code | Value Type | Sources |
|-------------|-------------|------|------------|---------|
| Pack Years Smoked | LOINC | (social history) | `valueQuantity` | GDC |
| Cigarettes Per Day | LOINC | `64218-1` | `valueQuantity` | GDC |
| Alcohol History | GDC/ICGC | (exposure) | `valueString` | GDC, ICGC |
| Smoking Status | SNOMED | Various | `valueCodeableConcept` | ICGC |
| Alcohol Intensity | SNOMED | Various | `valueCodeableConcept` | ICGC |

#### Smoking SNOMED Codes (ICGC)

| Display | SNOMED Code |
|---------|-------------|
| Current smoker | `77176002` |
| Non-smoker | `8392000` |

#### Alcohol SNOMED Codes (ICGC)

| Display | SNOMED Code |
|---------|-------------|
| Occasional drinker | `228276006` |
| Social drinker | `28127009` |
| Once a week | `225769003` |
| Daily | `69620002` |

### 5e. File/Read Group Observations (GDC)

~40+ components per read group including:

| Component Category | Example Fields |
|--------------------|----------------|
| Sequencing | `platform`, `instrument_model`, `sequencing_center` |
| Library | `library_name`, `library_strategy`, `library_selection` |
| Read Group | `read_group_id`, `read_length`, `is_paired_end` |
| QC | `RIN`, `adapter_sequence`, `target_capture_kit` |

### 5f. Mutation Observations (CDA)

| Field | FHIR Path | Value |
|-------|-----------|-------|
| Code | `Observation.code` | NCIT_C164934 (Mutation Type Code) |
| Components | `Observation.component[]` | Dynamic from all non-null mutation fields |

### Component Helper Function

```python
def get_component(key, value, component_type, system) -> dict:
    """
    component_type: 'string' | 'int' | 'float' | 'bool' | 'dateTime'
    Returns: {
        "code": {"coding": [{"system": system, "code": key, "display": key}]},
        "value{Type}": value
    }
    """
```

---

## 6. Specimen

**Purpose:** Represents biospecimens with hierarchical relationships.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| ID | `Specimen.id` | Minted from specimen UUID | Minted from specimen_id | Minted from HTAN Biospecimen ID | Minted from icgc_sample/specimen_id |
| Identifier | `Specimen.identifier[]` | sample/portion/analyte/aliquot_id | specimen_id | HTAN Biospecimen ID | icgc_sample_id, submitted_sample_id |
| Type | `Specimen.type` | `sample_type` | `source_material_type` | `Biospecimen Type` | `specimen_type` |
| Subject | `Specimen.subject` | `Reference(Patient/{id})` | Same | Same | Same |
| Parent | `Specimen.parent[]` | Hierarchical references | `derived_from_specimen` | `HTAN Parent ID` | Sample → Specimen |
| Collection | `Specimen.collection.collector` | `Reference(Practitioner/{id})` | — | — | — |
| Collection | `Specimen.collection.bodySite` | — | `CodeableReference(BodyStructure/{id})` | — | — |
| Collection | `Specimen.collection.duration` | — | — | — | `specimen_interval` |
| Processing | `Specimen.processing[].method` | Preservation method | — | `Preservation Method` | `specimen_processing` |
| Study Link | `Specimen.extension[]` | — | `part-of-study` (supports multiple) | — | — |

### Hierarchy Patterns

| Source | Hierarchy |
|--------|-----------|
| GDC | Sample → Portion → Analyte → Aliquot (4 levels) |
| CDA | Parent → Child (2 levels) |
| HTAN | Parent → Child (2 levels) |
| ICGC | Sample → Specimen (2 levels) |
| CELLO | Parent → Child (derived-from relationship) |

### Identifier Systems

| Source | System URL |
|--------|------------|
| GDC | `https://gdc.cancer.gov/{level}_id` |
| CDA | `https://cda.readthedocs.io/specimen` |
| HTAN | `https://data.humantumoratlas.org` |
| ICGC | `https://platform.icgc-argo.org/icgc_sample_id` |

---

## 7. BodyStructure

**Purpose:** Represents anatomical sites for specimens and conditions.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| ID | `BodyStructure.id` | Minted from patient + body site | Minted from anatomical_site | Minted from patient + organ | Minted from patient + body_site |
| Included Structure | `BodyStructure.includedStructure[]` | SNOMED anatomical codes | Parsed from string | Tissue or Organ of Origin | SNOMED body site |
| Patient | `BodyStructure.patient` | `Reference(Patient/{id})` | Same | Same | Same |

### SNOMED Body Site Codes (ICGC)

| Site | SNOMED Code |
|------|-------------|
| Esophagus | `32849002` |
| Bronchus and lung | `110736001` |

### Parsing Logic (CDA)

```
"code:display" → Split on ":"
"site1,site2"  → Multiple includedStructure entries
"plain text"   → Use as both code and display
```

---

## 8. ImagingStudy

**Purpose:** Represents slide microscopy imaging data.

### Field Mappings (GDC Only)

| Field | FHIR Path | Transformation |
|-------|-----------|----------------|
| ID | `ImagingStudy.id` | Minted from slide_id |
| Status | `ImagingStudy.status` | Always `"available"` |
| Subject | `ImagingStudy.subject` | `Reference(Patient/{id})` |
| Series UID | `ImagingStudy.series[].uid` | Parent specimen ID |
| Modality | `ImagingStudy.series[].modality` | DICOM `SM` (Slide Microscopy) |
| Specimen | `ImagingStudy.series[].specimen[]` | `Reference(Specimen/{id})` |

---

## 9. DocumentReference

**Purpose:** Represents data files with metadata and access information.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN | ICGC |
|-------|-----------|-----|-----|------|------|
| ID | `DocumentReference.id` | Minted from file_id | Same | Minted from HTAN Data File ID | Minted from File ID |
| Status | `DocumentReference.status` | `"current"` | Same | Same | Same |
| Doc Status | `DocumentReference.docStatus` | — | — | `"final"` | — |
| Type | `DocumentReference.type` | Data format | `file_format` | — | `file_type` |
| Category | `DocumentReference.category[]` | Multiple fields | Multiple fields | Assay, Level | Data Type, Experimental Strategy |
| Subject | `DocumentReference.subject` | Specimen/Patient/Group | Same | Same | Patient/Specimen |
| Version | `DocumentReference.version` | File version | `"1"` | — | — |
| Security Label | `DocumentReference.securityLabel[]` | — | — | Data Access | — |
| Relates To | `DocumentReference.relatesTo[]` | — | — | Parent Data File ID | — |
| Based On | `DocumentReference.basedOn[]` | — | — | — | Specimen reference |

### Attachment Fields

| Field | FHIR Path | Sources |
|-------|-----------|---------|
| URL | `content[].attachment.url` | GDC: API URL, HTAN: file:/// path, ICGC: platform URL |
| Title | `content[].attachment.title` | All |
| Size | `content[].attachment.size` | All |
| Hash | `content[].attachment.hash` | GDC, CDA (MD5) |
| Content Type | `content[].attachment.contentType` | GDC, HTAN (mime_type) |

### Profile Fields

| Field | FHIR Path | Sources |
|-------|-----------|---------|
| Value URI | `content[].profile[].valueUri` | CDA (drs_uri), HTAN (drs_uri) |
| Value Coding | `content[].profile[].valueCoding` | GDC (data_format) |

### Category Fields by Source

| Source | Category Fields |
|--------|-----------------|
| GDC | `data_category`, `data_type`, `platform`, `experimental_strategy`, `workflow_type`, `access` |
| CDA | `data_category`, `data_type`, `data_modality`, `imaging_modality`, `imaging_series` |
| HTAN | `Assay`, `Level` |
| ICGC | `Data Type`, `Experimental Strategy` |

### Subject Resolution Priority

1. Single specimen → Direct Specimen reference
2. Multiple specimens → Group reference (type: `"specimen"`)
3. Single patient (no specimens) → Direct Patient reference
4. Multiple patients (no specimens) → Group reference (type: `"patient"`)

---

## 10. Group

**Purpose:** Aggregates multiple subjects when DocumentReference has multiple associated entities.

### Field Mappings

| Field | FHIR Path | All Sources |
|-------|-----------|-------------|
| ID | `Group.id` | Minted from member IDs + document ID |
| Type | `Group.type` | `"specimen"` or `"person"`/`"patient"` |
| Membership | `Group.membership` | Always `"definitional"` |
| Members | `Group.member[].entity` | `Reference(Specimen/{id})` or `Reference(Patient/{id})` |
| Identifier | `Group.identifier[]` | Concatenation of document ID + member IDs |

---

## 11. Organization

**Purpose:** Represents tissue source sites.

### Field Mappings (GDC Only)

| Field | FHIR Path | Transformation |
|-------|-----------|----------------|
| ID | `Organization.id` | Minted from TSS code |
| Name | `Organization.name` | TSS name |
| Identifier (official) | `Organization.identifier[]` | TSS code |
| Identifier (secondary) | `Organization.identifier[]` | BCR ID |

---

## 12. Practitioner

**Purpose:** Represents PI associated with tissue source site.

### Field Mappings (GDC Only)

| Field | FHIR Path | Transformation |
|-------|-----------|----------------|
| ID | `Practitioner.id` | Minted from TSS code + "-PI" |
| Qualification Code | `Practitioner.qualification[].code` | `PHD` (Doctor of Philosophy) |
| Qualification Issuer | `Practitioner.qualification[].issuer` | `Reference(Organization/{id})` |

---

## 13. MedicationAdministration

**Purpose:** Represents treatment administration events.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN |
|-------|-----------|-----|-----|------|
| ID | `MedicationAdministration.id` | Minted from treatment_id | Minted from patient+drug | Minted from atlas+patient+treatment |
| Status | `MedicationAdministration.status` | Logic-based | Logic-based | Logic-based |
| Category | `MedicationAdministration.category[]` | Treatment type (CaDSR 5102381) | — | Treatment Type |
| Medication | `MedicationAdministration.medication` | CodeableReference | Same | Same |
| Subject | `MedicationAdministration.subject` | `Reference(Patient/{id})` | Same | Same |
| Occurrence | `MedicationAdministration.occurenceDateTime` | Placeholder | — | — |
| Occurrence | `MedicationAdministration.occurenceTiming` | — | `boundsRange` | `boundsRange` |

### Status Mapping

| Condition | Status | Sources |
|-----------|--------|---------|
| `treatment_or_therapy = "yes"` | `"completed"` | GDC |
| `treatment_or_therapy = "no"` | `"not-done"` | GDC |
| `days_to_treatment_end` exists | `"completed"` | CDA, HTAN |
| `days_to_treatment_end` null | `"in-progress"` or `"unknown"` | CDA, HTAN |

---

## 14. Medication

**Purpose:** Represents therapeutic agents.

### Field Mappings

| Field | FHIR Path | GDC | CDA | HTAN |
|-------|-----------|-----|-----|------|
| ID | `Medication.id` | Minted from drug name (uppercased) | Same | Same |
| Code | `Medication.code` | Drug name or treatment type | Same | Same |
| Ingredient | `Medication.ingredient[].item` | — | `Reference(Substance/{id})` | Same |

### Exclusions (GDC)

Skip medication creation for: `"UNKNOWN"`, `"NOT REPORTED"`, `"OTHER"`, `"CHEMOTHERAPY"`, `"NOT OTHERWISE SPECIFIED"`

### Default Unknown Code

```json
{
  "system": "http://snomed.info/sct",
  "code": "261665006",
  "display": "Unknown"
}
```

---

## 15. Substance & SubstanceDefinition

**Purpose:** Detailed chemical structure information from ChEMBL.

### SubstanceDefinition (CDA, HTAN)

| Field | FHIR Path | Transformation |
|-------|-----------|----------------|
| ID | `SubstanceDefinition.id` | Minted from compound name |
| Name | `SubstanceDefinition.name[].name` | Compound name from ChEMBL |
| Structure InChI | `SubstanceDefinition.structure.representation[]` | `STANDARD_INCHI` format |
| Structure SMILES | `SubstanceDefinition.structure.representation[]` | `CANONICAL_SMILES` format |

### Substance (CDA, HTAN)

| Field | FHIR Path | Transformation |
|-------|-----------|----------------|
| ID | `Substance.id` | Minted from compound name |
| Code | `Substance.code` | CodeableReference to SubstanceDefinition |
| Category | `Substance.category[]` | `"drug"` (Drug or Medicament) |
| Instance | `Substance.instance` | Always `true` (placeholder) |

### ChEMBL Query

```sql
SELECT DISTINCT 
    a.CHEMBL_ID,
    c.STANDARD_INCHI,
    c.CANONICAL_SMILES,
    cr.COMPOUND_NAME
FROM MOLECULE_DICTIONARY as a
LEFT JOIN COMPOUND_STRUCTURES as c ON a.MOLREGNO = c.MOLREGNO
LEFT JOIN compound_records as cr ON a.MOLREGNO = cr.MOLREGNO
WHERE cr.COMPOUND_NAME IN {drug_names}
```

---

## 16. Encounter

**Purpose:** Represents clinical encounters (limited use).

### Field Mappings (ICGC, HTAN)

| Field | FHIR Path | Transformation |
|-------|-----------|----------------|
| ID | `Encounter.id` | Minted from patient identifier |
| Status | `Encounter.status` | `"completed"` |
| Subject | `Encounter.subject` | `Reference(Patient/{id})` |

---

## 17. Cross-Cutting Patterns

### Part-of-Study Extension

```json
{
  "url": "http://fhir-aggregator.org/fhir/StructureDefinition/part-of-study",
  "valueReference": {
    "reference": "ResearchStudy/{id}"
  }
}
```

**Applied to (CDA):** Patient, ResearchStudy, ResearchSubject, Condition, Observation, Specimen, BodyStructure, DocumentReference, Group, MedicationAdministration

### HTAN ID Deciphering

```python
def decipher_htan_id(_id) -> dict:
    """
    Pattern: <htan_center_id>_<integer>[_<integer>...]
    Wild-card: '0000' = same file from multiple participants
    External: 'EXT' = external participants
    Returns: {"participant_id": str, "subsets": list}
    """
```

### Resource Deduplication

```python
# Pattern used across all transformers
entities = list({v['id']: v for v in entities}.values())
```

---

## 18. Validation & Error Handling

### Regex Patterns

| Validation | Regex | Sources |
|------------|-------|---------|
| Submitter ID | `^[A-Za-z0-9\-.]+$` | GDC |
| FHIR string fields | `^[^\s]+(\s[^\s]+)*$` | GDC, CDA |
| FHIR code fields | `[ \r\n\t\S]+` | GDC |

### Library-Based Validation

| Validation | Library/Method | Sources |
|------------|----------------|---------|
| UUID validity | `uuid.UUID(value, version=5)` | CDA |
| ICD-10 codes | `icd10.find()` | GDC |
| Stage codes | Lookup table validation | GDC |

### Logging Patterns

| Log File | Purpose | Sources |
|----------|---------|---------|
| `output.log` | Missing diagnoses, stage codes | GDC |
| `no_reference.log` | Missing subject references | GDC |
| `no_project_associations.log` | Missing project associations | CDA |
| `error.log` | General errors | CDA |
| `info.log` | General info | CDA |

### Skip Conditions

| Condition | Action | Sources |
|-----------|--------|---------|
| `primary_diagnosis` is null | Skip Condition creation | CDA |
| `anatomical_site` contains "Not specified" | Skip BodyStructure | CDA |
| No valid subject reference | Skip DocumentReference | HTAN |
| Failed regex validation | Skip field/resource | All |

---

## Appendix: System URLs Quick Reference

| System | URL |
|--------|-----|
| SNOMED CT | `http://snomed.info/sct` |
| LOINC | `http://loinc.org` |
| ICD-10-CM | `https://terminology.hl7.org/5.1.0/NamingSystem-icd10CM.html` |
| NCI Thesaurus | `https://ncit.nci.nih.gov` |
| MONDO | `https://www.ebi.ac.uk/ols4/ontologies/mondo` |
| ChEMBL | `https://www.ebi.ac.uk/chembl` |
| Ontobee | `https://ontobee.org/` |
| CaDSR | `https://cadsr.cancer.gov/` |
| US Core Birth Sex | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex` |
| US Core Race | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-race` |
| US Core Ethnicity | `http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity` |
| FHIR Aggregator Part-of-Study | `http://fhir-aggregator.org/fhir/StructureDefinition/part-of-study` |
| DICOM | `http://dicom.nema.org/resources/ontology/DCM` |

---

*Generated from analysis of GDC entity2fhir.py, CDA transformer.py, HTAN htan2fhir.py, and ICGC icgc2fhir.py*
