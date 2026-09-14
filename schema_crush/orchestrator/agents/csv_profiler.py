"""csv profiler for understanding use-case and column relationships."""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ContentType:
    """detected content type for a column."""
    dtype: str  # boolean, categorical, numeric, date, id, text, coded
    pattern: Optional[str] = None  # specific pattern detected
    vocabulary: Optional[str] = None  # detected vocabulary (e.g., "vital_status", "tnm_stage")
    nullable: bool = False


@dataclass
class CSVProfile:
    """profile of a csv file for context-aware mapping."""
    use_case: str  # survival_analysis, variant_tracking, lab_results, clinical_phenotyping, etc.
    analysis_purpose: str  # human-readable description
    data_provenance: str  # ehr, sequencing_lab, registry, clinical_trial, etc.
    primary_entity: str  # patient, sample, variant, lab_result
    column_groups: Dict[str, List[str]]  # semantic groupings
    content_types: Dict[str, ContentType]  # per-column content types
    column_relationships: Dict[str, str]  # column -> related column(s)
    recommendations: Dict[str, str]  # column -> mapping recommendation


class ContentTypeDetector:
    """zero-cost programmatic content type detection."""

    # patterns for common biomedical data
    BOOLEAN_VALUES = {
        '0', '1', 'true', 'false', 'yes', 'no', 'y', 'n',
        'alive', 'dead', 'positive', 'negative', 'pos', 'neg',
        'male', 'female', 'm', 'f'
    }

    VITAL_STATUS_VALUES = {'alive', 'dead', 'living', 'deceased', 'censored'}

    TNM_PATTERN = re.compile(r'^[pcy]?[TNM][0-4X]?[abc]?$', re.IGNORECASE)
    STAGE_PATTERN = re.compile(r'^(stage\s*)?(I{1,3}V?|[1-4])[ABC]?$', re.IGNORECASE)
    GRADE_PATTERN = re.compile(r'^G[1-4X]$|^(grade\s*)?[1-4]$', re.IGNORECASE)

    HGVS_PATTERN = re.compile(r'^[cgnmpr]\.[\d\-\+\*]+', re.IGNORECASE)
    CHROMOSOME_PATTERN = re.compile(r'^(chr)?\d{1,2}|X|Y$', re.IGNORECASE)

    GSM_PATTERN = re.compile(r'^GSM\d+$')
    UUID_PATTERN = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.IGNORECASE)

    @classmethod
    def detect(cls, column_name: str, sample_values: List[str]) -> ContentType:
        """detect content type from column name and sample values.

        args:
            column_name: name of the column
            sample_values: list of sample values (strings)

        returns:
            contenttype with detected dtype, pattern, and vocabulary
        """
        # clean values - remove empty/null
        values = [v.strip().lower() for v in sample_values if v and v.strip() and v.lower() not in ('na', 'nan', 'null', '')]

        if not values:
            return ContentType(dtype="unknown", nullable=True)

        col_lower = column_name.lower()

        # check for id patterns first (by column name)
        if any(x in col_lower for x in ['_id', 'identifier', 'accession', 'sample_id', 'participant_id', 'patient_id']):
            if all(cls.GSM_PATTERN.match(v) for v in values if v):
                return ContentType(dtype="id", pattern="geo_accession")
            if all(cls.UUID_PATTERN.match(v) for v in values if v):
                return ContentType(dtype="id", pattern="uuid")
            return ContentType(dtype="id", pattern="identifier")

        # check for boolean patterns
        unique_values = set(values)
        if unique_values.issubset(cls.BOOLEAN_VALUES):
            # determine specific vocabulary
            if unique_values.issubset({'0', '1'}):
                # check column name for hint
                if any(x in col_lower for x in ['death', 'deceased', 'dead', 'event', 'censor', 'status']):
                    return ContentType(dtype="boolean", pattern="binary_event", vocabulary="survival_event")
                return ContentType(dtype="boolean", pattern="binary")
            if unique_values.issubset({'alive', 'dead', 'living', 'deceased'}):
                return ContentType(dtype="boolean", pattern="vital_status", vocabulary="vital_status")
            if unique_values.issubset({'male', 'female', 'm', 'f'}):
                return ContentType(dtype="categorical", pattern="sex", vocabulary="administrative_gender")
            return ContentType(dtype="boolean", pattern="binary")

        # check for vital status in column name + categorical values
        if any(x in col_lower for x in ['vital', 'status', 'survival_status']):
            if any(v in cls.VITAL_STATUS_VALUES for v in unique_values):
                return ContentType(dtype="boolean", pattern="vital_status", vocabulary="vital_status")

        # check for tnm staging
        if any(x in col_lower for x in ['t_stage', 'n_stage', 'm_stage', 'tnm', 'pathologic_t', 'pathologic_n', 'pathologic_m', 'clinical_t', 'clinical_n', 'clinical_m']):
            return ContentType(dtype="coded", pattern="tnm_category", vocabulary="ajcc_tnm")
        if all(cls.TNM_PATTERN.match(v) for v in values if v):
            return ContentType(dtype="coded", pattern="tnm_category", vocabulary="ajcc_tnm")

        # check for stage
        if 'stage' in col_lower and 't_stage' not in col_lower and 'n_stage' not in col_lower and 'm_stage' not in col_lower:
            return ContentType(dtype="coded", pattern="cancer_stage", vocabulary="ajcc_stage")
        if all(cls.STAGE_PATTERN.match(v) for v in values if v):
            return ContentType(dtype="coded", pattern="cancer_stage", vocabulary="ajcc_stage")

        # check for grade
        if 'grade' in col_lower or 'grading' in col_lower:
            return ContentType(dtype="coded", pattern="tumor_grade", vocabulary="histologic_grade")
        if all(cls.GRADE_PATTERN.match(v) for v in values if v):
            return ContentType(dtype="coded", pattern="tumor_grade", vocabulary="histologic_grade")

        # check for hgvs notation (variants)
        if all(cls.HGVS_PATTERN.match(v) for v in values if v):
            return ContentType(dtype="coded", pattern="hgvs", vocabulary="hgvs_nomenclature")

        # check for chromosome
        if 'chromosome' in col_lower or 'chr' in col_lower:
            return ContentType(dtype="coded", pattern="chromosome", vocabulary="chromosome")
        if all(cls.CHROMOSOME_PATTERN.match(v) for v in values if v):
            return ContentType(dtype="coded", pattern="chromosome", vocabulary="chromosome")

        # check for numeric (survival time, positions, frequencies)
        try:
            floats = [float(v) for v in values if v]
            if floats:
                # check if integers
                if all(f == int(f) for f in floats):
                    if any(x in col_lower for x in ['position', 'start', 'end', 'stop']):
                        return ContentType(dtype="numeric", pattern="genomic_position")
                    if any(x in col_lower for x in ['days', 'day']):
                        return ContentType(dtype="numeric", pattern="duration_days")
                    return ContentType(dtype="numeric", pattern="integer")
                else:
                    if any(x in col_lower for x in ['survival', 'time', 'month']):
                        return ContentType(dtype="numeric", pattern="duration_time")
                    if any(x in col_lower for x in ['frequency', 'freq', 'vaf', 'percent', 'pct']):
                        return ContentType(dtype="numeric", pattern="frequency")
                    return ContentType(dtype="numeric", pattern="continuous")
        except ValueError:
            pass

        # check for categorical by low cardinality
        if len(unique_values) <= 10 and len(values) > len(unique_values) * 2:
            return ContentType(dtype="categorical", pattern="enumerated")

        # default to text
        return ContentType(dtype="text", pattern="free_text")


class CSVProfiler:
    """profiler for understanding csv use-case and structure."""

    # use-case detection patterns
    USE_CASE_INDICATORS = {
        "survival_analysis": [
            "survival", "os_time", "os_event", "vital_status", "days_to_death",
            "survival_months", "death_event", "censor", "overall_survival"
        ],
        "variant_tracking": [
            "variant", "mutation", "gene", "chromosome", "hgvs", "position",
            "variant_type", "allele", "frequency", "vaf", "snp", "cnv",
            "pathogenic", "clinical_significance"
        ],
        "lab_results": [
            "result", "value", "unit", "reference_range", "analyte",
            "concentration", "measurement", "test_name", "panel"
        ],
        "staging_grading": [
            "stage", "grade", "tnm", "t_stage", "n_stage", "m_stage",
            "pathologic", "clinical", "ajcc", "grading"
        ],
        "sample_metadata": [
            "sample", "specimen", "tissue", "tumor", "normal", "ffpe",
            "collection", "preservation", "biopsy"
        ],
        "clinical_phenotyping": [
            "diagnosis", "phenotype", "symptom", "finding", "condition",
            "disease", "histology", "morphology"
        ]
    }

    # provenance detection
    PROVENANCE_INDICATORS = {
        "geo_repository": ["gsm", "gse", "gpl"],
        "sequencing_lab": ["panel_name", "run", "flowcell", "lane", "barcode"],
        "clinical_registry": ["mrn", "encounter", "visit", "admission"],
        "research_study": ["participant", "subject", "cohort", "arm"]
    }

    def __init__(self, use_llm: bool = True, llm_model: str = "claude-opus-4-6"):
        """initialize profiler.

        args:
            use_llm: whether to use llm for enhanced profiling (costs ~$0.01-0.02/csv)
            llm_model: claude model to use for llm profiling
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.content_detector = ContentTypeDetector()

    def profile(self, columns: List[str], sample_data: Dict[str, List[str]]) -> CSVProfile:
        """profile a csv to understand its purpose and structure.

        args:
            columns: list of column names
            sample_data: dict mapping column names to sample values

        returns:
            csvprofile with use-case, column groups, and recommendations
        """
        # step 1: detect content types for all columns (zero cost)
        content_types = {}
        for col in columns:
            samples = sample_data.get(col, [])
            content_types[col] = self.content_detector.detect(col, samples)

        # step 2: detect use-case from column names (zero cost)
        use_case = self._detect_use_case(columns)

        # step 3: detect data provenance (zero cost)
        provenance = self._detect_provenance(columns, sample_data)

        # step 4: group related columns (zero cost)
        column_groups = self._group_columns(columns, content_types)

        # step 5: detect column relationships (zero cost)
        relationships = self._detect_relationships(columns, content_types)

        # step 6: generate recommendations (zero cost or llm)
        if self.use_llm:
            recommendations, analysis_purpose = self._llm_recommendations(
                columns, sample_data, content_types, use_case, column_groups
            )
        else:
            recommendations = self._heuristic_recommendations(columns, content_types, use_case)
            analysis_purpose = self._generate_purpose_description(use_case, provenance, columns)

        # determine primary entity
        primary_entity = self._determine_primary_entity(columns, content_types, use_case)

        return CSVProfile(
            use_case=use_case,
            analysis_purpose=analysis_purpose,
            data_provenance=provenance,
            primary_entity=primary_entity,
            column_groups=column_groups,
            content_types=content_types,
            column_relationships=relationships,
            recommendations=recommendations
        )

    def _detect_use_case(self, columns: List[str]) -> str:
        """detect primary use-case from column names."""
        col_lower = [c.lower() for c in columns]
        col_text = ' '.join(col_lower)

        scores = {}
        for use_case, indicators in self.USE_CASE_INDICATORS.items():
            score = sum(1 for ind in indicators if ind in col_text)
            if score > 0:
                scores[use_case] = score

        if not scores:
            return "unknown"

        # return highest scoring use-case
        return max(scores, key=scores.get)

    def _detect_provenance(self, columns: List[str], sample_data: Dict[str, List[str]]) -> str:
        """detect data provenance from column names and values."""
        col_lower = [c.lower() for c in columns]
        col_text = ' '.join(col_lower)

        # check sample values for geo accessions
        for col, values in sample_data.items():
            for v in values[:3]:
                if v and re.match(r'^GSM\d+$', v):
                    return "geo_repository"

        for provenance, indicators in self.PROVENANCE_INDICATORS.items():
            if any(ind in col_text for ind in indicators):
                return provenance

        return "unknown"

    def _group_columns(self, columns: List[str], content_types: Dict[str, ContentType]) -> Dict[str, List[str]]:
        """group semantically related columns."""
        groups = {
            "identifiers": [],
            "survival": [],
            "staging": [],
            "genomic_location": [],
            "variant_details": [],
            "sample_info": [],
            "clinical": [],
            "other": []
        }

        for col in columns:
            col_lower = col.lower()
            ct = content_types.get(col, ContentType(dtype="unknown"))

            # identifiers
            if ct.pattern in ["identifier", "geo_accession", "uuid"] or ct.dtype == "id":
                groups["identifiers"].append(col)
            # survival
            elif ct.vocabulary == "survival_event" or any(x in col_lower for x in ['survival', 'os_time', 'os_event', 'death', 'vital', 'censor']):
                groups["survival"].append(col)
            # staging
            elif ct.vocabulary in ["ajcc_tnm", "ajcc_stage", "histologic_grade"] or any(x in col_lower for x in ['stage', 'grade', 't_stage', 'n_stage', 'm_stage']):
                groups["staging"].append(col)
            # genomic location
            elif any(x in col_lower for x in ['chromosome', 'position', 'start', 'end', 'stop', 'genome_build']):
                groups["genomic_location"].append(col)
            # variant details
            elif any(x in col_lower for x in ['gene', 'variant', 'mutation', 'hgvs', 'allele', 'frequency', 'vaf']):
                groups["variant_details"].append(col)
            # sample info
            elif any(x in col_lower for x in ['sample', 'tissue', 'tumor', 'specimen', 'biopsy']):
                groups["sample_info"].append(col)
            # clinical
            elif any(x in col_lower for x in ['diagnosis', 'disease', 'condition', 'phenotype']):
                groups["clinical"].append(col)
            else:
                groups["other"].append(col)

        # remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _detect_relationships(self, columns: List[str], content_types: Dict[str, ContentType]) -> Dict[str, str]:
        """detect relationships between columns."""
        relationships = {}
        col_lower = {c.lower(): c for c in columns}

        # survival pairs (time + event)
        survival_time_cols = [c for c in columns if any(x in c.lower() for x in ['survival', 'os_time', '_time', '_months'])]
        survival_event_cols = [c for c in columns if any(x in c.lower() for x in ['event', 'death', 'vital', 'censor', 'status'])]

        for tc in survival_time_cols:
            for ec in survival_event_cols:
                # check if they share a prefix or are clearly related
                tc_base = re.sub(r'(time|months?|_time|survival)', '', tc.lower())
                ec_base = re.sub(r'(event|death|status|vital|censor)', '', ec.lower())
                if tc_base and tc_base in ec_base or ec_base and ec_base in tc_base or 'survival' in tc.lower():
                    relationships[tc] = f"survival_pair:{ec}"
                    relationships[ec] = f"survival_pair:{tc}"

        # genomic location group
        loc_cols = [c for c in columns if any(x in c.lower() for x in ['chromosome', 'position', 'start', 'end', 'stop'])]
        if len(loc_cols) > 1:
            for c in loc_cols:
                relationships[c] = f"genomic_location_group:{','.join(loc_cols)}"

        # tnm group
        tnm_cols = [c for c in columns if any(x in c.lower() for x in ['t_stage', 'n_stage', 'm_stage'])]
        if len(tnm_cols) > 1:
            for c in tnm_cols:
                relationships[c] = f"tnm_group:{','.join(tnm_cols)}"

        return relationships

    def _determine_primary_entity(self, columns: List[str], content_types: Dict[str, ContentType], use_case: str) -> str:
        """determine the primary entity type for the csv."""
        col_lower = ' '.join(c.lower() for c in columns)

        if 'variant' in use_case or any(x in col_lower for x in ['gene', 'variant', 'mutation', 'hgvs']):
            return "variant_observation"
        if 'lab' in use_case:
            return "lab_observation"
        if any(x in col_lower for x in ['sample_id', 'specimen', 'biopsy']):
            return "sample"
        if any(x in col_lower for x in ['patient_id', 'participant_id', 'subject_id']):
            return "patient"
        if 'survival' in use_case or 'staging' in use_case:
            return "patient"
        return "patient"  # default

    def _generate_purpose_description(self, use_case: str, provenance: str, columns: List[str]) -> str:
        """generate human-readable purpose description."""
        purposes = {
            "survival_analysis": "clinical survival/outcome data for patient cohort analysis",
            "variant_tracking": "molecular variant data for genomic analysis",
            "lab_results": "laboratory test results and measurements",
            "staging_grading": "cancer staging and grading assessment data",
            "sample_metadata": "biospecimen/sample metadata and characteristics",
            "clinical_phenotyping": "clinical phenotype and diagnosis data"
        }

        base = purposes.get(use_case, "biomedical data")

        if provenance == "geo_repository":
            base += " from geo repository"
        elif provenance == "sequencing_lab":
            base += " from sequencing laboratory"

        return base

    def _heuristic_recommendations(self, columns: List[str], content_types: Dict[str, ContentType], use_case: str) -> Dict[str, str]:
        """generate mapping recommendations using heuristics (zero cost)."""
        recommendations = {}

        for col in columns:
            ct = content_types.get(col, ContentType(dtype="unknown"))
            col_lower = col.lower()

            # boolean survival events -> Patient.deceasedBoolean
            if ct.vocabulary == "survival_event" or (ct.dtype == "boolean" and any(x in col_lower for x in ['death', 'deceased', 'dead', 'vital'])):
                recommendations[col] = "binary survival/death indicator -> Patient.deceasedBoolean (map 1/DEAD to true)"

            # survival time -> Observation.valueQuantity
            elif ct.pattern == "duration_time" or any(x in col_lower for x in ['survival_month', 'os_time', 'survival_time']):
                recommendations[col] = "survival duration -> Observation.valueQuantity with appropriate time unit"

            # stage -> Condition.stage
            elif ct.vocabulary == "ajcc_stage" or (ct.pattern == "cancer_stage"):
                recommendations[col] = "cancer stage -> Condition.stage[].summary with ajcc staging code"

            # tnm -> Observation.component under staging Observation
            elif ct.vocabulary == "ajcc_tnm":
                recommendations[col] = "tnm category -> Observation.valueCodeableConcept under staging Observation.hasMember"

            # grade -> Observation.valueCodeableConcept
            elif ct.vocabulary == "histologic_grade":
                recommendations[col] = "tumor grade -> Observation.valueCodeableConcept with snomed grade code"

            # identifiers
            elif ct.dtype == "id":
                if 'patient' in col_lower or 'participant' in col_lower:
                    recommendations[col] = "patient identifier -> Patient.identifier"
                elif 'sample' in col_lower or 'specimen' in col_lower:
                    recommendations[col] = "sample identifier -> Specimen.identifier"
                else:
                    recommendations[col] = "identifier field -> appropriate resource.identifier"

            # genomic positions
            elif ct.pattern == "genomic_position":
                recommendations[col] = "genomic coordinate -> Observation.component (as part of variant observation)"

            # variant details for complex genomic data
            elif any(x in col_lower for x in ['gene', 'chromosome', 'variant', 'hgvs']):
                recommendations[col] = "variant detail -> Observation.component with observation_subject: Specimen"

        return recommendations

    def _llm_recommendations(
        self,
        columns: List[str],
        sample_data: Dict[str, List[str]],
        content_types: Dict[str, ContentType],
        use_case: str,
        column_groups: Dict[str, List[str]]
    ) -> tuple[Dict[str, str], str]:
        """generate recommendations using llm for enhanced understanding.

        returns:
            tuple of (recommendations dict, analysis_purpose string)
        """
        import os
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # fall back to heuristics
            return self._heuristic_recommendations(columns, content_types, use_case), self._generate_purpose_description(use_case, "unknown", columns)

        llm = ChatAnthropic(model=self.llm_model, api_key=api_key, temperature=0)

        # build compact representation
        col_info = []
        for col in columns:
            ct = content_types.get(col, ContentType(dtype="unknown"))
            samples = sample_data.get(col, [])[:3]
            col_info.append(f"  {col}: {ct.dtype}({ct.pattern}) samples={samples}")

        prompt = f"""analyze this biomedical csv to understand its purpose and provide mapping recommendations.

columns and sample data:
{chr(10).join(col_info)}

detected use-case: {use_case}
detected column groups: {column_groups}

provide:
1. analysis_purpose: one sentence describing what this data is for (e.g., "clinical survival data from geo for pancreatic cancer cohort analysis")

2. mapping_recommendations: for columns that need special handling, provide a brief recommendation.
   focus on:
   - binary 0/1 columns that represent survival events (should map to Patient.deceasedBoolean)
   - time/duration columns (survival months, days to event)
   - cryptic column names that need interpretation
   - columns that work together as a group

format your response as:
analysis_purpose: [one sentence]

recommendations:
column_name: [brief recommendation]
...

only include columns that need guidance - skip obvious ones like sample_id -> identifier."""

        try:
            response = llm.invoke([
                SystemMessage(content="you are a biomedical data expert. be concise."),
                HumanMessage(content=prompt)
            ])

            content = response.content if isinstance(response.content, str) else str(response.content)

            # parse response
            analysis_purpose = ""
            recommendations = {}

            # extract analysis purpose
            purpose_match = re.search(r'analysis_purpose:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            if purpose_match:
                analysis_purpose = purpose_match.group(1).strip()

            # extract recommendations
            rec_section = re.search(r'recommendations:\s*\n(.+)', content, re.DOTALL | re.IGNORECASE)
            if rec_section:
                for line in rec_section.group(1).strip().split('\n'):
                    if ':' in line:
                        col, rec = line.split(':', 1)
                        col = col.strip()
                        if col in columns:
                            recommendations[col] = rec.strip()

            # merge with heuristics for columns not covered by llm
            heuristic_recs = self._heuristic_recommendations(columns, content_types, use_case)
            for col, rec in heuristic_recs.items():
                if col not in recommendations:
                    recommendations[col] = rec

            return recommendations, analysis_purpose or self._generate_purpose_description(use_case, "unknown", columns)

        except Exception as e:
            # fall back to heuristics on error
            return self._heuristic_recommendations(columns, content_types, use_case), self._generate_purpose_description(use_case, "unknown", columns)