"""csv/dataframe loader with claude data engineer integration."""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from .base import BaseLoader


class CSVLoader(BaseLoader):
    """
    csv loader for biomedical data (case, file, biospecimen tables).
    
    integrates with claude data engineer for intelligent data analysis
    and schema inference with up to 5 iterations for refinement.
    """
    
    def __init__(self, name: str = "csv_loader"):
        super().__init__(name)
        self.loaded_data = {}
        self.schema_info = {}
        self.claude_de_iterations = 0
        self.max_iterations = 5
    
    def load(
        self, 
        sources: Union[str, Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        load csv files for biomedical entities.
        
        args:
            sources: either a single csv path or dict of {entity_name: csv_path}
                    e.g., {"case": "cases.csv", "file": "files.csv", "biospecimen": "biospecimen.csv"}
        
        returns:
            dictionary with loaded dataframes and metadata
        """
        if isinstance(sources, str):
            # single file - infer entity name from filename
            entity_name = Path(sources).stem
            sources = {entity_name: sources}
        
        self.loaded_data = {}
        
        for entity_name, csv_path in sources.items():
            try:
                df = pd.read_csv(csv_path, **kwargs)
                self.loaded_data[entity_name] = {
                    "dataframe": df,
                    "source_path": csv_path,
                    "analysis": self._analyze_dataframe(df, entity_name)
                }
                print(f"Loaded {entity_name}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"✗ Failed to load {entity_name} from {csv_path}: {e}")
                self.loaded_data[entity_name] = {
                    "dataframe": None,
                    "source_path": csv_path,
                    "error": str(e)
                }
        
        return self.loaded_data
    
    def get_schema_info(self, loaded_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """extract comprehensive schema information."""
        data = loaded_data or self.loaded_data
        
        schema_info = {
            "entities": {},
            "total_entities": len(data),
            "successful_loads": sum(1 for v in data.values() if v.get("dataframe") is not None),
            "summary": {}
        }
        
        for entity_name, entity_data in data.items():
            if entity_data.get("dataframe") is not None:
                df = entity_data["dataframe"]
                analysis = entity_data["analysis"]
                
                schema_info["entities"][entity_name] = {
                    "fields": {
                        col: {
                            "type": str(df[col].dtype),
                            "samples": df[col].dropna().head(3).tolist(),
                            "null_count": int(df[col].isnull().sum()),
                            "unique_count": int(df[col].nunique()),
                            "is_identifier": self._is_identifier_field(col, df[col]),
                            "is_biomedical": self._is_biomedical_field(col, df[col])
                        }
                        for col in df.columns
                    },
                    "row_count": len(df),
                    "source": entity_data["source_path"]
                }
            else:
                schema_info["entities"][entity_name] = {
                    "fields": {},
                    "error": entity_data.get("error", "Unknown error")
                }
        
        # generate summary for claude de analysis
        schema_info["summary"] = self._generate_summary(schema_info)
        return schema_info
    
    def _is_identifier_field(self, column_name: str, series: pd.Series) -> bool:
        """heuristic to detect identifier fields."""
        col_lower = column_name.lower()

        # name-based detection
        id_indicators = ['id', 'identifier', 'code', 'key', 'uuid', 'guid', 'number']
        if any(indicator in col_lower for indicator in id_indicators):
            return True
        
        # pattern-based detection
        if series.dtype == 'object':
            sample_values = series.dropna().head(10).astype(str)
            if len(sample_values) > 0:
                # check for common id patterns
                avg_length = sample_values.str.len().mean()
                has_consistent_format = sample_values.str.contains(r'^[A-Z0-9-_]+$', na=False).mean() > 0.7
                
                if avg_length > 5 and has_consistent_format:
                    return True
        
        return False
    
    def _is_biomedical_field(self, column_name: str, series: pd.Series) -> bool:
        """heuristic to detect biomedical fields."""
        col_lower = column_name.lower()
        
        biomedical_terms = [
            'patient', 'specimen', 'tissue', 'sample', 'diagnosis', 'disease',
            'anatomy', 'pathology', 'medical', 'clinical', 'treatment', 'therapy',
            'biospecimen', 'histology', 'morphology', 'grade', 'stage'
        ]
        
        return any(term in col_lower for term in biomedical_terms)
    
    def _generate_summary(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """generate summary for claude de analysis."""
        total_fields = 0
        identifier_fields = 0
        biomedical_fields = 0
        
        entity_summaries = {}
        
        for entity_name, entity_data in schema_info["entities"].items():
            if "fields" in entity_data and entity_data["fields"]:
                fields = entity_data["fields"]
                total_fields += len(fields)
                
                entity_id_count = sum(1 for f in fields.values() if f.get("is_identifier", False))
                entity_bio_count = sum(1 for f in fields.values() if f.get("is_biomedical", False))
                
                identifier_fields += entity_id_count
                biomedical_fields += entity_bio_count
                
                entity_summaries[entity_name] = {
                    "field_count": len(fields),
                    "identifier_fields": entity_id_count,
                    "biomedical_fields": entity_bio_count,
                    "row_count": entity_data.get("row_count", 0),
                    "key_fields": [
                        name for name, info in fields.items()
                        if info.get("is_identifier") or info.get("is_biomedical")
                    ][:5]  # top 5 key fields
                }
        
        return {
            "total_fields": total_fields,
            "identifier_fields": identifier_fields,
            "biomedical_fields": biomedical_fields,
            "entities": entity_summaries,
            "analysis_ready": total_fields > 0
        }
    
    def analyze_with_claude_de(self, max_iterations: int = 5) -> Dict[str, Any]:
        """
        integrate with claude data engineer for enhanced analysis.
        
        this method would call claude de to:
        1. analyze the loaded data structure
        2. identify potential fhir mappings
        3. suggest data quality improvements
        4. refine schema understanding through iterations
        """
        self.max_iterations = max_iterations
        self.claude_de_iterations = 0
        
        schema_info = self.get_schema_info()
        
        # placeholder for claude de integration
        # in real implementation, this would call claude de api
        analysis_prompt = self._build_claude_de_prompt(schema_info)
        
        return {
            "schema_info": schema_info,
            "claude_de_prompt": analysis_prompt,
            "iterations_used": self.claude_de_iterations,
            "ready_for_mapping": True
        }
    
    def _build_claude_de_prompt(self, schema_info: Dict[str, Any]) -> str:
        """build prompt for claude data engineer analysis."""
        summary = schema_info["summary"]
        
        prompt = f"""analyze this biomedical csv dataset for fhir schema mapping:

dataset summary:
- {summary['total_fields']} total fields across {schema_info['total_entities']} entities
- {summary['identifier_fields']} identifier fields detected
- {summary['biomedical_fields']} biomedical fields detected

entities:"""
        
        for entity_name, entity_summary in summary["entities"].items():
            prompt += f"""

{entity_name.upper()}:
- {entity_summary['field_count']} fields, {entity_summary['row_count']} rows
- key fields: {', '.join(entity_summary['key_fields'])}"""
        
        prompt += """

tasks:
1. identify which entity corresponds to fhir patient, specimen, observation, etc.
2. map key identifier and biomedical fields to fhir paths
3. suggest data transformations needed
4. flag potential data quality issues
5. recommend mapping confidence levels

provide structured analysis for automated processing."""
        
        return prompt
    
    def get_mapping_candidates(self) -> Dict[str, List[str]]:
        """get field candidates for each fhir target based on heuristics."""
        schema_info = self.get_schema_info()
        candidates = {}
        
        fhir_targets = [
            "Patient.identifier", "Patient.birthDate", "Patient.deceasedBoolean",
            "Specimen.type", "Specimen.collection.method", "Specimen.collection.collectedDateTime",
            "Observation.code", "Observation.valueString", "Observation.valueQuantity",
            "Condition.code", "Condition.stage.summary", "Condition.onsetDateTime",
            "DocumentReference.content.attachment.url", "DocumentReference.type"
        ]
        
        for target in fhir_targets:
            candidates[target] = []
            
            for entity_name, entity_data in schema_info["entities"].items():
                if "fields" in entity_data:
                    for field_name, field_info in entity_data["fields"].items():
                        if self._is_candidate_for_target(field_name, field_info, target):
                            candidates[target].append(f"{entity_name}.{field_name}")
        
        return candidates
    
    def _is_candidate_for_target(
        self, 
        field_name: str, 
        field_info: Dict[str, Any], 
        target: str
    ) -> bool:
        """heuristic matching of fields to fhir targets."""
        field_lower = field_name.lower()
        target_lower = target.lower()
        
        # simple keyword matching
        if "identifier" in target_lower and field_info.get("is_identifier", False):
            return True
        
        if "specimen" in target_lower and any(
            term in field_lower for term in ["specimen", "tissue", "sample"]
        ):
            return True
        
        if "date" in target_lower and any(
            term in field_lower for term in ["date", "time", "collected", "onset"]
        ):
            return True
        
        if "url" in target_lower and any(
            sample and ("http" in str(sample) or "ftp" in str(sample))
            for sample in field_info.get("samples", [])
        ):
            return True
        
        return False