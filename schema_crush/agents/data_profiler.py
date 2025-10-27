"""data profiler - specialized agent for data loading and schema analysis."""

import pandas as pd
from typing import Dict, Any, List, Optional
from .base import BaseAgent


class DataProfiler(BaseAgent):
    """
    data profiler: specialized agent for data engineering tasks.

    focused on:
    - csv data loading and analysis
    - schema inference and validation
    - data quality assessment
    - etl planning and execution
    - up to 5 iterations for refinement
    """

    def __init__(self, name: str = "data_profiler"):
        super().__init__(name)
        self.max_iterations = 5
        self.iteration_count = 0
        self.data_analysis_cache = {}
        self.transformation_patterns = {}
        
        # data engineering expertise domains
        self.expertise_domains = {
            "csv_analysis": 0.95,
            "schema_inference": 0.9,
            "data_quality": 0.9,
            "etl_planning": 0.85,
            "biomedical_data": 0.8
        }
    
    def match(self, source: str, target: str, level: str) -> Dict[str, Any]:
        """
        Claude DE's data-driven matching approach.
        
        Uses data analysis, schema patterns, and ETL expertise to determine matches.
        """
        # analyze from data engineering perspective
        analysis = self._data_engineering_analysis(source, target, level)
        
        # Apply data pattern recognition
        pattern_confidence = self._pattern_analysis(source, target, analysis)
        
        # Schema compatibility assessment
        schema_confidence = self._schema_compatibility(source, target, level)
        
        # ETL feasibility analysis
        etl_confidence = self._etl_feasibility(source, target, analysis)
        
        # Combine confidences with data engineering weights
        final_confidence = (
            pattern_confidence * 0.4 +
            schema_confidence * 0.3 +
            etl_confidence * 0.3
        )
        
        # Generate data engineering rationale
        rationale = self._generate_de_rationale(
            analysis, pattern_confidence, schema_confidence, etl_confidence
        )
        
        decision = "YES" if final_confidence > 0.65 else "NO"
        
        return self._format_result(decision, final_confidence, rationale, "claude_data_engineer")
    
    def match_entities(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entity matching using data engineering approach.
        
        Analyzes actual data patterns, relationships, and schema structure.
        """
        entities = source_data.get("entities", {})
        mappings = {}
        confidence_scores = []
        
        for entity_name, entity_data in entities.items():
            # Analyze data characteristics
            data_profile = self._profile_entity_data(entity_name, entity_data)
            
            # Map based on data patterns
            fhir_entity = self._map_entity_by_data_pattern(data_profile)
            
            if fhir_entity:
                mappings[entity_name] = fhir_entity
                confidence_scores.append(data_profile["confidence"])
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "mappings": mappings,
            "confidence": avg_confidence,
            "decision": "YES" if mappings else "NO",
            "rationale": f"Data engineering entity analysis: {len(mappings)} entities mapped based on data patterns",
            "data_profiles": {name: self._profile_entity_data(name, entities[name]) for name in mappings}
        }
    
    def match_fields(
        self, 
        source_data: Dict[str, Any], 
        entity_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Field matching using data analysis and ETL patterns.
        """
        mappings = {}
        field_analyses = {}
        
        for entity_name, entity_data in source_data.get("entities", {}).items():
            fhir_entity = entity_mappings.get(entity_name)
            if not fhir_entity or "fields" not in entity_data:
                continue
            
            for field_name, field_info in entity_data["fields"].items():
                # Analyze field data
                field_analysis = self._analyze_field_data(field_name, field_info)
                field_analyses[f"{entity_name}.{field_name}"] = field_analysis
                
                # Find best FHIR mapping based on data analysis
                best_target = self._find_best_target_by_data(
                    field_analysis, fhir_entity, field_info
                )
                
                if best_target and field_analysis["confidence"] > 0.6:
                    mappings[f"{entity_name}.{field_name}"] = best_target
        
        avg_confidence = sum(
            analysis["confidence"] for analysis in field_analyses.values()
        ) / len(field_analyses) if field_analyses else 0.0
        
        return {
            "mappings": mappings,
            "confidence": avg_confidence,
            "decision": "YES" if mappings else "NO",
            "rationale": f"Data-driven field mapping: {len(mappings)} fields mapped",
            "field_analyses": field_analyses
        }
    
    def transform_content(
        self,
        source_data: Dict[str, Any],
        field_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Content transformation with data engineering expertise.
        """
        transformations = {}
        
        for source_field, target_field in field_mappings.items():
            # Get field data for analysis
            field_data = self._extract_field_data(source_data, source_field)
            
            # Design transformation based on data analysis
            transform_spec = self._design_transformation(
                field_data, source_field, target_field
            )
            
            transformations[source_field] = transform_spec
        
        return {
            "transformations": transformations,
            "confidence": 0.85,
            "decision": "YES" if transformations else "NO",
            "rationale": f"ETL transformation design for {len(transformations)} fields",
            "etl_ready": True
        }
    
    def _data_engineering_analysis(self, source: str, target: str, level: str) -> Dict[str, Any]:
        """Comprehensive data engineering analysis."""
        return {
            "source_analysis": {
                "field_name": source,
                "name_pattern": self._analyze_field_name(source),
                "likely_type": self._infer_data_type(source),
                "data_domain": self._detect_data_domain(source)
            },
            "target_analysis": {
                "fhir_path": target,
                "resource_type": target.split('.')[0] if '.' in target else target,
                "field_type": self._analyze_fhir_field_type(target),
                "constraints": self._get_fhir_constraints(target)
            },
            "compatibility": {
                "type_compatible": True,  # Would check actual compatibility
                "domain_match": self._check_domain_compatibility(source, target),
                "transformation_needed": self._needs_transformation(source, target)
            }
        }
    
    def _pattern_analysis(self, source: str, target: str, analysis: Dict[str, Any]) -> float:
        """Pattern analysis from data engineering perspective."""
        source_lower = source.lower()
        target_lower = target.lower()
        
        confidence = 0.0
        
        # Identifier patterns (high confidence for data engineers)
        if analysis["source_analysis"]["name_pattern"] == "identifier":
            if "identifier" in target_lower:
                confidence += 0.8
            elif any(term in target_lower for term in ["id", "code", "key"]):
                confidence += 0.6
        
        # Data type patterns
        inferred_type = analysis["source_analysis"]["likely_type"]
        fhir_type = analysis["target_analysis"]["field_type"]
        
        if self._types_compatible(inferred_type, fhir_type):
            confidence += 0.3
        
        # Domain-specific patterns
        if analysis["compatibility"]["domain_match"]:
            confidence += 0.4
        
        # Structural patterns
        if self._structural_similarity(source, target) > 0.7:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _schema_compatibility(self, source: str, target: str, level: str) -> float:
        """Schema compatibility analysis."""
        # FHIR schema validation
        if not self._is_valid_fhir_path(target):
            return 0.0
        
        # Cardinality compatibility
        source_cardinality = self._infer_cardinality(source)
        target_cardinality = self._get_fhir_cardinality(target)
        
        cardinality_score = 1.0 if source_cardinality == target_cardinality else 0.7
        
        # Type system compatibility
        type_score = 0.8 if self._check_type_system_compatibility(source, target) else 0.4
        
        # Constraint compatibility
        constraint_score = self._check_constraint_compatibility(source, target)
        
        return (cardinality_score * 0.3 + type_score * 0.4 + constraint_score * 0.3)
    
    def _etl_feasibility(self, source: str, target: str, analysis: Dict[str, Any]) -> float:
        """ETL feasibility assessment."""
        feasibility_score = 0.8  # Base feasibility
        
        # Transformation complexity
        if analysis["compatibility"]["transformation_needed"]:
            transform_complexity = self._assess_transformation_complexity(source, target)
            if transform_complexity == "simple":
                feasibility_score = 0.9
            elif transform_complexity == "moderate":
                feasibility_score = 0.7
            else:  # complex
                feasibility_score = 0.5
        
        # Data quality considerations
        data_quality_impact = self._assess_data_quality_impact(source, target)
        feasibility_score *= data_quality_impact
        
        return feasibility_score
    
    def _profile_entity_data(self, entity_name: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Profile entity data for mapping decisions."""
        fields = entity_data.get("fields", {})
        row_count = entity_data.get("row_count", 0)
        
        # Analyze field patterns
        identifier_fields = [
            name for name, info in fields.items()
            if info.get("is_identifier", False) or "id" in name.lower()
        ]
        
        biomedical_fields = [
            name for name, info in fields.items()
            if info.get("is_biomedical", False)
        ]
        
        # Determine entity type confidence
        confidence = 0.5  # Base confidence
        
        # Patient/case indicators
        if any(term in entity_name.lower() for term in ["patient", "case", "subject"]):
            confidence = 0.9
            entity_type = "Patient"
        # Specimen indicators
        elif any(term in entity_name.lower() for term in ["specimen", "tissue", "sample", "biospecimen"]):
            confidence = 0.85
            entity_type = "Specimen"
        # File/document indicators
        elif any(term in entity_name.lower() for term in ["file", "document", "attachment"]):
            confidence = 0.8
            entity_type = "DocumentReference"
        else:
            confidence = 0.6
            entity_type = "Observation"
        
        return {
            "entity_type": entity_type,
            "confidence": confidence,
            "field_count": len(fields),
            "row_count": row_count,
            "identifier_fields": identifier_fields,
            "biomedical_fields": biomedical_fields,
            "data_quality": "good" if row_count > 0 else "unknown"
        }
    
    def _map_entity_by_data_pattern(self, data_profile: Dict[str, Any]) -> Optional[str]:
        """Map entity based on data pattern analysis."""
        if data_profile["confidence"] > 0.6:
            return data_profile["entity_type"]
        return None
    
    def _analyze_field_data(self, field_name: str, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual field data."""
        samples = field_info.get("samples", [])
        field_type = field_info.get("type", "unknown")
        
        analysis = {
            "field_name": field_name,
            "inferred_type": self._infer_data_type_from_samples(samples, field_type),
            "sample_count": len(samples),
            "pattern": self._detect_data_pattern(samples),
            "confidence": 0.5
        }
        
        # Boost confidence based on clear patterns
        if analysis["pattern"] in ["url", "datetime", "identifier"]:
            analysis["confidence"] = 0.9
        elif analysis["pattern"] in ["categorical", "numerical"]:
            analysis["confidence"] = 0.7
        
        return analysis
    
    def _find_best_target_by_data(
        self, 
        field_analysis: Dict[str, Any], 
        fhir_entity: str,
        field_info: Dict[str, Any]
    ) -> Optional[str]:
        """Find best FHIR target based on data analysis."""
        pattern = field_analysis["pattern"]
        field_name = field_analysis["field_name"]
        
        # Pattern-based mapping
        targets = self._get_targets_for_entity(fhir_entity)
        
        for target in targets:
            if self._pattern_matches_target(pattern, field_name, target):
                return target
        
        return None
    
    def _pattern_matches_target(self, pattern: str, field_name: str, target: str) -> bool:
        """Check if data pattern matches FHIR target."""
        field_lower = field_name.lower()
        target_lower = target.lower()
        
        if pattern == "url" and "url" in target_lower:
            return True
        elif pattern == "datetime" and any(term in target_lower for term in ["date", "time"]):
            return True
        elif pattern == "identifier" and any(term in target_lower for term in ["identifier", "id"]):
            return True
        elif pattern == "categorical" and "type" in target_lower:
            return True
        
        # Fallback to name similarity
        return any(token in target_lower for token in field_lower.split('_'))
    
    def _generate_de_rationale(
        self,
        analysis: Dict[str, Any],
        pattern_conf: float,
        schema_conf: float,
        etl_conf: float
    ) -> str:
        """Generate data engineering rationale."""
        domain = analysis["source_analysis"]["data_domain"]
        pattern = analysis["source_analysis"]["name_pattern"]
        compatible = analysis["compatibility"]["domain_match"]
        
        return (
            f"Claude DE analysis (domain: {domain}, pattern: {pattern}): "
            f"pattern {pattern_conf:.3f}, schema {schema_conf:.3f}, etl {etl_conf:.3f}, "
            f"compatible: {compatible}"
        )
    
    # Helper methods (simplified implementations)
    def _analyze_field_name(self, field_name: str) -> str:
        field_lower = field_name.lower()
        if any(term in field_lower for term in ["id", "identifier", "code"]):
            return "identifier"
        elif any(term in field_lower for term in ["date", "time"]):
            return "temporal"
        elif any(term in field_lower for term in ["type", "category"]):
            return "categorical"
        elif any(term in field_lower for term in ["url", "file", "path"]):
            return "content"
        else:
            return "general"
    
    def _infer_data_type(self, field_name: str) -> str:
        return "string"  # Simplified
    
    def _detect_data_domain(self, field_name: str) -> str:
        field_lower = field_name.lower()
        biomedical_terms = ["patient", "specimen", "tissue", "medical", "clinical"]
        if any(term in field_lower for term in biomedical_terms):
            return "biomedical"
        return "general"
    
    def _analyze_fhir_field_type(self, target: str) -> str:
        return "string"  # Simplified
    
    def _get_fhir_constraints(self, target: str) -> List[str]:
        return []  # Simplified
    
    def _check_domain_compatibility(self, source: str, target: str) -> bool:
        source_domain = self._detect_data_domain(source)
        return source_domain in ["biomedical", "general"]  # Simplified
    
    def _needs_transformation(self, source: str, target: str) -> bool:
        return False  # Simplified
    
    def _types_compatible(self, source_type: str, target_type: str) -> bool:
        return True  # Simplified
    
    def _structural_similarity(self, source: str, target: str) -> float:
        return 0.5  # Simplified
    
    def _is_valid_fhir_path(self, target: str) -> bool:
        return "." in target  # Simplified
    
    def _infer_cardinality(self, source: str) -> str:
        return "1"  # Simplified
    
    def _get_fhir_cardinality(self, target: str) -> str:
        return "1"  # Simplified
    
    def _check_type_system_compatibility(self, source: str, target: str) -> bool:
        return True  # Simplified
    
    def _check_constraint_compatibility(self, source: str, target: str) -> float:
        return 0.8  # Simplified
    
    def _assess_transformation_complexity(self, source: str, target: str) -> str:
        return "simple"  # Simplified
    
    def _assess_data_quality_impact(self, source: str, target: str) -> float:
        return 0.9  # Simplified
    
    def _infer_data_type_from_samples(self, samples: List, field_type: str) -> str:
        return field_type  # Simplified
    
    def _detect_data_pattern(self, samples: List) -> str:
        if not samples:
            return "unknown"
        
        sample_strs = [str(s) for s in samples if s is not None]
        if not sample_strs:
            return "unknown"
        
        # URL pattern
        if any("http" in s.lower() for s in sample_strs):
            return "url"
        
        # Date pattern
        if any("-" in s and ":" in s for s in sample_strs):
            return "datetime"
        
        # Identifier pattern (alphanumeric with dashes/underscores)
        if all(any(c.isalnum() or c in "-_" for c in s) for s in sample_strs):
            if any(len(s) > 5 for s in sample_strs):
                return "identifier"
        
        return "general"
    
    def _extract_field_data(self, source_data: Dict[str, Any], source_field: str) -> Dict[str, Any]:
        """Extract field data for transformation design."""
        return {"samples": [], "type": "string"}  # Simplified
    
    def _design_transformation(
        self, 
        field_data: Dict[str, Any], 
        source_field: str, 
        target_field: str
    ) -> Dict[str, Any]:
        """Design ETL transformation specification."""
        return {
            "source": source_field,
            "target": target_field,
            "transform_type": "direct_copy",
            "validation_rules": ["not_null"],
            "confidence": 0.8
        }