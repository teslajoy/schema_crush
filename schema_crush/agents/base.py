"""base agent interface for schema crush v2."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """base class for all schema mapping agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.iteration_count = 0
        self.max_iterations = 5
    
    @abstractmethod
    def match(self, source: str, target: str, level: str) -> Dict[str, Any]:
        """
        standard match interface for field-level matching.
        
        args:
            source: source field/entity name
            target: target fhir path
            level: matching level ('entity', 'field', 'content')
            
        returns:
            dictionary with match result
        """
        pass
    
    def match_entities(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        match entities (csv tables) to fhir resources.
        
        args:
            source_data: loaded csv data with entity information
            
        returns:
            dictionary with entity mappings
        """
        # default implementation - can be overridden
        entities = source_data.get("entities", {})
        mappings = {}
        
        for entity_name in entities.keys():
            # simple heuristic mapping
            if "case" in entity_name.lower() or "patient" in entity_name.lower():
                mappings[entity_name] = "Patient"
            elif "specimen" in entity_name.lower() or "biospecimen" in entity_name.lower():
                mappings[entity_name] = "Specimen"
            elif "file" in entity_name.lower():
                mappings[entity_name] = "DocumentReference"
            else:
                mappings[entity_name] = "Observation"
        
        return {
            "mappings": mappings,
            "confidence": 0.6,
            "decision": "YES",
            "rationale": f"{self.name} basic entity mapping",
            "method": f"{self.name}_entity_match"
        }
    
    def match_fields(
        self, 
        source_data: Dict[str, Any], 
        entity_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        match fields within entities to fhir paths.
        
        args:
            source_data: loaded csv data
            entity_mappings: results from entity matching
            
        returns:
            dictionary with field mappings  
        """
        # default implementation - can be overridden
        mappings = {}
        confidences = []
        
        for entity_name, entity_data in source_data.get("entities", {}).items():
            fhir_entity = entity_mappings.get(entity_name)
            if not fhir_entity or "fields" not in entity_data:
                continue
                
            for field_name in entity_data["fields"]:
                targets = self._get_targets_for_entity(fhir_entity)
                
                best_target = None
                best_confidence = 0.0
                
                for target in targets:
                    result = self.match(f"{entity_name}.{field_name}", target, "field")
                    if result.get("confidence", 0.0) > best_confidence:
                        best_confidence = result["confidence"] 
                        best_target = target
                
                if best_target and best_confidence > 0.5:
                    mappings[f"{entity_name}.{field_name}"] = best_target
                    confidences.append(best_confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "mappings": mappings,
            "confidence": avg_confidence,
            "decision": "YES" if mappings else "NO",
            "rationale": f"{self.name} field matching: {len(mappings)} mappings",
            "method": f"{self.name}_field_match"
        }
    
    def transform_content(
        self,
        source_data: Dict[str, Any],
        field_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        transform content based on field mappings.
        
        args:
            source_data: loaded csv data
            field_mappings: results from field matching
            
        returns:
            dictionary with transformation specifications
        """
        transformations = {}
        
        for source_field, target_field in field_mappings.items():
            # basic transformation - can be overridden
            transformations[source_field] = {
                "target": target_field,
                "transform_type": "direct_copy",
                "validation_rules": [],
                "confidence": 0.8
            }
        
        return {
            "transformations": transformations,
            "confidence": 0.8,
            "decision": "YES" if transformations else "NO", 
            "rationale": f"{self.name} content transform: {len(transformations)} transformations",
            "method": f"{self.name}_content_transform"
        }
    
    def _get_targets_for_entity(self, fhir_entity: str) -> list:
        """get candidate fhir targets for an entity type."""
        targets_by_entity = {
            "Patient": [
                "Patient.identifier", "Patient.birthDate", "Patient.deceasedBoolean",
                "Patient.deceasedDateTime"
            ],
            "Specimen": [
                "Specimen.type", "Specimen.collection.method", 
                "Specimen.collection.collectedDateTime", "Specimen.subject"
            ],
            "DocumentReference": [
                "DocumentReference.content.attachment.url", "DocumentReference.type"
            ],
            "Observation": [
                "Observation.code", "Observation.valueString", 
                "Observation.valueQuantity", "Observation.subject"
            ],
            "Condition": [
                "Condition.code", "Condition.stage.summary", "Condition.onsetDateTime"
            ]
        }
        
        return targets_by_entity.get(fhir_entity, ["Observation.valueString"])
    
    def _format_result(
        self, 
        decision: str, 
        confidence: float, 
        rationale: str,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """format standard result dictionary."""
        return {
            "decision": decision,
            "confidence": float(confidence),
            "rationale": rationale,
            "method": method or self.name,
            "agent": self.name
        }