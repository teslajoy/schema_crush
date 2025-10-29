"""base dataclasses for composite fhir mapping rules."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Reference:
    """represents a fhir reference relationship."""

    resource: str           # ex. "specimen"
    field: str              # ex. "focus"

    def to_dict(self) -> Dict[str, str]:
        """convert to dictionary."""
        return {
            "resource": self.resource,
            "field": self.field
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Reference":
        """create from dictionary."""
        return cls(
            resource=data.get("resource", ""),
            field=data.get("field", "")
        )


@dataclass
class FieldMapping:
    """represents a fhir field mapping with type and terminology."""

    field: str                      # ex. "component"
    type: str                       # ex. "valuestring"
    system: Optional[str] = None    # ex. "https://humantumoratlas.org/..."
    code: Optional[str] = None      # ex. "urinebiospcimentype"
    category: Optional[str] = None  # ex. "laboratory"

    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary."""
        result = {
            "field": self.field,
            "type": self.type
        }
        if self.system:
            result["system"] = self.system
        if self.code:
            result["code"] = self.code
        if self.category:
            result["category"] = self.category
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldMapping":
        """create from dictionary."""
        return cls(
            field=data.get("field", ""),
            type=data.get("type", ""),
            system=data.get("system"),
            code=data.get("code"),
            category=data.get("category")
        )


@dataclass
class MappingRule:
    """
    composite mapping rule capturing multi-layered fhir logic.

    this represents a complete mapping from source concept to fhir resource
    with relationship context and field-level transformations.

    example:
        bts:urinebiospcimentype ->
            observation (resource) ->
                focus -> specimen (reference) ->
                    component[valuestring] (field mapping)
    """

    # tier 1: entity/concept level
    source_node: str                        # "bts:urinebiospcimentype"
    target_resource: str                    # "observation"
    subclass_of: Optional[str] = None       # "bts:biospecimen"

    # tier 2: relationship context
    references: List[Reference] = field(default_factory=list)
    category: List[str] = field(default_factory=list)

    # tier 3: field mappings
    field_mappings: List[FieldMapping] = field(default_factory=list)

    # metadata
    confidence: float = 1.0                 # from human-curated source
    source: str = "unknown"                 # provenance (htan/gdc)
    rule_id: str = ""                       # unique identifier

    # additional metadata
    ui_name: Optional[str] = None           # human-readable name
    missing_in_schematic: bool = False      # htan-specific flag
    is_range_value: bool = False            # htan-specific flag

    def __post_init__(self):
        """generate rule_id if not provided."""
        if not self.rule_id:
            # create id from source_node
            clean_node = self.source_node.replace(":", "_").replace("/", "_")
            self.rule_id = f"{self.source}_{clean_node}"

    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary."""
        return {
            "source_node": self.source_node,
            "target_resource": self.target_resource,
            "subclass_of": self.subclass_of,
            "references": [ref.to_dict() for ref in self.references],
            "category": self.category,
            "field_mappings": [fm.to_dict() for fm in self.field_mappings],
            "confidence": self.confidence,
            "source": self.source,
            "rule_id": self.rule_id,
            "ui_name": self.ui_name,
            "missing_in_schematic": self.missing_in_schematic,
            "is_range_value": self.is_range_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MappingRule":
        """create from dictionary."""
        return cls(
            source_node=data.get("source_node", ""),
            target_resource=data.get("target_resource", ""),
            subclass_of=data.get("subclass_of"),
            references=[Reference.from_dict(r) for r in data.get("references", [])],
            category=data.get("category", []),
            field_mappings=[FieldMapping.from_dict(fm) for fm in data.get("field_mappings", [])],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            rule_id=data.get("rule_id", ""),
            ui_name=data.get("ui_name"),
            missing_in_schematic=data.get("missing_in_schematic", False),
            is_range_value=data.get("is_range_value", False)
        )

    def matches_source(self, query: str) -> bool:
        """check if query matches source_node (case-insensitive)."""
        return query.lower() in self.source_node.lower()

    def matches_target(self, query: str) -> bool:
        """check if query matches target_resource (case-insensitive)."""
        return query.lower() in self.target_resource.lower()

    def has_reference_to(self, resource: str) -> bool:
        """check if rule has reference to given resource."""
        return any(ref.resource.lower() == resource.lower() for ref in self.references)

    def get_field_mapping(self, field_name: str) -> Optional[FieldMapping]:
        """get field mapping by field name."""
        for fm in self.field_mappings:
            if fm.field.lower() == field_name.lower():
                return fm
        return None