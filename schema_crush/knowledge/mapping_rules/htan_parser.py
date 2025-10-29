"""parser for htan mappings.json -> mappingrule objects."""

import json
from pathlib import Path
from typing import List, Dict, Any
from .base import MappingRule, Reference, FieldMapping


def parse_htan_mappings(json_path: Path) -> List[MappingRule]:
    """
    parse htan mappings.json into mappingrule objects.

    args:
        json_path: path to htan mappings.json file

    returns:
        list of mappingrule objects

    example input:
        {
          "node": "bts:urinebiospcimentype",
          "fhir:resourcetype": "observation",
          "fhir:reference": [{"fhir:resourcetype": "specimen", "fhir:field": "focus"}],
          "fhir:fieldmapping": [{
            "fhir:field": "component",
            "fhir:type": "valuestring",
            "fhir:code": "urinebiospcimentype",
            "fhir:category": "laboratory"
          }],
          "rdfs:subclassof": "bts:biospecimen",
          "category": ["specimen"]
        }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rules = []
    for idx, entry in enumerate(data):
        try:
            rule = _parse_htan_entry(entry, idx)
            if rule:
                rules.append(rule)
        except Exception as e:
            print(f"warning: failed to parse entry {idx}: {e}")
            continue

    return rules


def _parse_htan_entry(entry: Dict[str, Any], idx: int) -> MappingRule:
    """parse single htan mapping entry into mappingrule."""

    # extract basic fields
    source_node = entry.get("node", f"unknown_{idx}")
    target_resource = entry.get("fhir:resourceType", "")
    ui_name = entry.get("ui_name", "")
    subclass_of = entry.get("rdfs:subClassOf", "")

    # parse references
    references = []
    for ref_entry in entry.get("fhir:reference", []):
        if isinstance(ref_entry, dict):
            resource = ref_entry.get("fhir:resourceType", "")
            field = ref_entry.get("fhir:field", "")
            if resource or field:
                references.append(Reference(resource=resource, field=field))

    # parse field mappings
    field_mappings = []
    for fm_entry in entry.get("fhir:fieldMapping", []):
        if isinstance(fm_entry, dict):
            fm = FieldMapping(
                field=fm_entry.get("fhir:field", ""),
                type=fm_entry.get("fhir:type", ""),
                system=fm_entry.get("fhir:system"),
                code=fm_entry.get("fhir:code"),
                category=fm_entry.get("fhir:category")
            )
            field_mappings.append(fm)

    # extract category list
    category = entry.get("category", [])
    if not isinstance(category, list):
        category = [category] if category else []

    # create mapping rule
    rule = MappingRule(
        source_node=source_node,
        target_resource=target_resource,
        subclass_of=subclass_of if subclass_of else None,
        references=references,
        category=category,
        field_mappings=field_mappings,
        confidence=1.0,  # human-curated htan mappings
        source="htan",
        ui_name=ui_name if ui_name else None,
        missing_in_schematic=entry.get("missing_in_schematic", False),
        is_range_value=entry.get("is_range_value", False)
    )

    return rule


def load_htan_rules(mapping_dir: Path = None) -> List[MappingRule]:
    """
    convenience function to load htan rules from default location.

    args:
        mapping_dir: optional path to mapping directory
                    defaults to schema_crush/data/resources/htan_mapping/

    returns:
        list of mappingrule objects
    """
    if mapping_dir is None:
        # default to package data directory
        import importlib.resources
        base_path = Path(importlib.resources.files('schema_crush'))
        mapping_dir = base_path / 'data' / 'resources' / 'htan_mapping'

    json_path = mapping_dir / 'mappings.json'

    if not json_path.exists():
        raise FileNotFoundError(f"htan mappings not found at: {json_path}")

    print(f"loading htan mappings from: {json_path}")
    rules = parse_htan_mappings(json_path)
    print(f"loaded {len(rules)} htan mapping rules")

    return rules