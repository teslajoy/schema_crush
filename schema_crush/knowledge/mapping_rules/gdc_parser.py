"""parser for gdc case.json/file.json/project.json -> mappingrule objects."""

import json
from pathlib import Path
from typing import List, Dict, Any
from .base import MappingRule, Reference, FieldMapping


def parse_gdc_mappings(json_path: Path) -> List[MappingRule]:
    """
    parse gdc mapping file into mappingrule objects.

    args:
        json_path: path to gdc case.json, file.json, or project.json

    returns:
        list of mappingrule objects

    example gdc structure:
        {
          "mappings": [
            {
              "source": {"name": "case_id"},
              "destination": {"name": "Patient.id"}
            },
            {
              "source": {"name": "demographic.gender"},
              "destination": {"name": "Patient.gender"}
            }
          ]
        }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # determine gdc entity type from filename
    entity_type = json_path.stem  # "case", "file", or "project"

    # get target resource from obj_mapping
    obj_mapping = data.get("obj_mapping", {})
    destination = obj_mapping.get("destination", {})
    target_resource = destination.get("name", "")

    mappings_list = data.get("mappings", [])
    rules = []

    for idx, mapping in enumerate(mappings_list):
        try:
            rule = _parse_gdc_mapping(
                mapping,
                entity_type,
                target_resource,
                idx
            )
            if rule:
                rules.append(rule)
        except Exception as e:
            print(f"warning: failed to parse gdc mapping {idx}: {e}")
            continue

    return rules


def _parse_gdc_mapping(
    mapping: Dict[str, Any],
    entity_type: str,
    default_target_resource: str,
    idx: int
) -> MappingRule:
    """parse single gdc mapping entry into mappingrule."""

    source = mapping.get("source", {})
    destination = mapping.get("destination", {})

    # extract source information
    source_name = source.get("name", f"unknown_{idx}")
    source_description = source.get("description", "")

    # extract destination information
    dest_name = destination.get("name", "")

    # parse destination to extract resource and field
    # ex. "Patient.identifier" -> resource="Patient", field="identifier"
    if "." in dest_name:
        parts = dest_name.split(".", 1)
        target_resource = parts[0]
        target_field = parts[1]
    else:
        target_resource = default_target_resource
        target_field = dest_name

    # create field mapping
    field_mappings = []
    if target_field:
        fm = FieldMapping(
            field=target_field,
            type=destination.get("type", "string")
        )
        field_mappings.append(fm)

    # create mapping rule
    rule = MappingRule(
        source_node=f"gdc:{entity_type}.{source_name}",
        target_resource=target_resource,
        subclass_of=None,
        references=[],  # gdc mappings don't include references
        category=[entity_type],
        field_mappings=field_mappings,
        confidence=1.0,  # human-curated gdc mappings
        source="gdc",
        ui_name=source_description if source_description else None
    )

    return rule


def load_gdc_rules(mapping_dir: Path = None) -> List[MappingRule]:
    """
    convenience function to load all gdc rules from default location.

    args:
        mapping_dir: optional path to gdc_mapping directory
                    defaults to schema_crush/data/resources/gdc_mapping/

    returns:
        list of mappingrule objects from case.json, file.json, project.json
    """
    if mapping_dir is None:
        # default to package data directory
        import importlib.resources
        base_path = Path(importlib.resources.files('schema_crush'))
        mapping_dir = base_path / 'data' / 'resources' / 'gdc_mapping'

    if not mapping_dir.exists():
        raise FileNotFoundError(f"gdc mapping directory not found at: {mapping_dir}")

    all_rules = []

    # load each mapping file
    for json_file in ["case.json", "file.json", "project.json"]:
        json_path = mapping_dir / json_file
        if json_path.exists():
            print(f"loading gdc mappings from: {json_path}")
            rules = parse_gdc_mappings(json_path)
            print(f"  loaded {len(rules)} rules from {json_file}")
            all_rules.extend(rules)
        else:
            print(f"warning: {json_path} not found, skipping")

    print(f"total gdc rules loaded: {len(all_rules)}")
    return all_rules