"""loader for HTAN/GDC curated mapping JSON files.

parses the mapping JSON files and extracts:
- source->destination field mappings
- reference relationships between resources
- content/enum mappings (coded values)

these mappings are more specific than the fhir aggregator catch-all
(observation.component) and should take precedence.

consolidates functionality from legacy/gdc_loader.py and legacy/htan_loader.py.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional, List


@dataclass
class MappingEntry:
    """parsed field mapping from JSON files."""
    source_term: str          # ex. "Filename", "HTANParticipantID"
    source_schema: str        # "htan" or "gdc"
    source_context: str       # parent category ex. "DocumentReference", "Patient"
    destination: str          # full FHIR path ex. "DocumentReference.content.attachment.title"
    destination_resource: str # ex. "DocumentReference"
    dest_system: str          # coding system if any
    tier: str                 # "entity" or "field"
    relation: str = "maps_to" # "maps_to" or "references"
    description: str = ""     # source field description
    parent: str = ""          # parent class (for HTAN hierarchy)


@dataclass
class ContentEntry:
    """parsed content/enum mapping (coded values)."""
    source_value: str         # ex. "female", "Adenocarcinoma"
    source_category: str      # ex. "gender", "histology"
    source_schema: str        # "htan" or "gdc"
    dest_code: str            # FHIR code
    dest_system: str          # coding system URL
    dest_display: str = ""    # display name


def load_htan_mappings(path: str = None) -> tuple[list[MappingEntry], list[ContentEntry]]:
    """load mappings from HTAN mappings.json.

    extracts:
    - field mappings (fhir:fieldMapping)
    - reference relationships (fhir:reference)
    - content mappings (fhir:code/system)

    args:
        path: path to mappings.json (default: built-in)

    returns:
        tuple of (MappingEntry list, ContentEntry list)
    """
    if path is None:
        path = Path(__file__).parent.parent / "data/resources/htan_mapping/mappings.json"

    with open(path) as f:
        nodes = json.load(f)

    entries = []
    content_entries = []

    for node in nodes:
        # extract source term (remove bts: prefix)
        source = node.get("node", "").replace("bts:", "")
        if not source:
            continue

        # get resource type and category
        resource_type = node.get("fhir:resourceType", "")
        categories = node.get("category", [])
        context = categories[0] if categories else ""
        parent = node.get("rdfs:subClassOf", "").replace("bts:", "")
        description = node.get("ui_name", "")

        # get field mappings
        field_mappings = node.get("fhir:fieldMapping", [])

        for fm in field_mappings:
            field = fm.get("fhir:field", "")
            if not field:
                continue

            system = fm.get("fhir:system", "")
            fhir_code = fm.get("fhir:code", "")

            # build full destination path
            if resource_type and not field.startswith(resource_type):
                destination = f"{resource_type}.{field}"
            else:
                destination = field
                if "." in field:
                    resource_type = field.split(".")[0]

            # determine tier
            tier = "entity" if "." not in field else "field"

            entries.append(MappingEntry(
                source_term=source,
                source_schema="htan",
                source_context=context,
                destination=destination,
                destination_resource=resource_type,
                dest_system=system,
                tier=tier,
                relation="maps_to",
                description=description,
                parent=parent,
            ))

            # if there's a fhir:code, create content mapping
            if fhir_code:
                content_entries.append(ContentEntry(
                    source_value=source,
                    source_category=context,
                    source_schema="htan",
                    dest_code=fhir_code,
                    dest_system=system,
                    dest_display=fhir_code,
                ))

        # process fhir:reference entries (from legacy/htan_loader.py)
        references = node.get("fhir:reference", [])
        for ref in references:
            if not isinstance(ref, dict):
                continue

            ref_resource = ref.get("fhir:resourceType", "")
            ref_field = ref.get("fhir:field", "")

            if not ref_resource:
                continue

            destination = f"{ref_resource}.{ref_field}" if ref_field else ref_resource

            entries.append(MappingEntry(
                source_term=source,
                source_schema="htan",
                source_context=context,
                destination=destination,
                destination_resource=ref_resource,
                dest_system="",
                tier="field",
                relation="references",
                description=description,
                parent=parent,
            ))

    return entries, content_entries


def load_gdc_mappings(base_path: str = None) -> tuple[list[MappingEntry], list[ContentEntry]]:
    """load mappings from GDC JSON files (case.json, file.json, project.json).

    extracts:
    - entity mappings (obj_mapping)
    - field mappings (mappings list)
    - content/enum mappings (enums in source)
    - reference relationships

    args:
        base_path: path to gdc_mapping directory (default: built-in)

    returns:
        tuple of (MappingEntry list, ContentEntry list)
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "data/resources/gdc_mapping"
    else:
        base_path = Path(base_path)

    entries = []
    content_entries = []

    for json_file in base_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        entity_type = json_file.stem  # "case", "file", "project"

        # get entity-level mapping from obj_mapping
        obj_mapping = data.get("obj_mapping", {})
        source_info = obj_mapping.get("source", {})
        dest_info = obj_mapping.get("destination", {})
        source_entity = source_info.get("name", "")
        dest_entity = dest_info.get("name", "")
        entity_description = source_info.get("description", "")

        if source_entity and dest_entity:
            entries.append(MappingEntry(
                source_term=source_entity,
                source_schema="gdc",
                source_context="",
                destination=dest_entity,
                destination_resource=dest_entity,
                dest_system="",
                tier="entity",
                relation="maps_to",
                description=entity_description,
            ))

        # get field mappings from mappings list
        mappings = data.get("mappings", [])
        for m in mappings:
            if not isinstance(m, dict):
                continue

            src_info = m.get("source", {})
            dst_info = m.get("destination", {})

            src_name = src_info.get("name", "")
            dst_name = dst_info.get("name", "")
            description = src_info.get("description", "")

            if not src_name or not dst_name:
                continue

            # extract resource from destination path
            if "." in dst_name:
                dst_resource = dst_name.split(".")[0]
            else:
                dst_resource = dst_name

            # get context from source (e.g., "demographic.gender" -> "demographic")
            context = ""
            original_src_name = src_name
            if "." in src_name:
                context = src_name.split(".")[0]
                src_name = src_name.split(".")[-1]

            # check if destination is a reference
            relation = "maps_to"
            if dst_info.get("reference"):
                relation = "references"

            entries.append(MappingEntry(
                source_term=src_name,
                source_schema="gdc",
                source_context=context or source_entity,
                destination=dst_name,
                destination_resource=dst_resource,
                dest_system="",
                tier="field",
                relation=relation,
                description=description,
            ))

            # process enums into content mappings (from legacy/gdc_loader.py)
            enums = src_info.get("enums", [])
            for enum_entry in enums:
                values = enum_entry.get("enum", []) or enum_entry.get("enums", [])
                for value in values:
                    if isinstance(value, str):
                        content_entries.append(ContentEntry(
                            source_value=value,
                            source_category=src_name,
                            source_schema="gdc",
                            dest_code=value,  # same value by default
                            dest_system="",
                            dest_display=value,
                        ))

    return entries, content_entries


def merge_mappings_into_db(entries: list[MappingEntry], db, priority: int = 10):
    """merge mapping entries into flat mapping database.

    higher priority mappings will be preferred over lower priority ones
    when there are conflicts.

    args:
        entries: list of MappingEntry objects
        db: FlatMappingDatabase instance
        priority: priority level for these mappings (higher = more preferred)

    returns:
        dict with counts of added sources and destinations
    """
    from schema_crush.mappings.flat import Source, Destination, Tier

    added = {"sources": 0, "destinations": 0, "skipped": 0}

    for entry in entries:
        # determine tier enum
        tier = Tier.ENTITY if entry.tier == "entity" else Tier.FIELD

        # create source id with priority suffix
        source_id = f"{entry.source_schema}:{entry.source_term}"

        # check if source exists
        if source_id not in db.sources:
            source = Source(
                id=source_id,
                source=entry.source_term,
                source_schema=entry.source_schema,
                tier=tier,
                source_context=entry.source_context,
            )
            db.add_source(source)
            added["sources"] += 1

        # check if this destination already exists for this source
        existing_dests = db._dest_by_source.get(source_id, [])
        dest_exists = any(d.destination.lower() == entry.destination.lower() for d in existing_dests)

        if not dest_exists:
            dest = Destination(
                source_id=source_id,
                destination=entry.destination,
                dest_system=entry.dest_system,
            )
            db.add_destination(dest)
            added["destinations"] += 1
        else:
            added["skipped"] += 1

    return added


def merge_content_into_db(content_entries: list[ContentEntry], db) -> dict:
    """merge content/enum entries into flat mapping database.

    args:
        content_entries: list of ContentEntry objects
        db: FlatMappingDatabase instance

    returns:
        dict with counts
    """
    from schema_crush.mappings.flat import ContentValue

    added = {"content_values": 0, "skipped": 0}

    for entry in content_entries:
        # check if this content value already exists
        existing = db.lookup_content(entry.source_value, entry.source_category)
        if existing:
            added["skipped"] += 1
            continue

        cv = ContentValue(
            id=len(db.content_values),
            source_value=entry.source_value,
            source_category=entry.source_category,
            code=entry.dest_code,
            system=entry.dest_system,
            display=entry.dest_display,
        )
        db.add_content_value(cv)
        added["content_values"] += 1

    return added


def load_and_merge_all(db):
    """load all mapping JSON files and merge into database.

    args:
        db: FlatMappingDatabase instance

    returns:
        dict with total counts
    """
    print("loading HTAN mappings...")
    htan_entries, htan_content = load_htan_mappings()
    print(f"  parsed {len(htan_entries)} field mappings, {len(htan_content)} content mappings")

    print("loading GDC mappings...")
    gdc_entries, gdc_content = load_gdc_mappings()
    print(f"  parsed {len(gdc_entries)} field mappings, {len(gdc_content)} content mappings")

    all_entries = htan_entries + gdc_entries
    all_content = htan_content + gdc_content

    print(f"\nmerging {len(all_entries)} field mappings...")
    result = merge_mappings_into_db(all_entries, db)

    print(f"  added {result['sources']} sources")
    print(f"  added {result['destinations']} destinations")
    print(f"  skipped {result['skipped']} duplicates")

    if all_content:
        print(f"\nmerging {len(all_content)} content mappings...")
        content_result = merge_content_into_db(all_content, db)
        print(f"  added {content_result['content_values']} content values")
        result["content_values"] = content_result["content_values"]

    return result


if __name__ == "__main__":
    # demo: load and show mappings
    print("=== HTAN MAPPINGS ===")
    htan_entries, htan_content = load_htan_mappings()
    print(f"Field mappings: {len(htan_entries)}")
    print(f"Content mappings: {len(htan_content)}")
    for e in htan_entries[:10]:
        print(f"  {e.source_term} -> {e.destination} ({e.relation})")

    print("\n=== GDC MAPPINGS ===")
    gdc_entries, gdc_content = load_gdc_mappings()
    print(f"Field mappings: {len(gdc_entries)}")
    print(f"Content mappings: {len(gdc_content)}")
    for e in gdc_entries[:10]:
        print(f"  {e.source_term} ({e.source_context}) -> {e.destination}")