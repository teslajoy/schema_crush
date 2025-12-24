"""load GDC and HTAN mappings into FlatMappingDatabase.

builds once from JSON sources, caches to SQLite db for fast subsequent loads.
content mappings loaded from raw_content_maps/terminology YAML files.
"""

import json
from pathlib import Path
from typing import Optional

from .flat import Source, Destination, Tier, FlatMappingDatabase
from .content_loader import load_content_mappings


# default db path
_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "flat_mappings.db"

# in-memory cache
_cached_db: Optional[FlatMappingDatabase] = None


def load_flat_mappings(
    force_rebuild: bool = False,
    use_cache: bool = True,
) -> FlatMappingDatabase:
    """load FlatMappingDatabase from cached SQLite db or build from JSON sources.

    args:
        force_rebuild: if True, rebuild from JSON even if db exists
        use_cache: if True, use in-memory cache for repeated calls

    returns:
        FlatMappingDatabase with all mappings (entity, field, and content tiers)
    """
    global _cached_db

    # return in-memory cache if available
    if use_cache and _cached_db is not None and not force_rebuild:
        return _cached_db

    db = FlatMappingDatabase()

    # try loading from pre-built SQLite db
    if not force_rebuild and _DB_PATH.exists():
        db.load_sqlite(_DB_PATH)
        if use_cache:
            _cached_db = db
        return db

    # build from JSON sources (entity + field tiers)
    db = _build_from_json()

    # load content mappings from terminology YAML files
    load_content_mappings(db)

    # save to SQLite db for next time
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.save_sqlite(_DB_PATH)

    if use_cache:
        _cached_db = db

    return db


def rebuild_db() -> FlatMappingDatabase:
    """force rebuild db from JSON sources.

    use this after updating gdc_mapping/ or htan_mapping/ JSON files.

    returns:
        newly built FlatMappingDatabase
    """
    return load_flat_mappings(force_rebuild=True)


def get_db_path() -> Path:
    """return path to SQLite database file."""
    return _DB_PATH


def _build_from_json(
    gdc_dir: Optional[Path] = None,
    htan_dir: Optional[Path] = None,
) -> FlatMappingDatabase:
    """build FlatMappingDatabase from GDC and HTAN JSON files."""
    db = FlatMappingDatabase()

    # load GDC
    if gdc_dir is None:
        import importlib.resources
        base_path = Path(importlib.resources.files("schema_crush"))
        gdc_dir = base_path / "data" / "resources" / "gdc_mapping"

    if gdc_dir.exists():
        _load_gdc(db, gdc_dir)

    # load HTAN
    if htan_dir is None:
        import importlib.resources
        base_path = Path(importlib.resources.files("schema_crush"))
        htan_dir = base_path / "data" / "resources" / "htan_mapping"

    if htan_dir.exists():
        _load_htan(db, htan_dir)

    return db


def _load_gdc(db: FlatMappingDatabase, gdc_dir: Path) -> None:
    """load GDC case/file/project into flat db."""
    for json_file in ["case.json", "file.json", "project.json"]:
        json_path = gdc_dir / json_file
        if not json_path.exists():
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        entity_type = json_path.stem  # "case", "file", "project"

        # entity-level mapping from obj_mapping
        obj_mapping = data.get("obj_mapping", {})
        dest_info = obj_mapping.get("destination", {})
        dest_resource = dest_info.get("name", "")

        if dest_resource:
            # add entity source
            entity_id = f"gdc:{entity_type}"
            db.add_source(Source(
                id=entity_id,
                source=entity_type,
                source_schema="gdc",
                tier=Tier.ENTITY,
                source_context=""
            ))
            db.add_destination(Destination(
                source_id=entity_id,
                destination=dest_resource
            ))

        # field mappings
        for mapping in data.get("mappings", []):
            _process_gdc_field(db, entity_type, mapping)


def _process_gdc_field(db: FlatMappingDatabase, entity_type: str, mapping: dict) -> None:
    """process single GDC field mapping."""
    source_info = mapping.get("source", {})
    dest_info = mapping.get("destination", {})

    source_name = source_info.get("name", "")
    dest_name = dest_info.get("name", "")

    if not source_name or not dest_name:
        return

    # determine context from source path
    # ex, "demographic.gender" -> context="demographic", source="gender"
    if "." in source_name:
        parts = source_name.rsplit(".", 1)
        context = parts[0]
        field_name = parts[1]
    else:
        context = entity_type
        field_name = source_name

    # create source
    source_id = f"gdc:{entity_type}.{source_name}"
    db.add_source(Source(
        id=source_id,
        source=field_name,
        source_schema="gdc",
        tier=Tier.FIELD,
        source_context=context
    ))

    # determine references from destination
    subject_ref = ""
    focus_ref = ""
    specimen_ref = ""

    # extract resource from destination path
    if "." in dest_name:
        dest_resource = dest_name.split(".")[0]
        # common patterns
        if dest_resource == "Observation":
            subject_ref = "Patient"
        if "specimen" in source_name.lower() or "sample" in source_name.lower():
            specimen_ref = "Specimen"

    # add destination
    db.add_destination(Destination(
        source_id=source_id,
        destination=dest_name,
        dest_system="",
        dest_code="",
        dest_display="",
        subject_ref=subject_ref,
        focus_ref=focus_ref,
        specimen_ref=specimen_ref
    ))

    # NOTE: GDC enums removed - content mappings now loaded from
    # raw_content_maps/terminology YAML files with proper FHIR codes


def _load_htan(db: FlatMappingDatabase, htan_dir: Path) -> None:
    """load HTAN mappings.json into flat db."""
    json_path = htan_dir / "mappings.json"
    if not json_path.exists():
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    for entry in data:
        _process_htan_entry(db, entry)


def _process_htan_entry(db: FlatMappingDatabase, entry: dict) -> None:
    """process single HTAN mapping entry."""
    source_node = entry.get("node", "")
    target_resource = entry.get("fhir:resourceType", "")

    if not source_node or not target_resource:
        return

    # extract source name from node (e.g., "bts:UrineBiospecimenType" -> "UrineBiospecimenType")
    if ":" in source_node:
        source_name = source_node.split(":")[-1]
    else:
        source_name = source_node

    # determine tier based on structure
    field_mappings = entry.get("fhir:fieldMapping", [])
    references = entry.get("fhir:reference", [])

    # get category/context
    category = entry.get("category", [])
    if isinstance(category, list) and category:
        context = category[0]
    elif isinstance(category, str):
        context = category
    else:
        context = ""

    # add entity-level source
    entity_id = source_node
    db.add_source(Source(
        id=entity_id,
        source=source_name,
        source_schema="htan",
        tier=Tier.ENTITY,
        source_context=context
    ))

    # add entity destination
    db.add_destination(Destination(
        source_id=entity_id,
        destination=target_resource
    ))

    # process field mappings
    for fm in field_mappings:
        if not isinstance(fm, dict):
            continue

        fhir_field = fm.get("fhir:field", "")
        fhir_type = fm.get("fhir:type", "")
        fhir_code = fm.get("fhir:code", "")
        fhir_system = fm.get("fhir:system", "")
        fhir_category = fm.get("fhir:category", "")

        if not fhir_field:
            continue

        field_id = f"{entity_id}.{fhir_field}"
        destination = f"{target_resource}.{fhir_field}"

        # add field source
        db.add_source(Source(
            id=field_id,
            source=source_name,
            source_schema="htan",
            tier=Tier.FIELD,
            source_context=context
        ))

        # determine references
        subject_ref = ""
        focus_ref = ""
        specimen_ref = ""

        for ref in references:
            if isinstance(ref, dict):
                ref_resource = ref.get("fhir:resourceType", "")
                ref_field = ref.get("fhir:field", "")
                if ref_field == "subject":
                    subject_ref = ref_resource
                elif ref_field == "focus":
                    focus_ref = ref_resource
                elif ref_field == "specimen":
                    specimen_ref = ref_resource

        # add field destination
        db.add_destination(Destination(
            source_id=field_id,
            destination=destination,
            dest_system=fhir_system,
            dest_code=fhir_code,
            dest_display="",
            subject_ref=subject_ref,
            focus_ref=focus_ref,
            specimen_ref=specimen_ref
        ))

        # if there's a code, also add content mapping
        if fhir_code:
            content_id = f"{field_id}.{_safe_id(fhir_code)}"
            db.add_source(Source(
                id=content_id,
                source=fhir_code,
                source_schema="htan",
                tier=Tier.CONTENT,
                source_context=source_name
            ))
            db.add_destination(Destination(
                source_id=content_id,
                destination=fhir_code,
                dest_system=fhir_system,
                dest_code=fhir_code
            ))


def _safe_id(value: str) -> str:
    """make value safe for use in id."""
    return value.lower().replace(" ", "_").replace("-", "_")[:50]