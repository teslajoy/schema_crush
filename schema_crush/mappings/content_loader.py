"""load content mappings from raw_content_maps YAML files into FlatMappingDatabase."""

from pathlib import Path
from typing import Optional
import yaml

from .flat import FlatMappingDatabase, ContentValue, ContentFhirTarget


# default terminology directory
_TERMINOLOGY_DIR = Path(__file__).parent.parent / "data" / "raw_content_maps" / "terminology"


def load_content_mappings(
    db: FlatMappingDatabase,
    terminology_dir: Optional[Path] = None,
) -> int:
    """load all terminology YAML files into FlatMappingDatabase.

    args:
        db: FlatMappingDatabase to populate
        terminology_dir: path to terminology YAML files (default: data/raw_content_maps/terminology)

    returns:
        number of content values added
    """
    if terminology_dir is None:
        terminology_dir = _TERMINOLOGY_DIR

    if not terminology_dir.exists():
        return 0

    total_added = 0

    for yaml_file in terminology_dir.glob("*.yaml"):
        category = yaml_file.stem  # "histology", "staging_grade", etc.
        added = _load_terminology_file(db, yaml_file, category)
        total_added += added

    return total_added


def _load_terminology_file(
    db: FlatMappingDatabase,
    yaml_path: Path,
    category: str,
) -> int:
    """load single terminology YAML file.

    handles multiple YAML formats found in raw_content_maps:
    - histology.yaml: {value: {Condition.code: [{code, description}]}}
    - staging_grade.yaml: {value: {grade: [{code, description}]}}
    - sample_tissue_type.yaml: {value: [{code}]}
    - metastasis-site.yaml: {value: {body_structure: [{code, description}]}}
    - condition.yaml: {value: {Condition.code: [{code, description}]}}
    - anatomical_location.prime.yaml: {value: {Procedure.code, body_structure}}
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return 0

    added = 0

    for source_value, mapping in data.items():
        # skip comments (strings starting with #)
        if isinstance(source_value, str) and source_value.startswith("#"):
            continue

        if not isinstance(mapping, dict) and not isinstance(mapping, list):
            continue

        # handle different formats
        added += _process_mapping(db, category, source_value, mapping)

    return added


def _process_mapping(
    db: FlatMappingDatabase,
    category: str,
    source_value: str,
    mapping,
) -> int:
    """process a single source value mapping."""
    added = 0

    # format: {value: [{code, description}]} (simple list)
    if isinstance(mapping, list):
        for i, item in enumerate(mapping):
            if isinstance(item, dict) and "code" in item:
                cv_id = db.add_content_value(ContentValue(
                    source_value=str(source_value),
                    source_category=category,
                    code=str(item.get("code", "")),
                    display=item.get("description", ""),
                    is_primary=(i == 0),
                ))
                # infer FHIR path from category
                fhir_path, fhir_resource = _infer_fhir_path(category)
                db.add_content_fhir_target(ContentFhirTarget(
                    content_value_id=cv_id,
                    fhir_path=fhir_path,
                    fhir_resource=fhir_resource,
                    context=category,
                    is_primary=True,
                ))
                added += 1
        return added

    # format: {value: {key: [{code, description}]}}
    if isinstance(mapping, dict):
        for fhir_key, codes in mapping.items():
            # skip empty or None
            if not codes:
                continue

            # determine FHIR path from key
            fhir_path, fhir_resource = _parse_fhir_key(fhir_key, category)

            if isinstance(codes, list):
                for i, item in enumerate(codes):
                    if isinstance(item, dict):
                        code = item.get("code", item.get("body_structure_code", ""))
                        if not code:
                            continue

                        cv_id = db.add_content_value(ContentValue(
                            source_value=str(source_value),
                            source_category=category,
                            code=str(code),
                            display=item.get("description", ""),
                            is_primary=(i == 0),
                        ))
                        db.add_content_fhir_target(ContentFhirTarget(
                            content_value_id=cv_id,
                            fhir_path=fhir_path,
                            fhir_resource=fhir_resource,
                            context=category,
                            is_primary=(fhir_key == list(mapping.keys())[0]),
                        ))
                        added += 1

            elif isinstance(codes, dict):
                # nested dict with code
                code = codes.get("code", "")
                if code:
                    cv_id = db.add_content_value(ContentValue(
                        source_value=str(source_value),
                        source_category=category,
                        code=str(code),
                        display=codes.get("description", ""),
                        is_primary=True,
                    ))
                    db.add_content_fhir_target(ContentFhirTarget(
                        content_value_id=cv_id,
                        fhir_path=fhir_path,
                        fhir_resource=fhir_resource,
                        context=category,
                        is_primary=True,
                    ))
                    added += 1

    return added


def _parse_fhir_key(key: str, category: str) -> tuple[str, str]:
    """parse FHIR key to (fhir_path, fhir_resource)."""
    # explicit FHIR paths like "Condition.code", "Procedure.code"
    if "." in key:
        resource = key.split(".")[0]
        return key, resource

    # known keys
    key_lower = key.lower()
    if key_lower == "body_structure":
        return "Specimen.collection.bodySite", "Specimen"
    if key_lower == "grade":
        return "Observation.valueCodeableConcept", "Observation"
    if key_lower in ("procedure", "procedure_code"):
        return "Procedure.code", "Procedure"

    # default: infer from category
    return _infer_fhir_path(category)


def _infer_fhir_path(category: str) -> tuple[str, str]:
    """infer FHIR path from category name."""
    category_lower = category.lower()

    # condition/diagnosis categories
    if any(x in category_lower for x in ["condition", "diagnosis", "histology"]):
        return "Condition.code", "Condition"

    # staging categories -> Observation
    if any(x in category_lower for x in ["staging", "grade", "stage"]):
        return "Observation.valueCodeableConcept", "Observation"

    # tissue/specimen categories
    if any(x in category_lower for x in ["tissue", "specimen", "sample"]):
        return "Specimen.type", "Specimen"

    # anatomical/body structure
    if any(x in category_lower for x in ["anatomical", "location", "site", "metastasis"]):
        return "Specimen.collection.bodySite", "Specimen"

    # procedure
    if "procedure" in category_lower:
        return "Procedure.code", "Procedure"

    # default to Observation
    return "Observation.valueCodeableConcept", "Observation"


def get_terminology_categories() -> list[str]:
    """return list of available terminology categories."""
    if not _TERMINOLOGY_DIR.exists():
        return []
    return [f.stem for f in _TERMINOLOGY_DIR.glob("*.yaml")]