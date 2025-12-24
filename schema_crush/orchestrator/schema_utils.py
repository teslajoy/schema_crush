"""utility functions for loading target schemas from LinkML definitions."""

from typing import Dict, List
from functools import lru_cache
from schema_crush.tools.fhir_schema_tool import get_schema_explorer


# cache the schema explorer singleton
_schema_explorer_cache = None

def get_cached_schema_explorer():
    """get cached schema explorer instance to avoid reloading files ex. yaml"""
    global _schema_explorer_cache
    if _schema_explorer_cache is None:
        _schema_explorer_cache = get_schema_explorer()
    return _schema_explorer_cache


def load_fhir_schema_for_calibration(required_fields: bool = True) -> Dict[str, List[str]]:
    """
    load FHIR schema from LinkML definition for calibration candidate generation.

    args:
        required_fields: if True, also include required fields in addition to all fields;
                        if False, include all fields only (default: True)

    returns:
        dict mapping {resource_type: [field_names]}

    example:
        {
            "Patient": ["id", "identifier", "name", "gender", ...],
            "Specimen": ["id", "type", "subject", "collection", ...],
            ...
        }
    """
    explorer = get_schema_explorer()
    resources = explorer.get_all_resources()

    schema = {}
    for resource in resources:
        # Always get all fields
        all_fields = explorer.get_resource_fields(resource, required_only=False)

        if required_fields:
            # Also get required fields and ensure they're included
            req_fields = explorer.get_resource_fields(resource, required_only=True)
            # Combine: required fields first, then other fields
            fields = req_fields + [f for f in all_fields if f not in req_fields]
        else:
            # Just use all fields
            fields = all_fields

        if fields:  # only include resources with fields
            schema[resource] = fields

    return schema


@lru_cache(maxsize=128)
def _load_fhir_schema_subset_cached(
    resources_tuple: tuple,
    max_fields_per_resource: int,
    required_fields: bool
) -> Dict[str, List[str]]:
    """Internal cached version that uses tuples for hashability."""
    explorer = get_cached_schema_explorer()
    resources = list(resources_tuple) if resources_tuple else None

    if resources is None:
        # default to most common biomedical resources
        resources = [
            "Patient", "Specimen", "Observation", "Condition",
            "DocumentReference", "ResearchStudy", "ResearchSubject",
            "MedicationAdministration", "ImagingStudy", "Group"
        ]

    schema = {}
    for resource in resources:
        # Always get all fields (try case-insensitive match)
        all_fields = explorer.get_resource_fields(resource, required_only=False)
        if not all_fields:
            # try lowercase version
            all_fields = explorer.get_resource_fields(resource.lower(), required_only=False)

        if required_fields:
            # Also get required fields and ensure they're included first
            req_fields = explorer.get_resource_fields(resource, required_only=True)
            if not req_fields:
                req_fields = explorer.get_resource_fields(resource.lower(), required_only=True)

            # Combine: required fields first, then other fields
            fields = req_fields + [f for f in all_fields if f not in req_fields]
        else:
            fields = all_fields

        # Apply limit if specified
        if max_fields_per_resource:
            fields = fields[:max_fields_per_resource]

        if fields:
            schema[resource] = fields

    return schema


def load_fhir_schema_subset(
    resources: List[str] = None,
    max_fields_per_resource: int = None,
    required_fields: bool = True
) -> Dict[str, List[str]]:
    """
    load subset of FHIR schema (for faster calibration) with caching.

    args:
        resources: list of resource types to include (None = default biomedical resources)
        max_fields_per_resource: optional limit on fields per resource (None = all fields)
        required_fields: if True, also include required fields in addition to all fields (default: True)

    returns:
        dict mapping {resource_type: [field_names]}

    note:
        results are cached using lru_cache for performance. repeated calls with
        the same parameters will reuse cached results without reloading the schema.
    """
    # convert list to tuple for hashability in lru_cache
    resources_tuple = tuple(resources) if resources else None
    return _load_fhir_schema_subset_cached(
        resources_tuple,
        max_fields_per_resource,
        required_fields
    )