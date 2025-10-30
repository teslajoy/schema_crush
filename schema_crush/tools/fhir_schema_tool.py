"""fhir + iceberg schema exploration tool using linkml schema (HTAN TBD)."""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache


class FHIRSchemaExplorer:
    """query fhir schema from linkml definition."""

    def __init__(self, schema_path: Optional[str] = None):
        """initialize with linkml schema file.

        args:
            schema_path: path to linkml yaml schema file
        """
        if schema_path is None:
            # default to bmeg_iceberg schema
            # path from schema_crush/tools/ -> schema_crush/data/resources/
            base_path = Path(__file__).parent.parent
            schema_path = base_path / "data" / "resources" / "bmeg_iceberg_structural_linkml_schema.yaml"

        self.schema_path = Path(schema_path)
        self._schema = None
        self._load_schema()

    def _load_schema(self):
        """load linkml schema from yaml file."""
        with open(self.schema_path, 'r') as f:
            self._schema = yaml.safe_load(f)

    @lru_cache(maxsize=1)
    def get_all_resources(self) -> List[str]:
        """get list of all fhir resource types.

        returns:
            list of resource type names
        """
        if not self._schema or 'classes' not in self._schema:
            return []
        return list(self._schema['classes'].keys())

    def get_resource_fields(self, resource_type: str) -> List[str]:
        """get all fields for a specific fhir resource.

        args:
            resource_type: fhir resource name (case-insensitive)

        returns:
            list of field names (slots) for the resource
        """
        # try exact match first
        classes = self._schema.get('classes', {})
        if resource_type in classes:
            return classes[resource_type].get('slots', [])

        # try case-insensitive match
        resource_lower = resource_type.lower()
        for class_name, class_def in classes.items():
            if class_name.lower() == resource_lower:
                return class_def.get('slots', [])

        return []

    def get_resource_description(self, resource_type: str) -> str:
        """get description for a fhir resource.

        args:
            resource_type: fhir resource name

        returns:
            resource description
        """
        classes = self._schema.get('classes', {})

        # try exact match
        if resource_type in classes:
            return classes[resource_type].get('description', '')

        # try case-insensitive
        resource_lower = resource_type.lower()
        for class_name, class_def in classes.items():
            if class_name.lower() == resource_lower:
                return class_def.get('description', '')

        return ''

    def search_fields(self, search_term: str, resource_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """search for fields matching a term across resources.

        args:
            search_term: term to search for in field names
            resource_filter: optional resource to limit search

        returns:
            list of matching field paths with resources
        """
        results = []
        search_lower = search_term.lower()
        classes = self._schema.get('classes', {})

        for resource, class_def in classes.items():
            # apply resource filter
            if resource_filter and resource.lower() != resource_filter.lower():
                continue

            slots = class_def.get('slots', [])
            for field in slots:
                if search_lower in field.lower():
                    results.append({
                        'resource': resource,
                        'field': field,
                        'full_path': f"{resource}.{field}"
                    })

        return results

    def get_all_field_paths(self, limit: Optional[int] = None) -> List[str]:
        """get all field paths in resource.field format.

        args:
            limit: optional limit on number of paths to return

        returns:
            list of field paths
        """
        paths = []
        classes = self._schema.get('classes', {})

        for resource, class_def in classes.items():
            slots = class_def.get('slots', [])
            for field in slots:
                paths.append(f"{resource}.{field}")
                if limit and len(paths) >= limit:
                    return paths

        return paths


# global singleton
_schema_explorer = None


def get_schema_explorer() -> FHIRSchemaExplorer:
    """get singleton fhirschemaexplorer instance."""
    global _schema_explorer
    if _schema_explorer is None:
        _schema_explorer = FHIRSchemaExplorer()
    return _schema_explorer