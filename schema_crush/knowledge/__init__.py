"""knowledge infrastructure for domain-specific information.

this module contains:
- mapping_rules: composite fhir mapping rules (htan/gdc)
- terminology: (future) code systems, ontologies, value sets
- fhir_schema: (future) fhir resource definitions
"""

from .mapping_rules import MappingRule, Reference, FieldMapping

__all__ = ["MappingRule", "Reference", "FieldMapping"]