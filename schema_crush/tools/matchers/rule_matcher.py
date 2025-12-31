"""matcher using mapping rules - supports both flat and hierarchical databases."""

from typing import List, Tuple, Optional, Union
import numpy as np
from schema_crush.tools.matchers.base import BaseMatcher
from schema_crush.mappings.flat import FlatMappingDatabase, Tier

# old hierarchical imports - kept for backwards compatibility
# from schema_crush.knowledge.mapping_rules import RuleDatabase, MappingRule


class RuleMatcher(BaseMatcher):
    """matcher using mapping rules with O(1) lookup via flat database.

    supports both flat mappings (new) and hierarchical rules (legacy).
    flat mappings provide direct source term lookup with context awareness.
    """

    def __init__(
        self,
        db: FlatMappingDatabase,
        # mapping_ruledb: RuleDatabase = None,  # legacy hierarchical db
    ):
        """initialize rule matcher.

        args:
            db: flat mapping database with sources and destinations
        """
        self.db = db
        self._cache = {}

    @classmethod
    def from_builtin(cls) -> "RuleMatcher":
        """load matcher with builtin gdc/htan mappings."""
        from schema_crush.mappings.flat_loader import load_flat_mappings
        db = load_flat_mappings()
        return cls(db)

    def match(
        self,
        source: str,
        targets: List[str],
        context: Optional[str] = None,
        tier: Optional[Tier] = None,
        schema: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """match source to targets using flat database O(1) lookup.

        searches both sources table AND content_values table.

        scoring:
        - exact destination match -> 1.0
        - resource-only match (target has no field) -> 0.5
        - no match -> 0.0

        args:
            source: source field name
            targets: candidate target fields
            context: optional context for disambiguation
            tier: optional tier filter (entity, field, content)
            schema: optional schema filter (e.g., "gdc", "htan")

        returns:
            sorted list of (target, score) tuples
        """
        # O(1) lookup in flat database (sources table)
        mappings = self.db.lookup(source, context, schema=schema)

        # filter by tier if specified
        if tier and mappings:
            mappings = [(s, d) for s, d in mappings if s.tier == tier]

        # also check content_values table for content tier lookups
        content_results = self.db.lookup_content(source, category=context)

        # if no mappings from sources but found in content_values
        if not mappings and content_results:
            # content_values -> fhir_path matching
            # collect ALL fhir_paths from ALL matching content_values
            dest_set = set()
            for cv, fhir_targets in content_results:
                for t in fhir_targets:
                    dest_set.add(t.fhir_path.lower())

            # extract resources from FHIR paths
            dest_resources = set()
            for path in dest_set:
                parts = path.split(".", 1)
                if parts:
                    dest_resources.add(parts[0].lower())

            scores = []
            for target in targets:
                score = self._score_target_flat(target, dest_set, dest_resources)
                scores.append((target, score))

            return sorted(scores, key=lambda x: x[1], reverse=True)

        if not mappings:
            return [(t, 0.0) for t in targets]

        # collect destination strings and resources
        dest_set = {d.destination.lower() for _, d in mappings}
        dest_resources = set()
        for _, d in mappings:
            parts = d.destination.split(".", 1)
            if parts:
                dest_resources.add(parts[0].lower())

        # score each target
        scores = []
        for target in targets:
            score = self._score_target_flat(target, dest_set, dest_resources)
            scores.append((target, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _score_target_flat(
        self,
        target: str,
        dest_set: set,
        dest_resources: set,
    ) -> float:
        """score target against flat destination set.

        args:
            target: target field to score
            dest_set: set of destination strings (lowercase)
            dest_resources: set of resource names (lowercase)

        returns:
            confidence score 0.0-1.0
        """
        target_lower = target.lower()

        # exact match
        if target_lower in dest_set:
            return 1.0

        # parse target into resource.field
        parts = target.split(".", 1)
        target_resource = parts[0].lower() if parts else target_lower
        target_field = parts[1].lower() if len(parts) > 1 else None

        # resource-only match (no field specified in target)
        if target_resource in dest_resources and not target_field:
            return 0.5

        return 0.0

    def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """batch similarity matrix for evaluation.

        args:
            sources: list of source fields
            targets: list of target fields

        returns:
            similarity matrix [len(sources), len(targets)]
        """
        matrix = np.zeros((len(sources), len(targets)))
        for i, src in enumerate(sources):
            matches = self.match(src, targets)
            for j, tgt in enumerate(targets):
                score = next((s for t, s in matches if t == tgt), 0.0)
                matrix[i, j] = score
        return matrix

    def match_with_metadata(
        self,
        source: str,
        targets: List[str],
        context: Optional[str] = None,
    ) -> List[dict]:
        """match with full metadata from flat database.

        returns destination details including system, code, references.

        args:
            source: source field name
            targets: candidate target fields
            context: optional context for disambiguation

        returns:
            list of dicts with target, score, and metadata
        """
        mappings = self.db.lookup(source, context)

        if not mappings:
            return [{"target": t, "score": 0.0, "metadata": None} for t in targets]

        # build destination lookup
        dest_map = {}
        for src, dest in mappings:
            key = dest.destination.lower()
            if key not in dest_map:
                dest_map[key] = {
                    "tier": src.tier.value,
                    "context": src.source_context,
                    "system": dest.dest_system,
                    "code": dest.dest_code,
                    "display": dest.dest_display,
                    "subject_ref": dest.subject_ref,
                    "focus_ref": dest.focus_ref,
                    "specimen_ref": dest.specimen_ref,
                }

        results = []
        for target in targets:
            target_lower = target.lower()
            if target_lower in dest_map:
                results.append({
                    "target": target,
                    "score": 1.0,
                    "metadata": dest_map[target_lower],
                })
            else:
                results.append({
                    "target": target,
                    "score": 0.0,
                    "metadata": None,
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def lookup_reverse(self, destination: str) -> List[dict]:
        """reverse lookup: find sources that map to a destination.

        args:
            destination: fhir destination path

        returns:
            list of source info dicts
        """
        mappings = self.db.lookup_by_destination(destination)

        return [
            {
                "source": src.source,
                "schema": src.source_schema,
                "tier": src.tier.value,
                "context": src.source_context,
            }
            for src, _ in mappings
        ]

    def stats(self) -> dict:
        """return matcher statistics."""
        return {
            "type": "rule_matcher",
            "db_stats": self.db.stats(),
        }


# -----------------------------------------------------------------------------
# legacy hierarchical matcher - commented out, kept for reference
# -----------------------------------------------------------------------------
#
# class HierarchicalRuleMatcher(BaseMatcher):
#     """old matcher using hierarchical RuleDatabase.
#
#     uses structured rules (htan/gdc) to score source->target mappings
#     based on multi-tier matching (entity, references, field mappings).
#     """
#
#     def __init__(self, mapping_ruledb: RuleDatabase):
#         """initialize rule matcher.
#
#         args:
#             mapping_ruledb: database of composite mapping rules
#         """
#         self.mapping_ruledb = mapping_ruledb
#         self._cache = {}
#
#     def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
#         """match using rule database lookups.
#
#         args:
#             source: source field name
#             targets: candidate target fields
#
#         returns:
#             sorted list of (target, score) tuples
#         """
#         # query mapping_ruledb for source
#         rules = self.mapping_ruledb.find_by_source(source)
#
#         # score each target based on rule matches
#         scores = []
#         for target in targets:
#             score = self._score_target(source, target, rules)
#             scores.append((target, score))
#
#         return sorted(scores, key=lambda x: x[1], reverse=True)
#
#     def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
#         """batch version for evaluation compatibility."""
#         matrix = np.zeros((len(sources), len(targets)))
#         for i, src in enumerate(sources):
#             matches = self.match(src, targets)
#             for j, tgt in enumerate(targets):
#                 score = next((s for t, s in matches if t == tgt), 0.0)
#                 matrix[i, j] = score
#         return matrix
#
#     def _score_target(self, source: str, target: str, rules: List[MappingRule]) -> float:
#         """calculate confidence score for source->target based on rules.
#
#         scoring logic for composite rules (reliable tiers only):
#         - exact match (resource + field): 1.0 (perfect match)
#         - reference exact match: 0.9 (appears in reference chain)
#         - resource-only match: 0.5 (correct resource, unknown field)
#         - no match: 0.0
#         """
#         if not rules:
#             return 0.0
#
#         best_score = 0.0
#         target_lower = target.lower()
#
#         # parse target into resource and field
#         target_parts = target.split(".", 1)
#         target_resource = target_parts[0].lower() if target_parts else target_lower
#         target_field = target_parts[1].lower() if len(target_parts) > 1 else None
#
#         for rule in rules:
#             rule_resource = rule.target_resource.lower()
#
#             # exact resource match only (no substring matching)
#             resource_match = (rule_resource == target_resource)
#
#             # tier 1: exact field match (highest confidence)
#             if rule.field_mappings and target_field:
#                 for fm in rule.field_mappings:
#                     rule_field = fm.field.lower()
#
#                     # exact match: resource + field both correct
#                     if resource_match and rule_field == target_field:
#                         best_score = max(best_score, 1.0 * rule.confidence)
#
#             # tier 2: reference exact match (high confidence)
#             if rule.references and target_field:
#                 for ref in rule.references:
#                     ref_resource = ref.resource.lower()
#                     ref_field = ref.field.lower()
#
#                     # check if target exactly matches a reference
#                     if ref_resource == target_resource and ref_field == target_field:
#                         best_score = max(best_score, 0.9 * rule.confidence)
#
#             # tier 3: resource-only match (moderate confidence, no field info)
#             if resource_match and best_score < 0.5 and not target_field:
#                 best_score = max(best_score, 0.5 * rule.confidence)
#
#         return best_score