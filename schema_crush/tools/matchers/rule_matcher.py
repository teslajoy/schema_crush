"""matcher using composite mapping rules from knowledge base."""

from typing import List, Tuple
import numpy as np
from schema_crush.tools.matchers.base import BaseMatcher
from schema_crush.knowledge.mapping_rules import RuleDatabase, MappingRule


class RuleMatcher(BaseMatcher):
    """matcher using composite mapping rules from knowledge base.

    uses structured rules (htan/gdc) to score source->target mappings
    based on multi-tier matching (entity, references, field mappings).
    """

    def __init__(self, mapping_ruledb: RuleDatabase):
        """initialize rule matcher.

        args:
            mapping_ruledb: database of composite mapping rules
        """
        self.mapping_ruledb = mapping_ruledb
        self._cache = {}

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        """match using rule database lookups.

        ex. source="case_id" → finds rules where source_node contains "case"
            checks if target in {target_resource, field_mappings}

        args:
            source: source field name
            targets: candidate target fields

        returns:
            sorted list of (target, score) tuples
        """
        # query mapping_ruledb for source
        rules = self.mapping_ruledb.find_by_source(source)

        # score each target based on rule matches
        scores = []
        for target in targets:
            score = self._score_target(source, target, rules)
            scores.append((target, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """batch version for evaluation compatibility.

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

    def _score_target(self, source: str, target: str, rules: List[MappingRule]) -> float:
        """calculate confidence score for source->target based on rules.

        scoring logic for composite rules:
        - tier 1 (resource): exact match on target_resource (0.9)
        - tier 2 (references): match in references list (0.7)
        - tier 3 (field mappings): match in field_mappings (0.8)
        - partial match: substring match (0.5)
        - no match: 0.0

        note: tier weights could later be parameterized via settings.yaml
              to allow claude or hitl to tune scoring strategy.

        args:
            source: source field name
            target: target field name
            rules: matching rules from database

        returns:
            confidence score between 0.0 and 1.0
        """
        if not rules:
            return 0.0

        best_score = 0.0
        target_lower = target.lower()

        for rule in rules:
            # tier 1: resource-level match
            if rule.target_resource.lower() in target_lower:
                best_score = max(best_score, 0.9 * rule.confidence)

            # tier 2: reference match
            if rule.references:
                for ref in rule.references:
                    ref_path = f"{ref.resource}.{ref.field}".lower()
                    if ref_path in target_lower:
                        best_score = max(best_score, 0.7 * rule.confidence)

            # tier 3: field mapping match
            if rule.field_mappings:
                for fm in rule.field_mappings:
                    if fm.field.lower() in target_lower:
                        best_score = max(best_score, 0.8 * rule.confidence)

            # partial match: substring
            if rule.source_node.lower() in target_lower:
                best_score = max(best_score, 0.5 * rule.confidence)

        return best_score