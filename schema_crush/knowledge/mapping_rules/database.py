"""rule database for storing and querying mapping rules."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from .base import MappingRule


class RuleDatabase:
    """
    in-memory database for mapping rules with query capabilities.

    supports:
    - finding rules by source node (ex. "patient_id")
    - finding rules by target resource (ex. "Patient")
    - finding rules by category (ex. "case")
    - finding rules with specific references (ex. "Specimen")
    - loading from htan/gdc parsers
    - saving/loading from json
    """

    def __init__(self):
        """initialize empty rule database."""
        self.rules: List[MappingRule] = []
        self._index_by_source: Dict[str, List[MappingRule]] = {}
        self._index_by_target: Dict[str, List[MappingRule]] = {}
        self._index_by_id: Dict[str, MappingRule] = {}

    def add_rule(self, rule: MappingRule):
        """add a single rule to database."""
        self.rules.append(rule)
        self._rebuild_indexes()

    def add_rules(self, rules: List[MappingRule]):
        """add multiple rules to database."""
        self.rules.extend(rules)
        self._rebuild_indexes()

    def _rebuild_indexes(self):
        """rebuild all indexes for fast lookup."""
        self._index_by_source.clear()
        self._index_by_target.clear()
        self._index_by_id.clear()

        for rule in self.rules:
            # index by source node
            source_lower = rule.source_node.lower()
            if source_lower not in self._index_by_source:
                self._index_by_source[source_lower] = []
            self._index_by_source[source_lower].append(rule)

            # index by target resource
            target_lower = rule.target_resource.lower()
            if target_lower not in self._index_by_target:
                self._index_by_target[target_lower] = []
            self._index_by_target[target_lower].append(rule)

            # index by rule_id
            self._index_by_id[rule.rule_id] = rule

    def find_by_source(self, source_query: str) -> List[MappingRule]:
        """
        find all rules matching source node.

        supports partial matching (case-insensitive).

        args:
            source_query: source node to search for (ex. "patient_id")

        returns:
            list of matching rules (may be multiple for one-to-many)
        """
        query_lower = source_query.lower()
        matches = []

        for source_key, rules in self._index_by_source.items():
            if query_lower in source_key:
                matches.extend(rules)

        return matches

    def find_by_target(self, target_query: str) -> List[MappingRule]:
        """
        find all rules matching target resource.

        args:
            target_query: target resource (ex. "Patient", "Specimen")

        returns:
            list of matching rules
        """
        query_lower = target_query.lower()
        return self._index_by_target.get(query_lower, [])

    def find_by_category(self, category: str) -> List[MappingRule]:
        """
        find all rules in a category.

        args:
            category: category name (ex. "case", "specimen")

        returns:
            list of matching rules
        """
        category_lower = category.lower()
        return [
            rule for rule in self.rules
            if any(cat.lower() == category_lower for cat in rule.category)
        ]

    def find_by_reference(self, resource: str) -> List[MappingRule]:
        """
        find all rules that reference a specific resource.

        args:
            resource: resource name (ex. "Specimen")

        returns:
            list of rules with references to that resource
        """
        return [
            rule for rule in self.rules
            if rule.has_reference_to(resource)
        ]

    def find_by_id(self, rule_id: str) -> Optional[MappingRule]:
        """
        find rule by unique id.

        args:
            rule_id: unique rule identifier

        returns:
            matching rule or None
        """
        return self._index_by_id.get(rule_id)

    def filter(
        self,
        source: Optional[str] = None,
        target_resource: Optional[str] = None,
        category: Optional[str] = None,
        has_reference_to: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[MappingRule]:
        """
        filter rules by multiple criteria.

        args:
            source: filter by source node (partial match)
            target_resource: filter by target resource
            category: filter by category
            has_reference_to: filter by reference to resource
            min_confidence: minimum confidence threshold

        returns:
            list of rules matching all specified criteria
        """
        results = self.rules.copy()

        if source:
            results = [r for r in results if r.matches_source(source)]

        if target_resource:
            results = [r for r in results if r.matches_target(target_resource)]

        if category:
            cat_lower = category.lower()
            results = [
                r for r in results
                if any(c.lower() == cat_lower for c in r.category)
            ]

        if has_reference_to:
            results = [r for r in results if r.has_reference_to(has_reference_to)]

        if min_confidence is not None:
            results = [r for r in results if r.confidence >= min_confidence]

        return results

    def count(self) -> int:
        """return total number of rules."""
        return len(self.rules)

    def get_statistics(self) -> Dict[str, Any]:
        """get database statistics."""
        sources = set(r.source for r in self.rules)
        targets = set(r.target_resource for r in self.rules)
        categories = set(cat for r in self.rules for cat in r.category)

        return {
            "total_rules": len(self.rules),
            "sources": sorted(sources),
            "unique_targets": len(targets),
            "unique_categories": len(categories),
            "categories": sorted(categories)
        }

    def save(self, file_path: Path):
        """
        save database to json file.

        args:
            file_path: path to save json file
        """
        data = {
            "rules": [rule.to_dict() for rule in self.rules],
            "statistics": self.get_statistics()
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"saved {len(self.rules)} rules to {file_path}")

    def load(self, file_path: Path):
        """
        load database from json file.

        args:
            file_path: path to json file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        rules = [MappingRule.from_dict(r) for r in data.get("rules", [])]
        self.add_rules(rules)

        print(f"loaded {len(rules)} rules from {file_path}")

    def clear(self):
        """clear all rules from database."""
        self.rules.clear()
        self._index_by_source.clear()
        self._index_by_target.clear()
        self._index_by_id.clear()