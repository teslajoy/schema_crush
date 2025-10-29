"""base matcher interface for schema crush."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseMatcher(ABC):
    """base class for all schema mapping matchers.

    matchers score source-to-target field similarity using various approaches:
    - semantic similarity (ex. biobert, magneto)
    - rule-based matching (ex. knowledge base rules)
    - hybrid approaches
    """

    @abstractmethod
    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        """match single source to targets, return sorted (target, score) pairs.

        args:
            source: source field name
            targets: list of candidate target fields

        returns:
            list of (target, score) tuples sorted by score descending
        """
        pass

    @abstractmethod
    def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """compute similarity matrix for batch evaluation.

        args:
            sources: list of source field names
            targets: list of target field names

        returns:
            similarity matrix with shape [len(sources), len(targets)]
        """
        pass