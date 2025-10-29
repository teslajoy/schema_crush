"""magneto matcher wrapping magneto embedder."""

from typing import List, Tuple
import numpy as np
from schema_crush.tools.matchers.base import BaseMatcher
from schema_crush.tools.embeddings.magneto_embedder import MagnetoEmbedder


class MagnetoMatcher(BaseMatcher):
    """matcher using magneto embeddings for schema matching."""

    def __init__(self):
        """initialize magneto matcher."""
        self.embedder = MagnetoEmbedder()

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        """match single source to targets using magneto embeddings.

        args:
            source: source field name
            targets: candidate target fields

        returns:
            sorted list of (target, score) tuples
        """
        # use batch_similarity with single source
        scores = self.batch_similarity([source], targets)[0]

        # create (target, score) pairs and sort
        results = [(targets[i], float(scores[i])) for i in range(len(targets))]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """compute similarity matrix using magneto embeddings.

        args:
            sources: list of source field names
            targets: list of target field names

        returns:
            similarity matrix [len(sources), len(targets)]
        """
        return self.embedder.batch_similarity(sources, targets)