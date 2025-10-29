"""biobert matcher wrapping biobert embedder."""

from typing import List, Tuple
import numpy as np
from schema_crush.tools.matchers.base import BaseMatcher
from schema_crush.tools.embeddings.biobert_embedder import BioBERTEmbedder


class BioBERTMatcher(BaseMatcher):
    """matcher using biobert embeddings for biomedical semantic similarity."""

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        """initialize biobert matcher.

        args:
            model_name: huggingface model identifier
        """
        self.embedder = BioBERTEmbedder(model_name=model_name)

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        """match single source to targets using biobert embeddings.

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
        """compute similarity matrix using biobert embeddings.

        args:
            sources: list of source field names
            targets: list of target field names

        returns:
            similarity matrix [len(sources), len(targets)]
        """
        return self.embedder.batch_similarity(sources, targets)