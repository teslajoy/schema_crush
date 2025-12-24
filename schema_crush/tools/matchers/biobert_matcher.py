"""biobert matcher wrapping biobert embedder."""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from schema_crush.tools.matchers.base import BaseMatcher
from schema_crush.tools.embeddings.biobert_embedder import BioBERTEmbedder


class BioBERTMatcher(BaseMatcher):
    """matcher using biobert embeddings for biomedical semantic similarity."""

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        use_expert_embeddings: bool = False,
    ):
        """initialize biobert matcher.

        args:
            model_name: huggingface model identifier
            use_expert_embeddings: if True, use pre-computed expert embeddings for content matching
        """
        self.embedder = BioBERTEmbedder(model_name=model_name)
        self.use_expert_embeddings = use_expert_embeddings
        self._expert_data = None

        if use_expert_embeddings:
            self._load_expert_embeddings()

    def _load_expert_embeddings(self):
        """load pre-computed expert embeddings."""
        from schema_crush.mappings.expert_embeddings import load_expert_embeddings
        try:
            self._expert_data = load_expert_embeddings()
        except FileNotFoundError:
            print("warning: expert embeddings not found, run: python -m schema_crush.mappings.expert_embeddings")
            self.use_expert_embeddings = False

    def match(self, source: str, targets: List[str]) -> List[Tuple[str, float]]:
        """match single source to targets using biobert embeddings.

        args:
            source: source field name
            targets: candidate target fields

        returns:
            sorted list of (target, score) tuples
        """
        # try expert embeddings first for content matching
        if self.use_expert_embeddings and self._expert_data:
            expert_result = self._match_expert(source, targets)
            if expert_result:
                return expert_result

        # fallback to biobert embeddings
        scores = self.batch_similarity([source], targets)[0]
        results = [(targets[i], float(scores[i])) for i in range(len(targets))]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _match_expert(self, source: str, targets: List[str]) -> Optional[List[Tuple[str, float]]]:
        """match using expert embeddings if source is known."""
        source_lower = source.lower()
        expertise = self._expert_data["expertise"]

        # exact lookup - return known mappings with score 1.0
        if source_lower in expertise:
            known_paths = {p.lower() for p in expertise[source_lower]}
            results = []
            for t in targets:
                if t.lower() in known_paths:
                    results.append((t, 1.0))
                else:
                    results.append((t, 0.0))
            return sorted(results, key=lambda x: x[1], reverse=True)

        # fuzzy match using embeddings
        sources_list = self._expert_data["sources"]
        source_emb = self._expert_data["source_embeddings"]

        if source_lower not in sources_list:
            return None  # unknown source, fallback to biobert

        idx = sources_list.index(source_lower)
        src_vec = source_emb[idx]

        # find similar known sources
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([src_vec], source_emb)[0]
        top_idx = np.argmax(sims)

        if sims[top_idx] > 0.8:  # high similarity threshold
            similar_source = sources_list[top_idx]
            known_paths = {p.lower() for p in expertise[similar_source]}
            results = []
            for t in targets:
                if t.lower() in known_paths:
                    results.append((t, float(sims[top_idx])))
                else:
                    results.append((t, 0.0))
            return sorted(results, key=lambda x: x[1], reverse=True)

        return None

    def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """compute similarity matrix using biobert embeddings.

        args:
            sources: list of source field names
            targets: list of target field names

        returns:
            similarity matrix [len(sources), len(targets)]
        """
        return self.embedder.batch_similarity(sources, targets)