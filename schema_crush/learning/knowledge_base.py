"""unified knowledge base - single entry point for all learned knowledge."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KnowledgeBase:
    """unified access to flat mappings, calibrators, embeddings, and vector store."""

    db: Optional[object] = None
    calibrators: Optional[dict] = None
    embeddings: Optional[dict] = None
    vectors: Optional[object] = None

    @classmethod
    def load(cls, include_vectors: bool = True) -> "KnowledgeBase":
        """load all knowledge components."""
        from schema_crush.mappings.flat_loader import load_flat_mappings
        from schema_crush.orchestrator.calibration import load_calibrators

        kb = cls()
        kb.db = load_flat_mappings()
        kb.calibrators = load_calibrators()

        try:
            from schema_crush.mappings.expert_embeddings import load_expert_embeddings
            kb.embeddings = load_expert_embeddings()
        except FileNotFoundError:
            kb.embeddings = None

        if include_vectors:
            from schema_crush.learning.vector_store import MappingVectorStore
            kb.vectors = MappingVectorStore()
            kb.vectors.index(kb.db)

        return kb

    def lookup(self, source: str, context: str = None):
        """o(1) lookup from flat db."""
        if not self.db:
            return []
        return self.db.lookup(source, context)

    def lookup_content(self, source: str, category: str = None):
        """lookup content values."""
        if not self.db:
            return []
        return self.db.lookup_content(source, category)

    def calibrate(self, matcher: str, score: float, tier: str = None) -> float:
        """apply tier-aware calibration."""
        if not self.calibrators or matcher not in self.calibrators:
            return score
        key = f"{matcher}:{tier}" if tier else matcher
        cal = self.calibrators[matcher]
        if hasattr(cal, "calibration_curves") and key in cal.calibration_curves:
            return cal.calibrate_score(key, score)
        return cal.calibrate_score(matcher, score)

    def find_similar(self, query: str, k: int = 5) -> list:
        """find similar mappings via vector search."""
        if not self.vectors:
            return []
        return self.vectors.find_similar(query, k)

    def stats(self) -> dict:
        """return knowledge base statistics."""
        return {
            "db": self.db.stats() if self.db else None,
            "calibrators": list(self.calibrators.keys()) if self.calibrators else [],
            "embeddings": bool(self.embeddings),
            "vectors": self.vectors._indexed if self.vectors else False,
        }