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

    def _stem(self, word: str) -> str:
        """simple suffix stripping for matching."""
        w = word.lower()
        # common suffixes in field names
        for suffix in ["ing", "tion", "ment", "ness", "ity", "ies", "es", "s"]:
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                return w[:-len(suffix)]
        return w

    def _tokenize(self, text: str) -> set:
        """split field name into tokens."""
        # split on _ and camelCase
        import re
        # split camelCase: "TumorGrade" -> ["Tumor", "Grade"]
        tokens = re.sub(r'([a-z])([A-Z])', r'\1_\2', text).lower().split('_')
        return set(t for t in tokens if len(t) >= 3)

    def fuzzy_lookup(self, query: str, limit: int = 10) -> list:
        """smart fuzzy search on field names in flat db.

        uses stemming, token matching, and fuzzy distance.

        args:
            query: search term (e.g., "grade", "tissue", "stage")
            limit: max results to return

        returns:
            list of dicts with source, target, score (based on match quality)
        """
        if not self.db:
            return []

        from fuzzywuzzy import fuzz

        query_lower = query.lower()
        query_stem = self._stem(query_lower)
        query_tokens = self._tokenize(query)
        min_match_len = 3
        results = []
        seen = set()

        for source_obj in self.db.sources.values():
            source_name = source_obj.source
            source_lower = source_name.lower()

            if len(source_lower) < min_match_len:
                continue

            score = 0.0

            # 1. exact match
            if source_lower == query_lower:
                score = 1.0

            # 2. substring match
            elif query_lower in source_lower:
                score = len(query_lower) / len(source_lower)
            elif source_lower in query_lower:
                score = len(source_lower) / len(query_lower) * 0.8

            # 3. stem match (grading -> grad matches tumor_grade -> grad)
            elif query_stem in self._stem(source_lower):
                score = 0.7
            elif self._stem(source_lower) in query_stem:
                score = 0.65

            # 4. token overlap (survival matches survival_status)
            else:
                source_tokens = self._tokenize(source_name)
                overlap = query_tokens & source_tokens
                if overlap:
                    score = 0.6 * len(overlap) / max(len(query_tokens), len(source_tokens))

            # 5. fuzzy string distance as fallback
            if score == 0:
                ratio = fuzz.ratio(query_lower, source_lower) / 100
                if ratio >= 0.7:  # only accept high similarity
                    score = ratio * 0.5  # discount fuzzy matches

            if score < 0.3:  # threshold
                continue

            dests = self.db.lookup(source_name)
            for _, dest in dests:
                key = f"{source_name}|{dest.destination}"
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "source": source_name,
                    "target": dest.destination,
                    "score": score,
                    "schema": source_obj.source_schema
                })

        results.sort(key=lambda x: -x["score"])
        return results[:limit]

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