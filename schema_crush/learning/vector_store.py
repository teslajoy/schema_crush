"""chromadb vector store for semantic mapping retrieval."""

import chromadb
from sentence_transformers import SentenceTransformer


class MappingVectorStore:
    """vector store for finding similar mappings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("mappings")
        self.model = SentenceTransformer(model_name)
        self._indexed = False

    def index(self, db):
        """index all sources from flatmappingdatabase."""
        if self._indexed:
            return

        sources = []
        targets = []
        ids = []

        # entity + field tier
        for src in db.sources.values():
            dests = db._dest_by_source.get(src.id, [])
            for i, d in enumerate(dests):
                sources.append(src.source)
                targets.append(d.destination)
                ids.append(f"{src.id}_{i}")

        # content tier
        for cv in db.content_values:
            for t in db.content_fhir_targets:
                if t.content_value_id == cv.id:
                    sources.append(cv.source_value)
                    targets.append(t.fhir_path)
                    ids.append(f"cv_{cv.id}_{t.id}")

        if not sources:
            self._indexed = True
            return

        embeddings = self.model.encode(sources).tolist()
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=[{"source": s, "target": t} for s, t in zip(sources, targets)]
        )
        self._indexed = True

    def find_similar(self, query: str, k: int = 5) -> list:
        """find k most similar mappings."""
        if not self._indexed or self.collection.count() == 0:
            return []

        query_emb = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=k)

        out = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                out.append({
                    "source": meta["source"],
                    "target": meta["target"],
                    "score": 1 - results["distances"][0][i]
                })
        return out