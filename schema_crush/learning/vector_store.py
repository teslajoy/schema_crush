"""chromadb vector store for semantic mapping retrieval."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# default path relative to this file: schema_crush/data/db/chroma/
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent / "data" / "db" / "chroma"


class MappingVectorStore:
    """vector store for finding similar mappings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Path | str | None = None,
    ):
        persist_path = Path(persist_directory) if persist_directory else DEFAULT_CHROMA_PATH
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection("mappings")
        self.model = SentenceTransformer(model_name)
        # check if already indexed from persistence
        self._indexed = self.collection.count() > 0

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

    def clear(self):
        """clear the collection and reset indexed state."""
        self.client.delete_collection("mappings")
        self.collection = self.client.get_or_create_collection("mappings")
        self._indexed = False

    def reindex(self, db):
        """force reindex by clearing and rebuilding."""
        self.clear()
        self.index(db)

    def add_mapping(self, source: str, target: str, id_prefix: str = "user"):
        """add a single mapping (for feedback loop)."""
        doc_id = f"{id_prefix}_{source}_{target}".replace(" ", "_")[:63]
        embedding = self.model.encode([source]).tolist()
        self.collection.upsert(
            ids=[doc_id],
            embeddings=embedding,
            metadatas=[{"source": source, "target": target}]
        )