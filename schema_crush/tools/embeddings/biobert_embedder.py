"""biobert embedder for biomedical semantic similarity."""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class BioBERTEmbedder:
    """
    biobert embedder using sentence-transformers.

    uses dmis-lab/biobert-v1.1 model trained on pubmed literature.
    """

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        """
        initialize biobert embedder.

        args:
            model_name: huggingface model name
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """load the biobert model."""
        print(f"loading {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print(f"model loaded successfully, embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        generate embeddings for texts.

        args:
            texts: single text or list of texts

        returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_entity(self, entity_name: str) -> np.ndarray:
        """
        tier 1: embed entity/table name only.

        args:
            entity_name: name of entity (e.g., "case", "biospecimen")

        returns:
            numpy array of embedding (1, embedding_dim)
        """
        return self.embed(entity_name)

    def embed_field(self, field_name: str) -> np.ndarray:
        """
        tier 2: embed field/column name only.

        args:
            field_name: name of field (e.g., "participant_id", "primary_diagnosis")

        returns:
            numpy array of embedding (1, embedding_dim)
        """
        return self.embed(field_name)

    def embed_content(self, field_name: str, column_values: List[str], num_samples: int = 10) -> np.ndarray:
        """
        tier 3: embed field name + column values for content matching.

        args:
            field_name: name of the field
            column_values: full list of column values (vector)
            num_samples: number of random samples to use

        returns:
            numpy array of embedding (1, embedding_dim)
        """
        import random

        # filter to get complete, non-null, non-empty values
        valid_values = [
            str(v).strip()
            for v in column_values
            if v is not None
            and str(v).strip()
            and str(v).strip().lower() not in ['nan', 'null', 'none', '', 'na', 'n/a']
        ]

        # randomly sample from valid values
        if valid_values:
            sample_size = min(num_samples, len(valid_values))
            samples = random.sample(valid_values, sample_size)
            text = f"{field_name}: {', '.join(samples)}"
        else:
            text = field_name

        return self.embed(text)

    def similarity(self, text1: str, text2: str) -> float:
        """
        calculate cosine similarity between two texts.

        args:
            text1: first text
            text2: second text

        returns:
            similarity score (0.0-1.0)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)

    def batch_similarity(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """
        calculate similarity matrix between source and target texts.

        args:
            sources: list of source texts
            targets: list of target texts

        returns:
            similarity matrix (n_sources, n_targets)
        """
        source_embs = self.embed(sources)
        target_embs = self.embed(targets)

        similarity_matrix = cosine_similarity(source_embs, target_embs)
        return similarity_matrix