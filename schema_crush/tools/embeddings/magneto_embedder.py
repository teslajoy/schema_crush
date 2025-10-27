"""magneto embedder for schema matching using magneto library."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from magneto import Magneto
from magneto.embedding_matcher import EmbeddingMatcher, DEFAULT_MODELS
from magneto.column_encoder import ColumnEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch


class MagnetoEmbedder:
    """
    magneto embedder wrapper for schema matching.

    uses magneto's trained models and column encoding strategies.
    supports both default models (mpnet, roberta, e5, arctic, minilm)
    and fine-tuned models trained on schema matching benchmarks (including GDC).
    """

    def __init__(
        self,
        model_name: str = "mpnet",
        encoding_mode: str = "header_values_verbose",
        sampling_mode: str = "mixed",
        num_samples: int = 10,
        embedding_threshold: float = 0.1,
        topk: int = 20
    ):
        """
        initialize magneto embedder.

        args:
            model_name: key from DEFAULT_MODELS or path to fine-tuned model
            encoding_mode: column encoding strategy (header_only, header_values_verbose, etc.)
            sampling_mode: sampling strategy (random, frequent, mixed, etc.)
            num_samples: number of samples for content embedding
            embedding_threshold: minimum similarity threshold
            topk: number of top matches to return
        """
        self.model_name = model_name
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.num_samples = num_samples
        self.embedding_threshold = embedding_threshold
        self.topk = topk

        # create params dict for magneto
        self.params = {
            "embedding_model": model_name,
            "encoding_mode": encoding_mode,
            "sampling_mode": sampling_mode,
            "sampling_size": num_samples,
            "embedding_threshold": embedding_threshold,
            "topk": topk,
            "include_embedding_matches": True,
            "include_strsim_matches": False,
            "include_equal_matches": False,
            "use_bp_reranker": False,
            "use_gpt_reranker": False,
        }

        # initialize embedding matcher
        self.embedding_matcher = EmbeddingMatcher(self.params)
        self.model = self.embedding_matcher.model
        self.tokenizer = self.embedding_matcher.tokenizer

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        generate embeddings for texts using magneto's model.

        args:
            texts: list of texts to embed

        returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.embedding_matcher._get_embeddings(
            texts,
            use_prompt_query=self.embedding_matcher.use_prompt_query
        )

        # convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def embed_entity(self, entity_name: str) -> np.ndarray:
        """
        tier 1: embed entity/table name only.

        args:
            entity_name: name of entity (e.g., "case", "biospecimen")

        returns:
            numpy array of embedding (1, embedding_dim)
        """
        return self.embed([entity_name])

    def embed_field(self, field_name: str) -> np.ndarray:
        """
        tier 2: embed field/column name only.

        args:
            field_name: name of field (e.g., "participant_id", "primary_diagnosis")

        returns:
            numpy array of embedding (1, embedding_dim)
        """
        return self.embed([field_name])

    def embed_content(
        self,
        field_name: str,
        column_values: List[str],
        dataframe: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        tier 3: embed field name + column values using magneto's column encoder.

        args:
            field_name: name of the field
            column_values: full list of column values (vector)
            dataframe: optional dataframe containing the column (for proper encoding)

        returns:
            numpy array of embedding (1, embedding_dim)
        """
        # create encoder with current settings
        encoder = ColumnEncoder(
            self.tokenizer,
            encoding_mode=self.encoding_mode,
            sampling_mode=self.sampling_mode,
            n_samples=self.num_samples
        )

        # if dataframe provided, use magneto's proper encoding
        if dataframe is not None and field_name in dataframe.columns:
            encoded_text = encoder.encode(dataframe, field_name)
        else:
            # fallback: create temp dataframe
            df_temp = pd.DataFrame({field_name: column_values})
            encoded_text = encoder.encode(df_temp, field_name)

        return self.embed([encoded_text])

    def similarity(self, text1: str, text2: str) -> float:
        """
        calculate cosine similarity between two texts.

        args:
            text1: first text
            text2: second text

        returns:
            similarity score (0.0-1.0)
        """
        emb1 = self.embed([text1])
        emb2 = self.embed([text2])

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

    def match_schemas(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        use_reranking: bool = False
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        """
        match columns between source and target dataframes using magneto.

        args:
            source_df: source dataframe
            target_df: target dataframe
            use_reranking: whether to use bipartite reranking

        returns:
            valentine-format matches: {((src_table, src_col), (tgt_table, tgt_col)): score}
        """
        # create magneto matcher with current params
        params = self.params.copy()
        params["use_bp_reranker"] = use_reranking

        matcher = Magneto(**params)
        matches = matcher.get_matches(source_df, target_df)

        return matches