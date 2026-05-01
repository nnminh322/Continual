"""
Extract instruction embeddings using Sentence-BERT.

Uses the same model as SMoLoRA's ins_gen.py:
    sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
"""
from __future__ import annotations
from typing import List, Union

import numpy as np


try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


class SentenceBERTExtractor:
    """
    Extract instruction embeddings using Sentence-BERT.

    Matches SMoLoRA's ins_gen.py approach:
        1. Encode instruction strings via sentence-transformers/all-MiniLM-L6-v2
        2. Return 384-dim embeddings

    Args:
        model_name: HuggingFace model name. Default: all-MiniLM-L6-v2 (384-dim).
        device: 'cuda', 'cpu', or 'auto'.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
    ):
        if not HAS_SBERT:
            raise ImportError(
                "sentence-transformers is required. Install: pip install sentence-transformers"
            )

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_instructions(
        self,
        instructions: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of instruction strings.

        Args:
            instructions: List of instruction strings.
            batch_size: Batch size.
            normalize: If True, L2-normalize embeddings.

        Returns:
            (N, 384) float32 array.
        """
        embeddings = self.model.encode(
            instructions,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(
        self,
        instruction: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a single instruction string.

        Args:
            instruction: Instruction string.
            normalize: If True, L2-normalize.

        Returns:
            (384,) float32 array.
        """
        emb = self.model.encode(
            [instruction],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return emb[0].astype(np.float32)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def __repr__(self) -> str:
        return f"SentenceBERTExtractor(dim={self.embedding_dim})"
