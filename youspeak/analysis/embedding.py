"""Embedding-based similarity computation for cross-language subtitle alignment.

This module provides utilities for computing semantic similarity between subtitle texts
using pre-trained embedding models, enabling alignment across different languages.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class EmbeddingAlignmentConfig:
    """Configuration for embedding-based alignment."""
    # Base alignment parameters
    gap_penalty: float = -0.4
    min_similarity: float = 0.3
    grow_merge_threshold: float = 0.05
    use_banded: bool = True
    band_margin_pct: float = 0.10
    
    # Embedding-specific parameters
    # model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name: str = "thenlper/gte-small"
    model_name: str = "static-similarity-mrl-multilingual-v1" # brrrrr model
    batch_size: int = 32
    max_seq_length: int = 256
    use_gpu: bool = True


class EmbeddingModelManager:
    """Singleton for managing embedding model lifecycle and caching."""
    _instance = None
    _model = None
    _current_config = None
    
    @classmethod
    def get_model(cls, config: EmbeddingAlignmentConfig):
        """Get or create the embedding model for the given configuration."""
        if cls._instance is None:
            cls._instance = cls()
        
        # Check if we need to reload the model (different config)
        if (cls._model is None or 
            cls._current_config is None or 
            cls._current_config.model_name != config.model_name or
            cls._current_config.use_gpu != config.use_gpu):
            
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers package not found. "
                    "Install with: pip install sentence-transformers"
                )
            
            cls._model = SentenceTransformer(config.model_name)
            
            if config.use_gpu:
                try:
                    cls._model = cls._model.cuda()
                except Exception:
                    # Fallback to CPU if GPU not available
                    pass
            
            cls._current_config = config
        
        return cls._model
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache (useful for testing or memory management)."""
        cls._instance = None
        cls._model = None
        cls._current_config = None


def embedding_cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    norm_a: float,
    norm_b: float,
) -> float:
    """Compute cosine similarity between pre-normalized embeddings.
    
    Args:
        embedding_a: First embedding vector
        embedding_b: Second embedding vector  
        norm_a: Precomputed norm of embedding_a
        norm_b: Precomputed norm of embedding_b
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def build_embedding_vectors_batch(
    texts_list: List[List[str]],
    config: EmbeddingAlignmentConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build embedding vectors for all texts in batch for efficiency.
    
    This function precomputes embeddings for all subtitle texts, enabling
    fast vectorized similarity computation during alignment.
    
    Args:
        texts_list: List of text lists, one per subtitle file
        config: Embedding configuration
        
    Returns:
        (embeddings, norms) - embeddings[i] is the embedding matrix for texts_list[i],
                              norms[i] is the norm vector for embeddings[i]
    """
    model = EmbeddingModelManager.get_model(config)
    all_embeddings = []
    all_norms = []
    
    for texts in texts_list:
        if not texts:
            # Handle empty subtitle
            all_embeddings.append(np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32))
            all_norms.append(np.zeros(0, dtype=np.float32))
            continue
        
        # Batch encode all texts for this subtitle
        embeddings = model.encode(
            texts,
            batch_size=config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Compute norms for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
        
        all_embeddings.append(embeddings.astype(np.float32))
        all_norms.append(norms)
    
    return all_embeddings, all_norms


def compute_embedding_pair(
    text_a: str,
    text_b: str,
    config: EmbeddingAlignmentConfig,
) -> float:
    """Compute embedding similarity between two texts on-the-fly.
    
    Used for grow-merge operations where we need to compute similarity
    for concatenated text blocks that weren't precomputed.
    
    Args:
        text_a: First text
        text_b: Second text
        config: Embedding configuration
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    model = EmbeddingModelManager.get_model(config)
    
    # Batch encode both texts together for efficiency
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
    
    emb_a = embeddings[0]
    emb_b = embeddings[1]
    
    # Compute cosine similarity
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    
    return embedding_cosine_similarity(emb_a, emb_b, norm_a, norm_b)
