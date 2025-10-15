"""Clean alignment module for subtitle synchronization.

This module implements efficient subtitle alignment using:
- Needleman-Wunsch global alignment
- Local grow-merge for many-to-many block matching  
- Hard-anchor piecewise shifts for time synchronization
- Connected component analysis for candidate selection

The module works with Subtitle and BlockAlignment types and maintains
processing metadata throughout the pipeline.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
from collections import Counter

import numpy as np

from youspeak.util.types import Subtitle, BlockAlignment
from youspeak.analysis.embedding import EmbeddingAlignmentConfig, build_embedding_vectors_batch


@dataclass
class NGramAlignmentConfig:
    """Configuration for n-gram based alignment."""
    # Base alignment parameters
    gap_penalty: float = -0.4
    min_similarity: float = 0.3
    grow_merge_threshold: float = 0.05
    use_banded: bool = True
    band_margin_pct: float = 0.10
    
    # N-gram specific parameters
    n_gram_size: int = 3
    use_hashing: bool = True
    hash_dim: int = 32768


# Legacy alias for backward compatibility
BlockAlignmentConfig = NGramAlignmentConfig


# =============================================================================
# HELPER FUNCTIONS (Core algorithms)
# =============================================================================

def _compute_temporal_consistency(subtitle: Subtitle) -> Dict[str, Any]:
    """Compute temporal consistency metrics for a subtitle.
    
    Measures how much the subtitle intervals violate chronological order
    by computing the total "misplacement" - the sum of overlaps where
    a later cue starts before the previous one ends.
    
    Returns:
        Dictionary with:
            - total_misplacement: Total seconds of temporal disorder
            - num_cues: Number of cues in the subtitle
            - num_overlaps: Count of overlapping pairs
            - avg_misplacement_per_cue: Average misplacement per cue
    """
    intervals = subtitle.intervals
    num_cues = len(intervals)
    
    if num_cues <= 1:
        return {
            "total_misplacement": 0.0,
            "num_cues": num_cues,
            "num_overlaps": 0,
            "avg_misplacement_per_cue": 0.0
        }
    
    total_misplacement = 0.0
    num_overlaps = 0
    
    for i in range(num_cues - 1):
        end_current = intervals[i][1]
        start_next = intervals[i+1][0]
        
        if start_next < end_current:
            # Overlap or reversal detected
            misplacement = end_current - start_next
            total_misplacement += misplacement
            num_overlaps += 1
    
    avg_misplacement_per_cue = total_misplacement / num_cues if num_cues > 0 else 0.0
    
    return {
        "total_misplacement": round(total_misplacement, 3),
        "num_cues": num_cues,
        "num_overlaps": num_overlaps,
        "avg_misplacement_per_cue": round(avg_misplacement_per_cue, 6)
    }


def _char_ngram_cosine_similarity(
    text_a: str,
    text_b: str,
    n: int = 3,
    use_hashing: bool = False,
    hash_dim: int = 32768,
) -> float:
    """Compute character n-gram cosine similarity between two normalized texts."""
    if n <= 0:
        return 0.0
    if len(text_a) < n or len(text_b) < n:
        return 0.0

    if not use_hashing:
        # Exact n-gram bag via Counter
        grams_a = Counter(text_a[i : i + n] for i in range(len(text_a) - n + 1))
        grams_b = Counter(text_b[i : i + n] for i in range(len(text_b) - n + 1))
        # Dot product over intersection keys
        common_keys = grams_a.keys() & grams_b.keys()
        dot = sum(grams_a[k] * grams_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in grams_a.values()))
        norm_b = math.sqrt(sum(v * v for v in grams_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    else:
        # Hashed vector representation to fixed dimension
        vec_a = np.zeros(hash_dim, dtype=np.float32)
        vec_b = np.zeros(hash_dim, dtype=np.float32)
        for i in range(len(text_a) - n + 1):
            g = text_a[i : i + n]
            h = hash(g) % hash_dim
            vec_a[h] += 1.0
        for i in range(len(text_b) - n + 1):
            g = text_b[i : i + n]
            h = hash(g) % hash_dim
            vec_b[h] += 1.0
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _build_hashed_ngram_vectors_batch(
    texts_list: List[List[str]],
    n: int = 3,
    hash_dim: int = 32768,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build hashed n-gram vectors for all texts in batch for efficiency.
    
    Returns:
        (vectors, norms) - vectors[i] is the n-gram vector for texts_list[i],
                          norms[i] is the norm of vectors[i]
    """
    all_vectors = []
    all_norms = []
    
    for texts in texts_list:
        # Build vector for this subtitle's texts
        vectors = np.zeros((len(texts), hash_dim), dtype=np.float32)
        
        for i, text in enumerate(texts):
            if len(text) >= n:
                for j in range(len(text) - n + 1):
                    g = text[j : j + n]
                    h = hash(g) % hash_dim
                    vectors[i, h] += 1.0
        
        # Compute norms
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        
        all_vectors.append(vectors)
        all_norms.append(norms)
    
    return all_vectors, all_norms


def _calculate_band_width(m: int, n: int, config: BlockAlignmentConfig) -> int:
    """Calculate adaptive band width for banded Needleman-Wunsch.
    
    Band width = |m - n| + margin
    
    The |m - n| term ensures the path can reach from (0,0) to (m,n).
    The margin allows for local variations (gaps, different line splits).
    
    Args:
        m: Length of first sequence
        n: Length of second sequence
        config: Configuration with band_margin_pct
        
    Returns:
        Band width (distance from diagonal to compute)
    """
    length_diff = abs(m - n)
    avg_length = (m + n) / 2.0
    margin = int(config.band_margin_pct * avg_length)
    return length_diff + margin


def _needleman_wunsch_align(
    texts_a: List[str],
    texts_b: List[str],
    config: Union[NGramAlignmentConfig, EmbeddingAlignmentConfig],
    vectors_a: Optional[np.ndarray] = None,
    norms_a: Optional[np.ndarray] = None,
    vectors_b: Optional[np.ndarray] = None, 
    norms_b: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int | None, int | None]], np.ndarray, Dict[str, Any]]:
    """Needleman-Wunsch global alignment with optional banding and precomputed vectors."""
    m = len(texts_a)
    n = len(texts_b)
    
    # Calculate band width if using banded alignment
    band_width = None
    if config.use_banded:
        band_width = _calculate_band_width(m, n, config)
    
    # Build similarity matrix
    if isinstance(config, EmbeddingAlignmentConfig):
        # Embedding-based similarity
        if vectors_a is not None and vectors_b is not None:
            # Use precomputed embedding vectors with vectorized computation (FAST!)
            # Compute dot products: S[i,j] = dot(vectors_a[i], vectors_b[j])
            S = np.dot(vectors_a, vectors_b.T)
            
            # Normalize by outer product of norms: S[i,j] /= (norms_a[i] * norms_b[j])
            # Replace zero norms with 1 to avoid division by zero
            norms_a_safe = np.where(norms_a == 0.0, 1.0, norms_a)
            norms_b_safe = np.where(norms_b == 0.0, 1.0, norms_b)
            norm_product = np.outer(norms_a_safe, norms_b_safe)
            S = S / norm_product
            
            # Set similarity to 0 where either norm was originally 0
            zero_mask = (norms_a == 0.0)[:, np.newaxis] | (norms_b == 0.0)[np.newaxis, :]
            S = np.where(zero_mask, 0.0, S).astype(np.float32)
        else:
            # Compute embeddings on-the-fly (slower but works)
            from .embedding import EmbeddingModelManager, _embedding_cosine_similarity
            model = EmbeddingModelManager.get_model(config)
            
            # Compute embeddings for both sets of texts
            embeddings_a = model.encode(texts_a, convert_to_numpy=True, show_progress_bar=False)
            embeddings_b = model.encode(texts_b, convert_to_numpy=True, show_progress_bar=False)
            
            # Build similarity matrix manually
            S = np.zeros((m, n), dtype=np.float32)
            for i in range(m):
                for j in range(n):
                    S[i, j] = _embedding_cosine_similarity(embeddings_a[i], embeddings_b[j])
    
    elif isinstance(config, NGramAlignmentConfig):
        # N-gram based similarity
        if config.use_hashing and vectors_a is not None and vectors_b is not None:
            # Use precomputed n-gram vectors with vectorized computation (FAST!)
            # Compute dot products: S[i,j] = dot(vectors_a[i], vectors_b[j])
            S = np.dot(vectors_a, vectors_b.T)
            
            # Normalize by outer product of norms: S[i,j] /= (norms_a[i] * norms_b[j])
            # Replace zero norms with 1 to avoid division by zero
            norms_a_safe = np.where(norms_a == 0.0, 1.0, norms_a)
            norms_b_safe = np.where(norms_b == 0.0, 1.0, norms_b)
            norm_product = np.outer(norms_a_safe, norms_b_safe)
            S = S / norm_product
            
            # Set similarity to 0 where either norm was originally 0
            zero_mask = (norms_a == 0.0)[:, np.newaxis] | (norms_b == 0.0)[np.newaxis, :]
            S = np.where(zero_mask, 0.0, S).astype(np.float32)
        else:
            # Compute similarity on the fly (slower fallback)
            S = np.zeros((m, n), dtype=np.float32)
            for i in range(m):
                for j in range(n):
                    S[i, j] = _char_ngram_cosine_similarity(
                        texts_a[i], texts_b[j], 
                        n=config.n_gram_size,
                        use_hashing=config.use_hashing,
                        hash_dim=config.hash_dim
                    )
    else:
        raise ValueError(f"Unknown config type: {type(config)}")
    
    # Dynamic programming for alignment
    dp = np.full((m + 1, n + 1), -1e9, dtype=np.float32)
    bt_i = np.full((m + 1, n + 1), -1, dtype=np.int32)
    bt_j = np.full((m + 1, n + 1), -1, dtype=np.int32)
    
    dp[0, 0] = 0.0
    
    # Helper to check if cell is in band
    def in_band(i: int, j: int) -> bool:
        if not config.use_banded:
            return True
        return abs(i - j) <= band_width
    
    # Initialize boundaries (within band)
    for i in range(1, m + 1):
        if in_band(i, 0):
            dp[i, 0] = dp[i - 1, 0] + config.gap_penalty
            bt_i[i, 0] = i - 1
            bt_j[i, 0] = 0
    for j in range(1, n + 1):
        if in_band(0, j):
            dp[0, j] = dp[0, j - 1] + config.gap_penalty
            bt_i[0, j] = 0
            bt_j[0, j] = j - 1
    
    # Fill DP table (only cells within band)
    for i in range(1, m + 1):
        # Compute band bounds for this row
        if config.use_banded:
            j_start = max(1, i - band_width)
            j_end = min(n + 1, i + band_width + 1)
        else:
            j_start = 1
            j_end = n + 1
        
        for j in range(j_start, j_end):
            sim = float(S[i - 1, j - 1])
            match = dp[i - 1, j - 1] + sim if sim >= config.min_similarity else -1e12
            delete = dp[i - 1, j] + config.gap_penalty
            insert = dp[i, j - 1] + config.gap_penalty
            
            best = match
            pi, pj = i - 1, j - 1
            if delete > best:
                best = delete
                pi, pj = i - 1, j
            if insert > best:
                best = insert
                pi, pj = i, j - 1
            
            dp[i, j] = best
            bt_i[i, j] = pi
            bt_j[i, j] = pj
    
    # Backtrack to get alignment
    i, j = m, n
    alignment: List[Tuple[int | None, int | None]] = []
    while i > 0 or j > 0:
        pi, pj = bt_i[i, j], bt_j[i, j]
        if pi == i - 1 and pj == j - 1:
            alignment.append((i - 1, j - 1))
        elif pi == i - 1 and pj == j:
            alignment.append((i - 1, None))
        else:
            alignment.append((None, j - 1))
        i, j = pi, pj
    
    alignment.reverse()
    
    # Build minimal metadata
    metadata = {}
    if band_width is not None:
        metadata["band_width"] = band_width
    
    return alignment, S, metadata


def _compute_blocks_growmerge(
    alignment: List[Tuple[int | None, int | None]],
    similarity_matrix: np.ndarray,
    texts_a: List[str],
    texts_b: List[str],
    config: Union[NGramAlignmentConfig, EmbeddingAlignmentConfig],
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Two-pass grow-merge: expand left, then right, absorbing gaps on either side.

    For each seed match (i,j):
    1) LEFT PASS: If the immediate left neighbor is a gap on A or B side, pick the side
       with the higher immediate improvement (if both exist). While the next left neighbor
       on that side is a gap and concatenation improves by >= min_improve, absorb it.
    2) RIGHT PASS: Repeat the same to the right.

    Only gaps are consumed; blocks do not cross other matches. Covered alignment indices
    are marked used to avoid duplicate emission.
    """
    # Similarity computation for grow-merge
    def _strip(s: str) -> str:
        """Strip separators and punctuation for n-gram similarity."""
        import unicodedata
        return "".join(ch for ch in s if (unicodedata.category(ch))[0] not in ("Z", "P"))
    
    def _cos(a_txt: str, b_txt: str) -> float:
        """Compute similarity between concatenated text blocks."""
        if isinstance(config, EmbeddingAlignmentConfig):
            # Use embedding similarity without stripping (preserve natural text)
            from youspeak.analysis.embedding import compute_embedding_pair
            return compute_embedding_pair(a_txt, b_txt, config)
            #return _char_ngram_cosine_similarity(_strip(a_txt), _strip(b_txt), n=3)
        else:
            # Use n-gram similarity with stripping
            return _char_ngram_cosine_similarity(_strip(a_txt), _strip(b_txt), n=config.n_gram_size)
    
    used = [False] * len(alignment)
    blocks: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
    
    for idx, (ai, bj) in enumerate(alignment):
        if used[idx] or ai is None or bj is None:
            continue
        
        i0 = i1 = int(ai)
        j0 = j1 = int(bj)
        left_idx = right_idx = idx
        
        # Join texts with space (as in original)
        a_txt = " ".join(texts_a[i0:i1+1])
        b_txt = " ".join(texts_b[j0:j1+1])
        score = _cos(a_txt, b_txt)
        
        # LEFT PASS
        while True:
            if left_idx - 1 < 0 or used[left_idx - 1]:
                break
            lai, lbj = alignment[left_idx - 1]
            
            # Compute candidate deltas for available sides
            best_side = None
            best_new_score = score
            
            # A-left gap available?
            if lai is not None and lbj is None and int(lai) == i0 - 1:
                a_cand = " ".join(texts_a[i0-1:i1+1])
                scA = _cos(a_cand, b_txt)
                if scA - score >= config.grow_merge_threshold and scA > best_new_score:
                    best_side = "A_LEFT"
                    best_new_score = scA
            
            # B-left gap available?
            if lbj is not None and lai is None and int(lbj) == j0 - 1:
                b_cand = " ".join(texts_b[j0-1:j1+1])
                scB = _cos(a_txt, b_cand)
                if scB - score >= config.grow_merge_threshold and scB > best_new_score:
                    best_side = "B_LEFT"
                    best_new_score = scB
            
            if best_side is None:
                break
            
            if best_side == "A_LEFT":
                i0 -= 1
                left_idx -= 1
                a_txt = " ".join(texts_a[i0:i1+1])
            else:  # B_LEFT
                j0 -= 1
                left_idx -= 1
                b_txt = " ".join(texts_b[j0:j1+1])
            
            score = best_new_score
        
        # RIGHT PASS
        while True:
            if right_idx + 1 >= len(alignment) or used[right_idx + 1]:
                break
            rai, rbj = alignment[right_idx + 1]
            
            best_side = None
            best_new_score = score
            
            # A-right gap available?
            if rai is not None and rbj is None and int(rai) == i1 + 1:
                a_cand = " ".join(texts_a[i0:i1+2])
                scA = _cos(a_cand, b_txt)
                if scA - score >= config.grow_merge_threshold and scA > best_new_score:
                    best_side = "A_RIGHT"
                    best_new_score = scA
            
            # B-right gap available?
            if rbj is not None and rai is None and int(rbj) == j1 + 1:
                b_cand = " ".join(texts_b[j0:j1+2])
                scB = _cos(a_txt, b_cand)
                if scB - score >= config.grow_merge_threshold and scB > best_new_score:
                    best_side = "B_RIGHT"
                    best_new_score = scB
            
            if best_side is None:
                break
            
            if best_side == "A_RIGHT":
                i1 += 1
                right_idx += 1
                a_txt = " ".join(texts_a[i0:i1+1])
            else:  # B_RIGHT
                j1 += 1
                right_idx += 1
                b_txt = " ".join(texts_b[j0:j1+1])
            
            score = best_new_score
        
        # Mark all consumed indices as used
        for k in range(left_idx, right_idx + 1):
            used[k] = True
        
        blocks.append(((i0, i1), (j0, j1), float(score)))
    
    return blocks


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def align_subtitle_pair(
    subtitle_a: Subtitle,
    subtitle_b: Subtitle, 
    config: Union[NGramAlignmentConfig, EmbeddingAlignmentConfig],
) -> BlockAlignment:
    """Align two subtitles using NW + grow-merge to produce block alignment."""
    
    # Precompute similarity vectors based on config type
    if isinstance(config, EmbeddingAlignmentConfig):
        # Precompute embeddings for both subtitles
        from .embedding import build_embedding_vectors_batch
        all_vectors, all_norms = build_embedding_vectors_batch(
            [subtitle_a.texts, subtitle_b.texts], config
        )
        vectors_a, vectors_b = all_vectors[0], all_vectors[1]
        norms_a, norms_b = all_norms[0], all_norms[1]
    elif isinstance(config, NGramAlignmentConfig):
        # Precompute n-gram vectors for both subtitles
        all_vectors, all_norms = _build_hashed_ngram_vectors_batch(
            [subtitle_a.texts, subtitle_b.texts],
            n=config.n_gram_size,
            hash_dim=config.hash_dim
        )
        vectors_a, vectors_b = all_vectors[0], all_vectors[1]
        norms_a, norms_b = all_norms[0], all_norms[1]
    else:
        vectors_a = vectors_b = norms_a = norms_b = None
    
    # Run NW alignment with precomputed vectors
    alignment, similarity_matrix, _ = _needleman_wunsch_align(
        subtitle_a.texts, subtitle_b.texts, config,
        vectors_a, norms_a, vectors_b, norms_b
    )
    
    # Compute grow-merge blocks
    blocks = _compute_blocks_growmerge(
        alignment, similarity_matrix, subtitle_a.texts, subtitle_b.texts, config
    )
    
    # Convert to our clean data structure
    blocks_file_a = []
    blocks_file_b = []
    similarity_scores = []
    
    for (a_range, b_range, score) in blocks:
        blocks_file_a.append(a_range)
        blocks_file_b.append(b_range)
        similarity_scores.append(score)
    
    return BlockAlignment(
        file_a=subtitle_a.source_file,
        file_b=subtitle_b.source_file,
        num_blocks=len(blocks),
        blocks_file_a=blocks_file_a,
        blocks_file_b=blocks_file_b,
        similarity=similarity_scores
    )


def align_subtitle_matrix(
    subtitles: List[Subtitle],
    config: Union[NGramAlignmentConfig, EmbeddingAlignmentConfig],
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[List[BlockAlignment], Dict[str, Any]]:
    """MAIN: Compute all pairwise block alignments efficiently."""
    if metadata is None:
        metadata = {}
    
    start_time = time.time()
    
    # Precompute similarity vectors for efficiency
    precompute_start = time.time()
    if isinstance(config, EmbeddingAlignmentConfig):
        all_vectors, all_norms = build_embedding_vectors_batch(
            [sub.texts for sub in subtitles],
            config
        )
    elif isinstance(config, NGramAlignmentConfig):
        if config.use_hashing:
            all_vectors, all_norms = _build_hashed_ngram_vectors_batch(
                [sub.texts for sub in subtitles],
                n=config.n_gram_size,
                hash_dim=config.hash_dim
            )
        else:
            all_vectors = all_norms = None
    else:
        raise ValueError(f"Unknown config type: {type(config)}")
    precompute_time = time.time() - precompute_start
    
    # Compute all pairwise alignments
    alignments = []
    n = len(subtitles)
    
    nw_time = 0.0
    growmerge_time = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Use precomputed vectors if available
            vectors_a = all_vectors[i] if all_vectors else None
            vectors_b = all_vectors[j] if all_vectors else None
            norms_a = all_norms[i] if all_norms else None
            norms_b = all_norms[j] if all_norms else None
            
            # Run alignment
            nw_start = time.time()
            alignment, similarity_matrix, _ = _needleman_wunsch_align(
                subtitles[i].texts, subtitles[j].texts, config,
                vectors_a, norms_a, vectors_b, norms_b
            )
            nw_time += time.time() - nw_start
            
            # Compute blocks
            gm_start = time.time()
            blocks = _compute_blocks_growmerge(
                alignment, similarity_matrix, 
                subtitles[i].texts, subtitles[j].texts, config
            )
            growmerge_time += time.time() - gm_start
            
            # Convert to BlockAlignment
            blocks_file_a = []
            blocks_file_b = []
            similarity_scores = []
            
            for (a_range, b_range, score) in blocks:
                blocks_file_a.append(a_range)
                blocks_file_b.append(b_range)
                similarity_scores.append(score)
            
            block_alignment = BlockAlignment(
                file_a=subtitles[i].source_file,
                file_b=subtitles[j].source_file,
                num_blocks=len(blocks),
                blocks_file_a=blocks_file_a,
                blocks_file_b=blocks_file_b,
                similarity=similarity_scores
            )
            
            alignments.append(block_alignment)
    
    total_time = time.time() - start_time
    
    # Update metadata
    config_dict = config.__dict__.copy()
    if isinstance(config, EmbeddingAlignmentConfig):
        config_dict["alignment_type"] = "embedding"
    elif isinstance(config, NGramAlignmentConfig):
        config_dict["alignment_type"] = "n_gram"
    
    metadata.setdefault("block_alignment", {}).update({
        "config": config_dict,
        "total_pairs": len(alignments),
        "computation_time": round(total_time, 3),
        "precomputation_time": round(precompute_time, 3),
        "nw_time": round(nw_time, 3),
        "growmerge_time": round(growmerge_time, 3)
    })
    
    return alignments, metadata


def select_candidates(
    subtitles: List[Subtitle],
    block_alignments: List[BlockAlignment],
    threshold: float = 0.85,
    metadata: Optional[Dict[str, Any]] = None,
    return_similarity_matrix: bool = False
) -> Tuple[List[int], Dict[str, Any]]:
    """MAIN: Select largest connected component of high-quality alignments."""
    if metadata is None:
        metadata = {}
    
    n = len(subtitles)
    
    # Build similarity matrix from block alignments
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Create file to index mapping
    file_to_idx = {sub.source_file: i for i, sub in enumerate(subtitles)}
    
    # Fill similarity matrix
    for alignment in block_alignments:
        i = file_to_idx[alignment.file_a]
        j = file_to_idx[alignment.file_b]
        
        # Compute combined score (block similarity * coverage)
        combined_score = _compute_combined_score(alignment, subtitles[i], subtitles[j])
        
        similarity_matrix[i, j] = combined_score
        similarity_matrix[j, i] = combined_score
    
    # Set diagonal to 1.0
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Find largest connected component
    component_indices = _largest_connected_component(similarity_matrix, threshold)
    
    # Compute component quality metrics (edges only, excluding diagonal)
    component_edges = [
        similarity_matrix[i, j] 
        for i in component_indices 
        for j in component_indices 
        if i < j
    ]
    
    if component_edges:
        min_similarity = float(np.min(component_edges))
        max_similarity = float(np.max(component_edges))
        avg_similarity = float(np.mean(component_edges))
    else:
        # Single file in component - no edges
        min_similarity = max_similarity = avg_similarity = None
    
    # Update metadata
    metadata.setdefault("candidate_selection", {}).update({
        "threshold": threshold,
        "total_candidates": n,
        "selected_candidates": len(component_indices),
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "avg_similarity": avg_similarity,
    })
    
    # Add similarity matrix to metadata if requested
    if return_similarity_matrix:
        metadata["similarity_matrix"] = {
            "matrix": similarity_matrix.tolist(),
            "file_names": [sub.source_file for sub in subtitles],
            "threshold": threshold
        }
    
    return component_indices, metadata


def align_to_master(
    subtitles: List[Subtitle],
    candidate_indices: List[int],
    relevant_alignments: List[BlockAlignment],
    hard_anchor_threshold: float = 0.9,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[List[Subtitle], Dict[str, Any]]:
    """MAIN: Align selected candidates to master clock using hard anchors.
    
    Args:
        subtitles: All subtitle files
        candidate_indices: Indices of subtitles in the selected component
        relevant_alignments: BlockAlignment results between candidates
        hard_anchor_threshold: Minimum similarity score for a block to be used as anchor
        metadata: Optional metadata dict to populate
        
    Returns:
        Tuple of (aligned_subtitles, metadata)
    """
    if metadata is None:
        metadata = {}
    
    # Select master by median end time
    master_index = _select_master_by_median_duration(subtitles, candidate_indices)
    master_subtitle = subtitles[master_index]
    
    aligned_subtitles = []
    anchor_counts = {}
    successful_alignments = 0
    failed_alignments = 0
    
    for idx in candidate_indices:
        if idx == master_index:
            # Master stays unchanged
            aligned_subtitles.append(master_subtitle)
            successful_alignments += 1
        else:
            # Find alignment between master and this subtitle
            alignment = _find_alignment_between(
                master_subtitle.source_file,
                subtitles[idx].source_file,
                relevant_alignments
            )
            
            if alignment is None:
                # No alignment found - keep original
                aligned_subtitles.append(subtitles[idx])
                failed_alignments += 1
                continue
            
            # Extract hard anchors from high-confidence blocks
            hard_anchors = _extract_hard_anchors(
                alignment,
                master_subtitle,
                subtitles[idx],
                hard_anchor_threshold
            )
            
            if len(hard_anchors) == 0:
                # No anchors found - keep original
                aligned_subtitles.append(subtitles[idx])
                failed_alignments += 1
                continue
            
            # Apply piecewise shift
            aligned_subtitle = _apply_piecewise_shift(subtitles[idx], hard_anchors)
            aligned_subtitles.append(aligned_subtitle)
            anchor_counts[subtitles[idx].source_file] = len(hard_anchors)
            successful_alignments += 1
    
    # Update metadata
    metadata.setdefault("hard_anchor_alignment", {}).update({
        "hard_anchor_threshold": hard_anchor_threshold,
        "master_file": master_subtitle.source_file,
        "master_index": master_index,
        "successful_alignments": successful_alignments,
        "failed_alignments": failed_alignments,
        "anchor_counts": anchor_counts
    })
    
    return aligned_subtitles, metadata


def compute_temporal_consistency_batch(
    subtitles: List[Subtitle],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute temporal consistency metrics for a batch of subtitles.
    
    Args:
        subtitles: List of subtitle objects to analyze
        metadata: Optional metadata dict to populate
        
    Returns:
        Updated metadata dict with temporal_consistency section
    """
    if metadata is None:
        metadata = {}
    
    # Compute temporal consistency metrics for each subtitle
    temporal_consistency_files = {}
    for subtitle in subtitles:
        consistency = _compute_temporal_consistency(subtitle)
        temporal_consistency_files[subtitle.source_file] = consistency
    
    # Compute summary statistics
    if temporal_consistency_files:
        all_total_misplacements = [m["total_misplacement"] for m in temporal_consistency_files.values()]
        all_avg_per_cue = [m["avg_misplacement_per_cue"] for m in temporal_consistency_files.values()]
        files_with_overlaps = sum(1 for m in temporal_consistency_files.values() if m["num_overlaps"] > 0)
        
        temporal_summary = {
            "avg_total_misplacement": round(sum(all_total_misplacements) / len(all_total_misplacements), 3),
            "avg_misplacement_per_cue": round(sum(all_avg_per_cue) / len(all_avg_per_cue), 6),
            "total_files": len(temporal_consistency_files),
            "files_with_overlaps": files_with_overlaps,
            "max_total_misplacement": round(max(all_total_misplacements), 3),
            "min_total_misplacement": round(min(all_total_misplacements), 3)
        }
    else:
        temporal_summary = {}
    
    metadata["temporal_consistency"] = {
        "files": temporal_consistency_files,
        "summary": temporal_summary
    }
    
    return metadata


def clean_subtitle(
    subtitle: Subtitle,
    supporting_alignments: List[BlockAlignment],
    support_threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Subtitle, Dict[str, Any]]:
    """MAIN: Clean subtitle by removing cues without sufficient support.
    
    Conservative approach: Keep a cue if it has ANY block match with similarity
    >= support_threshold in any of the supporting alignments.
    
    If there are no supporting alignments (single file case), all cues are kept.
    """
    if metadata is None:
        metadata = {}
    
    num_cues = len(subtitle.texts)
    
    # Special case: if there are no supporting alignments (single file case),
    # keep all cues - there's nothing to compare against
    if not supporting_alignments:
        metadata.setdefault("cleaning", {}).update({
            "support_threshold": support_threshold,
            "supporting_alignments": 0,
            "original_cues": num_cues,
            "cleaned_cues": num_cues,
            "removed_cues": 0,
            "removed_indices": [],
            "note": "No cleaning performed - single file with no supporting alignments"
        })
        return subtitle, metadata
    
    # Initialize support mask - all cues are unsupported by default
    has_support = [False] * num_cues
    
    # Check each supporting alignment for block matches
    for alignment in supporting_alignments:
        # Determine which blocks belong to this subtitle
        if alignment.file_a == subtitle.source_file:
            # This subtitle is file_a
            for i in range(alignment.num_blocks):
                if alignment.similarity[i] >= support_threshold:
                    # Mark all cues in this block as supported
                    start_idx, end_idx = alignment.blocks_file_a[i]
                    for cue_idx in range(start_idx, min(end_idx + 1, num_cues)):
                        has_support[cue_idx] = True
        
        elif alignment.file_b == subtitle.source_file:
            # This subtitle is file_b
            for i in range(alignment.num_blocks):
                if alignment.similarity[i] >= support_threshold:
                    # Mark all cues in this block as supported
                    start_idx, end_idx = alignment.blocks_file_b[i]
                    for cue_idx in range(start_idx, min(end_idx + 1, num_cues)):
                        has_support[cue_idx] = True
    
    # Filter to keep only supported cues
    cleaned_intervals = []
    cleaned_texts = []
    cleaned_original_texts = [] if subtitle.original_texts else None
    removed_indices = []
    
    for i in range(num_cues):
        if has_support[i]:
            cleaned_intervals.append(subtitle.intervals[i])
            cleaned_texts.append(subtitle.texts[i])
            if subtitle.original_texts:
                cleaned_original_texts.append(subtitle.original_texts[i])
        else:
            removed_indices.append(i)
    
    # Sort by start time to maintain chronological order
    # (Cleaning may have removed cues, but remaining ones should stay in time order)
    if cleaned_intervals:
        # Create list of (start_time, interval, text, original_text) tuples, sort, then unzip
        if subtitle.original_texts:
            combined = list(zip([interval[0] for interval in cleaned_intervals], cleaned_intervals, cleaned_texts, cleaned_original_texts))
            combined.sort(key=lambda x: x[0])  # Sort by start time
            cleaned_intervals = [item[1] for item in combined]
            cleaned_texts = [item[2] for item in combined]
            cleaned_original_texts = [item[3] for item in combined]
        else:
            combined = list(zip([interval[0] for interval in cleaned_intervals], cleaned_intervals, cleaned_texts))
            combined.sort(key=lambda x: x[0])  # Sort by start time
            cleaned_intervals = [item[1] for item in combined]
            cleaned_texts = [item[2] for item in combined]
    
    # Create cleaned subtitle
    cleaned_subtitle = Subtitle(
        source_file=subtitle.source_file,
        intervals=cleaned_intervals,
        texts=cleaned_texts,
        original_texts=cleaned_original_texts
    )
    
    # Update metadata
    metadata.setdefault("cleaning", {}).update({
        "support_threshold": support_threshold,
        "supporting_alignments": len(supporting_alignments),
        "original_cues": num_cues,
        "cleaned_cues": len(cleaned_texts),
        "removed_cues": len(removed_indices),
        "removed_indices": removed_indices
    })
    
    return cleaned_subtitle, metadata


# =============================================================================
# HELPER FUNCTIONS (Continued)
# =============================================================================

def _compute_combined_score(
    alignment: BlockAlignment, 
    subtitle_a: Subtitle, 
    subtitle_b: Subtitle,
    min_sim_for_coverage: float = 0.3
) -> float:
    """Compute combined score (block similarity * coverage) for an alignment."""
    if alignment.num_blocks == 0:
        return 0.0
    
    # Compute block similarity (weighted average)
    total_weight = 0.0
    weighted_score = 0.0
    
    for i in range(alignment.num_blocks):
        block_score = alignment.similarity[i]
        if block_score >= min_sim_for_coverage:
            # Weight by block duration
            a_start, a_end = alignment.blocks_file_a[i]
            b_start, b_end = alignment.blocks_file_b[i]
            
            a_duration = sum(subtitle_a.intervals[j][1] - subtitle_a.intervals[j][0] 
                           for j in range(a_start, min(a_end + 1, len(subtitle_a.intervals))))
            b_duration = sum(subtitle_b.intervals[j][1] - subtitle_b.intervals[j][0]
                           for j in range(b_start, min(b_end + 1, len(subtitle_b.intervals))))
            
            weight = (a_duration + b_duration) / 2.0
            weighted_score += block_score * weight
            total_weight += weight
    
    if total_weight == 0.0:
        return 0.0
    
    block_similarity = weighted_score / total_weight
    
    # Compute coverage
    matched_duration_a = 0.0
    matched_duration_b = 0.0
    
    for i in range(alignment.num_blocks):
        if alignment.similarity[i] >= min_sim_for_coverage:
            a_start, a_end = alignment.blocks_file_a[i]
            b_start, b_end = alignment.blocks_file_b[i]
            
            a_duration = sum(subtitle_a.intervals[j][1] - subtitle_a.intervals[j][0]
                           for j in range(a_start, min(a_end + 1, len(subtitle_a.intervals))))
            b_duration = sum(subtitle_b.intervals[j][1] - subtitle_b.intervals[j][0]
                           for j in range(b_start, min(b_end + 1, len(subtitle_b.intervals))))
            
            matched_duration_a += a_duration
            matched_duration_b += b_duration
    
    total_duration_a = sum(end - start for start, end in subtitle_a.intervals)
    total_duration_b = sum(end - start for start, end in subtitle_b.intervals)
    
    if total_duration_a == 0.0 or total_duration_b == 0.0:
        return 0.0
    
    coverage_a = matched_duration_a / total_duration_a
    coverage_b = matched_duration_b / total_duration_b
    coverage = (coverage_a + coverage_b) / 2.0
    
    return block_similarity * coverage


def _largest_connected_component(
    similarity_matrix: np.ndarray, 
    threshold: float
) -> List[int]:
    """Find largest connected component in similarity graph."""
    n = similarity_matrix.shape[0]
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                adj[i].append(j)
                adj[j].append(i)
    
    # Find connected components using DFS
    visited = [False] * n
    components = []
    
    def dfs(node: int, component: List[int]):
        visited[node] = True
        component.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    # Return largest component
    if not components:
        return []
    
    return max(components, key=len)


def _select_master_by_median_duration(
    subtitles: List[Subtitle],
    candidate_indices: List[int]
) -> int:
    """Select the subtitle with median end time as the master clock.
    
    Args:
        subtitles: All subtitle files
        candidate_indices: Indices of candidate subtitles to consider
        
    Returns:
        Index of the subtitle with median end time (acts as master clock)
    """
    # Get the end time (last interval's end) for each candidate
    end_times = []
    for idx in candidate_indices:
        if subtitles[idx].intervals:
            last_end_time = subtitles[idx].intervals[-1][1]
            end_times.append((last_end_time, idx))
    
    # Sort by end time and pick the median
    end_times.sort()
    median_position = len(end_times) // 2
    
    return end_times[median_position][1]


def _find_alignment_between(
    file_a: str,
    file_b: str,
    alignments: List[BlockAlignment]
) -> Optional[BlockAlignment]:
    """Find the BlockAlignment between two specific files.
    
    Args:
        file_a: First subtitle file path
        file_b: Second subtitle file path
        alignments: List of all BlockAlignment results
        
    Returns:
        BlockAlignment between the two files, or None if not found
    """
    for alignment in alignments:
        if ((alignment.file_a == file_a and alignment.file_b == file_b) or
            (alignment.file_a == file_b and alignment.file_b == file_a)):
            return alignment
    return None


def _extract_hard_anchors(
    alignment: BlockAlignment,
    subtitle_master: Subtitle,
    subtitle_other: Subtitle,
    threshold: float
) -> List[Tuple[float, float]]:
    """Extract hard anchor time pairs from high-confidence blocks.
    
    Args:
        alignment: BlockAlignment between master and other subtitle
        subtitle_master: The master (reference) subtitle
        subtitle_other: The subtitle to be aligned
        threshold: Minimum similarity score for a block to be used as anchor
        
    Returns:
        List of (time_in_other, time_in_master) tuples, sorted by time_in_other
    """
    anchors = []
    
    for i in range(alignment.num_blocks):
        if alignment.similarity[i] >= threshold:
            # Determine which subtitle is which in the alignment
            if alignment.file_a == subtitle_master.source_file:
                master_range = alignment.blocks_file_a[i]
                other_range = alignment.blocks_file_b[i]
            else:
                master_range = alignment.blocks_file_b[i]
                other_range = alignment.blocks_file_a[i]
            
            master_start_idx, master_end_idx = master_range
            other_start_idx, other_end_idx = other_range
            
            # Validate indices
            if (master_start_idx >= len(subtitle_master.intervals) or 
                master_end_idx >= len(subtitle_master.intervals) or
                other_start_idx >= len(subtitle_other.intervals) or 
                other_end_idx >= len(subtitle_other.intervals)):
                continue
            
            # Compute center time of each block
            master_block_start = subtitle_master.intervals[master_start_idx][0]
            master_block_end = subtitle_master.intervals[master_end_idx][1]
            master_center = (master_block_start + master_block_end) / 2.0
            
            other_block_start = subtitle_other.intervals[other_start_idx][0]
            other_block_end = subtitle_other.intervals[other_end_idx][1]
            other_center = (other_block_start + other_block_end) / 2.0
            
            anchors.append((other_center, master_center))
    
    # Sort by time in other subtitle
    anchors.sort(key=lambda x: x[0])
    
    return anchors


def _apply_piecewise_shift(
    subtitle: Subtitle,
    hard_anchors: List[Tuple[float, float]]
) -> Subtitle:
    """Apply piecewise shift to subtitle based on hard anchors.
    
    Args:
        subtitle: The subtitle to be aligned
        hard_anchors: List of (time_in_subtitle, time_in_master) anchor points, sorted
        
    Returns:
        New Subtitle with shifted intervals
    """
    if len(hard_anchors) == 0:
        # No anchors, return unchanged
        return subtitle
    
    # Build shift values for each anchor
    shifts = [time_master - time_other for time_other, time_master in hard_anchors]
    anchor_times = [time_other for time_other, _ in hard_anchors]
    
    # Apply piecewise shift to each interval
    new_intervals = []
    for start, end in subtitle.intervals:
        # Determine which piece this interval belongs to based on its center
        center = (start + end) / 2.0
        
        # Find the appropriate shift
        shift = shifts[0]  # Default to first shift
        for i in range(len(anchor_times)):
            if center >= anchor_times[i]:
                shift = shifts[i]
            else:
                break
        
        # Apply shift
        new_start = start + shift
        new_end = end + shift
        new_intervals.append((new_start, new_end))
    
    return Subtitle(
        source_file=subtitle.source_file,
        intervals=new_intervals,
        texts=list(subtitle.texts),
        original_texts=subtitle.original_texts  # Pass through as-is
    )


