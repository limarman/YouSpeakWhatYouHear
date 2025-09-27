"""Core data types for the YouSpeak subtitle processing pipeline.

This module defines the fundamental data structures used throughout the
normalization and alignment pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Subtitle:
    """A subtitle file with timing and text content.
    
    This represents a single subtitle file at any stage of processing.
    The text content may be original, normalized, or otherwise transformed
    depending on the pipeline stage.
    
    Attributes:
        source_file: Path or identifier of the original subtitle file
        intervals: List of (start_seconds, end_seconds) timing pairs
        texts: List of text content for each subtitle cue
        
    Invariant: len(intervals) == len(texts)
    """
    source_file: str
    intervals: List[Tuple[float, float]]  # (start_seconds, end_seconds)
    texts: List[str]


@dataclass  
class BlockAlignment:
    """Complete pairwise alignment result between two subtitle files.
    
    This represents the output of the Needleman-Wunsch + local grow-merge
    algorithm, capturing many-to-many block matches between subtitle cues.
    
    Attributes:
        file_a: Identifier for the first subtitle file
        file_b: Identifier for the second subtitle file  
        num_blocks: Number of aligned blocks (for quick access)
        blocks_file_a: Start/end indices for blocks in file A
        blocks_file_b: Start/end indices for blocks in file B
        similarity: Similarity scores for each block pair
        
    Invariant: len(blocks_file_a) == len(blocks_file_b) == len(similarity) == num_blocks
    
    Example:
        Block i represents:
        file_a[blocks_file_a[i][0]:blocks_file_a[i][1]+1] ↔ 
        file_b[blocks_file_b[i][0]:blocks_file_b[i][1]+1]
        with similarity score similarity[i]
    """
    file_a: str
    file_b: str
    num_blocks: int
    blocks_file_a: List[Tuple[int, int]]  # [(start_idx, end_idx), ...]
    blocks_file_b: List[Tuple[int, int]]  # [(start_idx, end_idx), ...]  
    similarity: List[float]               # [score, ...]
