"""Consensus timeline computation across multiple subtitle candidates.

Takes multiple subtitle objects and computes a consensus timeline where a threshold
percentage of subtitles agree that speech is happening. Uses line-scan algorithm
to efficiently compute speech intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from youspeak.util.types import Subtitle


@dataclass
class ConsensusConfig:
    """Configuration for consensus computation."""
    target_agreement_pct: float = 0.66  # Target 66% agreement (2/3 majority)
    max_agreement_pct: float = 0.75     # Cap at 75% to avoid being too strict
    merge_micro_gaps: bool = True
    micro_gap_seconds: float = 0.2
    min_interval_seconds: float = 0.3


def _compute_threshold_k(n: int, target_pct: float, max_pct: float) -> int:
    """Compute optimal k threshold for n subtitles.
    
    Strategy:
    - Target target_pct agreement (e.g., 66% for 2/3 majority)
    - Cap at max_pct to avoid being too strict with many subtitles
    
    Args:
        n: Number of subtitles
        target_pct: Target agreement percentage (0.0 to 1.0)
        max_pct: Maximum agreement percentage cap (0.0 to 1.0)
        
    Returns:
        Number of subtitles that must agree (k)
    """
    if n <= 0:
        return 0
    
    # Target desired percentage
    k = math.ceil(n * target_pct)
    
    # Cap at maximum percentage
    if k / n > max_pct:
        k = math.floor(n * max_pct)
    
    # Ensure at least 1
    return max(1, k)


def _merge_intervals(
    intervals: List[Tuple[float, float]],
    micro_gap_seconds: float = 0.2,
) -> List[Tuple[float, float]]:
    """Merge overlapping or near-adjacent intervals.
    
    Args:
        intervals: List of (start, end) tuples
        micro_gap_seconds: Gaps smaller than this are merged
        
    Returns:
        Merged list of intervals
    """
    if not intervals:
        return []
    
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start <= cur_end + micro_gap_seconds:
            # Overlap or near-adjacent (micro-gap)
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    
    merged.append((cur_start, cur_end))
    return merged


def compute_consensus(
    subtitles: List[Subtitle],
    config: ConsensusConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Subtitle, float, Dict[str, Any]]:
    """Compute consensus timeline from multiple subtitle objects.
    
    Uses a line-scan algorithm to find time intervals where a threshold percentage
    of subtitles agree that speech is happening.
    
    Args:
        subtitles: List of subtitle objects to analyze
        config: Consensus computation configuration
        metadata: Optional metadata dict to update
        
    Returns:
        Tuple of:
        - Consensus subtitle with merged intervals (empty text content)
        - Speech time in seconds (float)
        - Updated metadata dict
    """
    if metadata is None:
        metadata = {}
    
    n = len(subtitles)
    if n == 0:
        raise ValueError("No subtitles provided")
    
    # Compute optimal k threshold
    k = _compute_threshold_k(n, config.target_agreement_pct, config.max_agreement_pct)
    
    # Extract intervals from each subtitle
    per_subtitle_intervals: List[List[Tuple[float, float]]] = []
    for subtitle in subtitles:
        intervals = [(start, end) for start, end in subtitle.intervals if end > start]
        
        # Optionally merge micro-gaps per subtitle first
        if config.merge_micro_gaps:
            intervals = _merge_intervals(intervals, config.micro_gap_seconds)
        
        per_subtitle_intervals.append(intervals)
    
    # Line-scan algorithm: create events for all interval boundaries
    events: List[Tuple[float, int]] = []  # (time, +1 for start / -1 for end)
    for intervals in per_subtitle_intervals:
        for start, end in intervals:
            events.append((start, +1))
            events.append((end, -1))
    
    if not events:
        # No speech detected in any subtitle
        consensus_subtitle = Subtitle(
            source_file="consensus",
            intervals=[],
            texts=[],
            original_texts=None
        )
        
        metadata["consensus"] = {
            "config": config.__dict__.copy(),
            "total_subtitles": n,
            "required_agreement": k,
            "agreement_percentage": round(k / n * 100, 2) if n > 0 else 0.0,
            "num_intervals": 0
        }
        
        return consensus_subtitle, 0.0, metadata
    
    # Sort events: starts before ends at same timestamp
    events.sort(key=lambda x: (x[0], -x[1]))
    
    # Sweep through events and track coverage count
    consensus_raw: List[Tuple[float, float]] = []
    coverage = 0
    current_start: Optional[float] = None
    
    for time, delta in events:
        prev_coverage = coverage
        coverage += delta
        
        # Entering consensus region (coverage reaches k)
        if prev_coverage < k and coverage >= k:
            current_start = time
        
        # Leaving consensus region (coverage drops below k)
        elif prev_coverage >= k and coverage < k:
            if current_start is not None and time > current_start:
                consensus_raw.append((current_start, time))
                current_start = None
    
    # Post-process: merge micro-gaps and filter short intervals
    if config.merge_micro_gaps:
        consensus_merged = _merge_intervals(consensus_raw, config.micro_gap_seconds)
    else:
        consensus_merged = consensus_raw
    
    consensus_filtered = [
        (start, end) 
        for start, end in consensus_merged 
        if (end - start) >= config.min_interval_seconds
    ]
    
    # Compute speech time
    total_seconds = sum(end - start for start, end in consensus_filtered)
    speech_seconds = round(total_seconds, 3)
    
    # Create consensus subtitle (with empty texts)
    consensus_subtitle = Subtitle(
        source_file="consensus",
        intervals=consensus_filtered,
        texts=["" for _ in consensus_filtered],
        original_texts=None
    )
    
    # Update metadata
    metadata["consensus"] = {
        "config": config.__dict__.copy(),
        "total_subtitles": n,
        "required_agreement": k,
        "agreement_percentage": round(k / n * 100, 2) if n > 0 else 0.0,
        "num_intervals": len(consensus_filtered)
    }
    
    return consensus_subtitle, speech_seconds, metadata


def export_consensus_srt(consensus_subtitle: Subtitle, placeholder_text: str = "-") -> str:
    """Export consensus subtitle as SRT format with placeholder text.
    
    Args:
        consensus_subtitle: Consensus subtitle with intervals
        placeholder_text: Text to use for each cue (default: "-")
        
    Returns:
        SRT formatted string
    """
    def fmt_timestamp(ts: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        ms = int(round((ts - int(ts)) * 1000))
        sec = int(ts) % 60
        minute = (int(ts) // 60) % 60
        hour = int(ts) // 3600
        return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"
    
    lines: List[str] = []
    for i, (start, end) in enumerate(consensus_subtitle.intervals, 1):
        lines.append(str(i))
        lines.append(f"{fmt_timestamp(start)} --> {fmt_timestamp(end)}")
        lines.append(placeholder_text)
        lines.append("")
    
    return "\n".join(lines)
