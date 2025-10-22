"""Speech analysis: compute total spoken duration from subtitles.

This module reads SRT/VTT subtitles, normalizes to segments, merges overlapping
intervals and optionally small gaps ("micro-gaps"), and computes total speaking
time in seconds/minutes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..parsers.subtitles import parse_srt_bytes, parse_vtt_bytes, parse_ass_bytes


def _merge_intervals(
	intervals: List[Tuple[float, float]],
	*,
	micro_gap_seconds: float = 0.2,
) -> List[Tuple[float, float]]:
	"""Merge overlapping or near-adjacent intervals.

	- Intervals are [start, end]
	- Overlaps are merged
	- Gaps less than or equal to ``micro_gap_seconds`` are merged to reduce
	  artificial fragmentation from subtitle timing
	"""
	if not intervals:
		return []
	intervals.sort(key=lambda x: x[0])
	merged: List[Tuple[float, float]] = []
	cur_start, cur_end = intervals[0]
	for s, e in intervals[1:]:
		if s <= cur_end + micro_gap_seconds:
			# overlap or near-adjacent (micro-gap)
			if e > cur_end:
				cur_end = e
		else:
			merged.append((cur_start, cur_end))
			cur_start, cur_end = s, e
	merged.append((cur_start, cur_end))
	return merged


def analyze_subtitle_bytes(
	data: bytes,
	ext: str,
	*,
	micro_gap_seconds: float = 0.2,
) -> Dict[str, float]:
	"""Compute speech duration metrics from subtitle bytes for given ext ('srt'|'vtt'|'ass')."""
	if ext.lower() == "srt":
		segments = parse_srt_bytes(data)
	elif ext.lower() == "vtt":
		segments = parse_vtt_bytes(data)
	elif ext.lower() == "ass":
		segments = parse_ass_bytes(data)
	else:
		raise ValueError(f"Unsupported subtitle extension: {ext}")

	intervals = [(s.start_seconds, s.end_seconds) for s in segments if s.end_seconds > s.start_seconds]
	merged = _merge_intervals(intervals, micro_gap_seconds=micro_gap_seconds)
	total_seconds = sum(e - s for s, e in merged)
	return {
		"speech_seconds": float(total_seconds),
		"speech_minutes": float(total_seconds / 60.0),
		"num_segments": float(len(segments)),
		"num_merged_intervals": float(len(merged)),
		"micro_gap_seconds": float(micro_gap_seconds),
	}


def analyze_subtitle_file(path: Path, *, micro_gap_seconds: float = 0.2) -> Dict[str, float]:
	"""Read a subtitle file and compute speech metrics based on its extension."""
	ext = path.suffix.lower().lstrip(".")
	data = path.read_bytes()
	return analyze_subtitle_bytes(data, ext, micro_gap_seconds=micro_gap_seconds)
