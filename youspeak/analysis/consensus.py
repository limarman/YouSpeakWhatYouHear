"""Consensus timeline (k-of-n) across multiple subtitle candidates.

Takes multiple subtitle files, normalizes segments, merges micro-gaps per
candidate, then computes a consensus timeline where at least k candidates are
"speaking". Outputs merged consensus intervals and basic stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

from .speech import _merge_intervals
from .alignment import align_intervals
from ..parsers.subtitles import parse_srt_bytes, parse_vtt_bytes, Segment


@dataclass
class ConsensusResult:
	intervals: List[Tuple[float, float]]
	num_candidates: int
	k: int
	speech_seconds: float
	speech_minutes: float
	shifts: Optional[List[float]] = None


def _load_segments_from_path(path: Path) -> List[Segment]:
	data = path.read_bytes()
	ext = path.suffix.lower().lstrip(".")
	if ext == "srt":
		return parse_srt_bytes(data)
	elif ext == "vtt":
		return parse_vtt_bytes(data)
	else:
		raise ValueError(f"Unsupported ext: {ext}")


def _segments_to_intervals(segments: Sequence[Segment]) -> List[Tuple[float, float]]:
	return [(s.start_seconds, s.end_seconds) for s in segments if s.end_seconds > s.start_seconds]


def compute_consensus(
	paths: Sequence[Path],
	*,
	k: int,
	micro_gap_seconds: float = 0.2,
	min_interval_seconds: float = 0.3,
	align_before_consensus: bool = False,
	align_dt: float = 0.1,
	align_max_lag_seconds: float = 300.0,
) -> ConsensusResult:
	"""Compute k-of-n consensus intervals from multiple subtitle files."""
	if not paths:
		raise ValueError("No subtitle files provided")
	if k < 1 or k > len(paths):
		raise ValueError("k must be between 1 and number of paths")

	# 1) Load and pre-merge per candidate
	per_candidate: List[List[Tuple[float, float]]] = []
	for p in paths:
		segments = _load_segments_from_path(p)
		intervals = _segments_to_intervals(segments)
		merged = _merge_intervals(intervals, micro_gap_seconds=micro_gap_seconds)
		per_candidate.append(merged)

	# Optional: align candidates by simple FFT lag + unweighted LS
	shifts_list: Optional[List[float]] = None
	if align_before_consensus and len(per_candidate) >= 2:
		aligned_intervals, _shifts = align_intervals(
			per_candidate,
			dt=align_dt,
			max_lag_seconds=align_max_lag_seconds,
		)
		per_candidate = aligned_intervals
		shifts_list = [float(x) for x in _shifts.tolist()]

	# 2) Line sweep over all boundaries with coverage counting
	events: List[Tuple[float, int]] = []  # (time, +1 start / -1 end)
	for intervals in per_candidate:
		for s, e in intervals:
			events.append((s, +1))
			events.append((e, -1))
	if not events:
		return ConsensusResult(intervals=[], num_candidates=len(paths), k=k, speech_seconds=0.0, speech_minutes=0.0, shifts=shifts_list)
	events.sort(key=lambda x: (x[0], -x[1]))  # starts before ends at same timestamp

	consensus_raw: List[Tuple[float, float]] = []
	coverage = 0
	current_start: float | None = None
	for t, delta in events:
		prev_coverage = coverage
		coverage += delta
		# entering consensus region
		if prev_coverage < k and coverage >= k:
			current_start = t
		# leaving consensus region
		elif prev_coverage >= k and coverage < k:
			if current_start is not None and t > current_start:
				consensus_raw.append((current_start, t))
				current_start = None

	# 3) Post-process consensus: merge micro-gaps and filter short intervals
	consensus_merged = _merge_intervals(consensus_raw, micro_gap_seconds=micro_gap_seconds)
	consensus_filtered = [(s, e) for s, e in consensus_merged if (e - s) >= min_interval_seconds]
	total_seconds = sum(e - s for s, e in consensus_filtered)
	return ConsensusResult(
		intervals=consensus_filtered,
		num_candidates=len(paths),
		k=k,
		speech_seconds=float(total_seconds),
		speech_minutes=float(total_seconds / 60.0),
		shifts=shifts_list,
	)


def export_srt(intervals: Sequence[Tuple[float, float]]) -> str:
	"""Generate a minimal SRT string from intervals with '-' as placeholder text."""
	def fmt(ts: float) -> str:
		ms = int(round((ts - int(ts)) * 1000))
		sec = int(ts) % 60
		minute = (int(ts) // 60) % 60
		hour = int(ts) // 3600
		return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"

	lines: List[str] = []
	for i, (s, e) in enumerate(intervals, 1):
		lines.append(str(i))
		lines.append(f"{fmt(s)} --> {fmt(e)}")
		lines.append("-")
		lines.append("")
	return "\n".join(lines)


def export_anchored_srt(
	paths: Sequence[Path],
	intervals: Sequence[Tuple[float, float]],
	*,
	anchor_index: int = 0,
	min_overlap_seconds: float = 0.05,
) -> str:
	"""Export consensus intervals as SRT using text from the anchor subtitle file.

	- The anchor is chosen by ``anchor_index`` in ``paths`` (default first file)
	- For each consensus interval, join texts of anchor cues that overlap the
	  interval by at least ``min_overlap_seconds``; fallback to '-' if none
	"""
	anchor_segments = _load_segments_from_path(paths[anchor_index])

	def fmt(ts: float) -> str:
		ms = int(round((ts - int(ts)) * 1000))
		sec = int(ts) % 60
		minute = (int(ts) // 60) % 60
		hour = int(ts) // 3600
		return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"

	def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
		start = max(a[0], b[0])
		end = min(a[1], b[1])
		return max(0.0, end - start)

	lines: List[str] = []
	for i, (s, e) in enumerate(intervals, 1):
		texts: List[str] = []
		for seg in anchor_segments:
			ov = overlap((s, e), (seg.start_seconds, seg.end_seconds))
			if ov >= min_overlap_seconds:
				if seg.text:
					texts.append(seg.text)
		joined = " ".join(texts).strip() if texts else "-"
		lines.append(str(i))
		lines.append(f"{fmt(s)} --> {fmt(e)}")
		lines.append(joined)
		lines.append("")
	return "\n".join(lines)
