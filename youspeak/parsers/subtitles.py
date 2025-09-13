"""Subtitle parsers for common formats (SRT, VTT) with normalization.

This module focuses on parsing/normalizing subtitle files, independent of how
files are fetched. A small ingestion helper is provided for convenience.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import srt
import webvtt

from ..data.storage import determine_subtitle_path, ensure_parent_dir
from ..data.db import upsert_content
from datetime import datetime


@dataclass
class Segment:
	start_seconds: float
	end_seconds: float
	text: str


def parse_srt_bytes(data: bytes) -> List[Segment]:
	"""Parse SRT bytes into normalized segments."""
	segments: List[Segment] = []
	for item in srt.parse(data.decode("utf-8", errors="replace")):
		start = item.start.total_seconds()
		end = item.end.total_seconds()
		text = (item.content or "").replace("\n", " ").strip()
		if text:
			segments.append(Segment(start, end, text))
	return segments


def parse_vtt_bytes(data: bytes) -> List[Segment]:
	"""Parse VTT bytes into normalized segments."""
	segments: List[Segment] = []
	vtt = webvtt.read_buffer(io.StringIO(data.decode("utf-8", errors="replace")))
	for caption in vtt:
		start = _vtt_ts_to_seconds(caption.start)
		end = _vtt_ts_to_seconds(caption.end)
		text = (caption.text or "").replace("\n", " ").strip()
		if text:
			segments.append(Segment(start, end, text))
	return segments


def _vtt_ts_to_seconds(ts: str) -> float:
	# WebVTT uses HH:MM:SS.mmm
	parts = ts.split(":")
	if len(parts) != 3:
		return 0.0
	h, m = int(parts[0]), int(parts[1])
	s = float(parts[2])
	return h * 3600 + m * 60 + s


def ingest_subtitle_from_source(
	*,
	source: str,
	language: str,
	platform: str,
	platform_id: Optional[str],
	title: Optional[str],
	ext: str,
	data: bytes,
) -> Tuple[int, Path]:
	"""Write subtitle bytes to storage and upsert content in DB.

	This helper assumes content was fetched elsewhere. It does not parse the
	segments, only validates basic format via parser and saves.
	"""
	# Validate by parsing
	if ext == "srt":
		_ = parse_srt_bytes(data)
	elif ext == "vtt":
		_ = parse_vtt_bytes(data)
	else:
		raise ValueError(f"Unsupported ext: {ext}")

	save_path = determine_subtitle_path(
		platform=platform,
		platform_id=platform_id,
		title=title,
		language=language,
		preferred_ext=ext,
	)
	ensure_parent_dir(save_path)
	save_path.write_bytes(data)

	content_id = upsert_content(
		{
			"platform": platform,
			"platform_id": platform_id,
			"url": source,
			"title": title,
			"language": language,
			"subtitle_path": str(save_path),
			"fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
			"extra_json": None,
		}
	)
	return content_id, save_path
