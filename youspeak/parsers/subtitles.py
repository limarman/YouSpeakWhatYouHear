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
import pysubs2

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
	# Decode and strip BOM if present
	text_content = data.decode("utf-8", errors="replace")
	if text_content.startswith('\ufeff'):
		text_content = text_content[1:]  # Remove BOM
	
	try:
		for item in srt.parse(text_content):
			start = item.start.total_seconds()
			end = item.end.total_seconds()
			text = (item.content or "").replace("\n", " ").strip()
			if text:
				segments.append(Segment(start, end, text))
	except srt.SRTParseError as e:
		# Handle malformed SRT files by parsing manually
		segments = _parse_srt_manually(text_content)
	
	return segments


def _parse_srt_manually(content: str) -> List[Segment]:
	"""Manually parse SRT content when the standard parser fails."""
	segments: List[Segment] = []
	lines = content.split('\n')
	i = 0
	
	while i < len(lines):
		line = lines[i].strip()
		
		# Skip empty lines
		if not line:
			i += 1
			continue
			
		# Check if this looks like a subtitle number
		if line.isdigit():
			i += 1
			if i >= len(lines):
				break
				
			# Next line should be the timestamp
			timestamp_line = lines[i].strip()
			if '-->' in timestamp_line:
				# Parse timestamp
				try:
					start_str, end_str = timestamp_line.split('-->')
					start_str = start_str.strip()
					end_str = end_str.strip()
					
					# Skip if timestamps are malformed (e.g., negative)
					if start_str.startswith('-') or end_str.startswith('-'):
						i += 1
						continue
					
					start_time = _parse_timestamp(start_str)
					end_time = _parse_timestamp(end_str)
					
					if start_time is None or end_time is None:
						i += 1
						continue
					
					# Collect subtitle text
					i += 1
					text_lines = []
					while i < len(lines) and lines[i].strip():
						text_lines.append(lines[i].strip())
						i += 1
					
					if text_lines:
						text = ' '.join(text_lines)
						segments.append(Segment(start_time, end_time, text))
						
				except (ValueError, IndexError):
					# Skip malformed entries
					i += 1
					continue
			else:
				i += 1
		else:
			i += 1
	
	return segments


def _parse_timestamp(timestamp_str: str) -> Optional[float]:
	"""Parse SRT timestamp string to seconds."""
	try:
		# Handle format: HH:MM:SS,mmm
		if ',' in timestamp_str:
			time_part, ms_part = timestamp_str.split(',')
			ms = int(ms_part)
		else:
			time_part = timestamp_str
			ms = 0
		
		parts = time_part.split(':')
		if len(parts) == 3:
			hours, minutes, seconds = map(int, parts)
			total_seconds = hours * 3600 + minutes * 60 + seconds + ms / 1000.0
			return total_seconds
	except (ValueError, IndexError):
		pass
	
	return None


def parse_vtt_bytes(data: bytes) -> List[Segment]:
	"""Parse VTT bytes into normalized segments."""
	segments: List[Segment] = []
	# Decode and strip BOM if present
	text_content = data.decode("utf-8", errors="replace")
	if text_content.startswith('\ufeff'):
		text_content = text_content[1:]  # Remove BOM
	
	vtt = webvtt.read_buffer(io.StringIO(text_content))
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


def parse_ass_bytes(data: bytes) -> List[Segment]:
	"""Parse ASS/SSA bytes into normalized segments."""
	segments: List[Segment] = []
	# Decode and strip BOM if present
	text_content = data.decode("utf-8", errors="replace")
	if text_content.startswith('\ufeff'):
		text_content = text_content[1:]  # Remove BOM
	
	# Parse using pysubs2
	subs = pysubs2.SSAFile.from_string(text_content)
	for line in subs:
		start = line.start / 1000.0  # pysubs2 uses milliseconds
		end = line.end / 1000.0
		# Remove ASS formatting tags and normalize text
		text = line.plaintext.replace("\n", " ").replace("\\N", " ").strip()
		if text:
			segments.append(Segment(start, end, text))
	
	return segments


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
	elif ext == "ass":
		_ = parse_ass_bytes(data)
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
