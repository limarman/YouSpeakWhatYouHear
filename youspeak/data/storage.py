"""Utilities for safe file naming and locating subtitle files under the data directory.

This module provides helpers to generate filesystem-safe slugs and to compute
canonical storage paths for subtitle files regardless of provider (YouTube,
Spotify, local, etc.).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from ..config import SUBTITLES_DIR


_slug_cleanup_re = re.compile(r"[^a-z0-9\-]+")


def safe_slug(text: str, max_len: int = 80) -> str:
	"""Return a filesystem-safe slug generated from arbitrary text.

	- Collapses whitespace into single dashes
	- Removes non-alphanumeric characters (except dashes)
	- Trims repeated dashes and leading/trailing dashes
	- Truncates to ``max_len`` characters
	"""
	base = text.lower().strip()
	base = re.sub(r"\s+", "-", base)
	base = _slug_cleanup_re.sub("-", base)
	base = re.sub(r"-+", "-", base).strip("-")
	return base[:max_len] if len(base) > max_len else base


def _short_hash(s: str, n: int = 8) -> str:
	"""Return a short deterministic hex hash for ``s`` used to disambiguate slugs."""
	return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def determine_subtitle_path(
	*,
	platform: str,
	platform_id: Optional[str],
	title: Optional[str],
	language: str,
	preferred_ext: str = "srt",
) -> Path:
	"""Compute the canonical storage path for a subtitle file.

	The path groups files under ``SUBTITLES_DIR/<platform>/<leaf>/<language>.<ext>``.
	If ``platform_id`` is not available, a slug of ``title`` plus a short hash is
	used as a stable fallback. This function does not create directories.
	"""
	platform_dir = SUBTITLES_DIR / safe_slug(platform or "unknown")
	if platform_id:
		leaf = safe_slug(platform_id)
	else:
		fallback = safe_slug(title or "untitled")
		leaf = f"{fallback}-{_short_hash(fallback)}"
	return platform_dir / leaf / f"{language}.{preferred_ext}"


def ensure_parent_dir(path: Path) -> None:
	"""Ensure the parent directory for ``path`` exists (idempotent)."""
	path.parent.mkdir(parents=True, exist_ok=True)
