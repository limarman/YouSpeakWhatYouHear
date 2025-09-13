"""Fetcher using Subliminal to retrieve subtitles for media, by path or search.

Subliminal searches subtitle providers (e.g., OpenSubtitles) for a given video
file or via an episode-like search query, downloading matching subtitles in the
requested language.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Sequence

from babelfish import Language  # type: ignore
from subliminal import download_best_subtitles, list_subtitles, download_subtitles, region, save_subtitles, compute_score, Video  # type: ignore

from ..parsers.subtitles import ingest_subtitle_from_source
from ..data.storage import determine_subtitle_path, ensure_parent_dir


# Configure cache for subliminal (in-memory is fine for MVP)
region.configure('dogpile.cache.memory')


def _bf_lang(code: str) -> Language:
	"""Convert common language codes (e.g., 'en', 'en-US', 'eng') to Babelfish Language."""
	try:
		return Language.fromietf(code)
	except Exception:
		return Language(code)


def _apply_imdb(video: Video, imdb_id: Optional[str]) -> None:
	"""Attach IMDb ID to the Video object to influence matching/scoring."""
	if imdb_id:
		setattr(video, "imdb_id", imdb_id)


def _subtitle_meta(sub, video) -> Dict[str, object]:  # type: ignore
	"""Extract extensive metadata from a Subliminal subtitle object, best-effort."""
	meta: Dict[str, object] = {}
	# Common, documented fields
	meta["provider_name"] = getattr(sub, "provider_name", None)
	meta["subtitle_id"] = getattr(sub, "id", None)
	meta["language"] = str(getattr(sub, "language", ""))
	meta["hearing_impaired"] = getattr(sub, "hearing_impaired", None)
	meta["page_link"] = getattr(sub, "page_link", None)
	meta["download_link"] = getattr(sub, "download_link", None)
	meta["date"] = str(getattr(sub, "date", "")) if getattr(sub, "date", None) else None
	meta["matched_by"] = getattr(sub, "matched_by", None)
	meta["release"] = getattr(sub, "release", None)
	meta["encoding"] = getattr(sub, "encoding", None)
	meta["title"] = getattr(sub, "title", None)
	# Matches and score
	try:
		matches = sorted(list(sub.get_matches(video)))  # type: ignore[attr-defined]
	except Exception:
		matches = []
	meta["matches"] = matches
	try:
		score = sub.compute_score(video)  # type: ignore[attr-defined]
	except Exception:
		score = getattr(sub, "score", None)
	meta["score"] = score
	# Fallback: include all public, non-callable attributes for debugging
	try:
		for name in dir(sub):
			if name.startswith("_"):
				continue
			try:
				value = getattr(sub, name)
			except Exception:
				continue
			if callable(value):
				continue
			meta.setdefault(name, value)
	except Exception:
		pass
	return meta


def _sorted_candidates(video: Video, languages: set[Language]) -> List[object]:
	"""List candidates and return them sorted by computed score descending, excluding HI."""
	print("Entering _sorted_candidates")
	all_subs_map = list_subtitles({video}, languages, providers=["opensubtitles"])  # type: ignore[arg-type]

	print("Total number of subtitles: ", len(all_subs_map.get(video, [])))

	candidates = [s for s in list(all_subs_map.get(video, [])) if not getattr(s, "hearing_impaired", False)]

	print("Total number of non-HI subtitles: ", len(candidates))

	def _score_of(s):
		try:
			return compute_score(s, video)
		except Exception:
			return getattr(s, "score", None) or -1
	candidates.sort(key=_score_of, reverse=True)
	return candidates


def fetch_with_subliminal(
	*,
	media_path: str,
	language: str,
	platform: str = "local",
	platform_id: Optional[str] = None,
	title: Optional[str] = None,
	imdb_id: Optional[str] = None,
) -> Tuple[int, Path, Dict[str, object]]:
	"""Fetch the highest-ranked non-HI subtitle for a local media file and register it.

	Returns (content_id, saved_subtitle_path, best_meta).
	"""
	video = Video.fromname(media_path)
	_apply_imdb(video, imdb_id)
	langs = {_bf_lang(language)}
	candidates = _sorted_candidates(video, langs)
	if not candidates:
		raise FileNotFoundError(f"No suitable (non-HI) subtitles found via Subliminal for {media_path} [{language}]")
	best_sub = candidates[0]
	# Download and save
	download_subtitles([best_sub])
	temp_dir = Path(media_path).parent
	save_subtitles(video, [best_sub], directory=str(temp_dir))
	candidate = _find_saved_srt(temp_dir, Path(media_path).stem, language)

	best_meta = _subtitle_meta(best_sub, video)

	content_id, path = ingest_subtitle_from_source(
		source=str(candidate),
		language=language,
		platform=platform,
		platform_id=platform_id or Path(media_path).stem,
		title=title or Path(media_path).stem,
		ext="srt",
		data=candidate.read_bytes(),
	)
	return content_id, path, best_meta


def fetch_with_subliminal_search(
	*,
	query: str,
	language: str,
	platform: str = "web",
	platform_id: Optional[str] = None,
	title: Optional[str] = None,
	imdb_id: Optional[str] = None,
) -> Tuple[int, Path, Dict[str, object]]:
	"""Fetch the highest-ranked non-HI subtitle by search query and register it.

	Uses subliminal.download_best_subtitles for a faster, provider-pooled flow.
	Returns (content_id, saved_subtitle_path, best_meta).
	"""
	video = Video.fromname(query)
	_apply_imdb(video, imdb_id)
	langs = {_bf_lang(language)}

	# Fast path: list+rank+download inside subliminal with a provider pool
	print("Downloading best subtitles for query '{query}' [{language}]")
	best_map = download_best_subtitles({video}, langs, hearing_impaired=False)
	subs = list(best_map.get(video, []))
	if not subs:
		raise FileNotFoundError(f"No suitable (non-HI) subtitles found via Subliminal for query '{query}' [{language}]")

	best_sub = subs[0]

	# Save next to a temp dir, then move into our storage layout
	temp_dir = Path('.')
	save_subtitles(video, [best_sub], directory=str(temp_dir))
	candidate = _find_saved_srt(temp_dir, Path(video.name).stem, language)

	best_meta = _subtitle_meta(best_sub, video)

	content_id, path = ingest_subtitle_from_source(
		source=str(candidate),
		language=language,
		platform=platform,
		platform_id=platform_id or Path(video.name).stem,
		title=title or query,
		ext="srt",
		data=candidate.read_bytes(),
	)
	return content_id, path, best_meta


def fetch_candidates_with_subliminal_search(
	*,
	query: str,
	language: str,
	top_n: int = 3,
	platform: str = "web",
	platform_id: Optional[str] = None,
	title: Optional[str] = None,
	imdb_id: Optional[str] = None,
) -> List[Path]:
	"""Download and save top-N non-HI subtitle candidates (by score) for consensus.

	Returns a list of saved candidate file paths.
	"""
	video = Video.fromname(query)
	_apply_imdb(video, imdb_id)
	langs = {_bf_lang(language)}
	candidates = _sorted_candidates(video, langs)
	if not candidates:
		raise FileNotFoundError(f"No subtitle candidates found via Subliminal for '{query}' [{language}]")
	selected = candidates[: max(1, top_n)]
	# Download their contents into Subtitle.content
	download_subtitles(selected)
	# Determine destination base directory under our storage
	best_path = determine_subtitle_path(
		platform=platform,
		platform_id=platform_id or Path(video.name).stem,
		title=title or query,
		language=language,
		preferred_ext="srt",
	)
	candidates_dir = best_path.parent / "candidates"
	ensure_parent_dir(candidates_dir / "placeholder")
	saved_paths: List[Path] = []
	for idx, sub in enumerate(selected, 1):
		prov = getattr(sub, "provider_name", "prov") or "prov"
		fname = f"{idx:02d}-{prov}.srt"
		out_path = candidates_dir / fname
		data = getattr(sub, "content", None)
		if not data:
			# fallback to text
			text = getattr(sub, "text", None)
			data = (text or "").encode("utf-8")
		out_path.write_bytes(data)
		saved_paths.append(out_path)
	return saved_paths


def _find_saved_srt(directory: Path, stem: str, language: str) -> Path:
	candidate = directory / f"{stem}.{language}.srt"
	if candidate.exists():
		return candidate
	fallback = directory / f"{stem}.srt"
	if fallback.exists():
		return fallback
	raise FileNotFoundError("Could not locate saved .srt file from Subliminal")
