"""Subtitle fetchers for various sources."""

from .subliminal_fetcher import (
    fetch_with_subliminal,
    fetch_with_subliminal_search,
    fetch_candidates_with_subliminal_search,
)

from .opensubtitles_fetcher import (
    fetch_subtitles_from_opensubtitles,
)

__all__ = [
    "fetch_with_subliminal",
    "fetch_with_subliminal_search",
    "fetch_candidates_with_subliminal_search",
    "fetch_subtitles_from_opensubtitles",
]

