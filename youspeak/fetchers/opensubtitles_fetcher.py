"""Direct web scraper for OpenSubtitles.org to retrieve subtitles by IMDB ID.

This module scrapes OpenSubtitles.org search results and downloads subtitle files
directly without using the Subliminal library.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import chardet  # type: ignore
import cloudscraper  # type: ignore
from bs4 import BeautifulSoup

from ..data.storage import determine_subtitle_path, ensure_parent_dir


# Common subtitle file extensions
SUBTITLE_EXTENSIONS = {'.srt', '.sub', '.ass', '.ssa', '.vtt', '.sbv', '.txt'}


def _normalize_imdb_id(imdb_id: str) -> str:
    """Convert IMDB ID to the format expected by OpenSubtitles (strip 'tt' and leading zeros).
    
    Examples:
        'tt0123456' -> '123456'
        '0123456' -> '123456'
        '123456' -> '123456'
    """
    # Remove 'tt' prefix if present
    if imdb_id.startswith('tt'):
        imdb_id = imdb_id[2:]
    
    # Remove leading zeros
    imdb_id = imdb_id.lstrip('0')
    
    # Handle edge case where all zeros
    if not imdb_id:
        imdb_id = '0'
    
    return imdb_id


def _build_search_url(imdb_id: str, language: str = "en") -> str:
    """Build the OpenSubtitles search URL for a given IMDB ID and language.
    
    Args:
        imdb_id: IMDB ID (with or without 'tt' prefix)
        language: ISO language code (default: 'en')
    
    Returns:
        Full search URL
    """
    normalized_id = _normalize_imdb_id(imdb_id)
    return f"https://www.opensubtitles.org/en/search/sublanguageid-{language}/imdbid-{normalized_id}/sort-7/asc-0"


def _create_scraper():
    """Create a reusable cloudscraper instance with browser settings.
    
    Returns:
        Configured cloudscraper instance
    """
    return cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )


def _scrape_subtitle_ids(search_url: str, max_count: int, scraper) -> List[Tuple[str, str]]:
    """Scrape subtitle IDs from OpenSubtitles search results page.
    
    Args:
        search_url: The OpenSubtitles search URL
        max_count: Maximum number of subtitle IDs to extract
        scraper: Cloudscraper instance (maintains session)
    
    Returns:
        List of tuples (subtitle_id, movie_name) extracted from the page
    """
    try:
        response = scraper.get(search_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Failed to fetch search results: {e}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links with class "bnone"
    bnone_links = soup.find_all('a', class_='bnone')
    
    subtitle_info: List[Tuple[str, str]] = []
    
    # Pattern to match: /en/subtitles/[SUB-ID]/[MOVIENAME]
    pattern = re.compile(r'/en/subtitles/(\d+)/(.+)$')
    
    for link in bnone_links:
        href = link.get('href', '')
        match = pattern.search(href)
        if match:
            sub_id = match.group(1)
            movie_name = match.group(2)
            subtitle_info.append((sub_id, movie_name))
            
            if len(subtitle_info) >= max_count:
                break
    
    return subtitle_info


def _extract_subtitle_from_archive(content: bytes, sub_id: str) -> Tuple[bytes, str]:
    """Extract subtitle file from a ZIP archive and convert to UTF-8.
    
    Args:
        content: Raw ZIP archive content
        sub_id: Subtitle ID (for error messages)
    
    Returns:
        Tuple of (subtitle_content_utf8, extension) where extension includes the dot (e.g., '.srt')
    
    Raises:
        ValueError: If no subtitle file is found in the archive
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        # List all files in the archive
        file_list = zf.namelist()
        print(f"  ZIP contains {len(file_list)} file(s): {file_list}")
        
        # Find subtitle files (prioritize .srt)
        subtitle_files = [
            f for f in file_list 
            if Path(f).suffix.lower() in SUBTITLE_EXTENSIONS
        ]
        
        if not subtitle_files:
            raise ValueError(f"No subtitle files found in ZIP for subtitle {sub_id}")
        
        # Prefer .srt files, otherwise take the first subtitle file
        chosen_file = next(
            (f for f in subtitle_files if Path(f).suffix.lower() == '.srt'),
            subtitle_files[0]
        )
        
        print(f"  Extracting: {chosen_file}")
        subtitle_content = zf.read(chosen_file)
        extension = Path(chosen_file).suffix.lower()
        
        # Detect encoding and convert to UTF-8
        detected = chardet.detect(subtitle_content)
        encoding = detected.get('encoding', 'utf-8')
        confidence = detected.get('confidence', 0)
        
        print(f"  Detected encoding: {encoding} (confidence: {confidence:.2%})")
        
        try:
            # Decode with detected encoding and re-encode as UTF-8
            text = subtitle_content.decode(encoding)
            subtitle_content_utf8 = text.encode('utf-8')
            return subtitle_content_utf8, extension
        except (UnicodeDecodeError, LookupError) as e:
            print(f"  Warning: Failed to decode with {encoding}, falling back to UTF-8 with error handling")
            # Fallback: decode with UTF-8, replacing errors
            text = subtitle_content.decode('utf-8', errors='replace')
            subtitle_content_utf8 = text.encode('utf-8')
            return subtitle_content_utf8, extension


def _download_subtitle(sub_id: str, scraper) -> bytes:
    """Download a subtitle file given its ID.
    
    Args:
        sub_id: The OpenSubtitles subtitle ID
        scraper: Cloudscraper instance (maintains session)
    
    Returns:
        Raw subtitle file content (typically compressed)
    
    Raises:
        ConnectionError: If download fails
    """
    # Note: The vrf token might change over time. If downloads fail in the future,
    # this might need to be updated or extracted from the page dynamically.
    download_url = f"https://www.opensubtitles.org/en/download/vrf-108d030f/sub/{sub_id}"
    
    try:
        response = scraper.get(download_url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise ConnectionError(f"Failed to download subtitle {sub_id}: {e}")


def fetch_subtitles_from_opensubtitles(
    imdb_id: str,
    language: str = "eng",
    max_count: int = 10,
    platform: str = "web",
    platform_id: Optional[str] = None,
    title: Optional[str] = None,
    delay_seconds: float = 0.7,
) -> List[Path]:
    """Fetch multiple subtitles from OpenSubtitles.org by IMDB ID.
    
    This function scrapes OpenSubtitles.org search results and downloads
    up to max_count subtitle files.
    
    Args:
        imdb_id: IMDB ID (with or without 'tt' prefix, e.g., 'tt0123456' or '123456')
        language: ISO language code (default: 'eng')
        max_count: Maximum number of subtitles to download (default: 10)
        platform: Platform identifier for storage (default: 'web')
        platform_id: Optional platform-specific ID
        title: Optional title for storage organization
        delay_seconds: Delay between downloads in seconds (default: 0.7, set to 0 to disable)
    
    Returns:
        List of Path objects pointing to saved subtitle files
    
    Raises:
        ConnectionError: If scraping or downloading fails
        FileNotFoundError: If no subtitles are found
    
    Example:
        >>> paths = fetch_subtitles_from_opensubtitles('tt0903747', language='eng', max_count=5)
        >>> print(f"Downloaded {len(paths)} subtitle files")
    """
    # Create a single scraper instance to reuse session/cookies
    scraper = _create_scraper()
    
    # Build search URL
    search_url = _build_search_url(imdb_id, language)
    print(f"Searching OpenSubtitles: {search_url}")
    
    # Scrape subtitle IDs
    subtitle_info = _scrape_subtitle_ids(search_url, max_count, scraper)
    
    if not subtitle_info:
        raise FileNotFoundError(
            f"No subtitles found for IMDB ID {imdb_id} in language '{language}'"
        )
    
    print(f"Found {len(subtitle_info)} subtitle(s) to download")
    
    # Determine storage location
    base_path = determine_subtitle_path(
        platform=platform,
        platform_id=platform_id or _normalize_imdb_id(imdb_id),
        title=title or f"imdb_{_normalize_imdb_id(imdb_id)}",
        language=language,
        preferred_ext="srt",
    )
    
    # Create candidates directory similar to subliminal_fetcher
    candidates_dir = base_path.parent / "opensubtitles_candidates"
    ensure_parent_dir(candidates_dir / "placeholder")
    
    saved_paths: List[Path] = []
    
    # Download each subtitle
    for idx, (sub_id, movie_name) in enumerate(subtitle_info, 1):
        try:
            print(f"Downloading subtitle {idx}/{len(subtitle_info)} (ID: {sub_id})...")
            
            # Download the subtitle archive (reusing scraper session)
            archive_content = _download_subtitle(sub_id, scraper)
            
            # Extract subtitle from archive (ZIP/GZIP)
            try:
                subtitle_content, extension = _extract_subtitle_from_archive(archive_content, sub_id)
            except Exception as extract_error:
                print(f"  Warning: Failed to extract subtitle from archive: {extract_error}")
                # Continue with remaining downloads
                continue
            
            # Save to disk with the correct extension
            filename = f"{idx:02d}-opensubtitles-{sub_id}{extension}"
            out_path = candidates_dir / filename
            
            out_path.write_bytes(subtitle_content)
            saved_paths.append(out_path)
            
            print(f"  Saved to: {out_path}")
            
            # Be polite to the server - add a delay between downloads
            if idx < len(subtitle_info) and delay_seconds > 0:
                time.sleep(delay_seconds)
                
        except Exception as e:
            print(f"  Warning: Failed to download subtitle {sub_id}: {e}")
            # Continue with remaining downloads
            continue
    
    if not saved_paths:
        raise FileNotFoundError(
            f"Failed to download any subtitles for IMDB ID {imdb_id}"
        )
    
    print(f"\nSuccessfully downloaded {len(saved_paths)} subtitle file(s)")
    return saved_paths

