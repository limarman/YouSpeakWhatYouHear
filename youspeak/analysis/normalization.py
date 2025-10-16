"""Subtitle text normalization for alignment processing.

This module handles all text preprocessing steps needed to prepare subtitles
for similarity comparison and alignment.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from youspeak.util.types import Subtitle


@dataclass
class NormalizationConfig:
    """Configuration for text normalization pipeline."""
    remove_hearing_impaired: bool = True
    strip_speaker_labels: bool = True
    normalize_numbers: str = "keep"  # "keep" | "token"
    drop_punctuation: bool = True
    collapse_whitespace: bool = True  # True for "helloworld", False for "hello world"


def _apply_basic_cleaning(text: str) -> str:
    """Apply basic text cleaning: newlines, unicode normalization, tag removal."""
    if not text:
        return ""
    
    # Unify newlines to spaces early
    s = text.replace("\n", " ")
    
    # Unicode normalization and casefold
    s = unicodedata.normalize("NFKC", s).casefold()
    
    # Remove HTML-like tags
    # Example: "<i>Hello</i>" → " Hello "
    s = re.sub(r"<[^>]+>", " ", s)
    
    # Remove ASS/SSA style tags like {\an8} or {italic}
    # Examples: "{\an8}Text" → " Text", "{italic}Hello" → " Hello"
    s = re.sub(r"\{\\[^}]*\}", " ", s)  # {\an8}, {\pos(10,20)}
    s = re.sub(r"\{[^}]*\}", " ", s)    # {italic}, {b1}
    
    # Remove music glyphs
    s = s.replace("♪", " ").replace("♫", " ")
    
    # Collapse repeated punctuation before other ops
    # Examples: "Hello!!!" → "Hello!", "What???" → "What?"
    s = re.sub(r"([!?.,])\1{1,}", r"\1", s)
    
    return s


def _remove_hearing_impaired_annotations(text: str) -> str:
    """Remove hearing-impaired annotations like [music], (laughs)."""
    s = text
    # Remove [ ... ] and ( ... ) (non-nested simple; repeat until stable)
    # Examples: "[music playing]" → " ", "(door slams)" → " "
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\[[^\[\]]*\]", " ", s)  # [music], [laughs]
        s = re.sub(r"\([^()]*\)", " ", s)    # (sighs), (door closes)
    return s


def _strip_speaker_labels(text: str) -> str:
    """Remove leading speaker labels and dashes (language-agnostic)."""
    s = text.strip()
    
    # Leading dashes like "- ", "– ", "— "
    # Examples: "- Hello there" → "Hello there", "– Yes, indeed" → "Yes, indeed"
    s = re.sub(r"^(?:[-–—]\s*)+", "", s)
    
    # Language-agnostic speaker labels: any sequence ending with ":"
    # Examples: "john: Hello" → "Hello", "王小明: 你好" → "你好", "José María: Hola" → "Hola"
    # Matches: letters, spaces, apostrophes, hyphens, followed by colon and whitespace
    s = re.sub(r"^[\w\s''-]+:\s+", "", s)
    
    return s


def _canonicalize_quotes_and_dashes(text: str) -> str:
    """Standardize various quote and dash characters."""
    replacements = {
        """: '"', """: '"', "„": '"', "«": '"', "»": '"',
        "'": "'", "'": "'",
        "—": "-", "–": "-",
    }
    s = text
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def _normalize_numbers(text: str, mode: str) -> str:
    """Normalize numbers according to the specified mode."""
    if mode == "token":
        # Examples: "I have 5 cats" → "I have <num> cats", "Call 555-1234" → "Call <num>-<num>"
        return re.sub(r"\d+", "<num>", text)
    return text  # "keep" mode - no changes


def _drop_punctuation(text: str) -> str:
    """Remove all punctuation including apostrophes and hyphens."""
    # Remove everything that's not letter/number/space
    # Examples: "Hello, world!" → "Hello  world ", "don't go" → "don t go", "twenty-one" → "twenty one"
    return re.sub(r"[^\w\s]", " ", text)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces into single spaces and trim."""
    # Examples: "Hello   world  " → "Hello world", "\t\nHi  there\n" → "Hi there"
    return re.sub(r"\s+", " ", text).strip()


def _collapse_whitespace(text: str) -> str:
    """Collapse ALL whitespace characters - remove completely for compact comparison."""
    # Examples: "Hello world" → "Helloworld", "don t go there" → "dontgothere"
    return re.sub(r"\s+", "", text)


def normalize_subtitle_text(
    text: str,
    *,
    remove_hearing_impaired: bool = True,
    strip_speaker_labels: bool = True,
    normalize_numbers: str = "keep",  # "keep" | "token"
    drop_punctuation: bool = True,
    collapse_whitespace: bool = False,
) -> str:
    """Normalize subtitle cue text for language-agnostic similarity.

    Steps (configurable):
    - Unicode NFKC, casefold
    - Strip HTML/ASS tags, music glyphs
    - Remove hearing-impaired annotations like [music], (laughs)
    - Drop leading speaker labels/dashes at cue start (language-agnostic)
    - Canonicalize quotes and dashes
    - Normalize numbers (optional)
    - Drop all punctuation (including apostrophes and hyphens)
    - Normalize or collapse whitespace
    """
    # Apply basic cleaning first
    s = _apply_basic_cleaning(text)
    
    # Apply conditional processing
    if remove_hearing_impaired:
        s = _remove_hearing_impaired_annotations(s)
    
    if strip_speaker_labels:
        s = _strip_speaker_labels(s)
    
    # Apply canonicalization
    s = _canonicalize_quotes_and_dashes(s)
    
    # Handle numbers
    s = _normalize_numbers(s, normalize_numbers)
    
    # Handle punctuation
    if drop_punctuation:
        s = _drop_punctuation(s)
    
    # Final whitespace handling
    if collapse_whitespace:
        s = _collapse_whitespace(s)  # Remove all whitespace
    else:
        s = _normalize_whitespace(s)  # Normalize to single spaces
    
    return s


def _filter_empty_cues(subtitle: Subtitle) -> tuple[Subtitle, Dict[str, Any]]:
    """Filter out cues without alphanumeric content.
    
    Removes cues that contain no letters or digits (empty, whitespace-only, or symbol-only).
    Keeps intervals, texts, and original_texts synchronized.
    
    Args:
        subtitle: Subtitle with normalized texts
        
    Returns:
        (filtered_subtitle, filter_stats)
    """
    filtered_intervals = []
    filtered_texts = []
    filtered_original_texts = [] if subtitle.original_texts else None
    removed_count = 0
    
    for i, text in enumerate(subtitle.texts):
        # Keep cue if it has at least one alphanumeric character
        if any(c.isalnum() for c in text):
            filtered_intervals.append(subtitle.intervals[i])
            filtered_texts.append(text)
            if subtitle.original_texts:
                filtered_original_texts.append(subtitle.original_texts[i])
        else:
            removed_count += 1
    
    filtered_subtitle = Subtitle(
        source_file=subtitle.source_file,
        intervals=filtered_intervals,
        texts=filtered_texts,
        original_texts=filtered_original_texts
    )
    
    stats = {
        "cues_before_filtering": len(subtitle.texts),
        "cues_after_filtering": len(filtered_texts),
        "filtered_cues": removed_count
    }
    
    return filtered_subtitle, stats


def normalize_subtitle(
    subtitle: Subtitle, 
    config: NormalizationConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple[Subtitle, Dict[str, Any]]:
    """Normalize all texts in a subtitle object.
    
    Args:
        subtitle: Input subtitle with original text
        config: Normalization configuration
        metadata: Optional metadata dict to update
        
    Returns:
        (normalized_subtitle, updated_metadata)
    """
    if metadata is None:
        metadata = {}
    
    # Normalize each text
    normalized_texts = []
    for text in subtitle.texts:
        normalized = normalize_subtitle_text(
            text,
            remove_hearing_impaired=config.remove_hearing_impaired,
            strip_speaker_labels=config.strip_speaker_labels,
            normalize_numbers=config.normalize_numbers,
            drop_punctuation=config.drop_punctuation,
            collapse_whitespace=config.collapse_whitespace,
        )
        normalized_texts.append(normalized)
    
    # Create new subtitle with normalized texts
    normalized_subtitle = Subtitle(
        source_file=subtitle.source_file,
        intervals=subtitle.intervals.copy(),
        texts=normalized_texts,
        original_texts=subtitle.original_texts
    )
    
    # Filter out empty cues (no alphanumeric content)
    filtered_subtitle, filter_stats = _filter_empty_cues(normalized_subtitle)
    
    # Track per-file stats
    file_stats = {
        "total_cues": len(subtitle.texts),
        "empty_cues_before": sum(1 for t in subtitle.texts if not t.strip()),
        "empty_cues_after": sum(1 for t in normalized_texts if not t.strip()),
        "cues_after_filtering": filter_stats["cues_after_filtering"],
        "filtered_cues": filter_stats["filtered_cues"]
    }
    
    # Initialize normalization metadata structure if needed
    if "normalization" not in metadata:
        metadata["normalization"] = {
            "config": config.__dict__.copy(),
            "files": {}
        }
    
    # Store per-file stats
    metadata["normalization"]["files"][subtitle.source_file] = file_stats
    
    return filtered_subtitle, metadata


def normalize_subtitles(
    subtitles: List[Subtitle],
    config: NormalizationConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple[List[Subtitle], Dict[str, Any]]:
    """Normalize a list of subtitle objects.
    
    Main entry point for the normalization pipeline.
    
    Args:
        subtitles: List of subtitle objects to normalize
        config: Normalization configuration
        metadata: Optional metadata dict to update
        
    Returns:
        (normalized_subtitles, updated_metadata)
    """
    if metadata is None:
        metadata = {}
    
    normalized_subtitles = []
    
    for subtitle in subtitles:
        normalized_sub, updated_meta = normalize_subtitle(subtitle, config, metadata)
        normalized_subtitles.append(normalized_sub)
        # metadata is updated in-place by normalize_subtitle
    
    return normalized_subtitles, metadata
