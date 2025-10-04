"""Token extraction and analysis for subtitle content.

This module handles tokenization of subtitle text for speech speed analysis
and vocabulary frequency metrics.

Example usage with custom frequency configuration:

    from youspeak.analysis.tokens import (
        tokenize_subtitle, 
        compute_frequency_statistics, 
        FrequencyConfig
    )
    
    # Create custom frequency configuration
    freq_config = FrequencyConfig(
        quantiles=[0.05, 0.25, 0.50, 0.75, 0.95],  # Custom percentiles
        bucket_borders=[1.5, 3, 4.5, 6]  # Custom zipf score boundaries
    )
    
    # Tokenize
    tokens, language = tokenize_subtitle(subtitle)
    
    # Compute frequency statistics with custom config
    metadata = {}
    stats, zipf_scores = compute_frequency_statistics(
        tokens, language, freq_config, metadata
    )
    
    # Results will use custom quantiles: p5, p25, median, p75, p95
    # And custom buckets: <1.5, [1.5,3), [3,4.5), [4.5,6), >=6
    # Configuration is logged in metadata['frequency_analysis']['config']
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from wordfreq import tokenize, zipf_frequency
from langdetect import detect, LangDetectException

from youspeak.util.types import Subtitle
from youspeak.analysis.normalization import normalize_subtitle_text


@dataclass
class FrequencyConfig:
    """Configuration for token frequency analysis."""
    quantiles: List[float] = None  # List of percentiles to compute (e.g., [0.10, 0.25, 0.50, 0.75, 0.90])
    bucket_borders: List[float] = None  # List of zipf score boundaries (e.g., [2, 3, 4, 5, 6])
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.quantiles is None:
            self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        if self.bucket_borders is None:
            self.bucket_borders = [2, 3, 4, 5, 6]


def tokenize_subtitle(
    subtitle: Subtitle,
    *,
    already_normalized: bool = False,
    language: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple[List[str], str]:
    """Extract tokens from a subtitle object.
    
    Tokenizes all cues in the subtitle after optional normalization.
    The normalization settings are optimized for tokenization (keeping
    punctuation and spaces for proper word boundary detection).
    
    Args:
        subtitle: Input subtitle object
        already_normalized: If True, skip normalization step
        language: Language code for tokenization (e.g., 'en', 'es', 'de').
                 If None, language will be automatically detected.
        metadata: Optional metadata dict to populate with settings
        
    Returns:
        (tokens, detected_language) tuple where tokens is the list of 
        extracted tokens and detected_language is the language code used
    """
    if metadata is None:
        metadata = {}
    
    # Prepare texts for tokenization
    if already_normalized:
        # Use texts as-is
        texts_to_tokenize = subtitle.texts
        norm_settings = None
    else:
        # Apply normalization with tokenization-friendly settings
        norm_settings = {
            'remove_hearing_impaired': True,
            'strip_speaker_labels': True,
            'normalize_numbers': 'token',
            'drop_punctuation': False,  # Keep for proper tokenization
            'collapse_whitespace': False,  # Keep spaces for word boundaries
        }
        
        texts_to_tokenize = [
            normalize_subtitle_text(text, **norm_settings)
            for text in subtitle.texts
        ]
    
    # Concatenate all cues with spaces
    concatenated_text = ' '.join(texts_to_tokenize)
    
    # Detect language if not provided
    language_detection_method = "manual"
    if language is None:
        try:
            detected_lang = detect(concatenated_text)
            language = detected_lang
            language_detection_method = "automatic"
        except LangDetectException:
            # Fallback to English if detection fails
            language = 'en'
            language_detection_method = "automatic_fallback"
    
    # Tokenize using wordfreq
    tokens = tokenize(concatenated_text, language)
    
    # Update metadata
    if "tokenization" not in metadata:
        metadata["tokenization"] = {
            "language": language,
            "language_detection": language_detection_method,
            "normalized": not already_normalized,
            "files": {}
        }
        if norm_settings:
            metadata["tokenization"]["normalization_config"] = norm_settings
    
    # Store per-file stats
    file_stats = {
        "total_cues": len(subtitle.texts),
        "total_tokens": len(tokens),
        "detected_language": language,
    }
    metadata["tokenization"]["files"][subtitle.source_file] = file_stats
    
    return tokens, language


def tokenize_subtitles(
    subtitles: List[Subtitle],
    *,
    already_normalized: bool = False,
    language: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple[List[List[str]], str, Dict[str, Any]]:
    """Extract tokens from multiple subtitle objects.
    
    Batch tokenization of subtitles. All subtitles are treated uniformly
    (same normalization settings). If language is None, each subtitle's
    language will be automatically detected independently, and the majority
    language will be returned.
    
    Args:
        subtitles: List of subtitle objects to tokenize
        already_normalized: If True, skip normalization for all subtitles
        language: Language code for tokenization (e.g., 'en', 'es', 'de').
                 If None, language will be automatically detected per subtitle.
        metadata: Optional metadata dict to populate with settings
        
    Returns:
        (list_of_token_lists, majority_language, updated_metadata) where:
        - list_of_token_lists: List of token lists (one per subtitle)
        - majority_language: The majority vote language across all subtitles
        - updated_metadata: Metadata including language agreement statistics
    """
    if metadata is None:
        metadata = {}
    
    all_tokens = []
    detected_languages = []
    
    for subtitle in subtitles:
        tokens, detected_lang = tokenize_subtitle(
            subtitle,
            already_normalized=already_normalized,
            language=language,
            metadata=metadata
        )
        all_tokens.append(tokens)
        detected_languages.append(detected_lang)
    
    # Determine majority language
    if detected_languages:
        lang_counter = Counter(detected_languages)
        majority_language, majority_count = lang_counter.most_common(1)[0]
        total_subtitles = len(detected_languages)
        agreement_percentage = (majority_count / total_subtitles) * 100
        
        # Add language agreement statistics to metadata
        if "tokenization" in metadata:
            metadata["tokenization"]["language_agreement"] = {
                "majority_language": majority_language,
                "agreement_count": majority_count,
                "total_subtitles": total_subtitles,
                "agreement_percentage": round(agreement_percentage, 2),
                "language_distribution": dict(lang_counter)
            }
    else:
        # Fallback if no subtitles
        majority_language = language if language else 'en'
    
    return all_tokens, majority_language, metadata


# =============================================================================
# Token Frequency Analysis (Zipf Scores)
# =============================================================================

def compute_token_zipf_scores(tokens: List[str], language: str) -> Dict[str, float]:
    """Compute zipf frequency scores for a list of tokens.
    
    Helper function that returns the zipf score for each unique token.
    Zipf scores typically range from 0 (very rare) to 8+ (very common).
    
    Args:
        tokens: List of tokens to analyze
        language: Language code for frequency lookup (e.g., 'en', 'es', 'de')
        
    Returns:
        Dictionary mapping each unique token to its zipf frequency score
    """
    unique_tokens = set(tokens)
    zipf_scores = {}
    
    for token in unique_tokens:
        zipf_scores[token] = zipf_frequency(token, language)
    
    return zipf_scores


def compute_frequency_statistics(
    tokens: List[str],
    language: str,
    config: FrequencyConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Compute descriptive statistics on token frequency (zipf scores).
    
    Computes robust statistics including quantiles and bucket distributions
    of zipf frequency scores for vocabulary analysis.
    
    Args:
        tokens: List of tokens to analyze
        language: Language code for frequency lookup
        config: Frequency analysis configuration
        metadata: Optional metadata dict to populate with settings
        
    Returns:
        (statistics_dict, zipf_scores_dict) where:
        - statistics_dict: Quantiles, bucket percentages, and other stats
        - zipf_scores_dict: Mapping of tokens to their zipf scores
    """
    if metadata is None:
        metadata = {}
    
    if not tokens:
        # Handle empty token list
        stats = {
            "total_tokens": 0,
            "unique_tokens": 0,
            "quantiles": {},
            "bucket_percentages": {},
            "mean": 0.0
        }
        return stats, {}
    
    # Use configuration from config object
    quantile_percentiles = config.quantiles
    bucket_borders = config.bucket_borders
    
    # Store configuration in metadata for logging
    if 'frequency_analysis' not in metadata:
        metadata['frequency_analysis'] = {}
    metadata['frequency_analysis']['config'] = {
        'quantiles': quantile_percentiles,
        'bucket_borders': bucket_borders
    }
    
    # Get zipf scores for all unique tokens
    zipf_scores = compute_token_zipf_scores(tokens, language)
    
    # Create list of scores (one per token occurrence, not just unique)
    all_scores = [zipf_scores[token] for token in tokens]
    all_scores.sort()
    
    n = len(all_scores)
    
    # Compute quantiles
    def get_quantile(sorted_list: List[float], percentile: float) -> float:
        """Get quantile from sorted list."""
        if not sorted_list:
            return 0.0
        idx = (len(sorted_list) - 1) * percentile
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, len(sorted_list) - 1)
        weight = idx - lower_idx
        return sorted_list[lower_idx] * (1 - weight) + sorted_list[upper_idx] * weight
    
    # Always include min and max
    quantiles = {
        "min": all_scores[0],
        "max": all_scores[-1]
    }
    
    # Add configured quantiles
    for p in quantile_percentiles:
        # Format as p10, p25, p75, p90, etc.
        key = str(p)
        quantiles[key] = get_quantile(all_scores, p)
    
    # Compute bucket distribution based on configured borders
    # bucket_borders = [2, 3, 4, 5, 6] creates buckets:
    # <2, [2,3), [3,4), [4,5), [5,6), >=6
    sorted_borders = sorted(bucket_borders)
    buckets = {}
    
    # Create bucket keys
    bucket_keys = []
    bucket_keys.append(f"zipf_<{sorted_borders[0]}")
    for i in range(len(sorted_borders) - 1):
        bucket_keys.append(f"zipf_{sorted_borders[i]}-{sorted_borders[i+1]}")
    bucket_keys.append(f"zipf_>={sorted_borders[-1]}")
    
    # Initialize bucket counts
    for key in bucket_keys:
        buckets[key] = 0
    
    # Count tokens in each bucket
    for score in all_scores:
        if score < sorted_borders[0]:
            buckets[bucket_keys[0]] += 1
        elif score >= sorted_borders[-1]:
            buckets[bucket_keys[-1]] += 1
        else:
            # Find the appropriate bucket
            for i in range(len(sorted_borders) - 1):
                if sorted_borders[i] <= score < sorted_borders[i + 1]:
                    buckets[bucket_keys[i + 1]] += 1
                    break
    
    # Convert counts to percentages (as decimals)
    bucket_percentages = {
        key: round(count / n, 4) for key, count in buckets.items()
    }
    
    # Compute mean (less robust but useful)
    mean_score = sum(all_scores) / n
    
    # Assemble statistics
    stats = {
        "total_tokens": n,
        "unique_tokens": len(zipf_scores),
        "quantiles": {k: round(v, 3) for k, v in quantiles.items()},
        "bucket_percentages": bucket_percentages,
        "mean": round(mean_score, 3)
    }
    
    return stats, zipf_scores


def compute_frequency_statistics_batch(
    all_tokens: List[List[str]],
    language: str,
    config: FrequencyConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute frequency statistics for multiple subtitle token sets.
    
    Computes per-file statistics and aggregate statistics across all subtitles.
    Uses the same configuration for all files.
    
    Args:
        all_tokens: List of token lists (one per subtitle)
        language: Language code for frequency lookup
        config: Frequency analysis configuration
        metadata: Optional metadata dict to populate with settings
        
    Returns:
        (aggregate_stats, per_file_stats) where:
        - aggregate_stats: Statistics computed over all tokens combined
        - per_file_stats: Dictionary mapping file index to its statistics
    """
    if metadata is None:
        metadata = {}
    
    if not all_tokens:
        return {}, {}
    
    # Compute per-file statistics (using same config)
    per_file_stats = {}
    for i, tokens in enumerate(all_tokens):
        stats, _ = compute_frequency_statistics(tokens, language, config, metadata)
        per_file_stats[f"file_{i}"] = stats
    
    # Compute aggregate statistics (all tokens combined)
    combined_tokens = [token for tokens in all_tokens for token in tokens]
    aggregate_stats, _ = compute_frequency_statistics(combined_tokens, language, config, metadata)
    
    return aggregate_stats, per_file_stats

