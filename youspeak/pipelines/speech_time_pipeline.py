"""Speech time analysis pipeline.

This pipeline takes an IMDB ID, fetches subtitles from OpenSubtitles,
computes individual speech times, and optionally computes consensus speech time.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from ..analysis.speech import analyze_subtitle_file
from ..analysis.consensus import compute_consensus, ConsensusConfig
from ..analysis.normalization import normalize_subtitle, NormalizationConfig
from ..fetchers.opensubtitles_fetcher import fetch_subtitles_from_opensubtitles
from ..parsers.subtitles import parse_srt_bytes, parse_vtt_bytes, parse_ass_bytes
from ..util.types import Subtitle


@dataclass
class SpeechTimePipelineConfig:
    """Configuration for the speech time pipeline."""
    imdb_id: str
    language: str = "all"  # Default to all languages
    max_subtitles: int = 5
    micro_gap_seconds: float = 0.2
    use_consensus: bool = False
    consensus_threshold: float = 0.66
    output_format: str = "json"  # json, table, csv
    output_file: Optional[str] = None
    delay_seconds: float = 0.7


@dataclass
class SpeechTimeResult:
    """Result from speech time analysis."""
    file_path: str
    speech_seconds: float
    speech_minutes: float
    num_segments: int
    num_merged_intervals: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result from consensus speech time computation."""
    speech_seconds: float
    speech_minutes: float
    num_intervals: int
    agreement_percentage: float
    required_agreement: int
    total_subtitles: int


@dataclass
class SpeechTimeStatistics:
    """Statistical summary of speech times."""
    median_speech_seconds: float
    mean_speech_seconds: float
    std_deviation: float
    min_speech_seconds: float
    max_speech_seconds: float
    median_speech_minutes: float
    mean_speech_minutes: float
    std_deviation_minutes: float
    min_speech_minutes: float
    max_speech_minutes: float
    total_subtitles: int
    successful_subtitles: int


class SpeechTimePipeline:
    """Pipeline for computing speech time from IMDB ID."""
    
    def __init__(self, config: SpeechTimePipelineConfig, console: Optional[Console] = None):
        self.config = config
        self.console = console or Console()
        self.metadata: Dict[str, Any] = {}
    
    def run(self) -> Dict[str, Any]:
        """Run the complete speech time pipeline."""
        start_time = time.time()
        
        try:
            # Phase 1: Fetch subtitles
            self._log_progress("Fetching subtitles from OpenSubtitles...")
            subtitle_paths = self._fetch_subtitles()
            
            if not subtitle_paths:
                raise ValueError(f"No subtitles found for IMDB ID {self.config.imdb_id}")
            
            self._log_progress(f"Found {len(subtitle_paths)} subtitle(s)")
            
            # Phase 2: Compute individual speech times
            self._log_progress("Computing individual speech times...")
            individual_results = self._compute_individual_speech_times(subtitle_paths)
            
            # Phase 3: Compute consensus (if enabled)
            consensus_result = None
            if self.config.use_consensus and len(subtitle_paths) > 1:
                self._log_progress("Computing consensus speech time...")
                consensus_result = self._compute_consensus_speech_time(subtitle_paths)
            
            # Phase 4: Generate statistics
            self._log_progress("Generating statistics...")
            statistics = self._generate_statistics(individual_results)
            
            # Phase 5: Prepare output
            processing_time = time.time() - start_time
            result = self._prepare_output(
                individual_results, consensus_result, statistics, processing_time
            )
            
            # Phase 6: Output results
            self._output_results(result)
            
            return result
            
        except Exception as e:
            self._log_error(f"Pipeline failed: {str(e)}")
            raise
    
    def _log_progress(self, message: str) -> None:
        """Log progress message."""
        if self.console:
            self.console.print(f"[blue]{message}[/blue]")
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.console:
            self.console.print(f"[red]{message}[/red]")
    
    def _fetch_subtitles(self) -> List[Path]:
        """Fetch subtitles from OpenSubtitles."""
        try:
            paths = fetch_subtitles_from_opensubtitles(
                imdb_id=self.config.imdb_id,
                language=self.config.language,
                max_count=self.config.max_subtitles,
                platform="web",
                platform_id=self.config.imdb_id,
                title=f"IMDB_{self.config.imdb_id}",
                delay_seconds=self.config.delay_seconds
            )
            
            self.metadata["fetch"] = {
                "imdb_id": self.config.imdb_id,
                "language": self.config.language,
                "max_subtitles": self.config.max_subtitles,
                "subtitle_paths": [str(p) for p in paths],
                "total_fetched": len(paths)
            }
            
            return paths
            
        except Exception as e:
            self.metadata["fetch"] = {
                "error": str(e),
                "imdb_id": self.config.imdb_id,
                "language": self.config.language
            }
            raise
    
    def _compute_individual_speech_times(self, subtitle_paths: List[Path]) -> List[SpeechTimeResult]:
        """Compute speech time for each subtitle file."""
        results = []
        
        for path in subtitle_paths:
            try:
                # Analyze the subtitle file
                analysis = analyze_subtitle_file(
                    path, 
                    micro_gap_seconds=self.config.micro_gap_seconds
                )
                
                result = SpeechTimeResult(
                    file_path=str(path),
                    speech_seconds=analysis["speech_seconds"],
                    speech_minutes=analysis["speech_minutes"],
                    num_segments=int(analysis["num_segments"]),
                    num_merged_intervals=int(analysis["num_merged_intervals"]),
                    success=True
                )
                
                results.append(result)
                
            except Exception as e:
                result = SpeechTimeResult(
                    file_path=str(path),
                    speech_seconds=0.0,
                    speech_minutes=0.0,
                    num_segments=0,
                    num_merged_intervals=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        self.metadata["individual_analysis"] = {
            "total_files": len(subtitle_paths),
            "successful_files": sum(1 for r in results if r.success),
            "failed_files": sum(1 for r in results if not r.success),
            "micro_gap_seconds": self.config.micro_gap_seconds
        }
        
        return results
    
    def _compute_consensus_speech_time(self, subtitle_paths: List[Path]) -> Optional[ConsensusResult]:
        """Compute consensus speech time from multiple subtitles."""
        try:
            # Load and normalize subtitles for consensus
            subtitles = []
            for path in subtitle_paths:
                try:
                    subtitle = self._load_subtitle(path)
                    if subtitle:
                        subtitles.append(subtitle)
                except Exception:
                    # Skip failed subtitles for consensus
                    continue
            
            if len(subtitles) < 2:
                return None
            
            # Configure consensus
            consensus_config = ConsensusConfig(
                target_agreement_pct=self.config.consensus_threshold,
                merge_micro_gaps=True,
                micro_gap_seconds=self.config.micro_gap_seconds,
                min_interval_seconds=0.3
            )
            
            # Compute consensus
            consensus_subtitle, speech_seconds, metadata = compute_consensus(
                subtitles, consensus_config
            )
            
            consensus_meta = metadata.get("consensus", {})
            
            result = ConsensusResult(
                speech_seconds=speech_seconds,
                speech_minutes=speech_seconds / 60.0,
                num_intervals=consensus_meta.get("num_intervals", 0),
                agreement_percentage=consensus_meta.get("agreement_percentage", 0.0),
                required_agreement=consensus_meta.get("required_agreement", 0),
                total_subtitles=consensus_meta.get("total_subtitles", 0)
            )
            
            self.metadata["consensus"] = {
                "config": consensus_config.__dict__,
                "result": result.__dict__,
                "metadata": metadata
            }
            
            return result
            
        except Exception as e:
            self.metadata["consensus"] = {
                "error": str(e),
                "config": self.config.__dict__
            }
            return None
    
    def _load_subtitle(self, path: Path) -> Optional[Subtitle]:
        """Load and normalize a subtitle file."""
        try:
            data = path.read_bytes()
            ext = path.suffix.lower().lstrip(".")
            
            if ext == "srt":
                segments = parse_srt_bytes(data)
            elif ext == "vtt":
                segments = parse_vtt_bytes(data)
            elif ext == "ass":
                segments = parse_ass_bytes(data)
            else:
                return None
            
            # Create subtitle object
            subtitle = Subtitle(
                source_file=str(path),
                intervals=[(s.start_seconds, s.end_seconds) for s in segments],
                texts=[s.text or "" for s in segments],
                original_texts=[s.text or "" for s in segments]
            )
            
            # Normalize
            norm_config = NormalizationConfig()
            normalized_subtitle, _ = normalize_subtitle(subtitle, norm_config)
            
            return normalized_subtitle
            
        except Exception:
            return None
    
    def _generate_statistics(self, results: List[SpeechTimeResult]) -> SpeechTimeStatistics:
        """Generate statistical summary of speech times."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return SpeechTimeStatistics(
                median_speech_seconds=0.0,
                mean_speech_seconds=0.0,
                std_deviation=0.0,
                min_speech_seconds=0.0,
                max_speech_seconds=0.0,
                median_speech_minutes=0.0,
                mean_speech_minutes=0.0,
                std_deviation_minutes=0.0,
                min_speech_minutes=0.0,
                max_speech_minutes=0.0,
                total_subtitles=len(results),
                successful_subtitles=0
            )
        
        speech_seconds = [r.speech_seconds for r in successful_results]
        speech_minutes = [r.speech_minutes for r in successful_results]
        
        return SpeechTimeStatistics(
            median_speech_seconds=statistics.median(speech_seconds),
            mean_speech_seconds=statistics.mean(speech_seconds),
            std_deviation=statistics.stdev(speech_seconds) if len(speech_seconds) > 1 else 0.0,
            min_speech_seconds=min(speech_seconds),
            max_speech_seconds=max(speech_seconds),
            median_speech_minutes=statistics.median(speech_minutes),
            mean_speech_minutes=statistics.mean(speech_minutes),
            std_deviation_minutes=statistics.stdev(speech_minutes) if len(speech_minutes) > 1 else 0.0,
            min_speech_minutes=min(speech_minutes),
            max_speech_minutes=max(speech_minutes),
            total_subtitles=len(results),
            successful_subtitles=len(successful_results)
        )
    
    def _prepare_output(self, individual_results: List[SpeechTimeResult], 
                       consensus_result: Optional[ConsensusResult],
                       statistics: SpeechTimeStatistics,
                       processing_time: float) -> Dict[str, Any]:
        """Prepare the final output structure."""
        return {
            "imdb_id": self.config.imdb_id,
            "language": self.config.language,
            "statistics": {
                "median_speech_time": statistics.median_speech_seconds,
                "mean_speech_time": statistics.mean_speech_seconds,
                "std_deviation": statistics.std_deviation,
                "min_speech_time": statistics.min_speech_seconds,
                "max_speech_time": statistics.max_speech_seconds,
                "median_speech_minutes": statistics.median_speech_minutes,
                "mean_speech_minutes": statistics.mean_speech_minutes,
                "std_deviation_minutes": statistics.std_deviation_minutes,
                "min_speech_minutes": statistics.min_speech_minutes,
                "max_speech_minutes": statistics.max_speech_minutes,
                "total_subtitles": statistics.total_subtitles,
                "successful_subtitles": statistics.successful_subtitles
            },
            "individual_times": [
                {
                    "file": Path(r.file_path).name,
                    "speech_seconds": r.speech_seconds,
                    "speech_minutes": r.speech_minutes,
                    "num_segments": r.num_segments,
                    "num_merged_intervals": r.num_merged_intervals,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in individual_results
            ],
            "consensus": {
                "speech_seconds": consensus_result.speech_seconds if consensus_result else None,
                "speech_minutes": consensus_result.speech_minutes if consensus_result else None,
                "num_intervals": consensus_result.num_intervals if consensus_result else None,
                "agreement_percentage": consensus_result.agreement_percentage if consensus_result else None,
                "required_agreement": consensus_result.required_agreement if consensus_result else None,
                "total_subtitles": consensus_result.total_subtitles if consensus_result else None
            } if consensus_result else None,
            "metadata": {
                "processing_time": processing_time,
                "config": self.config.__dict__,
                "pipeline_metadata": self.metadata
            }
        }
    
    def _output_results(self, result: Dict[str, Any]) -> None:
        """Output results in the specified format."""
        if self.config.output_format == "json":
            self._output_json(result)
        elif self.config.output_format == "table":
            self._output_table(result)
        elif self.config.output_format == "csv":
            self._output_csv(result)
        else:
            # Default to JSON
            self._output_json(result)
    
    def _output_json(self, result: Dict[str, Any]) -> None:
        """Output results as JSON."""
        if self.config.output_file:
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            self._log_progress(f"Results written to {self.config.output_file}")
        else:
            print(json.dumps(result, indent=2, default=str))
    
    def _output_table(self, result: Dict[str, Any]) -> None:
        """Output results as a formatted table."""
        stats = result["statistics"]
        consensus = result["consensus"]
        
        # Main statistics table
        table = Table(title=f"Speech Time Analysis - IMDB ID: {result['imdb_id']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Language", result["language"])
        table.add_row("Subtitles Analyzed", f"{stats['successful_subtitles']}/{stats['total_subtitles']}")
        table.add_row("", "")  # Empty row for spacing
        
        table.add_row("Median Speech Time", f"{stats['median_speech_minutes']:.1f} minutes ({stats['median_speech_time']:.1f} seconds)")
        table.add_row("Mean Speech Time", f"{stats['mean_speech_minutes']:.1f} minutes ({stats['mean_speech_time']:.1f} seconds)")
        table.add_row("Std Deviation", f"{stats['std_deviation_minutes']:.1f} minutes ({stats['std_deviation']:.1f} seconds)")
        table.add_row("Range", f"{stats['min_speech_minutes']:.1f} - {stats['max_speech_minutes']:.1f} minutes")
        
        if consensus and consensus["speech_seconds"] is not None:
            table.add_row("", "")  # Empty row for spacing
            table.add_row("Consensus Speech Time", f"{consensus['speech_minutes']:.1f} minutes ({consensus['speech_seconds']:.1f} seconds)")
            table.add_row("Agreement", f"{consensus['agreement_percentage']:.1f}% ({consensus['required_agreement']}/{consensus['total_subtitles']})")
        
        self.console.print(table)
        
        # Individual results table
        if len(result["individual_times"]) > 0:
            individual_table = Table(title="Individual Subtitle Results")
            individual_table.add_column("File", style="cyan")
            individual_table.add_column("Speech Time", style="green")
            individual_table.add_column("Segments", justify="right")
            individual_table.add_column("Status", style="red" if any(not r["success"] for r in result["individual_times"]) else "green")
            
            for item in result["individual_times"]:
                status = "✓" if item["success"] else f"✗ {item['error_message']}"
                individual_table.add_row(
                    item["file"],
                    f"{item['speech_minutes']:.1f} min" if item["success"] else "N/A",
                    str(item["num_segments"]) if item["success"] else "N/A",
                    status
                )
            
            self.console.print(individual_table)
    
    def _output_csv(self, result: Dict[str, Any]) -> None:
        """Output results as CSV."""
        import csv
        
        if not self.config.output_file:
            self.config.output_file = f"speech_times_{result['imdb_id']}.csv"
        
        with open(self.config.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "imdb_id", "language", "file", "speech_seconds", "speech_minutes", 
                "num_segments", "num_merged_intervals", "success", "error_message"
            ])
            
            # Write individual results
            for item in result["individual_times"]:
                writer.writerow([
                    result["imdb_id"],
                    result["language"],
                    item["file"],
                    item["speech_seconds"],
                    item["speech_minutes"],
                    item["num_segments"],
                    item["num_merged_intervals"],
                    item["success"],
                    item["error_message"] or ""
                ])
        
        self._log_progress(f"CSV results written to {self.config.output_file}")
