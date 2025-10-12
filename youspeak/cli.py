"""YouSpeak CLI - Command-line interface for subtitle processing and alignment.

Primary Commands:
  - align-pair: Align two subtitle files using NW + grow-merge
  - align-pipeline: Full pipeline with normalization, alignment, and cleaning
  - consensus: Compute speech time consensus from multiple subtitle files
  - tokenize: Extract tokens from a subtitle file for speech speed analysis
  - preview-html: Generate HTML preview of subtitles
  - init-db, list, ingest-subtitle: Database operations
  - fetch-subliminal-candidates: Fetch subtitle candidates from subliminal search
  - fetch-opensubtitles: Download subtitles directly from OpenSubtitles.org by IMDB ID
"""

from __future__ import annotations

import json
import logging
import typer
from rich import print
from rich.table import Table

from .config import ensure_data_dirs, DATA_DIR, DB_PATH
from .data.db import init_db, list_content
from .parsers.subtitles import ingest_subtitle_from_source
from .viewer.static_viewer import generate_static_preview
from .fetchers.subliminal_fetcher import fetch_candidates_with_subliminal_search
from .fetchers.opensubtitles_fetcher import fetch_subtitles_from_opensubtitles


# Enable INFO-level logging globally so Subliminal's logger.info messages are printed
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command(name="preview-html")
def preview_html_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt) to preview"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing .srt/.vtt"),
	recursive: bool = typer.Option(False, help="Recursively include files under --dir"),
	out: str = typer.Option("preview", help="Output directory for static viewer"),
	title: str = typer.Option("Subtitle Preview", help="Page title"),
	open_browser: bool = typer.Option(False, "--open", help="Open the viewer in your browser"),
) -> None:
	"""Generate HTML preview of subtitle files."""
	from pathlib import Path as _Path
	paths: list[_Path] = []
	if dir:
		d = _Path(dir)
		if not d.is_dir():
			raise typer.BadParameter(f"--dir path is not a directory: {dir}")
		if recursive:
			for p in sorted(d.rglob("*")):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
		else:
			for p in sorted(d.iterdir()):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
	if files:
		paths.extend([_Path(f) for f in files])
	# de-duplicate
	seen = set()
	uniq: list[_Path] = []
	for p in paths:
		if str(p) not in seen:
			seen.add(str(p))
			uniq.append(p)
	paths = uniq
	if len(paths) == 0:
		raise typer.BadParameter("Provide subtitle files via args or --dir")
	res = generate_static_preview(paths, _Path(out), title=title, open_browser=open_browser)
	print({"preview": res})


@app.command(name="init-db")
def init_db_cmd() -> None:
	"""Initialize data directories and SQLite database."""
	ensure_data_dirs()
	init_db()
	print(f"[green]Initialized data at[/green] {DATA_DIR} and DB {DB_PATH}")


@app.command(name="list")
def list_cmd(limit: int = typer.Option(50, help="Max rows to show")) -> None:
	"""List stored content entries."""
	rows = list_content(limit=limit)
	table = Table(title=f"Content (showing up to {limit})")
	for col in ["id", "platform", "platform_id", "language", "title", "subtitle_path"]:
		table.add_column(col)
	for r in rows:
		table.add_row(
			str(r.get("id", "")),
			str(r.get("platform", "")),
			str(r.get("platform_id", "")),
			str(r.get("language", "")),
			str(r.get("title", "")),
			str(r.get("subtitle_path", "")),
		)
	print(table)


@app.command(name="ingest-subtitle")
def ingest_subtitle_cmd(
	source: str = typer.Argument(..., help="URL or local path to .srt/.vtt"),
	language: str = typer.Option(..., "--lang", help="Subtitle language code, e.g., en, es"),
	platform: str = typer.Option("web", help="Platform name (e.g., youtube, web, local)"),
	platform_id: str | None = typer.Option(None, help="Platform-specific ID; auto/None if omitted"),
	title: str | None = typer.Option(None, help="Human title; derived from filename if omitted"),
	ext: str = typer.Option("srt", help="Subtitle extension: srt or vtt"),
) -> None:
	"""Ingest a subtitle file into storage and the database (no fetching)."""
	# Read bytes from file or URL
	if source.startswith("http://") or source.startswith("https://"):
		import requests
		resp = requests.get(source, timeout=30)
		resp.raise_for_status()
		data = resp.content
	else:
		from pathlib import Path
		p = Path(source)
		if not p.is_file():
			raise FileNotFoundError(f"Subtitle file not found: {source}")
		data = p.read_bytes()

	content_id, path = ingest_subtitle_from_source(
		source=source,
		language=language,
		platform=platform,
		platform_id=platform_id,
		title=title,
		ext=ext,
		data=data,
	)
	print(f"[green]Ingested[/green] content_id={content_id} -> {path}")


@app.command(name="fetch-subliminal-candidates")
def fetch_subliminal_candidates_cmd(
	query: str = typer.Argument(..., help="Search like 'The Last Airbender S01E01'"),
	language: str = typer.Option(..., "--lang", help="Subtitle language code (e.g., en, es)"),
	top_n: int = typer.Option(3, help="How many candidates to download"),
	platform: str = typer.Option("web", help="Platform name to organize storage"),
	platform_id: str | None = typer.Option(None, help="Platform-specific ID; optional"),
	title: str | None = typer.Option(None, help="Title to organize storage"),
	imdb_id: str | None = typer.Option(None, help="IMDb ID to influence matching"),
) -> None:
	"""Download top-N subtitle candidates (search flow) for consensus."""
	paths = fetch_candidates_with_subliminal_search(
		query=query,
		language=language,
		top_n=top_n,
		platform=platform,
		platform_id=platform_id,
		title=title,
		imdb_id=imdb_id,
	)
	print({"saved": [str(p) for p in paths]})


@app.command(name="fetch-opensubtitles")
def fetch_opensubtitles_cmd(
	imdb_id: str = typer.Argument(..., help="IMDB ID (e.g., 'tt0903747' or '903747')"),
	language: str = typer.Option("eng", "--lang", help="Subtitle language code (e.g., en, es)"),
	max_count: int = typer.Option(10, "--max-count", help="Maximum number of subtitles to download"),
	platform: str = typer.Option("web", help="Platform name to organize storage"),
	platform_id: str | None = typer.Option(None, help="Platform-specific ID; optional"),
	title: str | None = typer.Option(None, help="Title to organize storage"),
	delay: float = typer.Option(0.7, "--delay", help="Delay between downloads in seconds (0 to disable)"),
) -> None:
	"""Download subtitles directly from OpenSubtitles.org by IMDB ID."""
	paths = fetch_subtitles_from_opensubtitles(
		imdb_id=imdb_id,
		language=language,
		max_count=max_count,
		platform=platform,
		platform_id=platform_id,
		title=title,
		delay_seconds=delay,
	)
	print({"saved": [str(p) for p in paths]})


@app.command(name="align-pair")
def align_pair_cmd(
	file_a: str = typer.Argument(..., help="First subtitle file (.srt/.vtt)"),
	file_b: str = typer.Argument(..., help="Second subtitle file (.srt/.vtt)"),
	n: int = typer.Option(3, help="Character n-gram size for similarity"),
	gap_penalty: float = typer.Option(-0.4, help="Gap penalty for NW alignment"),
	min_sim: float = typer.Option(0.3, help="Hard floor for match acceptance (0..1)"),
	grow_threshold: float = typer.Option(0.05, help="Min improvement for grow-merge"),
	use_hashing: bool = typer.Option(True, help="Use hashed n-gram vectors (faster)"),
	use_banded: bool = typer.Option(True, help="Use banded NW alignment for speedup"),
	band_margin_pct: float = typer.Option(0.10, help="Band margin as percentage of avg length (0..1)"),
	show_nw: bool = typer.Option(True, help="Show Needleman-Wunsch alignment details"),
	show_nw_limit: int = typer.Option(50, help="Max NW alignment rows to show"),
	show_blocks: bool = typer.Option(True, help="Show block alignment results"),
) -> None:
	"""Test pairwise alignment using the new alignment module."""
	from pathlib import Path as _Path
	from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	from .analysis.normalization import normalize_subtitle, NormalizationConfig
	from .analysis.alignment import _needleman_wunsch_align, _compute_blocks_growmerge, BlockAlignmentConfig
	from .util.types import Subtitle, BlockAlignment
	
	# Load and normalize subtitles
	def load_and_normalize(file_path: str) -> Subtitle:
		p = _Path(file_path)
		data = p.read_bytes()
		ext = p.suffix.lower().lstrip(".")
		if ext == "srt":
			segs = parse_srt_bytes(data)
		elif ext == "vtt":
			segs = parse_vtt_bytes(data)
		else:
			raise typer.BadParameter(f"Unsupported extension: {ext}")
		
		# Capture original texts before normalization
		original_texts = [s.text or "" for s in segs]
		
		# Create subtitle object with original texts
		subtitle = Subtitle(
			source_file=str(p),
			intervals=[(s.start_seconds, s.end_seconds) for s in segs],
			texts=[s.text or "" for s in segs],
			original_texts=original_texts
		)
		
		# Normalize
		norm_config = NormalizationConfig()
		normalized_subtitle, _ = normalize_subtitle(subtitle, norm_config)
		return normalized_subtitle
	
	print("[blue]Loading and normalizing subtitles...[/blue]")
	sub_a = load_and_normalize(file_a)
	sub_b = load_and_normalize(file_b)
	
	print(f"[green]Loaded:[/green] {len(sub_a.texts)} cues from A, {len(sub_b.texts)} cues from B")
	
	# Configure alignment
	config = BlockAlignmentConfig(
		n_gram_size=n,
		gap_penalty=gap_penalty,
		min_similarity=min_sim,
		grow_merge_threshold=grow_threshold,
		use_hashing=use_hashing,
		use_banded=use_banded,
		band_margin_pct=band_margin_pct
	)
	
	print("[blue]Running NW alignment + block merging...[/blue]")
	
	# Track time for metadata
	import time
	start_time = time.time()
	
	# Run NW alignment to get raw alignment + similarity matrix
	nw_alignment, similarity_matrix, align_meta = _needleman_wunsch_align(
		sub_a.texts, sub_b.texts, config
	)
	
	# Run grow-merge to get blocks
	blocks = _compute_blocks_growmerge(
		nw_alignment, similarity_matrix, sub_a.texts, sub_b.texts, config
	)
	
	elapsed_time = time.time() - start_time
	
	# Convert blocks to BlockAlignment format (for consistency)
	blocks_file_a = []
	blocks_file_b = []
	similarity_scores = []
	for (a_range, b_range, score) in blocks:
		blocks_file_a.append(a_range)
		blocks_file_b.append(b_range)
		similarity_scores.append(score)
	
	block_alignment = BlockAlignment(
		file_a=sub_a.source_file,
		file_b=sub_b.source_file,
		num_blocks=len(blocks),
		blocks_file_a=blocks_file_a,
		blocks_file_b=blocks_file_b,
		similarity=similarity_scores
	)
	
	# Build metadata
	num_matches = sum(1 for ai, bj in nw_alignment if ai is not None and bj is not None)
	num_gaps_a = sum(1 for ai, bj in nw_alignment if ai is not None and bj is None)
	num_gaps_b = sum(1 for ai, bj in nw_alignment if ai is None and bj is not None)
	
	metadata = {
		"pairwise_alignment": {
			"file_a": block_alignment.file_a,
			"file_b": block_alignment.file_b,
			"nw_alignment": {
				"total_pairs": len(nw_alignment),
				"matches": num_matches,
				"gaps_a": num_gaps_a,
				"gaps_b": num_gaps_b
			},
			"num_blocks": block_alignment.num_blocks,
			"avg_similarity": round(sum(block_alignment.similarity) / max(1, block_alignment.num_blocks), 3),
			"config": {
				"n_gram_size": config.n_gram_size,
				"gap_penalty": config.gap_penalty,
				"min_similarity": config.min_similarity,
				"grow_merge_threshold": config.grow_merge_threshold,
				"use_hashing": config.use_hashing,
				"use_banded": config.use_banded,
				"band_margin_pct": config.band_margin_pct
			},
			"computation_time": round(elapsed_time, 3)
		}
	}
	
	# Add band_width if banded alignment was used
	if align_meta.get("band_width") is not None:
		metadata["pairwise_alignment"]["band_width"] = align_meta["band_width"]
	
	# Show NW alignment results
	if show_nw:
		from rich.table import Table as _Table
		
		# Helper to clip text for display
		def clip(s: str, width: int = 40) -> str:
			s = s.replace("\n", " ")
			return (s[: width - 1] + "…") if len(s) > width else s
		
		nw_table = _Table(title=f"Needleman-Wunsch Alignment (showing {min(show_nw_limit, len(nw_alignment))} of {len(nw_alignment)} pairs)")
		nw_table.add_column("Pair")
		nw_table.add_column("A idx")
		nw_table.add_column("A text")
		nw_table.add_column("Sim")
		nw_table.add_column("B idx")
		nw_table.add_column("B text")
		
		for i, (ai, bj) in enumerate(nw_alignment[:show_nw_limit]):
			a_txt = sub_a.texts[ai] if ai is not None and 0 <= ai < len(sub_a.texts) else "—"
			b_txt = sub_b.texts[bj] if bj is not None and 0 <= bj < len(sub_b.texts) else "—"
			sim = f"{similarity_matrix[ai, bj]:.3f}" if (ai is not None and bj is not None) else "—"
			
			nw_table.add_row(
				str(i),
				str(ai) if ai is not None else "—",
				clip(a_txt),
				sim,
				str(bj) if bj is not None else "—",
				clip(b_txt)
			)
		
		print(nw_table)
		if len(nw_alignment) > show_nw_limit:
			print(f"[dim]... {len(nw_alignment) - show_nw_limit} more pairs not shown (use --show-nw-limit to adjust)[/dim]")
	
	# Show block results
	if show_blocks:
		from rich.table import Table as _Table
		tbl = _Table(title=f"Block Alignment ({block_alignment.num_blocks} blocks)")
		tbl.add_column("Block")
		tbl.add_column("File A Range")
		tbl.add_column("File B Range")
		tbl.add_column("Similarity")
		
		for i in range(block_alignment.num_blocks):
			a_start, a_end = block_alignment.blocks_file_a[i]
			b_start, b_end = block_alignment.blocks_file_b[i]
			sim = block_alignment.similarity[i]
			tbl.add_row(
				str(i),
				f"[{a_start}..{a_end}]",
				f"[{b_start}..{b_end}]",
				f"{sim:.3f}"
			)
		print(tbl)
	
	# Show metadata summary
	print("\n[yellow]Alignment Metadata:[/yellow]")
	print(json.dumps(metadata, indent=2))


@app.command(name="align-pipeline")
def align_pipeline_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt) to align"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing candidate .srt/.vtt files"),
	recursive: bool = typer.Option(False, help="Recursively search for .srt/.vtt under --dir"),
	n: int = typer.Option(3, help="Character n-gram size for similarity"),
	gap_penalty: float = typer.Option(-0.4, help="Gap penalty for NW alignment"),
	min_sim: float = typer.Option(0.3, help="Hard floor for match acceptance (0..1)"),
	grow_threshold: float = typer.Option(0.05, help="Min improvement for grow-merge"),
	use_banded: bool = typer.Option(True, help="Use banded NW alignment for speedup"),
	band_margin_pct: float = typer.Option(0.10, help="Band margin as percentage of avg length (0..1)"),
	component_threshold: float | None = typer.Option(None, help="Edge threshold for candidate selection"),
	hard_anchor_threshold: float = typer.Option(0.9, help="Similarity threshold for hard anchors"),
	clean_threshold: float = typer.Option(0.5, help="Support threshold for cleaning"),
	output_dir: str | None = typer.Option(None, "--output-dir", help="Directory to write aligned SRT files"),
	show_metadata: bool = typer.Option(True, help="Show processing metadata"),
) -> None:
	"""Run the full alignment pipeline with the new alignment module."""
	from pathlib import Path as _Path
	from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	from .analysis.normalization import normalize_subtitle, NormalizationConfig
	from .analysis.alignment import (
		align_subtitle_matrix, 
		select_candidates,
		align_to_master,
		clean_subtitle,
		compute_temporal_consistency_batch,
		BlockAlignmentConfig
	)
	from .util.types import Subtitle
	
	# Collect files
	paths: list[_Path] = []
	if dir:
		d = _Path(dir)
		if not d.is_dir():
			raise typer.BadParameter(f"--dir path is not a directory: {dir}")
		if recursive:
			for p in sorted(d.rglob("*")):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
		else:
			for p in sorted(d.iterdir()):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
	if files:
		paths.extend([_Path(f) for f in files])
	
	# De-duplicate
	seen = set()
	unique_paths: list[_Path] = []
	for p in paths:
		if str(p) not in seen:
			seen.add(str(p))
			unique_paths.append(p)
	paths = unique_paths
	
	if len(paths) < 2:
		raise typer.BadParameter("Provide at least two subtitle files via arguments or --dir")
	
	print(f"[blue]Found {len(paths)} subtitle files[/blue]")
	
	# Load and normalize subtitles
	def load_and_normalize(file_path: _Path) -> Subtitle:
		data = file_path.read_bytes()
		ext = file_path.suffix.lower().lstrip(".")
		if ext == "srt":
			segs = parse_srt_bytes(data)
		elif ext == "vtt":
			segs = parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
		
		# Capture original texts before normalization
		original_texts = [s.text or "" for s in segs]
		
		subtitle = Subtitle(
			source_file=str(file_path),
			intervals=[(s.start_seconds, s.end_seconds) for s in segs],
			texts=[s.text or "" for s in segs],
			original_texts=original_texts
		)
		
		norm_config = NormalizationConfig()
		normalized_subtitle, _ = normalize_subtitle(subtitle, norm_config)
		return normalized_subtitle
	
	print("[blue]Loading and normalizing subtitles...[/blue]")
	subtitles = [load_and_normalize(p) for p in paths]
	print(f"[green]Loaded {len(subtitles)} subtitles[/green]")
	
	# Initialize metadata
	metadata = {}
	
	# Phase 1: Block alignment
	print("[blue]Phase 1: Computing block alignments...[/blue]")
	config = BlockAlignmentConfig(
		n_gram_size=n,
		gap_penalty=gap_penalty,
		min_similarity=min_sim,
		grow_merge_threshold=grow_threshold,
		use_hashing=True,
		use_banded=use_banded,
		band_margin_pct=band_margin_pct
	)
	alignments, metadata = align_subtitle_matrix(subtitles, config, metadata)
	print(f"[green]Computed {len(alignments)} pairwise alignments[/green]")
	
	# Phase 2: Candidate selection
	print("[blue]Phase 2: Selecting candidates...[/blue]")
	if component_threshold is not None:
		candidate_indices, metadata = select_candidates(subtitles, alignments, component_threshold, metadata)
	else:
		candidate_indices, metadata = select_candidates(subtitles, alignments, metadata=metadata)
	print(f"[green]Selected {len(candidate_indices)} candidates from component[/green]")
	for idx in candidate_indices:
		print(f"  • {paths[idx].name}")
	
	# Filter to relevant alignments only (selected <-> selected)
	candidate_files = {subtitles[idx].source_file for idx in candidate_indices}
	relevant_alignments = [
		a for a in alignments
		if a.file_a in candidate_files and a.file_b in candidate_files
	]
	print(f"[blue]Using {len(relevant_alignments)} relevant alignments[/blue]")
	
	# Phase 3: Hard-anchor alignment to master clock (using normalized subtitles directly)
	print("[blue]Phase 3: Aligning to master clock...[/blue]")
	aligned_subtitles, metadata = align_to_master(
		subtitles, candidate_indices, relevant_alignments, hard_anchor_threshold, metadata
	)
	print(f"[green]Aligned {len(aligned_subtitles)} subtitles to master[/green]")
	print(f"  Master: {metadata['hard_anchor_alignment']['master_file']}")
	print(f"  Successful: {metadata['hard_anchor_alignment']['successful_alignments']}")
	print(f"  Failed: {metadata['hard_anchor_alignment']['failed_alignments']}")
	
	# Phase 4: Cleaning unsupported cues (after alignment)
	print(f"[blue]Phase 4: Cleaning unsupported cues (threshold={clean_threshold})...[/blue]")
	cleaned_subtitles = []
	total_removed_cues = 0
	
	for aligned_sub in aligned_subtitles:
		# Get supporting alignments for this subtitle
		sub_file = aligned_sub.source_file
		supporting = [a for a in relevant_alignments if a.file_a == sub_file or a.file_b == sub_file]
		
		# Clean with per-file metadata tracking
		file_metadata = {}
		cleaned_sub, file_metadata = clean_subtitle(
			aligned_sub, supporting, clean_threshold, file_metadata
		)
		cleaned_subtitles.append(cleaned_sub)
		
		# Merge cleaning metadata into main metadata
		if "cleaning" not in metadata:
			metadata["cleaning"] = {"files": {}, "support_threshold": clean_threshold}
		metadata["cleaning"]["files"][sub_file] = file_metadata.get("cleaning", {})
		total_removed_cues += file_metadata.get("cleaning", {}).get("removed_cues", 0)
	
	print(f"[green]Removed {total_removed_cues} unsupported cues total[/green]")
	
	# Use cleaned subtitles for output
	final_subtitles = cleaned_subtitles
	
	# Phase 5: Compute temporal consistency metrics
	print("[blue]Phase 5: Computing temporal consistency metrics...[/blue]")
	metadata = compute_temporal_consistency_batch(final_subtitles, metadata)
	
	if metadata.get("temporal_consistency", {}).get("summary"):
		summary = metadata["temporal_consistency"]["summary"]
		print(f"[green]Temporal Consistency:[/green]")
		print(f"  Avg total misplacement: {summary['avg_total_misplacement']} seconds")
		print(f"  Avg per-cue misplacement: {summary['avg_misplacement_per_cue']} seconds")
		print(f"  Files with overlaps: {summary['files_with_overlaps']}/{summary['total_files']}")
		print(f"  Max misplacement: {summary['max_total_misplacement']} seconds")
	
	# Show metadata
	if show_metadata:
		print("\n[yellow]Processing Metadata:[/yellow]")
		print(json.dumps(metadata, indent=2, default=str))
	
	# Write output
	if output_dir:
		output_path = _Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)
		
		# Write metadata
		metadata_file = output_path / "alignment_metadata.json"
		with metadata_file.open("w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2, default=str)
		print(f"[green]Wrote metadata to[/green] {metadata_file}")
		
		# Write aligned and cleaned SRT files
		import srt
		from datetime import timedelta
		
		for i in range(len(final_subtitles)):
			subtitle = final_subtitles[i]
			output_file = output_path / f"{_Path(subtitle.source_file).stem}_aligned.srt"
			
			# Use original_texts if available, otherwise fall back to normalized texts
			output_texts = subtitle.original_texts if subtitle.original_texts else subtitle.texts
			
			srt_items = []
			for j, (start, end) in enumerate(subtitle.intervals):
				start_td = timedelta(seconds=start)
				end_td = timedelta(seconds=end)
				srt_items.append(srt.Subtitle(
					index=j+1,
					start=start_td,
					end=end_td,
					content=output_texts[j]
				))
			
			with output_file.open("w", encoding="utf-8") as f:
				f.write(srt.compose(srt_items))
			
			print(f"[green]Wrote aligned SRT to[/green] {output_file}")
	
	print("\n[green]Pipeline complete![/green]")


@app.command(name="tokenize")
def tokenize_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle file(s) (.srt/.vtt) to tokenize"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing subtitle files"),
	recursive: bool = typer.Option(False, help="Recursively search for .srt/.vtt under --dir"),
	language: str | None = typer.Option(None, "--lang", help="Language code for tokenization (e.g., en, es, de). If not specified, language will be auto-detected."),
	already_normalized: bool = typer.Option(False, help="Skip normalization if already normalized"),
	show_tokens: bool = typer.Option(False, help="Display the extracted tokens (only for single file)"),
	compute_frequency: bool = typer.Option(True, help="Compute token frequency statistics (zipf scores)"),
	show_metadata: bool = typer.Option(True, help="Show tokenization metadata"),
) -> None:
	"""Extract tokens from subtitle file(s) for speech speed analysis and vocabulary frequency."""
	from pathlib import Path as _Path
	from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	from .analysis.tokens import (
		tokenize_subtitle, 
		tokenize_subtitles,
		compute_frequency_statistics,
		compute_frequency_statistics_batch,
		FrequencyConfig
	)
	from .util.types import Subtitle
	from rich.table import Table as _Table
	
	# Collect files
	paths: list[_Path] = []
	if dir:
		d = _Path(dir)
		if not d.is_dir():
			raise typer.BadParameter(f"--dir path is not a directory: {dir}")
		if recursive:
			for p in sorted(d.rglob("*")):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
		else:
			for p in sorted(d.iterdir()):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
	if files:
		paths.extend([_Path(f) for f in files])
	
	# De-duplicate
	seen = set()
	unique_paths: list[_Path] = []
	for p in paths:
		if str(p) not in seen:
			seen.add(str(p))
			unique_paths.append(p)
	paths = unique_paths
	
	if len(paths) == 0:
		raise typer.BadParameter("Provide subtitle file(s) via arguments or --dir")
	
	print(f"[blue]Found {len(paths)} subtitle file(s)[/blue]")
	
	# Load subtitle files
	def load_subtitle(file_path: _Path) -> Subtitle:
		data = file_path.read_bytes()
		ext = file_path.suffix.lower().lstrip(".")
		if ext == "srt":
			segs = parse_srt_bytes(data)
		elif ext == "vtt":
			segs = parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
		
		subtitle = Subtitle(
			source_file=str(file_path),
			intervals=[(s.start_seconds, s.end_seconds) for s in segs],
			texts=[s.text or "" for s in segs],
			original_texts=None
		)
		return subtitle
	
	print("[blue]Loading subtitle files...[/blue]")
	subtitles = [load_subtitle(p) for p in paths]
	print(f"[green]Loaded {len(subtitles)} subtitle(s)[/green]")
	
	# Single file mode
	if len(subtitles) == 1:
		subtitle = subtitles[0]
		
		# Tokenize
		if language:
			print(f"[blue]Tokenizing (language={language}, already_normalized={already_normalized})...[/blue]")
		else:
			print(f"[blue]Auto-detecting language and tokenizing (already_normalized={already_normalized})...[/blue]")
		
		metadata = {}
		tokens, detected_lang = tokenize_subtitle(
			subtitle,
			already_normalized=already_normalized,
			language=language,
			metadata=metadata
		)
		
		# Extract detection method for display
		detection_method = metadata.get("tokenization", {}).get("language_detection", "unknown")
		
		# Display results
		print(f"\n[green]Tokenization complete![/green]")
		if detection_method == "automatic":
			print(f"  Detected language: {detected_lang}")
		elif detection_method == "automatic_fallback":
			print(f"  Language detection failed, using fallback: {detected_lang}")
		elif detection_method == "manual":
			print(f"  Language (manual): {detected_lang}")
		print(f"  Total tokens: {len(tokens)}")
		print(f"  Unique tokens: {len(set(tokens))}")
		
		# Compute frequency statistics
		if compute_frequency:
			print(f"\n[blue]Computing frequency statistics...[/blue]")
			freq_config = FrequencyConfig()
			freq_stats, zipf_scores = compute_frequency_statistics(tokens, detected_lang, freq_config, metadata)
			
			print(f"\n[yellow]Frequency Statistics (Zipf Scores):[/yellow]")
			print(f"  Quantiles:")
			for key, value in freq_stats["quantiles"].items():
				print(f"    {key}: {value}")
			
			print(f"\n  Bucket Distribution:")
			for bucket, percentage in freq_stats["bucket_percentages"].items():
				print(f"    {bucket}: {percentage:.4f}")
			
			print(f"\n  Mean zipf score: {freq_stats['mean']}")
		
		if show_tokens:
			print(f"\n[yellow]Tokens:[/yellow]")
			# Display tokens in a readable format (wrap at 80 chars)
			line = ""
			for token in tokens:
				if len(line) + len(token) + 2 > 80:
					print(f"  {line}")
					line = token
				else:
					if line:
						line += ", " + token
					else:
						line = token
			if line:
				print(f"  {line}")
		
		if show_metadata:
			print(f"\n[yellow]Metadata:[/yellow]")
			print(json.dumps(metadata, indent=2, default=str))
		
		# Print frequency results separately if computed
		if compute_frequency:
			print(f"\n[yellow]Frequency Analysis Results:[/yellow]")
			print(json.dumps({"stats": freq_stats}, indent=2, default=str))
	
	# Batch mode
	else:
		# Tokenize all subtitles
		if language:
			print(f"[blue]Tokenizing (language={language}, already_normalized={already_normalized})...[/blue]")
		else:
			print(f"[blue]Auto-detecting language and tokenizing (already_normalized={already_normalized})...[/blue]")
		
		metadata = {}
		all_tokens, majority_lang, metadata = tokenize_subtitles(
			subtitles,
			already_normalized=already_normalized,
			language=language,
			metadata=metadata
		)
		
		# Display results
		print(f"\n[green]Tokenization complete![/green]")
		
		# Show language agreement if auto-detected
		if language is None and "language_agreement" in metadata.get("tokenization", {}):
			lang_agreement = metadata["tokenization"]["language_agreement"]
			print(f"\n[yellow]Language Detection:[/yellow]")
			print(f"  Majority language: {lang_agreement['majority_language']}")
			print(f"  Agreement: {lang_agreement['agreement_count']}/{lang_agreement['total_subtitles']} ({lang_agreement['agreement_percentage']:.1f}%)")
			
			# Show distribution if there are multiple languages
			lang_dist = lang_agreement['language_distribution']
			if len(lang_dist) > 1:
				print(f"  Language distribution:")
				for lang, count in sorted(lang_dist.items(), key=lambda x: x[1], reverse=True):
					print(f"    {lang}: {count} file(s)")
		elif language:
			print(f"  Language (manual): {majority_lang}")
		
		# Create table with per-file stats
		table = _Table(title=f"Tokenization Results ({len(subtitles)} files)")
		table.add_column("File")
		table.add_column("Cues", justify="right")
		table.add_column("Tokens", justify="right")
		table.add_column("Unique", justify="right")
		table.add_column("Lang")
		
		file_stats = metadata.get("tokenization", {}).get("files", {})
		for i, (p, tokens) in enumerate(zip(paths, all_tokens)):
			file_key = str(p)
			stats = file_stats.get(file_key, {})
			cues = stats.get("total_cues", 0)
			total_tokens = len(tokens)
			unique_tokens = len(set(tokens))
			detected_lang = stats.get("detected_language", "?")
			
			table.add_row(
				p.name,
				str(cues),
				str(total_tokens),
				str(unique_tokens),
				detected_lang
			)
		
		print(f"\n{table}")
		
		# Show summary stats
		total_tokens = sum(len(tokens) for tokens in all_tokens)
		total_unique = len(set(token for tokens in all_tokens for token in tokens))
		print(f"\n[yellow]Summary:[/yellow]")
		print(f"  Total files: {len(subtitles)}")
		print(f"  Total tokens: {total_tokens}")
		print(f"  Total unique tokens: {total_unique}")
		
		# Compute frequency statistics for batch
		if compute_frequency:
			print(f"\n[blue]Computing frequency statistics for all files...[/blue]")
			freq_config = FrequencyConfig()
			aggregate_stats, per_file_stats = compute_frequency_statistics_batch(
				all_tokens, majority_lang, freq_config, metadata
			)
			
			print(f"\n[yellow]Aggregate Frequency Statistics (All Files Combined):[/yellow]")
			print(f"  Total tokens: {aggregate_stats['total_tokens']}")
			print(f"  Unique tokens: {aggregate_stats['unique_tokens']}")
			print(f"\n  Quantiles:")
			for key, value in aggregate_stats["quantiles"].items():
				print(f"    {key}: {value}")
			
			print(f"\n  Bucket Distribution:")
			for bucket, percentage in aggregate_stats["bucket_percentages"].items():
				print(f"    {bucket}: {percentage:.4f}")
			
			print(f"\n  Mean zipf score: {aggregate_stats['mean']}")
			
			# Create per-file frequency stats table
			# Get bucket keys dynamically (first and last buckets for display)
			bucket_keys = list(aggregate_stats["bucket_percentages"].keys())
			first_bucket = bucket_keys[0] if bucket_keys else "N/A"
			last_bucket = bucket_keys[-1] if bucket_keys else "N/A"
			
			freq_table = _Table(title="Per-File Frequency Statistics")
			freq_table.add_column("File")
			freq_table.add_column("Median", justify="right")
			freq_table.add_column("P25", justify="right")
			freq_table.add_column("P75", justify="right")
			freq_table.add_column("Mean", justify="right")
			freq_table.add_column(f"{last_bucket} %", justify="right")
			freq_table.add_column(f"{first_bucket} %", justify="right")
			
			for i, p in enumerate(paths):
				file_key = f"file_{i}"
				if file_key in per_file_stats:
					fstats = per_file_stats[file_key]
					
					# Get median and quantiles, with fallback
					median = fstats["quantiles"].get("0.5", "N/A")
					p25 = fstats["quantiles"].get("0.25", "N/A")
					p75 = fstats["quantiles"].get("0.75", "N/A")
					
					# Get bucket percentages
					last_bucket_pct = f"{fstats['bucket_percentages'].get(last_bucket, 0):.4f}"
					first_bucket_pct = f"{fstats['bucket_percentages'].get(first_bucket, 0):.4f}"
					
					freq_table.add_row(
						p.name,
						str(median),
						str(p25),
						str(p75),
						str(fstats["mean"]),
						f"{last_bucket_pct}%",
						f"{first_bucket_pct}%"
					)
			
			print(f"\n{freq_table}")
		
		if show_metadata:
			print(f"\n[yellow]Metadata:[/yellow]")
			print(json.dumps(metadata, indent=2, default=str))
		
		# Print frequency results separately if computed
		if compute_frequency:
			print(f"\n[yellow]Frequency Analysis Results:[/yellow]")
			print(json.dumps({
				"aggregate_stats": aggregate_stats,
				"per_file_stats": per_file_stats
			}, indent=2, default=str))


@app.command(name="consensus")
def consensus_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt) for consensus"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing candidate .srt/.vtt files"),
	recursive: bool = typer.Option(False, help="Recursively search for .srt/.vtt under --dir"),
	target_agreement_pct: float = typer.Option(0.66, help="Target agreement percentage (0.0-1.0)"),
	merge_micro_gaps: bool = typer.Option(True, help="Merge small gaps between intervals"),
	micro_gap_seconds: float = typer.Option(0.2, help="Maximum gap size to merge (seconds)"),
	min_interval_seconds: float = typer.Option(0.3, help="Minimum interval duration to keep (seconds)"),
	output_dir: str | None = typer.Option(None, "--output-dir", help="Directory to write consensus SRT and metadata"),
	show_metadata: bool = typer.Option(True, help="Show consensus metadata"),
) -> None:
	"""Compute speech time consensus from multiple subtitle files."""
	from pathlib import Path as _Path
	from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	from .analysis.consensus import compute_consensus, export_consensus_srt, ConsensusConfig
	from .util.types import Subtitle
	
	# Collect files
	paths: list[_Path] = []
	if dir:
		d = _Path(dir)
		if not d.is_dir():
			raise typer.BadParameter(f"--dir path is not a directory: {dir}")
		if recursive:
			for p in sorted(d.rglob("*")):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
		else:
			for p in sorted(d.iterdir()):
				if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
					paths.append(p)
	if files:
		paths.extend([_Path(f) for f in files])
	
	# De-duplicate
	seen = set()
	unique_paths: list[_Path] = []
	for p in paths:
		if str(p) not in seen:
			seen.add(str(p))
			unique_paths.append(p)
	paths = unique_paths
	
	if len(paths) < 1:
		raise typer.BadParameter("Provide at least one subtitle file via arguments or --dir")
	
	print(f"[blue]Found {len(paths)} subtitle files[/blue]")
	
	# Load subtitles
	def load_subtitle(file_path: _Path) -> Subtitle:
		data = file_path.read_bytes()
		ext = file_path.suffix.lower().lstrip(".")
		if ext == "srt":
			segs = parse_srt_bytes(data)
		elif ext == "vtt":
			segs = parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
		
		subtitle = Subtitle(
			source_file=str(file_path),
			intervals=[(s.start_seconds, s.end_seconds) for s in segs],
			texts=[s.text or "" for s in segs],
			original_texts=None
		)
		return subtitle
	
	print("[blue]Loading subtitles...[/blue]")
	subtitles = [load_subtitle(p) for p in paths]
	print(f"[green]Loaded {len(subtitles)} subtitles[/green]")
	for i, subtitle in enumerate(subtitles):
		print(f"  • {paths[i].name}: {len(subtitle.intervals)} cues")
	
	# Configure consensus
	config = ConsensusConfig(
		target_agreement_pct=target_agreement_pct,
		merge_micro_gaps=merge_micro_gaps,
		micro_gap_seconds=micro_gap_seconds,
		min_interval_seconds=min_interval_seconds
	)
	
	# Compute consensus
	print("\n[blue]Computing consensus...[/blue]")
	consensus_subtitle, speech_seconds, metadata = compute_consensus(subtitles, config)
	
	# Extract info from metadata for display
	consensus_meta = metadata.get("consensus", {})
	num_intervals = consensus_meta.get("num_intervals", 0)
	total_subtitles = consensus_meta.get("total_subtitles", 0)
	required_agreement = consensus_meta.get("required_agreement", 0)
	agreement_percentage = consensus_meta.get("agreement_percentage", 0.0)
	
	# Compute speech minutes from metric
	speech_minutes = speech_seconds / 60.0
	
	# Display results
	print("\n[green]Consensus Results:[/green]")
	print(f"  Speech Time: {speech_seconds:.2f} seconds ({speech_minutes:.2f} minutes)")
	print(f"  Speech Intervals: {num_intervals}")
	print(f"  Total Subtitles: {total_subtitles}")
	print(f"  Required Agreement: {required_agreement}/{total_subtitles} ({agreement_percentage:.1f}%)")
	
	# Show metadata
	if show_metadata:
		print("\n[yellow]Consensus Metadata:[/yellow]")
		print(json.dumps(metadata, indent=2, default=str))
	
	# Write output
	if output_dir:
		output_path = _Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)
		
		# Write metadata
		metadata_file = output_path / "consensus_metadata.json"
		with metadata_file.open("w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2, default=str)
		print(f"\n[green]Wrote metadata to[/green] {metadata_file}")
		
		# Write consensus SRT
		consensus_srt_file = output_path / "consensus.srt"
		consensus_srt = export_consensus_srt(consensus_subtitle)
		with consensus_srt_file.open("w", encoding="utf-8") as f:
			f.write(consensus_srt)
		print(f"[green]Wrote consensus SRT to[/green] {consensus_srt_file}")
	
	print("\n[green]Consensus computation complete![/green]")
