from __future__ import annotations

import json
import logging
import typer
from rich import print
from rich.table import Table

from .config import ensure_data_dirs, DATA_DIR, DB_PATH
from .data.db import init_db, list_content, get_content_by_id, insert_analysis
from .parsers.subtitles import ingest_subtitle_from_source
from .fetchers.subliminal_fetcher import fetch_with_subliminal, fetch_with_subliminal_search, fetch_candidates_with_subliminal_search
from .analysis.speech import analyze_subtitle_file
from .analysis.consensus import compute_consensus, export_srt, export_anchored_srt
from .analysis.alignment import (
    normalize_subtitle_text,
    char_ngram_cosine_similarity,
    needleman_wunsch_align,
    compute_match_blocks_localmerge,
    blocks_to_time_pairs,
    fit_affine_from_pairs,
    compute_block_similarity_and_coverage,
    compute_similarity_matrix,
    compute_similarity_matrix_precomputed,
)


# Enable INFO-level logging globally so Subliminal's logger.info messages are printed
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(add_completion=False, no_args_is_help=True)
@app.command(name="similarity-matrix")
def similarity_matrix_cmd(
    files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt) to compare"),
    n: int = typer.Option(3, help="Character n-gram size for similarity"),
    gap_penalty: float = typer.Option(-0.4, help="Gap penalty for NW alignment"),
    min_sim: float = typer.Option(0.3, help="Hard floor for match acceptance (0..1)"),
    min_improve: float = typer.Option(0.05, help="Min improvement to accept local merge"),
    dir: str | None = typer.Option(None, "--dir", help="Folder containing candidate .srt/.vtt files"),
    recursive: bool = typer.Option(False, help="Recursively search for .srt/.vtt under --dir"),
    precomputed: bool = typer.Option(False, help="Use precomputed hashed n-gram vectors (faster)"),
) -> None:
    """Compute and print a pairwise similarity matrix for multiple subtitle files."""
    from pathlib import Path as _Path
    from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes

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
    # de-duplicate while preserving order
    seen = set()
    unique_paths: list[_Path] = []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            unique_paths.append(p)
    paths = unique_paths
    if len(paths) < 2:
        raise typer.BadParameter("Provide at least two subtitle files via arguments or --dir")

    def load_norm_texts_and_intervals(p: _Path) -> tuple[list[str], list[tuple[float, float]]]:
        data = p.read_bytes()
        ext = p.suffix.lower().lstrip(".")
        if ext == "srt":
            segs = parse_srt_bytes(data)
        elif ext == "vtt":
            segs = parse_vtt_bytes(data)
        else:
            raise typer.BadParameter(f"Unsupported extension: {ext}")
        texts = [
            normalize_subtitle_text(s.text or "")
            for s in segs
        ]
        iv = [(s.start_seconds, s.end_seconds) for s in segs]
        return texts, iv

    texts_list = []
    intervals_list = []
    for p in paths:
        t, iv = load_norm_texts_and_intervals(p)
        texts_list.append(t)
        intervals_list.append(iv)

    # Compute chosen method and measure time
    import time as _time
    if precomputed:
        _t0 = _time.time()
        M, _dets, timing = compute_similarity_matrix_precomputed(
            texts_list,
            intervals_list,
            n=n,
            gap_penalty=gap_penalty,
            min_sim=min_sim,
            min_improve=min_improve,
        )
        total_seconds = timing.get("precompute_seconds", 0.0) + timing.get("align_seconds", 0.0)
        timing_out = {"precompute_seconds": round(timing.get("precompute_seconds", 0.0), 3), "align_seconds": round(timing.get("align_seconds", 0.0), 3), "total_seconds": round(total_seconds, 3)}
    else:
        _t0 = _time.time()
        M, _details = compute_similarity_matrix(
            texts_list,
            intervals_list,
            n=n,
            gap_penalty=gap_penalty,
            min_sim=min_sim,
            min_improve=min_improve,
        )
        timing_out = {"total_seconds": round(_time.time() - _t0, 3)}

    # Pretty print matrix with file basenames as headers
    from rich.table import Table as _Table
    tbl = _Table(title="Pairwise similarity (combined score)")
    headers = [p.name for p in paths]
    tbl.add_column("")
    for h in headers:
        tbl.add_column(h)
    for i, row in enumerate(M):
        tbl.add_row(headers[i], *[f"{float(v):.3f}" for v in row])
    print(tbl)
    print({"timing_seconds": timing_out})
@app.command(name="align-two")
def align_two_cmd(
    file_a: str = typer.Argument(..., help="First subtitle file (.srt/.vtt)"),
    file_b: str = typer.Argument(..., help="Second subtitle file (.srt/.vtt)"),
    n: int = typer.Option(3, help="Character n-gram size for similarity"),
    gap_penalty: float = typer.Option(-0.4, help="Gap penalty for NW alignment"),
    min_sim: float = typer.Option(0.3, help="Hard floor for match acceptance (0..1)"),
    normalize_numbers: str = typer.Option("keep", help="Number handling: keep|token"),
    drop_punctuation_mode: str = typer.Option("most", help="Punctuation handling: most|all"),
    keep_intraword_apostrophe_hyphen: bool = typer.Option(True, help="Keep ' and - inside words"),
    remove_hearing_impaired: bool = typer.Option(True, help="Drop [SFX]/(SFX) annotations"),
    strip_speaker_labels: bool = typer.Option(True, help="Drop leading dashes and NAME:"),
    fit_affine: bool = typer.Option(True, help="Fit y≈a*x+b between A and B using local-merged matches"),
    weight_pairs: bool = typer.Option(True, help="Weight pairs by merged block similarity"),
    centers_only: bool = typer.Option(False, help="Use only center→center pairs for fitting (no start/end)"),
    report_similarity: bool = typer.Option(True, help="Report block-based similarity, coverage, and combined score"),
) -> None:
    """Align two subtitle files by text with Needleman–Wunsch and print a simple table."""
    from pathlib import Path as _Path
    from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes

    def load_norm_texts(p: _Path) -> list[str]:
        data = p.read_bytes()
        ext = p.suffix.lower().lstrip(".")
        if ext == "srt":
            segs = parse_srt_bytes(data)
        elif ext == "vtt":
            segs = parse_vtt_bytes(data)
        else:
            raise typer.BadParameter(f"Unsupported extension: {ext}")
        texts: list[str] = []
        for s in segs:
            t = normalize_subtitle_text(
                s.text or "",
                remove_hearing_impaired=remove_hearing_impaired,
                strip_speaker_labels=strip_speaker_labels,
                normalize_numbers=normalize_numbers,
                drop_punctuation_mode=drop_punctuation_mode,
                keep_intraword_apostrophe_hyphen=keep_intraword_apostrophe_hyphen,
            )
            texts.append(t)
        return texts

    pa = _Path(file_a)
    pb = _Path(file_b)
    if not pa.is_file() or not pb.is_file():
        raise typer.BadParameter("Both file paths must exist")

    A = load_norm_texts(pa)
    B = load_norm_texts(pb)
    align, S = needleman_wunsch_align(A, B, n=n, gap_penalty=gap_penalty, min_sim=min_sim)

    # Simple illustration: print rows with A[i] | B[j] | sim
    # Limit each text cell to a reasonable width
    def clip(s: str, width: int = 40) -> str:
        s = s.replace("\n", " ")
        return (s[: width - 1] + "…") if len(s) > width else s

    from rich.table import Table as _Table
    tbl = _Table(title="Text alignment (Needleman–Wunsch)")
    tbl.add_column("A idx")
    tbl.add_column("A text")
    tbl.add_column("sim")
    tbl.add_column("B idx")
    tbl.add_column("B text")
    for ai, bj in align:
        a_txt = A[ai] if ai is not None and 0 <= ai < len(A) else "—"
        b_txt = B[bj] if bj is not None and 0 <= bj < len(B) else "—"
        sim = f"{(S[ai, bj] if (ai is not None and bj is not None) else 0.0):.3f}"
        tbl.add_row(
            (str(ai) if ai is not None else "-"),
            clip(a_txt),
            sim,
            (str(bj) if bj is not None else "-"),
            clip(b_txt),
        )
    print(tbl)

    # Local bidirectional merge (surroundings only)
    blocks_pm = compute_match_blocks_localmerge(align, S, A, B)
    if blocks_pm:
        tbl2 = _Table(title="Local-merged blocks (bidirectional gap absorb)")
        tbl2.add_column("A range [i0..i1]")
        tbl2.add_column("B range [j0..j1]")
        tbl2.add_column("score")
        for (i0, i1), (j0, j1), sc in blocks_pm:
            tbl2.add_row(f"[{i0}..{i1}]", f"[{j0}..{j1}]", f"{sc:.3f}")
        print(tbl2)

    if report_similarity:
        # For similarity we need intervals to compute coverage
        def load_intervals(p: _Path) -> list[tuple[float, float]]:
            data = p.read_bytes()
            ext = p.suffix.lower().lstrip(".")
            if ext == "srt":
                segs = parse_srt_bytes(data)
            elif ext == "vtt":
                segs = parse_vtt_bytes(data)
            else:
                raise typer.BadParameter(f"Unsupported extension: {ext}")
            return [(s.start_seconds, s.end_seconds) for s in segs]

        intervals_a = load_intervals(pa)
        intervals_b = load_intervals(pb)
        sb, covA, covB, cov, mA, mB, tA, tB = compute_block_similarity_and_coverage(
            blocks_pm,
            intervals_a,
            intervals_b,
        )
        print({
            "similarity": {
                "combined": round(sb * cov, 6),
                "score_blocks": round(sb, 6),
                "coverageA": round(covA, 6),
                "coverageB": round(covB, 6),
                "coverage": round(cov, 6),
                "matched_seconds": {"A": round(mA, 3), "B": round(mB, 3)},
                "total_seconds": {"A": round(tA, 3), "B": round(tB, 3)},
            }
        })

    # Optionally fit affine mapping using local-merged blocks
    if fit_affine:
        # Load intervals to extract times
        def load_intervals(p: _Path) -> list[tuple[float, float]]:
            data = p.read_bytes()
            ext = p.suffix.lower().lstrip(".")
            if ext == "srt":
                segs = parse_srt_bytes(data)
            elif ext == "vtt":
                segs = parse_vtt_bytes(data)
            else:
                raise typer.BadParameter(f"Unsupported extension: {ext}")
            return [(s.start_seconds, s.end_seconds) for s in segs]

        intervals_a = load_intervals(pa)
        intervals_b = load_intervals(pb)
        xs, ys, ws = blocks_to_time_pairs(
            blocks_pm,
            intervals_a,
            intervals_b,
            include_start_end=(not centers_only),
            include_center=centers_only,
        )
        a, b = fit_affine_from_pairs(xs, ys, ws if weight_pairs else None)
        # Clarify mapping direction: y (file_b) ≈ a * x (file_a) + b
        print({
            "affine": {"a": round(a, 6), "b": round(b, 6)},
            "pairs": len(xs),
            "mapping": {
                "from": str(pa),
                "to": str(pb),
                "equation": f"t_to ≈ a * t_from + b",
                "apply": {
                    "map_from_to": "t' = a*t + b",
                    "map_to_from": "t' = (t - b)/a",
                },
            },
        })


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


@app.command(name="fetch-subliminal")
def fetch_subliminal_cmd(
	media_path: str = typer.Argument(..., help="Local path to media file (e.g., .mp4/.mkv)"),
	language: str = typer.Option(..., "--lang", help="Subtitle language code (e.g., en, es)"),
	platform: str = typer.Option("local", help="Platform name to tag the entry with"),
	platform_id: str | None = typer.Option(None, help="Platform-specific ID; defaults to media stem"),
	title: str | None = typer.Option(None, help="Title; defaults to media stem"),
	imdb_id: str | None = typer.Option(None, help="IMDb ID to influence matching"),
) -> None:
	"""Fetch subtitles using Subliminal for a local media file and register them."""
	content_id, path, best_meta = fetch_with_subliminal(
		media_path=media_path,
		language=language,
		platform=platform,
		platform_id=platform_id,
		title=title,
		imdb_id=imdb_id,
	)
	filtered_meta = {k: v for k, v in best_meta.items() if k not in ("content", "text")}
	print({"content_id": content_id, "path": str(path), "best": filtered_meta})


@app.command(name="fetch-subliminal-search")
def fetch_subliminal_search_cmd(
	query: str = typer.Argument(..., help="Search like 'The Last Airbender S01E01'"),
	language: str = typer.Option(..., "--lang", help="Subtitle language code (e.g., en, es)"),
	platform: str = typer.Option("web", help="Platform name to tag the entry with"),
	platform_id: str | None = typer.Option(None, help="Platform-specific ID; optional"),
	title: str | None = typer.Option(None, help="Title to store; defaults to query"),
	imdb_id: str | None = typer.Option(None, help="IMDb ID to influence matching"),
) -> None:
	"""Fetch subtitles via Subliminal using a search query and register them."""
	content_id, path, best_meta = fetch_with_subliminal_search(
		query=query,
		language=language,
		platform=platform,
		platform_id=platform_id,
		title=title,
		imdb_id=imdb_id,
	)
	filtered_meta = {k: v for k, v in best_meta.items() if k not in ("content", "text")}
	print({"content_id": content_id, "path": str(path), "best": filtered_meta})


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


@app.command(name="analyze-speech")
def analyze_speech_cmd(
	content_id: int = typer.Option(None, "--content-id", help="Analyze by stored content id"),
	path: str | None = typer.Option(None, "--path", help="Analyze a specific subtitle path"),
	store: bool = typer.Option(True, help="Store result in DB when analyzing content-id"),
	micro_gap_seconds: float = typer.Option(0.2, help="Merge gaps <= this many seconds"),
) -> None:
	"""Compute total speech minutes from a subtitle file and optionally store it."""
	from pathlib import Path as _Path
	if not content_id and not path:
		raise typer.BadParameter("Provide --content-id or --path")

	if content_id:
		row = get_content_by_id(content_id)
		if not row:
			raise typer.BadParameter(f"No content with id {content_id}")
		p = _Path(row["subtitle_path"])
		metrics = analyze_subtitle_file(p, micro_gap_seconds=micro_gap_seconds)
		print({"content_id": content_id, **metrics})
		if store:
			insert_analysis(content_id, metrics)
	else:
		p = _Path(path)
		metrics = analyze_subtitle_file(p, micro_gap_seconds=micro_gap_seconds)
		print(metrics)


@app.command(name="consensus")
def consensus_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt); optional if --dir is used"),
	k: int = typer.Option(2, help="Require at least k candidates to overlap"),
	micro_gap_seconds: float = typer.Option(0.2, help="Merge gaps <= this many seconds"),
	min_interval_seconds: float = typer.Option(0.3, help="Drop consensus intervals shorter than this"),
	export_srt_path: str | None = typer.Option(None, help="Write consensus intervals as an empty SRT here"),
	export_anchored_srt_path: str | None = typer.Option(None, help="Write consensus SRT with text from the first file"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing candidate .srt/.vtt files"),
	align: bool = typer.Option(False, help="Align candidates before computing consensus"),
	align_dt: float = typer.Option(0.1, help="Alignment resolution in seconds (dt)"),
	align_max_lag_seconds: float = typer.Option(300.0, help="Max absolute lag to search (seconds)"),
	verbose: bool = typer.Option(False, help="Print discovered files and details"),
) -> None:
	"""Compute k-of-n consensus intervals from multiple subtitle files."""
	from pathlib import Path as _Path
	paths: list[_Path] = []
	if dir:
		d = _Path(dir)
		if not d.is_dir():
			raise typer.BadParameter(f"--dir path is not a directory: {dir}")
		for p in sorted(d.iterdir()):
			if p.suffix.lower() in (".srt", ".vtt") and p.is_file():
				paths.append(p)
	if files:
		paths.extend([_Path(f) for f in files])
	# de-duplicate while preserving order
	seen = set()
	unique_paths: list[_Path] = []
	for p in paths:
		if str(p) not in seen:
			seen.add(str(p))
			unique_paths.append(p)
	if verbose:
		print({"found": len(unique_paths), "paths": [str(p) for p in unique_paths]})
	if len(unique_paths) < 2:
		raise typer.BadParameter("Provide at least two subtitle files via arguments or --dir")
	res = compute_consensus(
		unique_paths,
		k=k,
		micro_gap_seconds=micro_gap_seconds,
		min_interval_seconds=min_interval_seconds,
		align_before_consensus=align,
		align_dt=align_dt,
		align_max_lag_seconds=align_max_lag_seconds,
	)
	print({
		"num_candidates": res.num_candidates,
		"k": res.k,
		"speech_seconds": res.speech_seconds,
		"speech_minutes": res.speech_minutes,
		"num_intervals": len(res.intervals),
	})
	if align and res.shifts is not None:
		print({"shifts_seconds": {str(p): s for p, s in zip(unique_paths, res.shifts)}})
	if export_srt_path:
		_Path(export_srt_path).write_text(export_srt(res.intervals), encoding="utf-8")
		print(f"[green]Wrote consensus SRT to[/green] {export_srt_path}")
	if export_anchored_srt_path:
		_Path(export_anchored_srt_path).write_text(export_anchored_srt(unique_paths, res.intervals), encoding="utf-8")
		print(f"[green]Wrote anchored consensus SRT to[/green] {export_anchored_srt_path}")
