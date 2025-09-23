from __future__ import annotations

import json
import logging
import typer
from rich import print
from rich.table import Table

from .config import ensure_data_dirs, DATA_DIR, DB_PATH
from .data.db import init_db, list_content, get_content_by_id, insert_analysis
from .parsers.subtitles import ingest_subtitle_from_source
from .viewer.static_viewer import generate_static_preview
from .fetchers.subliminal_fetcher import fetch_with_subliminal, fetch_with_subliminal_search, fetch_candidates_with_subliminal_search
from .analysis.speech import analyze_subtitle_file
from .analysis.consensus import compute_consensus, export_srt, export_anchored_srt
from .analysis.alignment import (
    normalize_subtitle_text,
    char_ngram_cosine_similarity,
    needleman_wunsch_align,
    compute_match_blocks_localmerge,
    compute_match_blocks_growmerge,
    blocks_to_time_pairs,
    fit_piecewise_affine,
    fit_affine_from_pairs,
    compute_block_similarity_and_coverage,
    compute_similarity_matrix,
    compute_similarity_matrix_precomputed,
    largest_connected_component,
    compute_hardshift_transform,
    apply_piecewise_shift_to_intervals,
    select_master_clock_by_median_duration,
    align_multiple_subtitles_to_master,
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
    component_threshold: float = typer.Option(0.65, help="Edge threshold for largest connected component"),
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
    # Largest connected component under threshold
    comp_idx = largest_connected_component(M, threshold=component_threshold)
    print({
        "largest_component": {
            "threshold": component_threshold,
            "size": len(comp_idx),
            "indices": comp_idx,
            "files": [str(paths[k]) for k in comp_idx],
        }
    })
    print({"timing_seconds": timing_out})


@app.command(name="preview-html")
def preview_html_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt) to preview"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing .srt/.vtt"),
	recursive: bool = typer.Option(False, help="Recursively include files under --dir"),
	out: str = typer.Option("preview", help="Output directory for static viewer"),
	title: str = typer.Option("Subtitle Preview", help="Page title"),
	open_browser: bool = typer.Option(False, "--open", help="Open the viewer in your browser"),
) -> None:
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
    piecewise: bool = typer.Option(False, help="Fit piecewise local affine segments from B to A"),
    pw_min_span_seconds: float = typer.Option(60.0, help="Min segment span in seconds"),
    pw_min_strong_matches: int = typer.Option(10, help="Min strong matches per segment"),
    pw_strong_sim_threshold: float = typer.Option(0.9, help="Similarity threshold for strong matches"),
    pw_shift_only: bool = typer.Option(False, help="Use shift-only (y≈x+b) per segment instead of affine"),
    emit_hardshift_transform: bool = typer.Option(False, help="Emit piecewise shift transform from B to A using hard anchors (sim>=threshold)"),
    hardshift_threshold: float = typer.Option(0.9, help="Similarity threshold for hard anchors (center-to-center)"),
    write_transformed: str | None = typer.Option(None, help="Write transformed B subtitle (mapped to A's clock) to this .srt path using hard anchors"),
    merge_grow: bool = typer.Option(True, help="Use iterative grow-merge (absorb consecutive gaps while similarity improves)"),
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

    # Merge blocks
    if merge_grow:
        blocks_pm = compute_match_blocks_growmerge(align, S, A, B)
    else:
        blocks_pm = compute_match_blocks_localmerge(align, S, A, B)
    if blocks_pm:
        tbl2 = _Table(title="Local-merged blocks (bidirectional gap absorb)")
        tbl2.add_column("block")
        tbl2.add_column("A range [i0..i1]")
        tbl2.add_column("B range [j0..j1]")
        tbl2.add_column("score")
        for bi, ((i0, i1), (j0, j1), sc) in enumerate(blocks_pm):
            tbl2.add_row(str(bi), f"[{i0}..{i1}]", f"[{j0}..{j1}]", f"{sc:.3f}")
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

    # Optionally fit affine mapping and/or piecewise mapping using local-merged blocks
    if fit_affine or piecewise:
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
        # Build equations with provenance per block
        equations = []  # list of dicts: x, y, w, block_idx, a_range, b_range, kind
        for bi, ((i0, i1), (j0, j1), sc) in enumerate(blocks_pm):
            if i0 < 0 or i1 >= len(intervals_a) or j0 < 0 or j1 >= len(intervals_b):
                continue
            a_start = min(intervals_a[i][0] for i in range(i0, i1 + 1))
            a_end = max(intervals_a[i][1] for i in range(i0, i1 + 1))
            b_start = min(intervals_b[j][0] for j in range(j0, j1 + 1))
            b_end = max(intervals_b[j][1] for j in range(j0, j1 + 1))
            w = float(sc) if weight_pairs else 1.0
            if not centers_only:
                equations.append({
                    "x": float(a_start), "y": float(b_start), "w": w,
                    "block": bi, "a_range": [i0, i1], "b_range": [j0, j1], "kind": "start",
                })
                equations.append({
                    "x": float(a_end), "y": float(b_end), "w": w,
                    "block": bi, "a_range": [i0, i1], "b_range": [j0, j1], "kind": "end",
                })
            else:
                a_c = 0.5 * (a_start + a_end)
                b_c = 0.5 * (b_start + b_end)
                equations.append({
                    "x": float(a_c), "y": float(b_c), "w": w,
                    "block": bi, "a_range": [i0, i1], "b_range": [j0, j1], "kind": "center",
                })
        xs = [e["x"] for e in equations]
        ys = [e["y"] for e in equations]
        ws = [e["w"] for e in equations]
        pair_blocks = [e.get("block", None) for e in equations]
        if piecewise:
            segs, seg_diags = fit_piecewise_affine(
                xs, ys, ws,
                min_span_seconds=pw_min_span_seconds,
                min_strong_matches=pw_min_strong_matches,
                strong_sim_threshold=pw_strong_sim_threshold,
                use_shift_only=pw_shift_only,
                pair_blocks=pair_blocks,
            )
            print({
                "piecewise": {
                    "segments": [{"x_start": round(s[0],3), "x_end": round(s[1],3), "a": round(s[2],6), "b": round(s[3],6)} for s in segs],
                    "diagnostics": seg_diags,
                },
                "pairs": len(equations),
            })
        if emit_hardshift_transform or write_transformed:
            # Build hard anchors (centers-only recommended)
            hard = [
                (e["y"], e["x"], e.get("i", idx))
                for idx, e in enumerate(equations)
                if e.get("kind") == "center" and float(e.get("w", 0.0)) >= hardshift_threshold
            ]
            hard.sort(key=lambda t: t[0])  # sort by B time (source)
            # Ensure strictly increasing in B; drop ties by keeping first
            dedup = []
            last_t = None
            for tB, tA, idx in hard:
                if last_t is None or tB > last_t + 1e-6:
                    dedup.append((tB, tA, idx))
                    last_t = tB
            hard = dedup
            # Compute boundaries on B clock (source)
            if hard:
                b_min = float(min(s for s,_ in intervals_b))
                b_max = float(max(e for _,e in intervals_b))
                tBs = [h[0] for h in hard]
                shifts = [float(h[1] - h[0]) for h in hard]  # shift = tA - tB
                # Midpoints between consecutive hard anchors
                mids = [0.5*(tBs[i] + tBs[i+1]) for i in range(len(tBs)-1)]
                boundaries = [b_min] + mids + [b_max]
                transform = {
                    "type": "piecewise_shift",
                    "from": str(pb),
                    "to": str(pa),
                    "threshold": hardshift_threshold,
                    "boundaries": [round(x, 3) for x in boundaries],
                    "shifts": [round(s, 6) for s in shifts],
                    "anchors": [
                        {"t_file": round(h[0],3), "t_ref": round(h[1],3), "shift": round(h[1]-h[0],6), "eq_index": int(h[2])}
                        for h in hard
                    ],
                }
            else:
                transform = {
                    "type": "piecewise_shift",
                    "from": str(pb),
                    "to": str(pa),
                    "threshold": hardshift_threshold,
                    "boundaries": [],
                    "shifts": [],
                    "anchors": [],
                }
            if emit_hardshift_transform:
                print({"hardshift_transform": transform})
            if write_transformed and hard:
                # Read full B segments with text
                def load_segments(p: _Path):
                    data = p.read_bytes()
                    ext = p.suffix.lower().lstrip(".")
                    if ext == "srt":
                        return parse_srt_bytes(data)
                    elif ext == "vtt":
                        return parse_vtt_bytes(data)
                    else:
                        raise typer.BadParameter(f"Unsupported extension: {ext}")
                segs_b = load_segments(pb)
                # Map and build SRT using alignment.apply_piecewise_shift_to_intervals
                def fmt(ts: float) -> str:
                    ms = int(round((ts - int(ts)) * 1000))
                    sec = int(ts) % 60
                    minute = (int(ts) // 60) % 60
                    hour = int(ts) // 3600
                    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"
                lines: list[str] = []
                idx_out = 1
                for s in segs_b:
                    sb = float(s.start_seconds); eb = float(s.end_seconds)
                    sa, ea = apply_piecewise_shift_to_intervals([(sb, eb)], transform)[0]
                    if ea <= sa:  # skip zero/negative durations
                        continue
                    lines.append(str(idx_out)); idx_out += 1
                    lines.append(f"{fmt(sa)} --> {fmt(ea)}")
                    lines.append((s.text or "-").strip())
                    lines.append("")
                _Path(write_transformed).write_text("\n".join(lines), encoding="utf-8")
                print({"written_transformed": write_transformed, "segments": len(segs_b), "kept": idx_out-1})
        if fit_affine:
            a, b = fit_affine_from_pairs(xs, ys, (ws if weight_pairs else None))
            # Clarify mapping direction: y (file_b) ≈ a * x (file_a) + b
            # Spread samples across the full index range (early→late)
            sample_n = min(24, len(equations))
            if sample_n > 1:
                idxs = sorted({int(round(k*(len(equations)-1)/(sample_n-1))) for k in range(sample_n)})
            else:
                idxs = [0]
            sample = [{
                "i": int(i),
                "x": round(float(equations[i]["x"]), 3),
                "y": round(float(equations[i]["y"]), 3),
                "w": (round(float(equations[i]["w"]), 3) if weight_pairs else 1.0),
                "a_range": equations[i]["a_range"],
                "b_range": equations[i]["b_range"],
                "kind": equations[i]["kind"],
            } for i in idxs]
            print({
                "affine": {"a": round(a, 6), "b": round(b, 6)},
                "pairs": len(equations),
                "equations_sample": sample,
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


@app.command(name="align-multiple")
def align_multiple_cmd(
	files: list[str] | None = typer.Argument(None, help="Subtitle files (.srt/.vtt) to align"),
	n: int = typer.Option(3, help="Character n-gram size for similarity"),
	gap_penalty: float = typer.Option(-0.4, help="Gap penalty for NW alignment"),
	min_sim: float = typer.Option(0.3, help="Hard floor for match acceptance (0..1)"),
	dir: str | None = typer.Option(None, "--dir", help="Folder containing candidate .srt/.vtt files"),
	recursive: bool = typer.Option(False, help="Recursively search for .srt/.vtt under --dir"),
	precomputed: bool = typer.Option(True, help="Use precomputed hashed n-gram vectors (faster)"),
	component_threshold: float = typer.Option(0.65, help="Edge threshold for largest connected component"),
	hardshift_threshold: float = typer.Option(0.9, help="Similarity threshold for hard anchors"),
	output_dir: str | None = typer.Option(None, "--output-dir", help="Directory to write aligned SRT files and metadata"),
) -> None:
	"""Align multiple subtitle files to a median-duration master clock using hard-anchor piecewise shifts."""
	import os
	import json
	from pathlib import Path as _Path
	from .parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	
	# Collect files
	if files is None:
		files = []
	if dir:
		import glob
		pattern = "**/*.srt" if recursive else "*.srt"
		dir_files = glob.glob(os.path.join(dir, pattern), recursive=recursive)
		pattern = "**/*.vtt" if recursive else "*.vtt"
		dir_files.extend(glob.glob(os.path.join(dir, pattern), recursive=recursive))
		files.extend(dir_files)
	
	if len(files) < 2:
		print("[red]Error:[/red] Need at least 2 subtitle files")
		raise typer.Exit(1)
	
	print(f"[blue]Computing similarity matrix for {len(files)} files...[/blue]")
	
	# Load texts and intervals from files
	def load_norm_texts_and_intervals(file_path: str) -> tuple[list[str], list[tuple[float, float]]]:
		p = _Path(file_path)
		data = p.read_bytes()
		ext = p.suffix.lower().lstrip(".")
		if ext == "srt":
			segs = parse_srt_bytes(data)
		elif ext == "vtt":
			segs = parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
		texts = [
			normalize_subtitle_text(s.text or "")
			for s in segs
		]
		intervals = [(s.start_seconds, s.end_seconds) for s in segs]
		return texts, intervals
	
	texts_list = []
	intervals_list = []
	for file_path in files:
		t, iv = load_norm_texts_and_intervals(file_path)
		texts_list.append(t)
		intervals_list.append(iv)
	
	# Compute similarity matrix
	if precomputed:
		similarity_matrix, _details, _timing = compute_similarity_matrix_precomputed(
			texts_list, intervals_list, n=n, gap_penalty=gap_penalty, min_sim=min_sim
		)
	else:
		similarity_matrix, _details = compute_similarity_matrix(
			texts_list, intervals_list, n=n, gap_penalty=gap_penalty, min_sim=min_sim
		)
	
	# Find largest connected component
	component_indices = largest_connected_component(similarity_matrix, threshold=component_threshold)
	
	if len(component_indices) < 2:
		print(f"[red]Error:[/red] Largest connected component has only {len(component_indices)} files (need ≥2)")
		print("Consider lowering --component-threshold")
		raise typer.Exit(1)
	
	print(f"[green]Found connected component with {len(component_indices)} files[/green]")
	for idx in component_indices:
		print(f"  • {files[idx]}")
	
	# Select master clock based on median duration
	master_index, master_file = select_master_clock_by_median_duration(files, component_indices)
	print(f"[yellow]Master clock:[/yellow] {master_file}")
	
	# Align all files to master clock
	print(f"[blue]Aligning {len(component_indices)} files to master clock...[/blue]")
	results = align_multiple_subtitles_to_master(
		files,
		component_indices,
		master_index,
		n=n,
		gap_penalty=gap_penalty,
		min_sim=min_sim,
		hardshift_threshold=hardshift_threshold,
	)
	
	# Print summary
	print("\n[green]Alignment Results:[/green]")
	successful = 0
	failed = 0
	for file_path, transform in results["transforms"].items():
		if transform["type"] == "failed":
			print(f"  [red]✗[/red] {os.path.basename(file_path)} (failed: {transform['error']})")
			failed += 1
		elif transform["type"] == "identity":
			print(f"  [yellow]◯[/yellow] {os.path.basename(file_path)} (master)")
			successful += 1
		else:
			num_anchors = transform.get("num_anchors", 0)
			print(f"  [green]✓[/green] {os.path.basename(file_path)} ({num_anchors} anchors)")
			successful += 1
	
	print(f"\n[blue]Summary:[/blue] {successful} successful, {failed} failed")
	
	# Write output if requested
	if output_dir:
		output_path = _Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)
		
		# Write metadata
		metadata_file = output_path / "alignment_metadata.json"
		with metadata_file.open("w", encoding="utf-8") as f:
			json.dump(results, f, indent=2, default=str)
		print(f"[green]Wrote alignment metadata to[/green] {metadata_file}")
		
		# Write transformed SRT files
		for file_path, transform in results["transforms"].items():
			if transform["type"] in ["identity", "piecewise_shift"]:
				try:
					# Load original segments
					def load_segments(file_path: str):
						p = _Path(file_path)
						data = p.read_bytes()
						ext = p.suffix.lower().lstrip(".")
						if ext == "srt":
							return parse_srt_bytes(data)
						elif ext == "vtt":
							return parse_vtt_bytes(data)
						else:
							raise ValueError(f"Unsupported extension: {ext}")
					
					original_segments = load_segments(file_path)
					
					if transform["type"] == "piecewise_shift":
						# Apply transform
						transformed_intervals = apply_piecewise_shift_to_intervals(
							[(seg.start_seconds, seg.end_seconds) for seg in original_segments],
							transform,
						)
						# Create new segments with transformed times
						from .parsers.subtitles import Segment
						transformed_segments = [
							Segment(
								start_seconds=new_start,
								end_seconds=new_end,
								text=original_segments[i].text
							)
							for i, (new_start, new_end) in enumerate(transformed_intervals)
							if i < len(original_segments)
						]
					else:
						# Identity transform - use original
						transformed_segments = original_segments
					
					# Write transformed SRT
					output_file = output_path / f"{_Path(file_path).stem}_aligned.srt"
					
					# Convert segments to SRT format and write
					import srt
					from datetime import timedelta
					
					srt_items = []
					for i, seg in enumerate(transformed_segments):
						start_td = timedelta(seconds=seg.start_seconds)
						end_td = timedelta(seconds=seg.end_seconds)
						srt_items.append(srt.Subtitle(
							index=i+1,
							start=start_td,
							end=end_td,
							content=seg.text
						))
					
					with output_file.open("w", encoding="utf-8") as f:
						f.write(srt.compose(srt_items))
					
					print(f"[green]Wrote aligned SRT to[/green] {output_file}")
					
				except Exception as e:
					print(f"[red]Warning: Failed to write transformed SRT for {os.path.basename(file_path)}: {e}[/red]")
