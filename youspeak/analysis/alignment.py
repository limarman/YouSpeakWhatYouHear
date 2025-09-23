"""Simple FFT-based alignment utilities for subtitle intervals.

This module provides a minimal pipeline to estimate per-candidate time shifts
from binary activity timelines derived from intervals, using:

- Pairwise cross-correlation (via FFT) to find best lag at resolution ``dt``
- Unweighted global least-squares to solve per-candidate shifts

Conventions:
- For an edge (i, j), the estimated lag ``d_ij`` approximates ``s_j - s_i``.
- To align, subtract each candidate's shift from its timestamps.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import re
import unicodedata
from collections import Counter
import math
import time


def normalize_subtitle_text(
	text: str,
	*,
	remove_hearing_impaired: bool = True,
	strip_speaker_labels: bool = True,
	normalize_numbers: str = "keep",  # "keep" | "token"
	drop_punctuation_mode: str = "most",  # "most" | "all"
	keep_intraword_apostrophe_hyphen: bool = True,
) -> str:
	"""Normalize subtitle cue text for language-agnostic similarity.

	Steps (configurable):
	- Unicode NFKC, casefold
	- Strip HTML/ASS tags, music glyphs
	- Remove hearing-impaired annotations like [music], (laughs)
	- Drop leading speaker labels/dashes at cue start
	- Canonicalize/trim punctuation; optionally drop most punctuation
	- Normalize numbers (optional)
	- Collapse whitespace
	"""
	if not text:
		return ""
	# unify newlines to spaces early
	s = text.replace("\n", " ")
	# Unicode normalization and casefold
	s = unicodedata.normalize("NFKC", s).casefold()
	# Remove HTML-like tags
	s = re.sub(r"<[^>]+>", " ", s)
	# Remove ASS/SSA style tags like {\an8} or {italic}
	s = re.sub(r"\{\\[^}]*\}", " ", s)
	s = re.sub(r"\{[^}]*\}", " ", s)
	# Remove music glyphs
	s = s.replace("♪", " ").replace("♫", " ")
	# Collapse repeated punctuation before other ops
	s = re.sub(r"([!?.,])\1{1,}", r"\1", s)
	# Remove hearing-impaired annotations: [ ... ] and ( ... ) (non-nested simple; repeat until stable)
	if remove_hearing_impaired:
		prev = None
		while prev != s:
			prev = s
			s = re.sub(r"\[[^\[\]]*\]", " ", s)
			s = re.sub(r"\([^()]*\)", " ", s)
	# Drop leading dashes/speaker labels at the very start
	if strip_speaker_labels:
		# Leading dashes like "- ", "– ", "— "
		s = re.sub(r"^(?:[-–—]\s*)+", "", s)
		# Leading NAME: label (simple Latin heuristic)
		s = re.sub(r"^[a-z][a-z .'-]{0,20}:\s+", "", s)
	# Canonicalize quotes/dashes
	replacements = {
		"“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
		"’": "'", "‘": "'",
		"—": "-", "–": "-",
	}
	for k, v in replacements.items():
		s = s.replace(k, v)
	# Numbers
	if normalize_numbers == "token":
		s = re.sub(r"\d+", "<num>", s)
	# Punctuation handling
	if drop_punctuation_mode not in ("most", "all"):
		drop_punctuation_mode = "most"
	if drop_punctuation_mode == "all":
		# Remove everything that's not letter/number/space
		s = re.sub(r"[^\w\s]", " ", s)
	else:
		# Remove punctuation except apostrophe and hyphen initially
		s = re.sub(r"[^\w\s'\-]", " ", s)
		if keep_intraword_apostrophe_hyphen:
			# Remove apostrophes/hyphens not between alphanumerics
			s = re.sub(r"(?<![\w])'(?![\w])", " ", s)  # lone '
			s = re.sub(r"(?<![\w])-(?![\w])", " ", s)  # lone -
			s = re.sub(r"(?<![\w])'(?=[\w])", " ", s)
			s = re.sub(r"(?<=[\w])'(?![\w])", " ", s)
			s = re.sub(r"(?<![\w])-(?=[\w])", " ", s)
			s = re.sub(r"(?<=[\w])-(?![\w])", " ", s)
		else:
			# Remove all apostrophes/hyphens too
			s = re.sub(r"['\-]", " ", s)
	# Collapse whitespace
	s = re.sub(r"\s+", " ", s).strip()
	return s


def char_ngram_cosine_similarity(
	text_a: str,
	text_b: str,
	*,
	n: int = 3,
	use_hashing: bool = False,
	hash_dim: int = 1 << 15,
) -> float:
	"""Compute character n-gram cosine similarity between two already-normalized texts.

	This helper does NOT perform normalization. Call ``normalize_subtitle_text``
	beforehand if needed.
	"""

	if n <= 0:
		return 0.0
	if len(text_a) < n or len(text_b) < n:
		return 0.0

	if not use_hashing:
		# Exact n-gram bag via Counter
		grams_a = Counter(text_a[i : i + n] for i in range(len(text_a) - n + 1))
		grams_b = Counter(text_b[i : i + n] for i in range(len(text_b) - n + 1))
		# Dot product over intersection keys
		common_keys = grams_a.keys() & grams_b.keys()
		dot = sum(grams_a[k] * grams_b[k] for k in common_keys)
		norm_a = math.sqrt(sum(v * v for v in grams_a.values()))
		norm_b = math.sqrt(sum(v * v for v in grams_b.values()))
		if norm_a == 0.0 or norm_b == 0.0:
			return 0.0
		return float(dot / (norm_a * norm_b))
	else:
		# Hashed vector representation to fixed dimension
		vec_a = np.zeros(hash_dim, dtype=np.float32)
		vec_b = np.zeros(hash_dim, dtype=np.float32)
		for i in range(len(text_a) - n + 1):
			g = text_a[i : i + n]
			h = hash(g) % hash_dim
			vec_a[h] += 1.0
		for i in range(len(text_b) - n + 1):
			g = text_b[i : i + n]
			h = hash(g) % hash_dim
			vec_b[h] += 1.0
		norm_a = float(np.linalg.norm(vec_a))
		norm_b = float(np.linalg.norm(vec_b))
		if norm_a == 0.0 or norm_b == 0.0:
			return 0.0
		return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def needleman_wunsch_align(
	texts_a: Sequence[str],
	texts_b: Sequence[str],
	*,
	n: int = 3,
	gap_penalty: float = -0.4,
	use_hashing: bool = False,
	hash_dim: int = 1 << 15,
	min_sim: float = 0.3,
) -> Tuple[List[Tuple[int | None, int | None]], np.ndarray]:
	"""Simple Needleman–Wunsch global alignment for cue texts.

	Returns (alignment, S) where alignment is a list of (i,j) index pairs with
	None indicating a gap, and S is the similarity matrix of shape (m,n).
	"""
	m = len(texts_a)
	nn = len(texts_b)
	# similarity matrix
	S = np.zeros((m, nn), dtype=np.float32)
	for i in range(m):
		ai = texts_a[i]
		for j in range(nn):
			bj = texts_b[j]
			S[i, j] = char_ngram_cosine_similarity(
				ai, bj, n=n, use_hashing=use_hashing, hash_dim=hash_dim
			)
	# DP
	dp = np.full((m + 1, nn + 1), -1e9, dtype=np.float32)
	bt_i = np.full((m + 1, nn + 1), -1, dtype=np.int32)
	bt_j = np.full((m + 1, nn + 1), -1, dtype=np.int32)
	dp[0, 0] = 0.0
	for i in range(1, m + 1):
		dp[i, 0] = dp[i - 1, 0] + gap_penalty
		bt_i[i, 0] = i - 1
		bt_j[i, 0] = 0
	for j in range(1, nn + 1):
		dp[0, j] = dp[0, j - 1] + gap_penalty
		bt_i[0, j] = 0
		bt_j[0, j] = j - 1
	for i in range(1, m + 1):
		for j in range(1, nn + 1):
			# Apply hard floor: forbid matches below min_sim
			sim = float(S[i - 1, j - 1])
			match = dp[i - 1, j - 1] + sim if sim >= min_sim else -1e12
			delete = dp[i - 1, j] + gap_penalty
			insert = dp[i, j - 1] + gap_penalty
			best = match
			pi, pj = i - 1, j - 1
			if delete > best:
				best = delete
				pi, pj = i - 1, j
			if insert > best:
				best = insert
				pi, pj = i, j - 1
			dp[i, j] = best
			bt_i[i, j] = pi
			bt_j[i, j] = pj
	# backtrack
	i, j = m, nn
	align: List[Tuple[int | None, int | None]] = []
	while i > 0 or j > 0:
		pi, pj = bt_i[i, j], bt_j[i, j]
		if pi == i - 1 and pj == j - 1:
			align.append((i - 1, j - 1))
		elif pi == i - 1 and pj == j:
			align.append((i - 1, None))
		else:
			align.append((None, j - 1))
		i, j = pi, pj
	align.reverse()
	return align, S


def needleman_wunsch_from_similarity(
	S: np.ndarray,
	*,
	gap_penalty: float = -0.4,
	min_sim: float = 0.3,
) -> List[Tuple[int | None, int | None]]:
	"""Run Needleman–Wunsch DP given a precomputed similarity matrix S (m x n)."""
	m, nn = S.shape
	dp = np.full((m + 1, nn + 1), -1e9, dtype=np.float32)
	bt_i = np.full((m + 1, nn + 1), -1, dtype=np.int32)
	bt_j = np.full((m + 1, nn + 1), -1, dtype=np.int32)
	dp[0, 0] = 0.0
	for i in range(1, m + 1):
		dp[i, 0] = dp[i - 1, 0] + gap_penalty
		bt_i[i, 0] = i - 1
		bt_j[i, 0] = 0
	for j in range(1, nn + 1):
		dp[0, j] = dp[0, j - 1] + gap_penalty
		bt_i[0, j] = 0
		bt_j[0, j] = j - 1
	for i in range(1, m + 1):
		for j in range(1, nn + 1):
			sim = float(S[i - 1, j - 1])
			match = dp[i - 1, j - 1] + sim if sim >= min_sim else -1e12
			delete = dp[i - 1, j] + gap_penalty
			insert = dp[i, j - 1] + gap_penalty
			best = match
			pi, pj = i - 1, j - 1
			if delete > best:
				best = delete
				pi, pj = i - 1, j
			if insert > best:
				best = insert
				pi, pj = i, j - 1
			dp[i, j] = best
			bt_i[i, j] = pi
			bt_j[i, j] = pj
	# backtrack
	i, j = m, nn
	align: List[Tuple[int | None, int | None]] = []
	while i > 0 or j > 0:
		pi, pj = bt_i[i, j], bt_j[i, j]
		if pi == i - 1 and pj == j - 1:
			align.append((i - 1, j - 1))
		elif pi == i - 1 and pj == j:
			align.append((i - 1, None))
		else:
			align.append((None, j - 1))
		i, j = pi, pj
	align.reverse()
	return align


def compute_match_blocks(
	alignment: Sequence[Tuple[int | None, int | None]],
	S: np.ndarray,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
	"""Collapse contiguous diagonal matches into simple blocks.

	Returns a list of blocks as ((i_start, i_end), (j_start, j_end), mean_sim).
	Only strictly consecutive matches (i and j advancing by 1 each step) are
	merged. Gaps break blocks. This is intentionally simple.
	"""
	blocks: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
	i0 = j0 = None  # type: ignore
	prev_i = prev_j = None
	sims_sum = 0.0
	count = 0
	for pair in alignment:
		ai, bj = pair
		if ai is not None and bj is not None:
			if (
				prev_i is not None
				and prev_j is not None
				and ai == prev_i + 1
				and bj == prev_j + 1
			):
				# continue current block
				prev_i, prev_j = ai, bj
				sims_sum += float(S[ai, bj])
				count += 1
			else:
				# close previous block if exists
				if count > 0 and i0 is not None and j0 is not None and prev_i is not None and prev_j is not None:
					blocks.append(((i0, prev_i), (j0, prev_j), sims_sum / max(1, count)))
				# start new block
				i0, j0 = ai, bj
				prev_i, prev_j = ai, bj
				sims_sum = float(S[ai, bj])
				count = 1
		else:
			# gap breaks block
			if count > 0 and i0 is not None and j0 is not None and prev_i is not None and prev_j is not None:
				blocks.append(((i0, prev_i), (j0, prev_j), sims_sum / max(1, count)))
			i0 = j0 = None  # type: ignore
			prev_i = prev_j = None
			sims_sum = 0.0
			count = 0
	# close tail
	if count > 0 and i0 is not None and j0 is not None and prev_i is not None and prev_j is not None:
		blocks.append(((i0, prev_i), (j0, prev_j), sims_sum / max(1, count)))
	return blocks


def compute_match_blocks_postmerge(
	alignment: Sequence[Tuple[int | None, int | None]],
	S: np.ndarray,
	texts_a: Sequence[str],
	texts_b: Sequence[str],
	*,
	n: int = 3,
	min_improve: float = 0.05,
	use_hashing: bool = False,
	hash_dim: int = 1 << 15,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
	"""Build blocks with simple post-merge of adjacent gaps into one-to-many or many-to-one.

	Heuristic: if match (i,j) is immediately followed by a run of insertions (None, j+1..j+k),
	merge as A[i..i] ↔ B[j..j+k] if cosine(A[i], concat(B[j..j+k])) improves over S[i,j] by >= min_improve.
	Similarly for deletions after (i,j): merge A[i..i+k] ↔ B[j..j]. Otherwise, also merge consecutive
	strict diagonal matches into a block averaging S.
	"""
	blocks: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
	L = len(alignment)
	pos = 0
	while pos < L:
		ai, bj = alignment[pos]
		if ai is not None and bj is not None:
			# Try one-to-many: gather subsequent insertions in B
			ins_count = 0
			p = pos + 1
			cur_j = bj
			while p < L and alignment[p][0] is None and alignment[p][1] is not None and alignment[p][1] == cur_j + 1:
				cur_j = alignment[p][1]  # type: ignore
				ins_count += 1
				p += 1
			merged = False
			if ins_count > 0:
				# score concatenated B[j..j+ins_count] vs A[ai]
				b_concat = " ".join(texts_b[bj : bj + ins_count + 1])
				base = float(S[ai, bj])
				comb = char_ngram_cosine_similarity(texts_a[ai], b_concat, n=n, use_hashing=use_hashing, hash_dim=hash_dim)
				if comb >= base + min_improve:
					blocks.append(((ai, ai), (bj, bj + ins_count), float(comb)))
					pos = p
					merged = True
			if merged:
				continue
			# Try many-to-one: gather subsequent deletions in B (i advances, j fixed)
			del_count = 0
			p = pos + 1
			cur_i = ai
			while p < L and alignment[p][1] is None and alignment[p][0] is not None and alignment[p][0] == cur_i + 1:
				cur_i = alignment[p][0]  # type: ignore
				del_count += 1
				p += 1
			if del_count > 0:
				a_concat = " ".join(texts_a[ai : ai + del_count + 1])
				base = float(S[ai, bj])
				comb = char_ngram_cosine_similarity(a_concat, texts_b[bj], n=n, use_hashing=use_hashing, hash_dim=hash_dim)
				if comb >= base + min_improve:
					blocks.append(((ai, ai + del_count), (bj, bj), float(comb)))
					pos = p
					continue
			# Else, also try to merge straight diagonal run starting here
			i0, j0 = ai, bj
			sum_sim = float(S[ai, bj])
			cnt = 1
			p = pos + 1
			prev_i, prev_j = ai, bj
			while p < L:
				nai, nbj = alignment[p]
				if nai is not None and nbj is not None and nai == prev_i + 1 and nbj == prev_j + 1:
					sum_sim += float(S[nai, nbj])
					cnt += 1
					prev_i, prev_j = nai, nbj
					p += 1
				else:
					break
			blocks.append(((i0, prev_i), (j0, prev_j), sum_sim / max(1, cnt)))
			pos = p
		else:
			# gap; skip
			pos += 1
	return blocks


def compute_match_blocks_localmerge(
	alignment: Sequence[Tuple[int | None, int | None]],
	S: np.ndarray,
	texts_a: Sequence[str],
	texts_b: Sequence[str],
	*,
	n: int = 3,
	min_improve: float = 0.05,
	use_hashing: bool = False,
	hash_dim: int = 1 << 15,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
	"""Local bidirectional merge around each matched pair, checking immediate gaps.

	For a seed match (i,j), try to absorb at most one neighbor from the left and/or
	from the right if concatenating improves cosine similarity by >= min_improve.
	Similarity for merge decisions is computed on texts with all Unicode
	separators and punctuation removed (language-agnostic), to avoid space issues
	with CJK and punctuation noise. The reported block score is also the stripped
	version.
	Does not chain across multiple blocks; intended for visualization and light grouping.
	"""
	blocks: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []

	def _strip_sep_punct(s: str) -> str:
		# Remove Unicode separators (Z*) and punctuation (P*)
		return "".join(ch for ch in s if (unicodedata.category(ch))[0] not in ("Z", "P"))

	def _cos(a_txt: str, b_txt: str) -> float:
		return char_ngram_cosine_similarity(_strip_sep_punct(a_txt), _strip_sep_punct(b_txt), n=n, use_hashing=use_hashing, hash_dim=hash_dim)
	L = len(alignment)
	for pos, pair in enumerate(alignment):
		ai, bj = pair
		if ai is None or bj is None:
			continue
		# seed
		a_i0 = ai
		a_i1 = ai
		b_j0 = bj
		b_j1 = bj
		# base score (stripped)
		base = _cos(texts_a[ai], texts_b[bj])
		# Left neighbor check (prefer the better of A-left or B-left if both present)
		best_left = (0.0, None)  # (improvement, which_side)
		if pos - 1 >= 0:
			lai, lbj = alignment[pos - 1]
			# B-left: (None, bj-1)
			if lai is None and lbj is not None and lbj == bj - 1:
				b_concat = (texts_b[lbj] + " " + (texts_b[b_j0] if b_j0 == b_j1 else " ".join(texts_b[b_j0:b_j1 + 1]))).strip()
				sc = _cos(texts_a[ai], b_concat)
				best_left = max(best_left, (sc - base, ("B", lbj)), key=lambda x: x[0])
			# A-left: (ai-1, None)
			if lbj is None and lai is not None and lai == ai - 1:
				a_concat = (texts_a[lai] + " " + (texts_a[a_i0] if a_i0 == a_i1 else " ".join(texts_a[a_i0:a_i1 + 1]))).strip()
				sc = _cos(a_concat, texts_b[bj])
				best_left = max(best_left, (sc - base, ("A", lai)), key=lambda x: x[0])
		if best_left[0] >= min_improve and best_left[1] is not None:
			which, idx = best_left[1]
			if which == "B":
				b_j0 = int(idx)
				base = _cos(texts_a[ai], " ".join(texts_b[b_j0:b_j1 + 1]))
			else:
				a_i0 = int(idx)
				base = _cos(" ".join(texts_a[a_i0:a_i1 + 1]), texts_b[bj])
		# Right neighbor check
		best_right = (0.0, None)
		if pos + 1 < L:
			rai, rbj = alignment[pos + 1]
			# B-right: (None, bj+1)
			if rai is None and rbj is not None and rbj == bj + 1:
				b_concat = (" ".join(texts_b[b_j0:b_j1 + 1]) + " " + texts_b[rbj]).strip()
				sc = _cos(texts_a[ai] if a_i0 == a_i1 else " ".join(texts_a[a_i0:a_i1 + 1]), b_concat)
				best_right = max(best_right, (sc - base, ("B", rbj)), key=lambda x: x[0])
			# A-right: (ai+1, None)
			if rbj is None and rai is not None and rai == ai + 1:
				a_concat = (" ".join(texts_a[a_i0:a_i1 + 1]) + " " + texts_a[rai]).strip()
				sc = _cos(a_concat, texts_b[bj] if b_j0 == b_j1 else " ".join(texts_b[b_j0:b_j1 + 1]))
				best_right = max(best_right, (sc - base, ("A", rai)), key=lambda x: x[0])
		if best_right[0] >= min_improve and best_right[1] is not None:
			which, idx = best_right[1]
			if which == "B":
				b_j1 = int(idx)
			else:
				a_i1 = int(idx)
		# Final score on the (possibly) merged local block
		a_text = texts_a[ai] if (a_i0 == a_i1) else " ".join(texts_a[a_i0:a_i1 + 1])
		b_text = texts_b[bj] if (b_j0 == b_j1) else " ".join(texts_b[b_j0:b_j1 + 1])
		score = _cos(a_text, b_text)
		blocks.append(((a_i0, a_i1), (b_j0, b_j1), float(score)))
	return blocks


def compute_match_blocks_growmerge(
    alignment: Sequence[Tuple[int | None, int | None]],
    S: np.ndarray,
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    *,
    n: int = 3,
    min_improve: float = 0.02,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Two-pass grow-merge: expand left, then right, only along one side per pass.

    For each seed match (i,j):
    1) LEFT PASS: If the immediate left neighbor is a gap on A or B side, pick the side
       with the higher immediate improvement (if both exist). While the next left neighbor
       on that side is a gap and concatenation improves by >= min_improve, absorb it.
    2) RIGHT PASS: Repeat the same to the right.

    Only gaps are consumed; blocks do not cross other matches. Covered alignment indices
    are marked used to avoid duplicate emission.
    """
    def _strip(s: str) -> str:
        return strip_separators_and_punct(s)

    def _cos(a_txt: str, b_txt: str) -> float:
        return char_ngram_cosine_similarity(_strip(a_txt), _strip(b_txt), n=n)

    used = [False] * len(alignment)
    blocks: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
    for idx, (ai, bj) in enumerate(alignment):
        if used[idx] or ai is None or bj is None:
            continue
        i0 = i1 = int(ai)
        j0 = j1 = int(bj)
        left_idx = right_idx = idx
        a_txt = " ".join(texts_a[i0 : i1 + 1])
        b_txt = " ".join(texts_b[j0 : j1 + 1])
        score = _cos(a_txt, b_txt)

        # LEFT PASS
        while True:
            if left_idx - 1 < 0 or used[left_idx - 1]:
                break
            lai, lbj = alignment[left_idx - 1]
            # compute candidate deltas for available sides
            best_side = None
            best_new_score = score
            # A-left gap available?
            if lai is not None and lbj is None and int(lai) == i0 - 1:
                a_cand = " ".join(texts_a[i0 - 1 : i1 + 1])
                scA = _cos(a_cand, b_txt)
                if scA - score >= min_improve and scA > best_new_score:
                    best_side = "A_LEFT"
                    best_new_score = scA
            # B-left gap available?
            if lbj is not None and lai is None and int(lbj) == j0 - 1:
                b_cand = " ".join(texts_b[j0 - 1 : j1 + 1])
                scB = _cos(a_txt, b_cand)
                if scB - score >= min_improve and scB > best_new_score:
                    best_side = "B_LEFT"
                    best_new_score = scB
            if best_side is None:
                break
            if best_side == "A_LEFT":
                i0 -= 1
                left_idx -= 1
                a_txt = " ".join(texts_a[i0 : i1 + 1])
            else:
                j0 -= 1
                left_idx -= 1
                b_txt = " ".join(texts_b[j0 : j1 + 1])
            score = best_new_score

        # RIGHT PASS
        while True:
            if right_idx + 1 >= len(alignment) or used[right_idx + 1]:
                break
            rai, rbj = alignment[right_idx + 1]
            best_side = None
            best_new_score = score
            # A-right gap?
            if rai is not None and rbj is None and int(rai) == i1 + 1:
                a_cand = " ".join(texts_a[i0 : i1 + 2])
                scA = _cos(a_cand, b_txt)
                if scA - score >= min_improve and scA > best_new_score:
                    best_side = "A_RIGHT"
                    best_new_score = scA
            # B-right gap?
            if rbj is not None and rai is None and int(rbj) == j1 + 1:
                b_cand = " ".join(texts_b[j0 : j1 + 2])
                scB = _cos(a_txt, b_cand)
                if scB - score >= min_improve and scB > best_new_score:
                    best_side = "B_RIGHT"
                    best_new_score = scB
            if best_side is None:
                break
            if best_side == "A_RIGHT":
                i1 += 1
                right_idx += 1
                a_txt = " ".join(texts_a[i0 : i1 + 1])
            else:
                j1 += 1
                right_idx += 1
                b_txt = " ".join(texts_b[j0 : j1 + 1])
            score = best_new_score

        blocks.append(((i0, i1), (j0, j1), float(score)))
        for k in range(left_idx, right_idx + 1):
            used[k] = True
    return blocks


def blocks_to_time_pairs(
	blocks: Sequence[Tuple[Tuple[int, int], Tuple[int, int], float]],
	intervals_a: Sequence[Tuple[float, float]],
	intervals_b: Sequence[Tuple[float, float]],
	*,
	include_start_end: bool = True,
	include_center: bool = False,
) -> Tuple[List[float], List[float], List[float]]:
	"""Convert merged blocks to time point pairs for affine fitting.

	For each block ((i0,i1),(j0,j1), score), compute union intervals in A and B and
	return pairs (x=tA, y=tB) with weights=score. If include_start_end, emits
	start→start and end→end. If include_center, emits center→center as well.
	"""
	xs: List[float] = []
	ys: List[float] = []
	ws: List[float] = []
	for (i0, i1), (j0, j1), sc in blocks:
		if i0 < 0 or i1 >= len(intervals_a) or j0 < 0 or j1 >= len(intervals_b):
			continue
		a_start = min(intervals_a[i][0] for i in range(i0, i1 + 1))
		a_end = max(intervals_a[i][1] for i in range(i0, i1 + 1))
		b_start = min(intervals_b[j][0] for j in range(j0, j1 + 1))
		b_end = max(intervals_b[j][1] for j in range(j0, j1 + 1))
		if include_start_end:
			xs.append(float(a_start)); ys.append(float(b_start)); ws.append(float(sc))
			xs.append(float(a_end)); ys.append(float(b_end)); ws.append(float(sc))
		if include_center:
			a_c = 0.5 * (a_start + a_end)
			b_c = 0.5 * (b_start + b_end)
			xs.append(float(a_c)); ys.append(float(b_c)); ws.append(float(sc))
	return xs, ys, ws


def blocks_to_center_pairs(
	blocks: Sequence[Tuple[Tuple[int, int], Tuple[int, int], float]],
	intervals_a: Sequence[Tuple[float, float]],
	intervals_b: Sequence[Tuple[float, float]],
) -> Tuple[List[float], List[float], List[float]]:
	"""Extract centers-only (x=tA_center, y=tB_center) with weights from blocks."""
	xs: List[float] = []
	ys: List[float] = []
	ws: List[float] = []
	for (i0, i1), (j0, j1), sc in blocks:
		if i0 < 0 or i1 >= len(intervals_a) or j0 < 0 or j1 >= len(intervals_b):
			continue
		a_start = min(intervals_a[i][0] for i in range(i0, i1 + 1))
		a_end = max(intervals_a[i][1] for i in range(i0, i1 + 1))
		b_start = min(intervals_b[j][0] for j in range(j0, j1 + 1))
		b_end = max(intervals_b[j][1] for j in range(j0, j1 + 1))
		a_c = 0.5 * (a_start + a_end)
		b_c = 0.5 * (b_start + b_end)
		xs.append(float(a_c))
		ys.append(float(b_c))
		ws.append(float(sc))
	return xs, ys, ws


def fit_piecewise_affine(
	xs: Sequence[float],
	ys: Sequence[float],
	ws: Sequence[float],
	*,
	min_span_seconds: float = 60.0,
	min_strong_matches: int = 10,
	strong_sim_threshold: float = 0.9,
) -> Tuple[List[Tuple[float, float, float, float]], List[dict]]:
	"""Deprecated signature; use the version below with use_shift_only flag."""
	return fit_piecewise_affine(xs, ys, ws,
		min_span_seconds=min_span_seconds,
		min_strong_matches=min_strong_matches,
		strong_sim_threshold=strong_sim_threshold,
		use_shift_only=False,
		pair_blocks=None)


def fit_piecewise_affine(
	xs: Sequence[float],
	ys: Sequence[float],
	ws: Sequence[float],
	*,
	min_span_seconds: float = 60.0,
	min_strong_matches: int = 10,
	strong_sim_threshold: float = 0.9,
	use_shift_only: bool = False,
	pair_blocks: Sequence[int] | None = None,
) -> Tuple[List[Tuple[float, float, float, float]], List[dict]]:
	"""Fit piecewise affine y≈a*x+b over sorted pairs, left-to-right.

	Segments are grown until both conditions are met:
	- span_seconds >= min_span_seconds
	- count_strong (w>=strong_sim_threshold) >= min_strong_matches

	Returns (segments, diag) where segments is a list of (x_start, x_end, a, b)
	and diag is a list of per-segment diagnostics dicts.
	"""
	if not xs:
		return [], []
	# sort by x
	if pair_blocks is None:
		pairs = sorted([(xs[i], ys[i], ws[i], None) for i in range(len(xs))], key=lambda t: t[0])
	else:
		pairs = sorted([(xs[i], ys[i], ws[i], pair_blocks[i]) for i in range(len(xs))], key=lambda t: t[0])
	segments: List[Tuple[float, float, float, float]] = []
	diags: List[dict] = []
	N = len(pairs)
	start_idx = 0
	while start_idx < N:
		end_idx = start_idx
		x_start = pairs[start_idx][0]
		strong = 0
		while end_idx < N:
			x_end = pairs[end_idx][0]
			w = pairs[end_idx][2]
			if w >= strong_sim_threshold:
				strong += 1
			span_ok = (x_end - x_start) >= min_span_seconds
			count_ok = strong >= min_strong_matches
			end_idx += 1
			if span_ok and count_ok:
				break
		# If we hit EOF without satisfying, merge with previous (or take all remaining if first)
		if end_idx > N and not segments:
			end_idx = N
		elif end_idx > N and segments:
			# merge remainder into last segment by extending its x_end only; skip creating new
			last_xs, last_ys, last_ws = [], [], []
			for i in range(start_idx, N):
				last_xs.append(pairs[i][0])
				last_ys.append(pairs[i][1])
				last_ws.append(pairs[i][2])
			# Refit last segment using its original span plus remainder
			# Extract previous
			lx0, lx1, la, lb = segments[-1]
			# Rebuild combined arrays within [lx0.. pairs[N-1][0]]
			combined_xs = [x for x,y,w in pairs if lx0 <= x <= pairs[N-1][0]]
			combined_ys = [y for x,y,w in pairs if lx0 <= x <= pairs[N-1][0]]
			combined_ws = [w for x,y,w in pairs if lx0 <= x <= pairs[N-1][0]]
			# Weighted LS for y≈a*x+b
			A = np.stack([np.asarray(combined_xs), np.ones(len(combined_xs))], axis=1).astype(np.float64)
			W = np.sqrt(np.asarray(combined_ws, dtype=np.float64))
			Aw = A * W[:, None]
			Yw = np.asarray(combined_ys, dtype=np.float64) * W
			sol, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)
			la2 = float(sol[0]); lb2 = float(sol[1])
			segments[-1] = (lx0, pairs[N-1][0], la2, lb2)
			# diagnostics
			pred = la2 * np.asarray(combined_xs) + lb2
			res = (pred - np.asarray(combined_ys))
			diags[-1] = {**diags[-1], "refit_extended": True, "rmse": float(np.sqrt(np.mean(res*res))), "count": int(len(combined_xs))}
			break
		# Normal segment creation
		seg_pairs = pairs[start_idx:end_idx]
		seg_xs = [p[0] for p in seg_pairs]
		seg_ys = [p[1] for p in seg_pairs]
		seg_ws = [p[2] for p in seg_pairs]
		seg_b_start = seg_pairs[0][3]
		seg_b_end = seg_pairs[-1][3]
		if use_shift_only:
			# y ≈ x + b => b = weighted mean of (y - x)
			delta = np.asarray(seg_ys, dtype=np.float64) - np.asarray(seg_xs, dtype=np.float64)
			w_arr = np.asarray(seg_ws, dtype=np.float64)
			w_sum = float(np.sum(w_arr)) if np.isfinite(np.sum(w_arr)) and np.sum(w_arr) > 0 else float(len(seg_xs))
			b = float(np.sum(w_arr * delta) / w_sum)
			a = 1.0
		else:
			A = np.stack([np.asarray(seg_xs), np.ones(len(seg_xs))], axis=1).astype(np.float64)
			W = np.sqrt(np.asarray(seg_ws, dtype=np.float64))
			Aw = A * W[:, None]
			Yw = np.asarray(seg_ys, dtype=np.float64) * W
			sol, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)
			a = float(sol[0])
			b = float(sol[1])
		segments.append((seg_xs[0], seg_xs[-1], a, b))
		pred = a * np.asarray(seg_xs) + b
		res = (pred - np.asarray(seg_ys))
		diags.append({
			"x_start": float(seg_xs[0]), "x_end": float(seg_xs[-1]),
			"a": a, "b": b,
			"count": int(len(seg_xs)),
			"strong": int(sum(1 for w in seg_ws if w >= strong_sim_threshold)),
			"rmse": float(np.sqrt(np.mean(res*res))),
			"block_start": (int(seg_b_start) if seg_b_start is not None else None),
			"block_end": (int(seg_b_end) if seg_b_end is not None else None),
		})
		start_idx = end_idx
	return segments, diags


def fit_affine_from_pairs(
	xs: Sequence[float],
	ys: Sequence[float],
	ws: Sequence[float] | None = None,
) -> Tuple[float, float]:
	"""Fit y ≈ a*x + b with (optional) weights via least squares.

	If not enough variation in x, fallback to a=1, b=median(y-x).
	"""
	if not xs or not ys or len(xs) != len(ys):
		return 1.0, 0.0
	X = np.asarray(xs, dtype=np.float64)
	Y = np.asarray(ys, dtype=np.float64)
	if np.allclose(np.var(X), 0.0):
		return 1.0, float(np.median(Y - X))
	A = np.stack([X, np.ones_like(X)], axis=1)
	if ws is not None:
		W = np.sqrt(np.asarray(ws, dtype=np.float64))
		Aw = A * W[:, None]
		Yw = Y * W
		sol, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)
	else:
		sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
	a = float(sol[0])
	b = float(sol[1])
	return a, b


def _merge_and_sum(intervals: Sequence[Tuple[float, float]]) -> float:
	"""Compute total duration of the union of intervals."""
	if not intervals:
		return 0.0
	sorted_iv = sorted(((float(s), float(e)) for s, e in intervals if e > s), key=lambda x: x[0])
	merged: List[Tuple[float, float]] = []
	cs, ce = sorted_iv[0]
	for s, e in sorted_iv[1:]:
		if s <= ce:
			ce = max(ce, e)
		else:
			merged.append((cs, ce))
			cs, ce = s, e
	merged.append((cs, ce))
	return float(sum(e - s for s, e in merged))


def compute_block_similarity_and_coverage(
	blocks: Sequence[Tuple[Tuple[int, int], Tuple[int, int], float]],
	intervals_a: Sequence[Tuple[float, float]],
	intervals_b: Sequence[Tuple[float, float]],
	*,
	min_sim_for_coverage: float = 0.3,
) -> Tuple[float, float, float, float, float, float, float, float]:
	"""Compute (score_blocks, coverageA, coverageB, coverage, matchedA, matchedB, totalA, totalB).

	- score_blocks: weighted mean of block similarities (weights = average of A/B union durations per block)
	- coverageX: matched_time_X / total_time_X using union durations and only blocks with s_k >= min_sim_for_coverage
	"""
	# Total times (union over all intervals)
	totalA = _merge_and_sum(intervals_a)
	totalB = _merge_and_sum(intervals_b)
	# Weighted average over blocks
	num = 0.0
	den = 0.0
	# Collect matched intervals for coverage
	match_iv_a: List[Tuple[float, float]] = []
	match_iv_b: List[Tuple[float, float]] = []
	for (i0, i1), (j0, j1), sc in blocks:
		if i0 < 0 or i1 >= len(intervals_a) or j0 < 0 or j1 >= len(intervals_b):
			continue
		a_start = min(intervals_a[i][0] for i in range(i0, i1 + 1))
		a_end = max(intervals_a[i][1] for i in range(i0, i1 + 1))
		b_start = min(intervals_b[j][0] for j in range(j0, j1 + 1))
		b_end = max(intervals_b[j][1] for j in range(j0, j1 + 1))
		durA = max(0.0, float(a_end - a_start))
		durB = max(0.0, float(b_end - b_start))
		w = 0.5 * (durA + durB)
		if w > 0.0:
			num += w * float(sc)
			den += w
		if float(sc) >= min_sim_for_coverage:
			# For coverage, count only actual cue intervals, not the envelope, to avoid
			# bridging silent gaps. Append all constituent intervals.
			for i in range(i0, i1 + 1):
				match_iv_a.append((float(intervals_a[i][0]), float(intervals_a[i][1])))
			for j in range(j0, j1 + 1):
				match_iv_b.append((float(intervals_b[j][0]), float(intervals_b[j][1])))
	score_blocks = float(num / den) if den > 0 else 0.0
	matchedA = _merge_and_sum(match_iv_a)
	matchedB = _merge_and_sum(match_iv_b)
	coverageA = float(min(1.0, matchedA / totalA)) if totalA > 0 else 0.0
	coverageB = float(min(1.0, matchedB / totalB)) if totalB > 0 else 0.0
	coverage = float(min(coverageA, coverageB))
	return score_blocks, coverageA, coverageB, coverage, matchedA, matchedB, totalA, totalB


def compute_similarity_matrix(
	texts_list: Sequence[Sequence[str]],
	intervals_list: Sequence[Sequence[Tuple[float, float]]],
	*,
	n: int = 3,
	gap_penalty: float = -0.4,
	min_sim: float = 0.3,
	min_improve: float = 0.05,
) -> Tuple[np.ndarray, List[List[dict]]]:
	"""Compute pairwise similarity matrix using NW + local-merge blocks.

	Returns (M, details) where M[i,j] is combined=score_blocks*coverage in [0,1], and
	"details" holds dictionaries per pair with components like score_blocks and coverage.
	"""
	N = len(texts_list)
	M = np.zeros((N, N), dtype=np.float32)
	details: List[List[dict]] = [[{} for _ in range(N)] for __ in range(N)]
	for i in range(N):
		M[i, i] = 1.0
		details[i][i] = {"combined": 1.0, "score_blocks": 1.0, "coverage": 1.0, "coverageA": 1.0, "coverageB": 1.0}
	for i in range(N):
		for j in range(i + 1, N):
			Ai = texts_list[i]; Bi = texts_list[j]
			align, S = needleman_wunsch_align(Ai, Bi, n=n, gap_penalty=gap_penalty, min_sim=min_sim)
			blocks = compute_match_blocks_localmerge(align, S, Ai, Bi, n=n, min_improve=min_improve)
			sb, covA, covB, cov, *_ = compute_block_similarity_and_coverage(blocks, intervals_list[i], intervals_list[j])
			comb = float(sb * cov)
			M[i, j] = comb
			M[j, i] = comb
			det = {"combined": comb, "score_blocks": float(sb), "coverage": float(cov), "coverageA": float(covA), "coverageB": float(covB)}
			details[i][j] = det
			details[j][i] = det
	return M, details


def strip_separators_and_punct(text: str) -> str:
	"""Remove Unicode separators (Z*) and punctuation (P*) from text."""
	return "".join(ch for ch in text if (unicodedata.category(ch))[0] not in ("Z", "P"))


def build_hashed_ngram_vectors(
	texts: Sequence[str],
	*,
	n: int = 3,
	dim: int = 8192,
	strip_sep_punct: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Build hashed n-gram count vectors and norms for a list of texts."""
	m = len(texts)
	V = np.zeros((m, dim), dtype=np.float32)
	for i, s in enumerate(texts):
		ss = strip_separators_and_punct(s) if strip_sep_punct else s
		L = len(ss)
		if L < n:
			continue
		for k in range(L - n + 1):
			g = ss[k : k + n]
			h = hash(g) % dim
			V[i, h] += 1.0
	norms = np.linalg.norm(V, axis=1).astype(np.float32)
	return V, norms


def similarity_matrix_from_vectors(
	VA: np.ndarray,
	NA: np.ndarray,
	VB: np.ndarray,
	NB: np.ndarray,
) -> np.ndarray:
	"""Compute cosine similarity matrix between two hashed vector sets."""
	# Avoid divide-by-zero by replacing zero norms with 1
	NA_safe = NA.copy()
	NB_safe = NB.copy()
	NA_safe[NA_safe == 0] = 1.0
	NB_safe[NB_safe == 0] = 1.0
	Snum = VA @ VB.T  # (m x d) · (d x n) = (m x n)
	S = Snum / (NA_safe[:, None] * NB_safe[None, :])
	return S.astype(np.float32)


def compute_similarity_matrix_precomputed(
	texts_list: Sequence[Sequence[str]],
	intervals_list: Sequence[Sequence[Tuple[float, float]]],
	*,
	n: int = 3,
	gap_penalty: float = -0.4,
	min_sim: float = 0.3,
	min_improve: float = 0.05,
	vec_dim: int = 8192,
) -> Tuple[np.ndarray, List[List[dict]], dict]:
	"""Like compute_similarity_matrix, but uses precomputed hashed vectors for speed.

	Returns (M, details, timing) where timing includes per-phase elapsed seconds.
	"""
	N = len(texts_list)
	M = np.zeros((N, N), dtype=np.float32)
	details: List[List[dict]] = [[{} for _ in range(N)] for __ in range(N)]
	# Precompute vectors for all files
	t0 = time.time()
	vecs = []
	norms = []
	for texts in texts_list:
		V, Nrm = build_hashed_ngram_vectors(texts, n=n, dim=vec_dim, strip_sep_punct=False)
		vecs.append(V)
		norms.append(Nrm)
	prep_time = time.time() - t0
	# Pairwise
	align_time = 0.0
	for i in range(N):
		M[i, i] = 1.0
		details[i][i] = {"combined": 1.0, "score_blocks": 1.0, "coverage": 1.0, "coverageA": 1.0, "coverageB": 1.0}
	for i in range(N):
		for j in range(i + 1, N):
			S = similarity_matrix_from_vectors(vecs[i], norms[i], vecs[j], norms[j])
			t1 = time.time()
			align = needleman_wunsch_from_similarity(S, gap_penalty=gap_penalty, min_sim=min_sim)
			align_time += (time.time() - t1)
			blocks = compute_match_blocks_localmerge(align, S, texts_list[i], texts_list[j], n=n, min_improve=min_improve)
			sb, covA, covB, cov, *_ = compute_block_similarity_and_coverage(blocks, intervals_list[i], intervals_list[j])
			comb = float(sb * cov)
			M[i, j] = comb
			M[j, i] = comb
			det = {"combined": comb, "score_blocks": float(sb), "coverage": float(cov), "coverageA": float(covA), "coverageB": float(covB)}
			details[i][j] = det
			details[j][i] = det
	return M, details, {"precompute_seconds": prep_time, "align_seconds": align_time}


def largest_connected_component(
	M: np.ndarray,
	*,
	threshold: float,
) -> List[int]:
	"""Return indices of the largest connected component under similarity threshold.

	Undirected graph: edge between i and j (i!=j) if M[i,j] >= threshold.
	"""
	N = M.shape[0]
	visited = [False] * N
	best_comp: List[int] = []
	for s in range(N):
		if visited[s]:
			continue
		# BFS
		queue = [s]
		visited[s] = True
		comp = [s]
		while queue:
			u = queue.pop(0)
			row = M[u]
			for v in range(N):
				if u == v or visited[v]:
					continue
				if float(row[v]) >= threshold:
					visited[v] = True
					queue.append(v)
					comp.append(v)
		if len(comp) > len(best_comp):
			best_comp = comp
	return sorted(best_comp)


def compute_hardshift_transform(
	equations: Sequence[dict],
	intervals_a: Sequence[Tuple[float, float]],
	intervals_b: Sequence[Tuple[float, float]],
	*,
	threshold: float = 0.9,
) -> dict:
	"""Build a piecewise shift transform from B->A using hard anchors from equations.

	equations: list of dicts emitted by align-two with fields: x (tA), y (tB), w (sim), kind ('center'), i (optional index)
	threshold: minimum similarity to consider an anchor 'hard'.
	Returns a dict with boundaries (on B clock) and shifts (A-B) per segment.
	"""
	# Collect hard anchors with centers only
	hard: List[Tuple[float, float, int]] = []
	for idx, e in enumerate(equations):
		kind = e.get("kind")
		w = float(e.get("w", 0.0))
		if kind == "center" and w >= threshold:
			x = float(e.get("x", 0.0))  # tA
			y = float(e.get("y", 0.0))  # tB
			hard.append((y, x, int(e.get("i", idx))))
	hard.sort(key=lambda t: t[0])
	# Deduplicate equal B-times
	dedup: List[Tuple[float, float, int]] = []
	last_t: float | None = None
	for tB, tA, idx in hard:
		if last_t is None or tB > last_t + 1e-6:
			dedup.append((tB, tA, idx))
			last_t = tB
	hard = dedup
	if intervals_b:
		b_min = float(min(s for s, _ in intervals_b))
		b_max = float(max(e for _, e in intervals_b))
	else:
		b_min = 0.0
		b_max = 0.0
	if not hard:
		return {
			"type": "piecewise_shift",
			"from": "B",
			"to": "A",
			"threshold": threshold,
			"boundaries": [],
			"shifts": [],
			"anchors": [],
		}
	tBs = [h[0] for h in hard]
	shifts = [float(h[1] - h[0]) for h in hard]
	mids = [0.5 * (tBs[i] + tBs[i + 1]) for i in range(len(tBs) - 1)]
	boundaries = [b_min] + mids + [b_max]
	return {
		"type": "piecewise_shift",
		"from": "B",
		"to": "A",
		"threshold": threshold,
		"boundaries": [float(x) for x in boundaries],
		"shifts": [float(s) for s in shifts],
		"anchors": [
			{"t_file": float(h[0]), "t_ref": float(h[1]), "shift": float(h[1] - h[0]), "eq_index": int(h[2])}
			for h in hard
		],
	}


def find_shift_for_time(transform: dict, t: float) -> float:
	"""Return the shift (seconds) for time t using a piecewise_shift transform."""
	bounds = transform.get("boundaries", [])
	shifts = transform.get("shifts", [])
	if not bounds or not shifts:
		return 0.0
	if t <= bounds[0]:
		return float(shifts[0])
	for i in range(len(shifts)):
		if bounds[i] <= t <= bounds[i + 1]:
			return float(shifts[i])
	return float(shifts[-1])


def apply_piecewise_shift_to_intervals(
	intervals: Sequence[Tuple[float, float]],
	transform: dict,
) -> List[Tuple[float, float]]:
	"""Map intervals [s,e] by t' = t + shift(midpoint)."""
	mapped: List[Tuple[float, float]] = []
	for s, e in intervals:
		mid = 0.5 * (float(s) + float(e))
		b = find_shift_for_time(transform, mid)
		mapped.append((float(s) + b, float(e) + b))
	return mapped


def _build_binary_timeline(
	intervals: Sequence[Tuple[float, float]],
	t0: float,
	t1: float,
	dt: float,
) -> np.ndarray:
	"""Build a binary on/off array sampled at ``dt`` over [t0, t1]."""
	L = int(np.floor((t1 - t0) / dt)) + 1
	arr = np.zeros(L, dtype=np.float32)
	for a, b in intervals:
		if b <= a:
			continue
		i0 = int(np.floor((a - t0) / dt))
		i1 = int(np.ceil((b - t0) / dt))
		i0 = max(i0, 0)
		i1 = min(i1, L - 1)
		if i0 <= i1:
			arr[i0 : i1 + 1] = 1.0
	return arr


def _next_pow2(n: int) -> int:
	p = 1
	while p < n:
		p <<= 1
	return p


def _cross_correlation_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Compute full cross-correlation y ⋆ x (lags from -(N-1)..+(N-1)) via FFT."""
	n = len(x)
	m = 2 * n - 1
	M = _next_pow2(m)
	X = np.fft.rfft(x, n=M)
	Y = np.fft.rfft(y, n=M)
	corr = np.fft.irfft(np.conj(X) * Y, n=M)
	# center
	corr = np.concatenate([corr[-(n - 1) :], corr[:n]])
	return corr


def _estimate_pairwise_edges(
	S_list: Sequence[np.ndarray],
	dt: float,
	*,
	max_lag_seconds: float,
) -> List[Tuple[int, int, float]]:
	"""Estimate pairwise lags between binaries without filtering or weighting.

	Returns a list of edges (i, j, d_ij_seconds) where positive d_ij means j is
	after i (so shifting j by -d_ij aligns it to i).
	"""
	N = len(S_list)
	if N <= 1:
		return []
	L = len(S_list[0])
	max_lag = int(round(max_lag_seconds / dt))
	# zero-mean for better correlation contrast
	Z = [s.astype(np.float32) - float(np.mean(s)) for s in S_list]
	edges: List[Tuple[int, int, float]] = []
	for i in range(N):
		for j in range(i + 1, N):
			corr = _cross_correlation_fft(Z[i], Z[j])
			center = len(corr) // 2
			lo = max(0, center - max_lag)
			hi = min(len(corr), center + max_lag + 1)
			slice_ = corr[lo:hi]
			if slice_.size == 0:
				continue
			best_idx = int(np.argmax(slice_))
			lag_samples = (lo + best_idx) - center
			d_ij = float(lag_samples) * dt
			edges.append((i, j, d_ij))
	return edges


def _solve_global_shifts_unweighted(
	N: int,
	edges: Sequence[Tuple[int, int, float]],
) -> np.ndarray:
	"""Solve least squares for shifts s (length N) from edges (i,j,d_ij).

	We add a tiny anchor (sum(s)=0 approx) to fix the gauge, and finally
	median-center the solution for interpretability.
	"""
	if N == 0:
		return np.zeros(0, dtype=np.float32)
	if N == 1 or not edges:
		return np.zeros(N, dtype=np.float32)
	A = np.zeros((len(edges), N), dtype=np.float32)
	b = np.zeros((len(edges),), dtype=np.float32)
	for r, (i, j, d_ij) in enumerate(edges):
		A[r, i] = -1.0
		A[r, j] = +1.0
		b[r] = float(d_ij)
	# tiny anchor row
	Aw = np.vstack([A, 1e-6 * np.ones((1, N), dtype=np.float32)])
	bw = np.concatenate([b, np.array([0.0], dtype=np.float32)])
	sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
	sol = sol.astype(np.float32)
	# center to median
	sol = sol - np.median(sol)
	return sol


def estimate_shifts_from_intervals(
	intervals_per_candidate: Sequence[Sequence[Tuple[float, float]]],
	*,
	dt: float = 0.1,
	max_lag_seconds: float = 300.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
	"""Estimate per-candidate shifts from intervals.

	Returns (shifts_seconds, (t0, t1)).
	"""
	if not intervals_per_candidate:
		return np.zeros(0, dtype=np.float32), (0.0, 0.0)
	# timeline bounds
	starts = [min((s for s, _ in iv), default=0.0) for iv in intervals_per_candidate]
	ends = [max((e for _, e in iv), default=0.0) for iv in intervals_per_candidate]
	t0 = float(min(starts))
	t1 = float(max(ends))
	if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
		return np.zeros(len(intervals_per_candidate), dtype=np.float32), (t0, t1)
	# binaries
	S_list = [
		_build_binary_timeline(iv, t0=t0, t1=t1, dt=dt)
		for iv in intervals_per_candidate
	]
	edges = _estimate_pairwise_edges(S_list, dt, max_lag_seconds=max_lag_seconds)
	shifts = _solve_global_shifts_unweighted(len(S_list), edges)
	return shifts, (t0, t1)


def apply_shifts_to_intervals(
	intervals_per_candidate: Sequence[Sequence[Tuple[float, float]]],
	shifts_seconds: Sequence[float],
) -> List[List[Tuple[float, float]]]:
	"""Shift each candidate's intervals by -shift to align into a common time base."""
	aligned: List[List[Tuple[float, float]]] = []
	for iv, s in zip(intervals_per_candidate, shifts_seconds):
		shift = float(s)
		aligned.append([(a - shift, b - shift) for a, b in iv])
	return aligned


def align_intervals(
	intervals_per_candidate: Sequence[Sequence[Tuple[float, float]]],
	*,
	dt: float = 0.1,
	max_lag_seconds: float = 300.0,
) -> Tuple[List[List[Tuple[float, float]]], np.ndarray]:
	"""Convenience function: estimate shifts and return aligned intervals + shifts."""
	shifts, _ = estimate_shifts_from_intervals(
		intervals_per_candidate,
		dt=dt,
		max_lag_seconds=max_lag_seconds,
	)
	aligned = apply_shifts_to_intervals(intervals_per_candidate, shifts)
	return aligned, shifts


def select_master_clock_by_median_duration(
	subtitle_files: Sequence[str],
	component_indices: Sequence[int],
) -> Tuple[int, str]:
	"""Select master clock file based on median duration from connected component.
	
	Args:
		subtitle_files: List of all subtitle file paths
		component_indices: Indices of files in the largest connected component
	
	Returns:
		(master_index, master_file_path) where master_index is the index in subtitle_files
	"""
	from pathlib import Path as _Path
	from ..parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	
	def load_segments(file_path: str):
		"""Load segments from a subtitle file."""
		p = _Path(file_path)
		data = p.read_bytes()
		ext = p.suffix.lower().lstrip(".")
		if ext == "srt":
			return parse_srt_bytes(data)
		elif ext == "vtt":
			return parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
	
	durations = []
	valid_indices = []
	
	for idx in component_indices:
		try:
			segments = load_segments(subtitle_files[idx])
			if segments:
				# Compute effective duration (last end - first start)
				duration = segments[-1].end_seconds - segments[0].start_seconds
				durations.append(duration)
				valid_indices.append(idx)
		except Exception:
			continue  # Skip files that can't be parsed
	
	if not durations:
		raise ValueError("No valid subtitle files found in connected component")
	
	# Find file with median duration
	durations_array = np.array(durations)
	median_duration = np.median(durations_array)
	closest_idx = np.argmin(np.abs(durations_array - median_duration))
	master_index = valid_indices[closest_idx]
	
	return master_index, subtitle_files[master_index]


def align_multiple_subtitles_to_master(
	subtitle_files: Sequence[str],
	component_indices: Sequence[int],
	master_index: int,
	*,
	n: int = 3,
	gap_penalty: float = -0.4,
	min_sim: float = 0.3,
	hardshift_threshold: float = 0.9,
) -> Dict[str, Any]:
	"""Align multiple subtitle files to a master clock using hard-anchor piecewise shifts.
	
	Args:
		subtitle_files: List of all subtitle file paths
		component_indices: Indices of files in the largest connected component
		master_index: Index of the master clock file
		n: Character n-gram size
		gap_penalty: Gap penalty for NW alignment
		min_sim: Hard floor for match acceptance
		hardshift_threshold: Similarity threshold for hard anchors
	
	Returns:
		Dictionary with alignment results including transforms and metadata
	"""
	from pathlib import Path as _Path
	from ..parsers.subtitles import parse_srt_bytes, parse_vtt_bytes
	
	def load_segments(file_path: str):
		"""Load segments from a subtitle file."""
		p = _Path(file_path)
		data = p.read_bytes()
		ext = p.suffix.lower().lstrip(".")
		if ext == "srt":
			return parse_srt_bytes(data)
		elif ext == "vtt":
			return parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
	
	master_file = subtitle_files[master_index]
	master_segments = load_segments(master_file)
	
	results = {
		"master_file": master_file,
		"master_index": master_index,
		"transforms": {},
		"metadata": {
			"n": n,
			"gap_penalty": gap_penalty,
			"min_sim": min_sim,
			"hardshift_threshold": hardshift_threshold,
		}
	}
	
	for idx in component_indices:
		if idx == master_index:
			# Identity transform for master
			results["transforms"][subtitle_files[idx]] = {
				"type": "identity",
				"shifts": [],
				"boundaries": [],
			}
			continue
		
		try:
			other_file = subtitle_files[idx]
			other_segments = load_segments(other_file)
			
			# Normalize texts
			master_texts = [normalize_subtitle_text(seg.text) for seg in master_segments]
			other_texts = [normalize_subtitle_text(seg.text) for seg in other_segments]
			
			# Align to master using NW + hard anchors
			aligned_pairs, similarity_matrix = needleman_wunsch_align(
				master_texts,
				other_texts,
				n=n,
				gap_penalty=gap_penalty,
				min_sim=min_sim,
			)
			
			# Compute merged blocks
			blocks = compute_match_blocks_growmerge(
				aligned_pairs,
				similarity_matrix,
				master_texts,
				other_texts,
				n=n,
			)
			
			# Convert blocks to equations format (center points only for hard anchors)
			equations = []
			master_intervals = [(seg.start_seconds, seg.end_seconds) for seg in master_segments]
			other_intervals = [(seg.start_seconds, seg.end_seconds) for seg in other_segments]
			
			for bi, ((i0, i1), (j0, j1), score) in enumerate(blocks):
				# Get time intervals for the block
				master_start = master_intervals[i0][0]
				master_end = master_intervals[i1][1]
				other_start = other_intervals[j0][0]
				other_end = other_intervals[j1][1]
				
				# Add center point (recommended for hard anchors)
				master_center = 0.5 * (master_start + master_end)
				other_center = 0.5 * (other_start + other_end)
				equations.append({
					"x": float(master_center),  # A time (master)
					"y": float(other_center),   # B time (other)
					"w": float(score),          # similarity weight
					"kind": "center",
					"i": bi,
				})
			
			# Create hard-anchor piecewise shift transform
			hardshift_transform = compute_hardshift_transform(
				equations, 
				master_intervals, 
				other_intervals, 
				threshold=hardshift_threshold
			)
			
			results["transforms"][other_file] = {
				"type": "piecewise_shift",
				"shifts": hardshift_transform["shifts"],
				"boundaries": hardshift_transform["boundaries"],
				"num_anchors": len([eq for eq in equations if eq.get("w", 0) >= hardshift_threshold]),
			}
			
		except Exception as e:
			print(f"[red]Warning: Failed to align {subtitle_files[idx]} to master: {e}[/red]")
			results["transforms"][subtitle_files[idx]] = {
				"type": "failed",
				"error": str(e),
			}
	
	return results


