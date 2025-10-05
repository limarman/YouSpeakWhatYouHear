## YouSpeakWhatYouHear – Project Overview

### Overall Goal
- Build a website that presents descriptive statistics about speech in episodes/movies (per title/episode), using subtitles as the data source.
- Example metrics the site aims to show:
  - Total speech time (seconds)
  - Words per second (speech rate)
  - Vocabulary coverage/rarity (e.g., share of words in top‑1k/2k/5k, median Zipf)

### Approach (high level)
- We generally don’t have raw audio, so we leverage multiple subtitle files per title/episode to robustly estimate speech timing and text statistics.
- Steps:
  1) Align all candidate subtitles to a common clock (robust to drift and edits)
  2) Optionally clean obviously unsupported cues
  3) Compute metrics from the aligned/cleaned subtitles and present them on the website

---

## Established Pipeline Details

This section summarizes the alignment subsystem that enables the metrics above.

### Text Normalization
- Unicode NFKC + casefold; remove tags/hearing‑impaired markers; collapse whitespace; strip punctuation for scoring.
- Similarity uses character n‑gram (default trigrams, n=3) cosine; fast precomputed hashed vectors are supported.

### Matching: Needleman–Wunsch + Local Grow‑Merge
- Global alignment (NW) with parameters such as gap_penalty (default −0.4) and a hard floor on cue‑cue similarity (default min_sim=0.3).
- Local grow‑merge: starting from a seed match, iteratively extend over consecutive gaps on either side if stripped 3‑gram cosine strictly improves (tunable threshold). This produces many↔many merged “blocks”.
- Each merged block carries: A‑range [i0..i1], B‑range [j0..j1], and a similarity score.

### Selecting Relevant Subtitles (Largest Connected Component)
- For each file pair, compute a combined score from merged blocks:
  - score_blocks: average block similarity (weighted by block duration/envelope).
  - coverage: fraction of speech time covered by matched blocks (symmetric across sides).
  - combined = score_blocks × coverage.
- Build a graph from the similarity matrix (edge if combined ≥ threshold) and keep the largest connected component. These files form the coherent group used for downstream alignment.

---

## Cleaning (Simple Support Rule)
- Goal: drop cues that are not corroborated by any other file in the component.

## Alignment with Hard Anchors (Piecewise Shift)
- Hard anchors: high‑confidence merged blocks (e.g., similarity ≥ 0.9). For each, use center‑to‑center times to define a shift b = t_master − t_file.
- Master clock: choose the file with median effective duration within the component; align all others to this clock via their hard‑anchor transforms.

---


## TODOs:

There seem to be many subtitles in a different format than the typical srt. We need to be able to distinguish this at load time and use a different parser.