# E-025 (E-020) — LLM-written mixed IT–EN text for the LM

**Pre-registered 2026-11-23, before any generation or LM build.** Control is
the product acoustic (R1 s0) with E-034c.

## Question

Does an E-034c LM rebuilt with in-domain mixed IT–EN dictation text generated
by an open-weight LLM raise S1-dev ETA without a WER regression, and survive
the R-S1-cal ranking, the S1-dev veto, the dev-macro guard and the 50 MB
budget?

## Generator (pinned)

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0 |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Files | `data/sources.lock.json` (`qwen2.5-1.5b-instruct`) |
| Decoding | temperature 0.8, top-p 0.95, seed 0, max 40 new tokens |
| Prompts | committed in `experiments/E-025-prompts.json` |

The generator is inference-only and never enters a release. Its outputs are
filtered text, not a teacher for audio.

## Leakage control (fixed before generation)

- Prompt seeds come only from **training-side** terms: `../moth-stt/data/en_terms.txt`
  and the gold train scripts (`manifests/train.jsonl`). Never S1-dev, T or A
  sentences or their term lists.
- Every generated sentence is normalized with `moth_text.normalize`. Drop it if
  it shares a **word 4-gram** with any S1-dev, T or A script, if it is an exact
  or near duplicate of another generated sentence, or if it is under 4 words
  or over 30 words.
- Report generated/dropped counts and the 4-gram overlap before building. The
  filter is `tools/e020_filter.py`.

## Arm

The E-034c clean corpus (185,103 rows, sha256 `569c8233…`) plus the filtered
generated sentences; everything else identical to E-034c: unpruned N2_1_1,
`term_weight = 1`, `min_bigram = 1`, `min_trigram = 1`, `backoff = 0.4`, same
tokenizer. One variable: the added text.

- Build with `tools/build_lm_from_corpus.py`; pack lossless to MOTHLMB2 10-bit
  with `tools/pack_lm.py`.
- **Budget:** the packed LM must keep the full package ≤ 50,000,000 B. If the
  unpruned build is over, the arm is recorded as over-budget; no post-hoc
  pruning or recipe change.
- Control: E-034c packed 7,787,184 B; package 49,203,776 B.

## Control numbers (R1 s0 + E-034c, INT8, frozen decoder)

| Set | WER | ETA | Artifact |
|---|---:|---:|---|
| R-S1-cal (400) | **12.8469** | **75.4967** | `runs/E-023/s1-cal-s0.json` |
| S1-dev (960) | **13.3820** | **75.1465** | `runs/E-023/s1-dev-s0.json` |
| dev-macro | **14.3274** | — | `runs/E-023/dev-macro-s0.json` |

## Gate (fixed before generation)

1. **R-S1-cal ranking.** ETA better than the control with a paired 95 % CI
   excluding zero and WER not worse (CI not entirely above zero), **or** WER
   better with a CI excluding zero and ETA not worse (CI not entirely below
   zero). If this fails, stop; do not read S1-dev.
2. **S1-dev (E-020 gate).** ETA better with a paired CI excluding zero and no
   WER regression, **or** WER better with a CI excluding zero and no ETA
   regression. Report acronym and brand ETA.
3. **dev-macro guard.** Macro-WER difference CI not entirely above zero.
4. **Budget.** Lossless pack and full package ≤ 50,000,000 B.
5. **T-S3 report-only, only after 1–4 pass.** A check against Moth D42; not a
   gate.

If 1–4 pass, the result is a **new dated LM decision candidate** (D33) and
needs the package/browser check before adoption. If any gate fails, E-034c
stays.

## Risk

S1-dev and the gold scripts come from a script pipeline that may be
LLM-assisted, so LLM-generated LM text can look better on S1-dev than on real
dictation. T-600 at release and, later, unscripted speech (E-019 arm B audio
with reviewed transcripts) check it. The generated text is synthetic; it must
not be described as new speech data.

## Artifacts

`runs/E-025/`: prompts, generated and filtered text reports, LM build and pack
reports, cal/S1-dev/dev-macro readouts, and the result. No training.

## Result (2026-11-23) — not adopted

- 6,000 generated → **3,521 filtered** (249 eval 4-gram drops, 1,385 seed
  misses, 139 duplicates, etc.); combined corpus 188,624 rows.
- LM: 12,375,336 B MOTHLMB1; packed **7,833,032 B**, lossless; package
  49,249,624 B (750,376 B room).
- **R-S1-cal: 12.8917 / 75.6291 vs 12.8469 / 75.4967**; WER +0.0448 pp
  [−0.0900, +0.1807], ETA +0.1325 pp [0, +0.4178]; both **ties**, 7/400 clips
  differ. **Gate 1 fails.**
- S1-dev, dev-macro and T were not read (pre-registered stop). E-034c stays.
  A 1.9 % corpus addition is too small to move the 2.3M-trigram LM at the frozen
  decoder; a larger generation is a new experiment. `runs/E-025/RESULT.md`.
