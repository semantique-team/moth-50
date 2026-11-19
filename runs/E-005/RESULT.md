# E-005 — RESULT: long fine-tune of the E-003 winner

> Executed 2026-11-18 07:43–14:27 CET on the M5 Pro. `base+6`, 16,000 optimizer
> steps, pretrained Whisper-base encoder, identity-initialized extra blocks,
> frozen E-002 data mixture. The final A3 readout, package and S1-dev session
> were completed on 2026-11-19.

## Erratum (2026-11-19): the first readout and package used the pre-A3 LM

Every "A3" in the first version of this file was wrong.
`tools/frozen_readout.py` and `tools/export_release.py` loaded
`../moth-stt/release/lm_bpe3.bin`, a stale, gitignored copy of the LM that Moth
shipped from 0.5.0 to 0.8.0 (sha256 `dcf1d59f…`, 5,116,464 B). The 0.8.1 A3 LM,
which decoded the 23.0170 % reference, is `3be1177b…` (5,381,992 B;
`data/sources.lock.json`). The package also shipped Moth 0.8.1's NOTICE byte for
byte (sha256 `6ca7672f…`), which names `openai/whisper-tiny` as the encoder.

- **The gate verdict stands.** A3 against the pre-A3 LM was −0.097 pp on Moth's
  dev (Moth E-031: 22.9463 → 22.8494 %), and the comparison here handicapped
  MOTH-50, not the reference; the margin is 5.35 pp.
- **The first numbers are not A3 numbers.** They are MOTH-50 + pre-A3 LM against
  0.8.1 + A3. The package lacked A3's ×3 term weighting (rare-term ETA 28.4 →
  46.9 % in Moth E-031), so it must not be used for S1-dev, T-600 or ETA claims.
- **With A3 the package is 47,836,502 B** (2,163,498 B of room, not 2,429,915),
  and its NOTICE is the MOTH-50 one.
- **16,000 steps are 1.19 epochs**, not ~1.4: epoch 0 ended after 13,458
  batches (`base+6-16k/train.log.jsonl`).

Fixed in the tools (D17): the LM is pinned by sha256 and fetched by
`tools/release_assets.py`, any other LM needs `--allow-other-lm`, and
`tools/release_notice.py` writes the MOTH-50 NOTICE. The A3 measurements follow.

## Final result (2026-11-19): the E-002 gate is complete

### Dev macro, A3 LM

Decoder: beam 12, top-k 10, lm_weight 0.55, token_bonus 2.7, A3 LM
(`3be1177b…`). Reference hypotheses: Moth E-037 0.8.1 INT8 on the same sets.

| Set | n | FP32 WER | 95 % CI | INT8 WER | 95 % CI | 0.8.1 |
|---|---:|---:|---|---:|---|---:|
| MLS-IT dev | 1,248 | **23.3474 %** | [22.6548, 24.0504] | **23.4415 %** | [22.7275, 24.1361] | 28.1040 % |
| LS-EN dev | 2,691 | **11.7656 %** | [11.4132, 12.1439] | **11.8010 %** | [11.4518, 12.1820] | 17.9299 % |
| **Macro** | 3,939 | **17.5565 %** | — | **17.6212 %** | — | **23.0170 %** |

Paired macro bootstrap (1,000 resamples, seed 0):

| Weights | Δ vs 0.8.1 | 95 % CI | INT8 cost | Gate |
|---|---:|---|---:|---|
| FP32 | **−5.4605 pp** | [−5.7237, −5.1921] | — | **pass** |
| INT8 | **−5.3957 pp** | [−5.6634, −5.1031] | +0.0647 pp | **pass** |

### S1-dev, no-regression half

The complete held-out session: **960/960 takes, 51.30 min**, mean 3.21 s,
peaks −17.6 dBFS mean (max −3.8), no hard QC errors. The reference is the
pinned 0.8.1 INT8 graph (`7bbfad7e…`) decoded by the same tool on the same
manifest with the A3 LM; the candidate uses the frozen decoder unchanged.

| System | WER | 95 % CI | ETA | ETA 95 % CI |
|---|---:|---|---:|---|
| 0.8.1 INT8 + A3 | 22.2498 % | [21.0424, 23.5277] | 62.1094 % | — |
| MOTH-50 FP32 | **13.9956 %** | [13.1255, 14.9123] | **75.3418 %** | — |
| MOTH-50 INT8 | **14.0541 %** | [13.1863, 15.0037] | **75.5371 %** | — |

Paired, candidate − reference (negative WER is better; positive ETA is better):

| Weights | Δ WER | 95 % CI | Δ ETA | 95 % CI | No regression |
|---|---:|---|---:|---|---|
| FP32 | **−8.2542 pp** | [−9.3534, −7.0728] | **+13.2324 pp** | [10.9405, 15.5926] | **pass** |
| INT8 | **−8.1958 pp** | [−9.3067, −7.0108] | **+13.4277 pp** | [11.2150, 15.7687] | **pass** |

Per-category WER/ETA on S1-dev (INT8 vs 0.8.1):

| Category | n | WER | 0.8.1 | ETA | 0.8.1 |
|---|---:|---:|---:|---:|---:|
| phrase | 142 | **14.78 %** | 22.33 % | **64.60 %** | 56.70 % |
| single_word | 112 | **7.09 %** | 11.34 % | **87.50 %** | 77.68 % |
| control_it | 121 | **8.24 %** | 14.49 % | — | — |
| acronym | 113 | **22.65 %** | 28.69 % | **29.20 %** | 21.24 % |
| brand | 113 | **19.62 %** | 28.82 % | **48.06 %** | 32.56 % |
| control_en | 119 | **16.82 %** | 33.41 % | **85.41 %** | 69.65 % |
| switch | 128 | **13.22 %** | 23.34 % | **81.41 %** | 64.40 % |
| italianized | 112 | **11.89 %** | 14.86 % | **72.32 %** | 69.64 % |

Both halves of the E-002 gate are now met: dev-macro ≥ 1.0 pp better with a CI
excluding zero, and no S1-dev WER or ETA regression outside its CI.

## S1-dev error analysis (post-hoc, 2026-11-19)

Descriptive, not a gate: it reads the 2,048 English reference words of S1-dev
and classifies every miss in the frozen readouts with `tools/eta_breakdown.py`
(classes and flags are defined in its docstring). Evidence:
`runs/E-005/eta-breakdown.json`.

```bash
python tools/eta_breakdown.py runs/E-005/s1-dev-a3-int8.json \
  --compare runs/E-005/s1-dev-0.8.1.json --out runs/E-005/eta-breakdown.json
# with per-flag denominators and per-category classes (needs the manifest):
#   --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl
```

| Miss class | 0.8.1 INT8 + A3 | MOTH-50 INT8 + A3 |
|---|---:|---:|
| deleted (no word aligned) | 151 | **53** |
| substituted | 518 | **335** |
| truncated (prefix of the term: `path` → `pat`) | 64 | 73 |
| glued to a neighbour (`bucket` → `storagebucket`) | 41 | 37 |
| apostrophe only (`l'opex` → `lopex`) | 2 | 3 |
| **Total misses** | **776** | **501** |

| Flag (any class) | 0.8.1 | MOTH-50 |
|---|---:|---:|
| English function word (`too`, `it`, `a`, …) | 140 | **59** |
| elided term (`l'sdk`, `l'hr`, `let's`) | 32 | 34 |
| no vowel after the elision (`mcv`, `l'hr`; acronym lower bound) | 30 | 24 |

Paired by (clip, term): MOTH-50 fixes 361 of 0.8.1's misses and adds 86.

- **Term deletions are mostly gone** (151 → 53). The old failure mode, dropping
  the English word, is no longer the main error.
- **Function words fell from 140 to 59.** The remainder is largely homophones
  (`too` → `to` five times), where little is left to gain.
- **Elided acronyms did not improve** (32 → 34). `l'sdk`, `l'ebitda`, `l'rfid`
  fail as a unit: the article, the letter names and the orthography all have to
  be right. With the manifest the tool reports how many elided terms S1-dev has
  in total, i.e. their accuracy.
- **Acronyms and brands are the weakest categories**: ETA 29.2 % and 48.1 %,
  with the highest insertion rates (4.70 and 3.13 per 100 words against 0.71 on
  `control_it`) — spelled letters turn into extra tokens.
- **Truncations rose** (64 → 73): final consonants or suffixes lost (`growth` →
  `grow`, `planned` → `plan`, `benchmarkato` → `benchmark`). Whether this is the
  speaker's pronunciation or the model needs listening, not more statistics.

These are data and LM problems rather than schedule problems. They motivate
E-019 (teacher labels), E-020 (LM text) and E-021 (targeted acronym, brand and
elision data) in `experiments/next-experiments.md`.

## Greedy curve (monitoring)

| Step | Greedy dev-macro WER |
|---:|---:|
| 2,000 | 42.37 % |
| 4,000 | 35.54 % |
| 6,000 | 30.73 % |
| 8,000 | 27.21 % |
| 10,000 | 24.95 % |
| 12,000 | 22.92 % |
| 14,000 | 22.23 % |
| **16,000** | **21.75 %** |

The last 2,000 steps bought 0.48 pp, so the curve is flattening but not flat.

## INT8 release package (A3, final)

Built with `tools/export_release.py` using Moth's 0.8.1 recipe (fp32 ONNX →
dynamic per-channel QInt8), the pinned A3 LM, the full Silero VAD graph and the
MOTH-50 NOTICE. Version `0.9.1-e005`; `config.budget_bytes = 50,000,000`.

| File | Bytes |
|---|---:|
| `model.int8.onnx` | 40,039,997 |
| `lm_bpe3.bin` (A3) | 5,381,992 |
| `silero_vad.onnx` | 2,327,524 |
| `mel_filters.bin` | 64,320 |
| `vocab.json` | 17,193 |
| `config.json` + `NOTICE` | 5,476 |
| **Total** | **47,836,502** |

Budget 50,000,000 B → **2,163,498 B (2.16 MB) remaining**. The vendored
`moth_release.validate_release` accepts the folder and verifies every hash. The
NOTICE names `openai/whisper-base`, the 6 extra blocks, Silero commit
`bfdc019`, the reused A3 LM/tokenizer and the MOTH-50 training sources.

Export fidelity: FP32 max logit gap 4.0e-3, top-1 agreement 1.0; INT8 max 31.1,
top-1 agreement **1.0**. The large INT8 max is a confidence outlier on one
frame, not a decoding failure — the measured cost is 0.0647 pp macro.

## Browser run (A3, final)

The 47.84 MB package loaded in the in-app browser (Chromium, WASM, 4 threads,
cross-origin isolated, SIMD) and transcribed the T-600 clip
`audio/T/S1/S1_0001.wav` exactly: **"ho spostato il meeting a domani"**.

| Measurement | Value |
|---|---|
| Release load | 280 ms |
| Cross-origin isolated / threads / SIMD | true / 4 / true |
| Single 3.99 s clip | 291 ms total, RTFx **13.7** |
| Benchmark (2 warm-ups + 5 runs) | median 179.0 ms, RTFx **22.3** (inference 44.3) |
| VAD / features / inference / decode (median) | 13.5 / 58.5 / 90.0 / 15.4 ms |

One compatibility change is required: `web/ort.js` reads an optional
`config.budget_bytes` (default 20,000,000, so the 0.8.1 release is unchanged);
this release declares 50,000,000. Without it the page rejects the package at the
old 20 MB ceiling. Evidence: `runs/E-005/browser-run-a3.json`.

## Pre-A3 artifacts (superseded)

| Artifact | Pre-A3 value | Final A3 value |
|---|---:|---:|
| Dev macro FP32 | 17.5816 % | **17.5565 %** |
| Dev macro INT8 | 17.6628 % | **17.6212 %** |
| Paired INT8 vs 0.8.1 | −5.3542 pp | **−5.3957 pp** |
| Package | 47,570,114 B | **47,836,502 B** |
| Browser benchmark RTFx | 22.7 | **22.3** |

Kept for the erratum: `readout.json`, `readout-int8.json`, `release/`,
`browser-run.json`. The final artifacts are `readout-a3.json`,
`readout-a3-int8.json`, `release-a3/`, `browser-run-a3.json`, plus the S1-dev
`*-a3-*.json` files.

## Run

- Model: 39,259,649 params, base+6, `model.init` = converted
  `openai/whisper-base` (`data/raw/whisper-base-encoder.pt`).
- Schedule: 16,000 steps, warmup 1,000, cosine to 0.05; AdamW head 3e-4 /
  encoder 3e-5; 192 audio-s / 32-clip cap; per-block checkpointing.
- Data: the frozen E-002 mixture (273,714 manifest rows, ~595 h with the gold
  ×50 repeat).
- Wall time **6.73 h**, peak MPS memory **8.604 GB**, 0 non-finite steps.
- Checkpoints every 100 steps; `ckpt_best.pt` is step 16,000.
- Final readout on CPU: dev macro **459.9 s** FP32 / **438.4 s** INT8; S1-dev
  **28.0 s** (0.8.1), **39.2 s** (FP32) and **37.0 s** (INT8).

## Caveats

- The browser run uses a whole clip through the demo page; it is not a
  microphone or VAD-segmentation test, and the readout does not apply the VAD.
- Dev was used for checkpoint selection (`ckpt_best`), as the protocol allows.
  S1-dev was **not** used for training or decoder tuning; the frozen settings
  are unchanged.
- Two S1-dev takes are short (`S1_0563` 1.49 s, `S1_0886` 1.27 s) but their cues
  are four words; the QC has no hard errors. `S1_0726` was re-recorded after a
  0.0 dBFS peak.
- S1-dev is one speaker and one session; the per-category numbers are
  descriptive, not a speaker-independent estimate.

## Next

1. E-009 decoder recalibration and E-010 encoder-LR screen are **not run**;
   nothing else starts without the go-ahead. See
   `experiments/next-experiments.md`.
2. If the product adopts MOTH-50, carry the `budget_bytes` runtime change in
   the MOTH-50 tree: `moth-stt` is archived and read-only, so the local commit
   there (`b0fa11c`) cannot be pushed.
