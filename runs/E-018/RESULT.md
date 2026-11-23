# E-018 — Teacher bake-off on the gold train set

> **Status: done 2026-11-23 — no teacher qualifies.** Parakeet TDT 0.6B v3,
> Whisper large-v3-turbo (forced Italian and auto language) and Canary-1B-v2
> ran on all 1,413 gold-train clips. **None** meets the pre-registered rule
> (WER ≤ 7 %, ETA ≥ 85 %, `italianized` ETA ≥ 72.3 %). Voxtral Mini 3B is
> excluded for resource cost on the 16 GB M5. E-019 therefore keeps only
> monolingual Italian sources; no teacher labels mixed IT–EN audio.

## Setup

- **Data:** gold train set v1.0.0, 1,413 clips, 3,457.79 s (57.6 min), S1+S2,
  reviewed code-switching transcripts. No candidate trained on it.
- **Scoring:** Moth's `stt/eval.py`, the one normalizer, `--breakdown
  category,speaker`; not the MOTH-50 scorer.
- **Models (pinned in `data/sources.lock.json`):**

| Candidate | Revision | License | Size | Engine |
|---|---|---|---:|---|
| Parakeet TDT 0.6B v3 int8 | `ed5d7456…` | CC-BY-4.0 | 670,576,784 B | sherpa-onnx 2.0.3, 4 threads |
| Whisper large-v3-turbo | `41f01f3f…` | MIT | 1.5 GB | Transformers 5.20.1, MPS fp32 |
| Canary-1B-v2 | `d4557063…` | CC-BY-4.0 | 3.9 GB safetensors | Transformers 5.20.1, MPS fp32 |
| Voxtral Mini 3B | `3060fe34…` | Apache-2.0 | 9.36 GB | excluded, see below |

## Results (full gold set, 1,413 clips, 8,425 ref words)

| Candidate | WER | ETA | `italianized` ETA | Ins/100w | RTFx | Qualifies |
|---|---:|---:|---:|---:|---:|---|
| Parakeet TDT 0.6B v3 | 9.95 % | 79.47 % | 62.71 % | 1.32 | 25.48 | **no** |
| Whisper large-v3-turbo, forced it | 9.92 % | 80.12 % | 70.76 % | 1.13 | 2.73 | **no** |
| Whisper large-v3-turbo, auto | 10.99 % | 82.60 % | 69.92 % | 1.04 | 1.51 | **no** |
| Canary-1B-v2, forced it | 13.60 % | 68.62 % | 61.44 % | 1.19 | 14.08 | **no** |
| Voxtral Mini 3B | — | — | — | — | 0.009 | **excluded** |

**Gates:** WER ≤ 7.0 %, ETA ≥ 85 %, `italianized` ETA ≥ 72.3 %. Every candidate
misses at least two; Whisper forced-it comes closest on ETA/italianized but is
2.9 pp over the WER gate and 1.5 pp under the italianized gate. Auto-language
fixes most of Whisper's English-only failure (`control_en` WER 54.5 → 7.5 %) but
loses on the Italian categories and ends at 10.99 % WER.

### Per-category (WER / ETA)

| Category | Parakeet | Whisper it | Whisper auto | Canary it |
|---|---:|---:|---:|---:|
| single_word (369) | 9.4 / 75.3 | 7.6 / 82.7 | 10.0 / 80.3 | 9.3 / 69.6 |
| phrase (269) | 11.9 / 81.4 | 9.4 / 85.0 | 11.4 / 84.9 | 11.3 / 79.5 |
| brand (178) | 15.9 / 72.2 | 8.6 / 88.2 | 14.7 / 86.3 | 11.8 / 76.4 |
| acronym (154) | 10.1 / 71.2 | 6.0 / 84.0 | 8.9 / 82.1 | 8.2 / 75.0 |
| italianized (221) | 10.0 / 62.7 | 9.0 / 70.8 | 10.3 / 69.9 | 10.2 / 61.4 |
| switch (112) | 6.3 / 93.7 | 10.0 / 82.9 | 12.3 / 83.7 | 13.2 / 74.3 |
| control_it (63) | 4.2 / — | 2.7 / — | 2.7 / — | 2.2 / — |
| control_en (47) | 2.8 / 95.7 | **54.5 / 46.0** | 7.5 / 90.0 | **102.2 / 1.4** |

**Why they fail.** Parakeet is excellent on English-only and switching runs
(`control_en` 2.8 %, `switch` 93.7 % ETA) but weak on Italianized terms
(62.7 %) and brands/acronyms. Whisper forced-it is the most balanced on
Italian categories but catastrophically translates/transliterates English-only
clips (`control_en` 54.5 % WER); auto-language fixes the English control
(7.5 %) but loses 1.1 pp WER and 1.5 pp ETA overall. Canary is strong on
Italian-only (2.2 % WER) but translates English-only audio outright (102 % WER,
ETA 1.4 %) and is the weakest overall. None is a safe teacher for mixed IT–EN
dictation: a teacher that writes Italian for an English-only cue teaches the
wrong mapping even when its Italian WER is good.

## Voxtral Mini 3B — excluded for cost, not accuracy

One 2.62 s clip took **304 s on MPS** (RTFx 0.009; 130 s on CPU), projecting to
over 100 h for the 57.6 min set on the 16 GB M5. The model is pinned, licensed
Apache-2.0, and transcribed the sample correctly; it remains a candidate for a
rented GPU. Evidence: `voxtral-mini-3b-2507/EXCLUDED.md`,
`voxtral-mini-3b-2507/smoke-1clip.*`.

## Selection

**No mixed IT–EN teacher qualifies.** Under the pre-registered rule E-019
therefore keeps only **monolingual Italian sources** (Granary Italian,
VoxPopuli Italian), where the candidates are far ahead of MOTH-50; it does not
use a teacher to label mixed English/Italian audio. No teacher is selected, and
the runner-up/agreement filter is empty.

The practical constraint is the English side: Parakeet is the best on
English-only clips (2.8 % WER) but poor on Italianized terms; Whisper/Canary are
good on Italian-only but translate English. A future language-routed ensemble
could combine them, but that is outside the pre-registered E-018 rule and is
not adopted.

## Artifacts

- `parakeet-tdt-0.6b-v3-int8/` — `gold_train.jsonl`, `meta.json`,
  `gold_train.metrics.json`.
- `whisper-large-v3-turbo-it/` — same.
- `whisper-large-v3-turbo-auto/` — same.
- `canary-1b-v2-it/` — same.
- `voxtral-mini-3b-2507/` — exclusion note and 1-clip smoke.
- `summary.json` — generated by `tools/e018_summary.py` (the table, the
  qualification flags and the empty selection).

## Reproduce

```bash
# Parakeet, from ../moth-stt
PYTHONPATH=. uv run --frozen --project . --extra training --with sherpa-onnx -- \
  python baselines/parakeet_sherpa.py \
  --model-dir "$ROOT/models/parakeet-tdt-0.6b-v3-int8" \
  --manifests-dir "$ROOT/manifests" --audio-root ../moth-cs-dataset \
  --sets gold_train --num-threads 4 --out runs/E-018/parakeet-tdt-0.6b-v3-int8

# Whisper (it and auto), Canary
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/e018_whisper.py --model-dir "$ROOT/models/whisper-large-v3-turbo" \
  --manifest ../moth-cs-dataset/manifests/train.jsonl --audio-root ../moth-cs-dataset \
  --language it --device mps --out-dir runs/E-018/whisper-large-v3-turbo-it
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/e018_whisper.py --model-dir "$ROOT/models/whisper-large-v3-turbo" \
  --manifest ../moth-cs-dataset/manifests/train.jsonl --audio-root ../moth-cs-dataset \
  --language auto --device mps --out-dir runs/E-018/whisper-large-v3-turbo-auto
uv run --frozen --project ../moth-stt --extra training --with librosa -- \
  python tools/e018_teacher.py --family canary --model-dir "$ROOT/models/canary-1b-v2" \
  --manifest ../moth-cs-dataset/manifests/train.jsonl --audio-root ../moth-cs-dataset \
  --language it --device mps --out-dir runs/E-018/canary-1b-v2-it
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/e018_summary.py --root runs/E-018 --out runs/E-018/summary.json
```
