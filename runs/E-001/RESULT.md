# E-001 — RESULT: the canonical manifests reproduce Moth 0.8.1 exactly

> Executed 2026-11-17 on the M5 Pro. Decode-free: the retained 0.8.1 INT8
> hypotheses from Moth E-037 are re-scored with the vendored evaluator against
> the canonical `moth-cs` v1.0.0 test split. No training, no new audio, no
> release change.

## Verdict

The vendored evaluator and the canonical dataset reproduce the published
Moth 0.8.1 product numbers: **24.0826 % WER / 55.2253 % ETA** over the 600 T
utterances, with a per-utterance 95 % CI of **[22.0392, 25.9442]** for WER.
These are identical to Moth E-038's figures, and the per-speaker WER/ETA are
identical too. E-001 passes its gate.

## Numbers

Reference: `../moth-cs-dataset/manifests/test.jsonl` (600 rows, sha256
`f9147c85…`). Hypotheses:
`../moth-stt/runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl` (sha256
`a07fc630…`; 0.8.1 INT8 graph + A3 LM, beam 12, top-k 10, LM weight 0.55,
token bonus 2.7). Evaluator: `vendor/moth-stt/moth_eval.py`, per-utterance
bootstrap, 1,000 resamples, seed 0.

| Set | n | WER | 95 % CI | ETA |
|---|---:|---:|---|---:|
| pooled T | 600 | **24.0826** | [22.0392, 25.9442] | **55.2253** |
| S1 (trained) | 200 | 20.9802 | — | 63.2836 |
| S2 (trained) | 200 | 26.3249 | — | 51.5580 |
| S3 (never trained) | 200 | 24.9574 | — | 51.2676 |

Counts: 3,488 reference words, 526 substitutions, 281 deletions, 33 insertions
(0.946 insertions per 100 words); 467 ETA errors.

## Consistency checks

- `../moth-cs-dataset/scripts/verify.py`: manifests, audio, hashes, durations
  and splits are consistent (train 1,413 clips / 57.6 min; test 600 / 23.2 min;
  anchor 180 / 7.0 min; 16 excluded files).
- The canonical `test.jsonl` and Moth's `recordings/manifests/T.jsonl` have
  different bytes (`f9147c85…` vs `11e002d8…`) but identical ids, `norm`,
  `en_spans` and `speaker` for all 600 rows, so the scores are directly
  comparable.
- The pooled and per-speaker WER/ETA match E-038 to the last displayed digit.
  E-038's `metrics.json` records the same hypothesis sha256.

## Artifacts

- `runs/E-001/metrics.json` — full metrics, per-utterance scores and ETA errors.
- Evaluated at `git_sha` `492d3f4`; normalizer `stt.text.normalize` sha256
  `926d2464…`.
- No checkpoints; the hypotheses are the retained E-037 artifacts.

## What this establishes

- The evaluation path is reproducible on the canonical dataset, so later
  MOTH-50 results can be compared with 0.8.1 without a harness change.
- It does not make T a clean test: T is reused, and S3 was inspected repeatedly.
  It also does not re-decode the package; it re-scores the retained hypotheses,
  as E-001 allows.
