# E-001 — Reproduce Moth 0.8.1 on the canonical dataset

**Status: planned.** This is the first experiment of `MOTH-50`: establish that
the vendored evaluation code and the canonical dataset reproduce the published
Moth 0.8.1 numbers before any new model is trained.

## Hypothesis

Scoring the retained Moth 0.8.1 hypotheses (or re-decoding the 0.8.1 package)
with `vendor/moth-stt/moth_eval.py` on `manifests/test.jsonl` reproduces
**24.08% WER / 55.23% ETA** over the 600 T utterances, within the published
bootstrap interval and the decoder configuration (beam 12, top-k 10, LM weight
0.55, token bonus 2.7, E-031 LM).

If the numbers differ, the difference must be explained by a documented cause
(manifest version, normalizer, decoder settings, audio resampling), not ignored.

## Method

1. `python ../moth-cs-dataset/scripts/verify.py` — dataset integrity.
2. Load `manifests/test.jsonl`; confirm 600 rows, 200 per speaker.
3. Score the retained Moth 0.8.1 hypotheses from
   `../moth-stt/runs/E-037` (or re-decode the INT8 package) with the vendored
   evaluator.
4. Report pooled and per-speaker WER/ETA, with bootstrap CIs.
5. Compare with `../moth-stt/runs/E-038/metrics.json` and the paper's table.

## Gate

- Pass: WER and ETA match the published values within the reported interval and
  the per-speaker ordering is preserved.
- Fail: investigate before training; do not "fix" the baseline by changing the
  evaluator silently.

## Why it matters

The new model will be compared with 0.8.1. If the baseline is not reproducible
on the canonical manifests, every later comparison is suspect.
