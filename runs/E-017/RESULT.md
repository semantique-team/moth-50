# E-017 option A — R-S1-cal calibration script (2026-11-21)

> **Status: script built and deduped, unreviewed and not recorded.** This is the
> calibration set that P5/E-017 (review 2026-11-21) put before R1/R2; it is not
> training data and not a gate.

## What was built

- `../moth-cs-dataset/text-2026-11-cal/R-S1.jsonl`: **400 cues**, 50 per
  category, IDs `R-S1-CAL-0001…0400`, `reviewed: false`.
- Generator: `deepseek-v4.1-flash`, date 2026-11-21; same schema, normalizer
  and category rules as S1-dev (`tools/build_cal_script.py` runs the dataset's
  own `validate_candidate`).
- 4–9 words, mean **5.58**; **755** English-term words.
- Sources: `candidates_a.py` (single_word, phrase, brand, acronym),
  `candidates_b.py` (italianized, switch, control_it, control_en).

## Dedupe and overlap

Checked against `T`, `A`, `R-S1`, `R-S2`, `pool` and the 960 S1-dev cues:

| Check | Result |
|---|---|
| exact duplicates | **0** |
| near duplicates (word-3-gram Jaccard > 0.8) | **0** |
| exact sentences in the E-031 LM corpus | **0** |
| word-4-gram share vs the LM corpus | **8.75 %** (T 9.75 %, clean S1-dev 4.09 %) |
| multi-word terms with corpus count 1 | 27.5 % (T 23.6 %, clean S1-dev 9.9 %) |

Reports: `dedupe-report.json`, `lm-corpus-overlap.json` (also copied into the
dataset directory). No candidate was accepted by weakening the check; 54
near-duplicates found in the first drafts were replaced.

## Power caveat

755 English-term words against 2,048 on S1-dev. The set ranks; it does not
gate. Overall ETA differences below ~1.5 pp may not be resolvable, and
category-level ETA (`control_en`, `switch`) is report-only. That is why
E-023's gate keeps the S1-dev veto and uses the calibration set for ranking.

## Next steps

1. **Review** all 400 cues in `../moth-cs-dataset/text-2026-11-cal/` and mark
   them reviewed (commands in its README). Freeze the file before any readout.
2. **Record** in a separate S1 session, `audio/R-S1-cal/S1/S1_0001.wav` …
   `S1_0400.wav`; ~13 min of speech, ~25–30 min with pauses.
3. Build the manifest (`split: dev`) and mel shards like S1-dev; never train.
4. Write the dated **E-002 amendment §8** naming `R-S1-cal` and the rule
   (rank on calibration, dev-macro guard, S1-dev veto) **before** the first
   calibration readout.
5. Then R1 (E-023) can start with the calibration set in place; no training
   starts without an explicit go-ahead.

## Reproduce

```bash
python tools/build_cal_script.py \
  --candidates runs/E-017/candidates_a.py \
  --candidates runs/E-017/candidates_b.py \
  --scripts ../moth-cs-dataset/text/T.jsonl \
  --scripts ../moth-cs-dataset/text/A.jsonl \
  --scripts ../moth-cs-dataset/text/R-S1.jsonl \
  --scripts ../moth-cs-dataset/text/R-S2.jsonl \
  --scripts ../moth-cs-dataset/text/pool.jsonl \
  --scripts ../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \
  --lm-corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \
  --out ../moth-cs-dataset/text-2026-11-cal/R-S1.jsonl \
  --report runs/E-017/dedupe-report.json
python tools/lm_corpus_overlap.py \
  --corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \
  --set s1-cal=../moth-cs-dataset/text-2026-11-cal/R-S1.jsonl \
  --set s1-dev=../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \
  --out runs/E-017/lm-corpus-overlap.json
```
