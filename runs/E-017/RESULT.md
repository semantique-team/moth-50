# E-017 option A — R-S1-cal calibration set (2026-11-21)

> **Status: reviewed, recorded, level-matched, manifest and mel shards built,
> control E-005 + E-034c read once.** The E-002 §8 amendment was written before
> that readout; no R1 readout has run. This is the set that P5/E-017 (review
> 2026-11-21) put before R1/R2; it is not training data and not a gate.

## What was built

- `../moth-cs-dataset/text-2026-11-cal/R-S1.jsonl`: **400 cues**, 50 per
  category, IDs `R-S1-CAL-0001…0400`, **reviewed by Tancredi** on 2026-11-21.
  Frozen at sha256 `ae5a26f4a0cc5925c35dd82bfa695fbc3408adccadd8f34678970aff3367bcc2`.
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

## Recording and level match

- Separate S1 session on 2026-11-21: **400/400 takes**, **16.95 min**, mean
  2.54 s (range 1.73–5.14 s), 19 retakes, 0 gaps. `session.log.jsonl` is next
  to the audio.
- The raw session was **16.58 dB hotter** than S1-dev (median RMS −17.76 vs
  −34.34 dBFS; mel offset +0.394). Same microphone/room (spectral profile
  correlation 0.998), so this was input gain, not a different channel.
- One fixed, set-wide **−16.58 dB** gain produced
  `audio/R-S1-cal-norm/` (`tools/match_audio_level.py`). The raw takes are
  untouched. After the gain: median RMS **−34.34 dBFS** (matches S1-dev),
  peak mean −18.5 dBFS, **0 clipped samples**, mel offset −0.014.
- The raw session had 69 files with at least one sample at full scale (max
  0.18 %, 50 with flat-topped runs ≥3 samples). The attenuation removes the
  level problem; the residual distortion is tiny and documented here.

## Artifacts

| Artifact | Value |
|---|---|
| script | `../moth-cs-dataset/text-2026-11-cal/R-S1.jsonl`, sha256 `ae5a26f4…` |
| manifest | `../moth-cs-dataset/manifests/R-S1-cal.jsonl`, sha256 `ef34c3f8…`, 400 rows, `split: dev` |
| audio | `../moth-cs-dataset/audio/R-S1-cal-norm/S1/S1_0001.wav…S1_0400.wav` |
| mel shards | `data/shards/s1-cal`: 400 clips, 1 shard, 0 errors, 0.28 h |
| QC | `runs/E-017/s1-cal-qc.json` (raw), `s1-cal-norm-qc.json` (level-matched) |
| level match | `runs/E-017/level-match.json` |
| dataset commit | `69d584e` in `semantique-team/moth-cs-dataset` (script, manifest, reports) |

Loader dry-run: 400 manifest rows, 400 shard-index entries, 0 missing ids, 0
rows with `split: train`; sampled clips 1/200/400 load as `(80, 227)`,
`(80, 278)`, `(80, 277)`.

## Control readout (E-005 + E-034c, 2026-11-21)

The product system was decoded once on `R-S1-cal` with the frozen decoder and
E-034c, before any R1 readout. This is the control side of the R1 paired gate,
not a gate itself.

| Metric | Value |
|---|---:|
| clips | 400 |
| ref words | 2,234 |
| WER | **13.9212 %** (95 % CI 12.2212–15.7753) |
| ETA | **75.6291 %** |
| insertion rate | 2.1934 % |

| Category | WER | ETA |
|---|---:|---:|
| single_word | 8.5911 % | 80.00 % |
| phrase | 9.8901 % | 79.63 % |
| brand | 28.5714 % | 30.77 % |
| acronym | 19.0840 % | 49.02 % |
| switch | 16.4948 % | 73.61 % |
| italianized | 9.3633 % | 66.00 % |
| control_it | 7.2165 % | — |
| control_en | 13.6667 % | 88.33 % |

Artifacts: `runs/E-017/control-e005-e034c.json` (metrics, per-category table,
`eta_errors`) and `.hyp.jsonl` (400 hypotheses; 400/400 ids, none empty).
`gate_eligible` is `false` by design: the LM is the product E-034c, not the
historical A3 gate LM. The graph
is the product INT8 graph (`14ba2034…`), the LM is E-034c (`4b122fbe…`,
packed `7a4332f1…`), and the decoder is the frozen 0.55 / 2.7, beam 12. When a
candidate is read with `--reference-hyp` pointing at this file,
`frozen_readout` still fills `paired`; R1 reads those paired CIs, not the
`gate_eligible` flag.

## Power caveat

755 English-term words against 2,048 on S1-dev. The set ranks; it does not
gate. Overall ETA differences below ~1.5 pp may not be resolvable, and
category-level ETA (`control_en`, `switch`) is report-only. That is why
E-023's gate keeps the S1-dev veto and uses the calibration set for ranking.

## Next steps

1. **Freeze is done.** The script is reviewed and frozen, the manifest and the
   mel shards exist, and **E-002 §8** names `R-S1-cal` and the rule (rank on
   calibration, dev-macro guard, S1-dev veto) before any calibration readout.
2. **Control readout is done** (E-005 + E-034c: WER 13.9212 %, ETA 75.6291 %).
   R1 (E-023) is authorized and seed 0 starts in `runs/E-023/r1-s0`, ranking
   against this control with
   `--reference-hyp runs/E-017/control-e005-e034c.hyp.jsonl`.
3. Keep `R-S1-cal` out of every training and dev-loader configuration. The
   `split: dev` label is a trainer guard, not a gate role.
4. T-600 S3 still checks speaker generalisation at release; T and FLEURS never
   select.

## Reproduce

```bash
# script + dedupe (already done; edits invalidate the freeze)
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

# level match, manifest, shards
uv run --no-project --with numpy --with soundfile python tools/match_audio_level.py \
  --source ../moth-cs-dataset/audio/R-S1-cal/S1 \
  --reference ../moth-cs-dataset/audio/R-S1-dev/S1 \
  --out ../moth-cs-dataset/audio/R-S1-cal-norm/S1 \
  --report runs/E-017/level-match.json
cd ../moth-cs-dataset
uv run --no-project --with numpy --with soundfile --with num2words \
  python -m scripts.build_manifest build --kind R --speaker S1 \
  --scripts-dir text-2026-11-cal --audio-dir audio/R-S1-cal-norm \
  --out manifests/R-S1-cal.raw.jsonl
uv run --no-project python ../moth-50/tools/finalize_s1_dev.py \
  --fragment manifests/R-S1-cal.raw.jsonl --dataset-root . \
  --source-prefix audio/R-S1/ --target-prefix audio/R-S1-cal-norm/ \
  --split dev --budget-minutes 10 20 --out manifests/R-S1-cal.jsonl
cd ../moth-50
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel \
  --manifest ../moth-cs-dataset/manifests/R-S1-cal.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/s1-cal --workers 8
```

