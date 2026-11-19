# MOTH-50 — session status and handoff (2026-11-19)

This is the first file to read when resuming. It records what is done, what is
running, and the exact next commands. Update it at the end of each session.

- **Remote:** `semantique-team/moth-50`, branch `main`
- **Model: E-005 is complete and the E-002 gate passes on both halves.**
  `base+6`, 16,000 steps, pinned A3 LM. Frozen beam+A3 dev-macro **17.5565 %
  FP32 / 17.6212 % INT8** against the **23.0170 %** reference (paired
  **−5.4605 / −5.3957 pp**, CIs exclude zero). Held-out S1-dev (960/960 takes,
  51.30 min): **13.9956 / 14.0541 %** vs **22.2498 %** for 0.8.1, ETA
  **75.34 / 75.54 %** vs **62.11 %**, no WER or ETA regression outside its CI.
  INT8 release **47,836,502 B (47.84 MB)**, **2.16 MB under** the 50 MB ceiling,
  A3 LM + MOTH-50 NOTICE. Browser run exact transcript at **RTFx 22.3**
  (4 threads, SIMD, isolated). `runs/E-005/RESULT.md`.
- **Erratum (2026-11-19, D17):** the first E-005 readout/package used the
  **pre-A3 LM** (`dcf1d59f…`) and Moth 0.8.1's NOTICE. The verdict stood, and
  the final A3 numbers above supersede the old ones. The tools pin A3 by sha256
  (`tools/release_assets.py`).
- **Plan:** `experiments/next-experiments.md` pre-registers E-009–E-022.
  **E-009 ran on 2026-11-19 and kept the frozen decoder; E-010–E-022 have not
  run.** E-017–E-022 come from the frontier-model review
  (`docs/frontier-stt-2026-11-19.md`, D19 proposed). No training starts without
  an explicit go-ahead.
- **Tests:** 98 passed in the training env, 88 passed + 2 skipped (the torch
  tests) in a plain venv; `ruff check` and `ruff format --check` clean.
- **Tokenizer:** pinned at `data/tokenizer/moth_bpe.model`
  (sha256 `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`).

## What is done

| Item | State | Where |
|---|---|---|
| E-001 baseline reproduction | done: 24.0826 WER / 55.2253 ETA, matches E-038 | `runs/E-001/`, `experiments/E-001-baseline.md` |
| E-002 protocol | frozen, §6 amendment 2026-11-19 | `experiments/E-002-protocol.md` |
| E-002 memory probe | 200 steps: 13.368 GB no-ckpt / 3.680 GB ckpt; no non-finite | `runs/E-002/RESULT.md` |
| E-002 dev reference | 23.0170 % macro (MLS-IT 28.1040, LS-EN 17.9299) | `runs/E-002/dev-reference.json` |
| Tokenizer pin (D14) | vendored and tested | `data/tokenizer/`, `docs/decisions.md` |
| S1-dev script | 960 cues, all reviewed: 900 short-form + 60 long (0901–0960) | `../moth-cs-dataset/text-2026-11-dev/` |
| S1-dev recording | **960/960 WAV, 51.30 min**, mean 3.21 s, no hard QC errors | `../moth-cs-dataset/audio/R-S1-dev/S1/` |
| S1-dev QC | 960/960, 0 errors, 2 advisory short takes (563, 886) | `runs/E-002/s1-dev-qc.json` |
| S1-dev manifest | 960 rows, `split: dev`, `audio/R-S1-dev/` | `../moth-cs-dataset/manifests/R-S1-dev.jsonl` |
| S1-dev mel shards | 960 clips, 1 shard, 0 errors | `data/shards/s1-dev/` |
| Public data | downloaded, verified, shards built | `data/raw/`, `data/shards/` |
| Trainer port | base+6, pretrained init, identity heads, checkpointing, resume | `training/`, `docs/decisions.md` D16 |
| E-003 preflight | **ready: true**, train guard ok | `runs/E-003/preflight.json` |
| E-003 base+0 | **done**: greedy dev 48.12 %, 56.2 min, peak 7.397 GB | `runs/E-003/base+0/` |
| E-003 base+2 | **done**: greedy dev 36.90 %, 70.7 min, peak 8.406 GB | `runs/E-003/base+2/` |
| E-003 base+4 | **done**: 34.73 % / 34.74 % (2 seeds), ~85 min, peak 9.525 GB | `runs/E-003/base+4*/` |
| E-003 base+6 | **done, winner**: 33.46 % / 33.55 % (2 seeds), ~99 min, peak 8.594 GB | `runs/E-003/base+6*/` |
| E-005 base+6 long | **done**: 16,000 steps, greedy dev 21.7540 %, 6.73 h, peak 8.604 GB | `runs/E-005/base+6-16k/` |
| E-005 A3 readout | **gate PASS**: 17.5565 % FP32 / 17.6212 % INT8 vs 23.0170 | `runs/E-005/readout-a3*.json` |
| E-005 A3 INT8 export | **done**: 47,836,502 B (47.84 MB), 2.16 MB headroom | `runs/E-005/release-a3/`, `export-report-a3.json` |
| E-005 A3 browser run | **done**: exact transcript, RTFx 22.3 (4 threads, SIMD, isolated) | `runs/E-005/browser-run-a3.json` |
| E-009 decoder grid | **done, no adoption**: dev-macro 0.45/2.0 17.3427 % (−0.2786 pp), but S1-dev veto +0.5844 pp WER / −1.1230 pp ETA; frozen 0.55/2.7 stays | `runs/E-009/` |
| **S1-dev no-regression** | **PASS**: 13.9956 % FP32 / 14.0541 % INT8 vs 22.2498; ETA +13.23/+13.43 pp | `runs/E-005/s1-dev-a3-*.json` |
| Release asset pins (D17) | A3 LM, 0.8.1 graph, Silero graphs pinned and fetched by sha256 | `tools/release_assets.py`, `data/sources.lock.json` |
| S1-dev readout tool | `frozen_readout.py --mode mixed`: WER/ETA + paired no-regression vs 0.8.1 | `tools/frozen_readout.py`, E-002 §6 |
| Post-E-005 tooling | decoder grid (E-009), EMA (`train.ema_decay`), E-010 config, VAD probe | `experiments/next-experiments.md` |
| VAD 16 kHz preflight | bit-identical on 23 synthetic clips, −1,037,921 B | `runs/probe-2026-11-19/` |
| S1-dev term-error analysis | 501 misses (0.8.1: 776); deletions 151 → 53; acronyms 29.2 % / brands 48.1 % ETA; elided acronyms not improved | `runs/E-005/RESULT.md`, `runs/E-005/eta-breakdown.json`, `tools/eta_breakdown.py` |
| Frontier-model review (D19, proposed) | teachers, LM text and ceilings, not architectures; E-017–E-022 pre-registered | `docs/frontier-stt-2026-11-19.md` |

The pre-A3 artifacts (`readout.json`, `readout-int8.json`, `release/`,
`browser-run.json`) are kept for the erratum; they are **not** the final numbers.

## Next step: the T-600 readout, then E-017/E-018; no training without go-ahead

The gate is complete and **E-009 kept the frozen decoder**: the dev-macro winner
0.45 / 2.0 (−0.2786 pp) was rejected by the S1-dev veto (+0.5844 pp WER,
−1.1230 pp ETA), so 0.55 / 2.7 stays (`runs/E-009/RESULT.md`). Nothing is
running. The order, with the pre-registered gates, is in
`experiments/next-experiments.md`:

1. **E-005 product readout** — T-600 and FLEURS once, as a report, with 0.8.1
   and Parakeet (10.98 % / 83.60 % on T-600) as references; ~20 min CPU after
   the T/FLEURS mel shards. The only measurement on speakers other than S1.
2. **E-017 calibration set** — owner decision: a new ~15–20 min `R-S1-cal`
   session (recommended) or a fixed split of S1-dev. Needed before any LM or
   decoder change.
3. **E-018 teacher bake-off** — permissive open models on the gold train set;
   no training.
4. **E-010 encoder-LR screen** — 2 arms × 4,000 steps, ~3 h on MPS.
5. **E-012 bytes** — 16 kHz VAD on real audio + browser, and lossless LM packing;
   no training, can run in parallel.
6. **E-011 long schedule + EMA** — ~17 h per seed, 2 seeds.
7. **E-019–E-021** — teacher labels, LLM text for the LM, targeted acronym/brand
   data; then E-004, E-016, the architecture screens and E-022.

Reproduction of the final readouts (assets must be fetched once):

```bash
python tools/release_assets.py fetch
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed \
  --onnx runs/E-005/release-a3/model.int8.onnx \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --reference-hyp runs/E-005/s1-dev-0.8.1.hyp.jsonl \
  --out runs/E-005/s1-dev-a3-int8-rerun.json
```

## Data inventory

| Source | Clips | Hours | Path |
|---|---:|---:|---|
| CV clean (full `cv_it_train`) | 113,118 | 167.31 | `data/shards/cv_it_train` |
| CV clean-50 manifest | 56,769 | 84.22 | `../moth-stt/data/manifests/cv_it_train_clean50.jsonl` |
| CV noisy-50 | 56,349 | 83.33 | `data/shards/cv_it_noisy50` |
| MLS Italian train | 40,953 | 169.89 | `data/shards/mls_it_train` |
| MLS Italian dev | 1,248 | 5.18 | `data/shards/mls_it_dev` |
| LibriSpeech train-clean-100 | 28,370 | 100.00 | `data/shards/ls_en_train` |
| LibriSpeech dev-clean | 2,691 | 5.31 | `data/shards/ls_en_dev` |
| Spliced IT+EN | 20,623 | 109.90 | `data/shards/spliced_it_en` |
| Gold v1.0.0 train | 1,413 | 0.96 | `data/shards/gold-train` |
| MUSAN noise set | 480 | 1.33 | `data/raw/noise`, `data/manifests/noise.jsonl` |
| S1-dev | **960** | **0.855** | `data/shards/s1-dev`, `../moth-cs-dataset/audio/R-S1-dev/S1` |

The CV 27.0 Italian archive is pinned in `data/sources.lock.json` (sha256
`27e1ef0e…`, 10,542,669,746 B). It was extracted to `data/raw/cv_it_27`.

## Gotchas

- **Clean-50 uses the full `cv_it_train` shards**, not a `cv_it_train_clean50`
  shard directory. The config is already correct.
- **Never build the noisy half with `build_mel`.** Use
  `tools/build_noisy_cv.py`, which mixes MUSAN/room tone at the E-024 SNR.
- Do not run `scripts/verify.py` on the v1.0.0 checkout with S1-dev audio: it
  checks fixed v1.0.0 counts. Use it only after a new dataset version.
- S1-dev manifest rows are `split: dev` and point to
  `audio/R-S1-dev/S1/...`; `build_manifest` writes the canonical R-S1 path and
  `split: train`, so `tools/finalize_s1_dev.py` must patch them. The final
  manifest is committed in `../moth-cs-dataset`.
- **S1-dev is held out.** It was not used for training or decoder tuning. Do not
  turn it into a calibration set without amending E-002 first (see E-009 above).
- S1-dev levels: 960 takes, 51.30 min, peak −17.6 dBFS mean (max −3.8), drift
  1.6 dB, no hard errors. Two advisory warnings: `S1_0563.wav` 1.49 s and
  `S1_0886.wav` 1.27 s, both four-word cues. `S1_0726` was re-recorded after a
  0.0 dBFS peak; the final max is −3.8 dBFS.
- **Every training run needs `model.init`.** `base6.toml` points at
  `data/raw/whisper-base-encoder.pt` (gitignored; rebuild with
  `tools/convert_whisper_encoder.py`). A random-init run is invalid: measured
  99 % dev WER after 4,000 steps. The extra blocks are identity-initialized
  (D16), so base+0/+2/+4/+6 start as the same function.
- **The web runtime reads `config.budget_bytes`.** `moth-stt/web/ort.js` uses it
  when present (default 20,000,000, so 0.8.1 is unchanged); the MOTH-50 release
  declares 50,000,000. The patch is local commit `b0fa11c` in the `moth-stt`
  checkout: that repository is **archived and read-only**, so it cannot be
  pushed. If MOTH-50 ships its own web runtime, carry this one-line change
  there (or vendor the runtime); without it the page rejects the 47.84 MB
  package. Vendoring the runtime is part of E-016.
- **Never take the LM from `../moth-stt/release/`.** That gitignored copy is the
  pre-A3 LM (`dcf1d59f…`). The readout, calibration and export tools resolve the
  LM through `tools/release_assets.py` (A3, `3be1177b…`) and refuse anything
  else without `--allow-other-lm`.
- **E-009 kept the frozen decoder.** The dev-macro grid preferred 0.45 / 2.0
  (−0.28 pp), but S1-dev regressed (+0.58 pp WER, −1.12 pp ETA), so 0.55 / 2.7
  stays. Do not adopt a decoder on dev-macro without the S1-dev veto.
- **The gate readout is `tools/frozen_readout.py`.** It loads `ckpt_best.pt`
  directly, or the INT8 graph with `--onnx`; `--mode mixed` reads one mixed set
  (S1-dev) and `--reference-hyp` adds the paired no-regression verdict. FP32 and
  INT8 differ by 0.06 pp macro on the A3 readout.
- **Measured training (2026-11-17/19):** 4,000 steps at 56.2–99.5 min
  (`base+0`…`base+6`), peak 7.397–9.525 GB; 16,000 steps at 6.73 h, peak
  8.604 GB; 0 non-finite. The E-002 4.47 s/step was the padded worst case.
- **4,000 steps is ~0.30 epoch; 16,000 steps is 1.19 epochs.** E-005's epoch 0
  ended after 13,458 batches (the earlier ~11.2k steps/epoch estimate assumed
  full 192 s batches). The E-003 numbers are a depth screen, not the ceiling.
- Checkpoints every 100 steps (`ckpt_last.pt`); resume with
  `--resume runs/<run>/checkpoints/ckpt_last.pt`.
- The trainer's greedy dev WER is monitoring only; the gate is beam 12 + A3 LM.
- Checkpoints and release binaries are gitignored; do not commit audio.

## Docs map

- `README.md` — project goal and current decisions.
- `docs/decisions.md` — D1–D19, including checkpointing (D15), init (D16),
  release asset pins (D17), the post-E-005 plan (D18) and frontier models as
  teachers (D19, proposed).
- `docs/architecture-options.md` — architecture, budget, data and protocol.
- `docs/frontier-stt-2026-11-19.md` — what the frontier STT models can and
  cannot give MOTH-50; licences; teachers.
- `docs/review-2026-11-17.md` — evidence corrections and P1–P7.
- `experiments/E-001-baseline.md` — baseline reproduction.
- `experiments/E-002-protocol.md` — frozen evaluation/data protocol.
- `experiments/E-002-s1-dev.md` — S1-dev recording and manifest runbook.
- `experiments/E-003-plan.md` — arms, screen result and E-005 follow-up.
- `experiments/next-experiments.md` — post-E-005 plan: E-005 addendum and
  product readout, E-009–E-022, re-scoped E-004/E-006, gates and commands.
- `runs/E-005/RESULT.md` — long fine-tune, A3 readout, S1-dev, term-error
  analysis, package, browser.
- `runs/E-009/RESULT.md` — decoder grid, S1-dev veto, post-hoc analysis.
- `runs/probe-2026-11-19/` — 16 kHz Silero graph parity preflight.
- `training/README.md` — trainer, data build, initialization, preflight, resume.
- `data/README.md`, `data/sources.lock.json` — pins and provenance.
