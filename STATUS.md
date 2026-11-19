# MOTH-50 — session status and handoff (2026-11-19)

This is the first file to read when resuming. It records what is done, what is
running, and the exact next commands. Update it at the end of each session.

- **Remote:** `semantique-team/moth-50`, branch `main`
- **Model:** E-005 is complete on the acoustic, package and browser side.
  `base+6`, 16,000 steps. Frozen beam+LM dev-macro **17.5816 % FP32 /
  17.6628 % INT8** vs the **23.0170 %** reference (paired **−5.4354 /
  −5.3542 pp**, CIs exclude zero). INT8 release **47,570,114 B (47.57 MB)**,
  **2.43 MB under** the 50 MB ceiling. The browser run loaded it in 63 ms and
  transcribed `T/S1/S1_0001.wav` exactly at **RTFx 22.7** (4 threads, SIMD,
  isolated). `runs/E-005/RESULT.md`.
- **Erratum (2026-11-19, D17):** that readout and package used the **pre-A3 LM**
  (`dcf1d59f…`) from a stale `../moth-stt/release/`, and Moth 0.8.1's NOTICE
  (`openai/whisper-tiny`). The dev-macro verdict stands; the A3 readout, the A3
  release (~47.84 MB, 2.16 MB of room) and S1-dev are the **E-005 addendum**.
  The tools now pin A3 by sha256 (`tools/release_assets.py`).
- **Plan:** `experiments/next-experiments.md` pre-registers E-009–E-016 (decoder
  recalibration, encoder-LR screen, ~3 epochs + EMA, VAD/LM bytes, browser
  checks, small architecture variants) and re-scopes E-004/E-006.
- **Tests:** 79 passed in the training env, 69 passed + 2 skipped (the torch
  tests) in a plain venv; `ruff check` and `ruff format --check` clean.
- **Tokenizer:** pinned at `data/tokenizer/moth_bpe.model`
  (sha256 `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`).

## What is done

| Item | State | Where |
|---|---|---|
| E-001 baseline reproduction | done: 24.0826 WER / 55.2253 ETA, matches E-038 | `runs/E-001/`, `experiments/E-001-baseline.md` |
| E-002 protocol | frozen | `experiments/E-002-protocol.md` |
| E-002 memory probe | 200 steps: 13.368 GB no-ckpt / 3.680 GB ckpt; no non-finite | `runs/E-002/RESULT.md` |
| E-002 dev reference | 23.0170 % macro (MLS-IT 28.1040, LS-EN 17.9299) | `runs/E-002/dev-reference.json` |
| Tokenizer pin (D14) | vendored and tested | `data/tokenizer/`, `docs/decisions.md` |
| S1-dev script | 960 cues, all reviewed: 900 short-form + 60 long (0901–0960) | `../moth-cs-dataset/text-2026-11-dev/` |
| S1-dev recording | **801/960 WAV** (159 left on 2026-11-19), 37.09 min, ~19 min projected, in progress | `../moth-cs-dataset/audio/R-S1-dev/S1/` |
| S1-dev pipeline | QC dry-run on 801 takes; manifest/mel/dev-loader dry-run on 300 | `tools/s1_dev_qc.py`, `tools/finalize_s1_dev.py` |
| Public data | downloaded, verified, shards built | `data/raw/`, `data/shards/` |
| Trainer port | base+6, pretrained init, identity heads, checkpointing, resume | `training/`, `docs/decisions.md` D16 |
| E-003 preflight | **ready: true**, train guard ok | `runs/E-003/preflight.json` |
| E-003 base+0 | **done**: greedy dev 48.12 %, 56.2 min, peak 7.397 GB | `runs/E-003/base+0/` |
| E-003 base+2 | **done**: greedy dev 36.90 %, 70.7 min, peak 8.406 GB | `runs/E-003/base+2/` |
| E-003 base+4 | **done**: 34.73 % / 34.74 % (2 seeds), ~85 min, peak 9.525 GB | `runs/E-003/base+4*/` |
| E-003 base+6 | **done, winner**: 33.46 % / 33.55 % (2 seeds), ~99 min, peak 8.594 GB | `runs/E-003/base+6*/` |
| E-005 base+6 long | **done**: 16,000 steps, greedy dev 21.7540 %, 6.73 h, peak 8.604 GB | `runs/E-005/base+6-16k/` |
| E-005 frozen readout | **gate PASS**: 17.5816 % FP32 / 17.6628 % INT8 vs 23.0170 (pre-A3 LM, see erratum) | `runs/E-005/readout*.json` |
| E-005 INT8 export | **done**: 47,570,114 B (47.57 MB), 2.43 MB headroom; pre-A3 LM, Moth NOTICE: rebuild | `runs/E-005/release/`, `export-report.json` |
| E-005 browser run | **done**: exact transcript, RTFx 22.7 (4 threads, SIMD, isolated) | `runs/E-005/browser-run.json` |
| Release asset pins (D17) | A3 LM, 0.8.1 graph, Silero graphs pinned and fetched by sha256 | `tools/release_assets.py`, `data/sources.lock.json` |
| S1-dev readout tool | `frozen_readout.py --mode mixed`: WER/ETA + paired no-regression vs 0.8.1 | `tools/frozen_readout.py`, E-002 §6 |
| Post-E-005 tooling | decoder grid (E-009), EMA (`train.ema_decay`), E-010 config, VAD probe | `experiments/next-experiments.md` |
| VAD 16 kHz preflight | bit-identical on 23 synthetic clips, −1,037,921 B | `runs/probe-2026-11-19/` |

## Next step: finish S1-dev, then its readout

1. Resume the recording. The recorder skips existing takes, so it resumes at
   the first missing cue (~661 of 960 on 2026-11-19):

```bash
cd ../moth-cs-dataset
uv run --no-project --with numpy --with soundfile --with num2words -- \
  python -m scripts.record --speaker S1 --kind R \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev
```

2. Check the session at any point (read-only; exits 1 on a real audio error):

```bash
uv run --no-project --with numpy --with soundfile -- \
  python ../moth-50/tools/s1_dev_qc.py \
  --scripts text-2026-11-dev/R-S1.jsonl --audio audio/R-S1-dev/S1
```

3. Build the dev manifest, patch it to `split: dev` + `audio/R-S1-dev/`, and
   build the mel shards:

```bash
uv run --no-project --with numpy --with soundfile --with num2words -- \
  python -m scripts.build_manifest build --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev \
  --out manifests/R-S1-dev.jsonl
uv run --no-project -- \
  python ../moth-50/tools/finalize_s1_dev.py \
  --fragment manifests/R-S1-dev.jsonl --dataset-root . --check   # then drop --check
```

```bash
cd ../moth-50
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/s1-dev --workers 8
```

4. Fetch the pinned assets once (A3 LM, 0.8.1 graph, Silero graphs; sha256
   checked):

```bash
python tools/release_assets.py fetch
```

5. Run the **E-005 addendum** (`experiments/next-experiments.md`): dev-macro
   readout with A3, the A3 release with the MOTH-50 NOTICE, then S1-dev with
   `tools/frozen_readout.py --mode mixed` — first the 0.8.1 reference
   (`--onnx data/raw/moth-0.8.1/model.int8.onnx --hyp-out …`), then MOTH-50 with
   `--reference-hyp`. The verdict rule is E-002 §6. Then E-009 (decoder grid) and
   E-010 (encoder-LR screen); no MPS training during a recording session.

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
| S1-dev | 801/960 | 0.62 / ~0.94 | `../moth-cs-dataset/audio/R-S1-dev/S1` |

The CV 27.0 Italian archive is pinned in `data/sources.lock.json` (sha256
`27e1ef0e…`, 10,542,669,746 B). It was extracted to `data/raw/cv_it_27`.

## Gotchas

- **Clean-50 uses the full `cv_it_train` shards**, not a `cv_it_train_clean50`
  shard directory. The config is already correct.
- **Never build the noisy half with `build_mel`.** Use
  `tools/build_noisy_cv.py`, which mixes MUSAN/room tone at the E-024 SNR.
- Do not run `scripts/verify.py` on the v1.0.0 checkout with S1-dev audio: it
  checks fixed v1.0.0 counts. Use it only after a new dataset version.
- S1-dev manifest rows must be `split: dev` and point to
  `audio/R-S1-dev/S1/...`; `build_manifest` writes the canonical R-S1 path and
  `split: train`. Use `tools/finalize_s1_dev.py`, which patches and verifies
  every path instead of a heredoc.
- The first 900 cues are **short-form only** (5.7 words, ~3 s). Sixty long cues
  (`R-S1-DEV-0901`–`0960`, 24–38 words, ~8–17 s) were appended and reviewed to
  cover the 8–20 s dictation mix. The projection is ~58 min, inside the 30–60
  min protocol; if it lands over 60, skip a few `control_en` cues.
- S1-dev levels match the S1 corpus: peak −17.9 dBFS mean vs −19.0 on
  `audio/R-S1/S1`, drift 1.2 dB over 800 takes. Do not change the input gain
  mid-session; the feature path has no loudness normalization. Two QC warnings:
  `S1_0563.wav` is 1.49 s (the cue is four words, so likely fine) and
  `S1_0726.wav` peaks at 0.0 dBFS with ~58 full-scale samples in one run —
  re-record that take before finalizing.
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
  there (or vendor the runtime); without it the page rejects the 47.57 MB
  package. Vendoring the runtime is part of E-016.
- **Never take the LM from `../moth-stt/release/`.** That gitignored copy is the
  pre-A3 LM (`dcf1d59f…`). The readout, calibration and export tools resolve the
  LM through `tools/release_assets.py` (A3, `3be1177b…`) and refuse anything
  else without `--allow-other-lm`.
- **The gate readout is `tools/frozen_readout.py`.** It loads `ckpt_best.pt`
  directly, or the INT8 graph with `--onnx`, and writes `readout.json` +
  hypotheses; `--mode mixed` reads one mixed set (S1-dev). FP32 and INT8 differ
  by 0.08 pp. Run it on CPU to keep the machine quiet.
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
- `docs/decisions.md` — D1–D18, including checkpointing (D15), init (D16),
  release asset pins (D17) and the post-E-005 plan (D18).
- `docs/architecture-options.md` — architecture, budget, data and protocol.
- `docs/review-2026-11-17.md` — evidence corrections and P1–P7.
- `experiments/E-001-baseline.md` — baseline reproduction.
- `experiments/E-002-protocol.md` — frozen evaluation/data protocol.
- `experiments/E-002-s1-dev.md` — S1-dev recording and manifest runbook.
- `experiments/E-003-plan.md` — arms, screen result and E-005 follow-up.
- `experiments/next-experiments.md` — post-E-005 plan: E-005 addendum,
  E-009–E-016, re-scoped E-004/E-006, gates and commands.
- `runs/E-005/RESULT.md` — long fine-tune, INT8 package, browser run, erratum.
- `runs/probe-2026-11-19/` — 16 kHz Silero graph parity preflight.
- `training/README.md` — trainer, data build, initialization, preflight, resume.
- `data/README.md`, `data/sources.lock.json` — pins and provenance.
