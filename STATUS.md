# MOTH-50 — session status and handoff (2026-11-19)

This is the first file to read when resuming. It records what is done, what is
running, and the exact next commands. Update it at the end of each session.

- **Remote:** `semantique-team/moth-50`, branch `main`
- **Model:** E-005 is complete on the acoustic, package and browser side.
  `base+6`, 16,000 steps. Frozen beam+A3 dev-macro **17.5816 % FP32 /
  17.6628 % INT8** vs the **23.0170 %** reference (paired **−5.4354 /
  −5.3542 pp**, CIs exclude zero). INT8 release **47,570,114 B (47.57 MB)**,
  **2.43 MB under** the 50 MB ceiling. The browser run loaded it in 63 ms and
  transcribed `T/S1/S1_0001.wav` exactly at **RTFx 22.7** (4 threads, SIMD,
  isolated). Only the S1-dev no-regression half is pending.
  `runs/E-005/RESULT.md`.
- **Tests:** 44 passed + 1 skipped in the plain venv (the torch test runs in the
  training env: 49 passed); `ruff check` and `ruff format --check` clean.
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
| S1-dev recording | **300/960 WAV**, 14.26 min, ~58 min projected, in progress | `../moth-cs-dataset/audio/R-S1-dev/S1/` |
| S1-dev pipeline | QC + dev patch + mel/dev-loader dry-run on 300 takes | `tools/s1_dev_qc.py`, `tools/finalize_s1_dev.py` |
| Public data | downloaded, verified, shards built | `data/raw/`, `data/shards/` |
| Trainer port | base+6, pretrained init, identity heads, checkpointing, resume | `training/`, `docs/decisions.md` D16 |
| E-003 preflight | **ready: true**, train guard ok | `runs/E-003/preflight.json` |
| E-003 base+0 | **done**: greedy dev 48.12 %, 56.2 min, peak 7.397 GB | `runs/E-003/base+0/` |
| E-003 base+2 | **done**: greedy dev 36.90 %, 70.7 min, peak 8.406 GB | `runs/E-003/base+2/` |
| E-003 base+4 | **done**: 34.73 % / 34.74 % (2 seeds), ~85 min, peak 9.525 GB | `runs/E-003/base+4*/` |
| E-003 base+6 | **done, winner**: 33.46 % / 33.55 % (2 seeds), ~99 min, peak 8.594 GB | `runs/E-003/base+6*/` |
| E-005 base+6 long | **done**: 16,000 steps, greedy dev 21.7540 %, 6.73 h, peak 8.604 GB | `runs/E-005/base+6-16k/` |
| E-005 frozen readout | **gate PASS**: 17.5816 % FP32 / 17.6628 % INT8 vs 23.0170 | `runs/E-005/readout*.json` |
| E-005 INT8 export | **done**: 47,570,114 B (47.57 MB), 2.43 MB headroom | `runs/E-005/release/`, `export-report.json` |
| E-005 browser run | **done**: exact transcript, RTFx 22.7 (4 threads, SIMD, isolated) | `runs/E-005/browser-run.json` |

## Next step: finish S1-dev, then its readout

1. Resume the recording (it skips existing files). The 60 long cues are already
   reviewed, so the recorder plans 301–960:

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

4. S1-dev readout: `tools/frozen_readout.py` currently evaluates the two
   monolingual dev sets (`config.data.dev[0]` = MLS-IT, `[1]` = LS-EN). The
   S1-dev session is a mixed IT–EN product set, so the readout needs a mixed-set
   mode (same beam 12 / top-k 10 / lm_weight 0.55 / token_bonus 2.7) before the
   "no S1-dev regression" comparison. Do that extension when the session is
   complete; do not reuse the monolingual macro for it.

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
| S1-dev | 300/960 | 0.24 / ~0.97 | `../moth-cs-dataset/audio/R-S1-dev/S1` |

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
- S1-dev levels match the S1 corpus: peak −19.0 dBFS mean vs −19.0 on
  `audio/R-S1/S1`, 0 clipped, drift 0.2 dB over 300 takes. Do not change the
  input gain mid-session; the feature path has no loudness normalization.
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
  package.
- **The gate readout is `tools/frozen_readout.py`.** It loads `ckpt_best.pt`
  directly, or the INT8 graph with `--onnx`, and writes `readout.json` +
  hypotheses. FP32 and INT8 differ by 0.08 pp; the release is 47.57 MB. Run it
  on CPU to keep the machine quiet.
- **Measured training (2026-11-17/19):** 4,000 steps at 56.2–99.5 min
  (`base+0`…`base+6`), peak 7.397–9.525 GB; 16,000 steps at 6.73 h, peak
  8.604 GB; 0 non-finite. The E-002 4.47 s/step was the padded worst case.
- **4,000 steps is ~0.36 epoch.** The ~595 h mixture at 192 audio-s per step is
  ~11.2k steps/epoch. The E-003 numbers are a depth screen, not the ceiling.
- Checkpoints every 100 steps (`ckpt_last.pt`); resume with
  `--resume runs/<run>/checkpoints/ckpt_last.pt`.
- The trainer's greedy dev WER is monitoring only; the gate is beam 12 + A3 LM.
- Checkpoints and release binaries are gitignored; do not commit audio.

## Docs map

- `README.md` — project goal and current decisions.
- `docs/decisions.md` — D1–D16, including checkpointing (D15) and init (D16).
- `docs/architecture-options.md` — architecture, budget, data and protocol.
- `docs/review-2026-11-17.md` — evidence corrections and P1–P7.
- `experiments/E-001-baseline.md` — baseline reproduction.
- `experiments/E-002-protocol.md` — frozen evaluation/data protocol.
- `experiments/E-002-s1-dev.md` — S1-dev recording and manifest runbook.
- `experiments/E-003-plan.md` — arms, screen result and E-005 follow-up.
- `runs/E-005/RESULT.md` — long fine-tune, INT8 package, browser run.
- `training/README.md` — trainer, data build, initialization, preflight, resume.
- `data/README.md`, `data/sources.lock.json` — pins and provenance.
