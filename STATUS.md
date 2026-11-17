# MOTH-50 — session status and handoff (2026-11-17)

This is the first file to read when resuming. It records what is done, what is
running, and the exact next commands. Update it at the end of each session.

- **Remote:** `semantique-team/moth-50`, branch `main`
- **Model:** not trained yet. No E-003 run has started.
- **Tests:** 26 passed; `ruff check` and `ruff format --check` clean.
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
| S1-dev script | 900 sentences selected from pool, all reviewed | `../moth-cs-dataset/text-2026-11-dev/` |
| S1-dev recording | **200/900 WAV**, in progress | `../moth-cs-dataset/audio/R-S1-dev/S1/` |
| Public data | downloaded, verified, shards built | `data/raw/`, `data/shards/` |
| Trainer port | base+6, checkpointing, resume, greedy dev | `training/` |
| E-003 preflight | **ready: true**, train guard ok; no training | `runs/E-003/preflight.json`, `experiments/E-003-plan.md` |

## Next step: finish S1-dev, then E-003

1. Resume the S1-dev recording (it skips existing files):

```bash
cd ../moth-cs-dataset
uv run --no-project --with numpy --with soundfile --with num2words -- \
  python -m scripts.record --speaker S1 --kind R \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev
```

2. Build the dev manifest:

```bash
uv run --no-project --with numpy --with soundfile --with num2words -- \
  python -m scripts.build_manifest build --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev \
  --out manifests/R-S1-dev.jsonl
```

3. Patch `split: dev` and the audio paths (`audio/R-S1/` →
   `audio/R-S1-dev/`) exactly as in `experiments/E-002-s1-dev.md`, then build
   the dev mel shards from `moth-50`:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/s1-dev --workers 8
```

4. Uncomment the S1-dev block at the bottom of `training/configs/base6.toml`
   and rerun the preflight:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.preflight --config training/configs/base6.toml \
  --moth-stt ../moth-stt --out runs/E-003/preflight.json
```

5. Only after that, run the four E-003 arms (see `experiments/E-003-plan.md`):

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/base6.toml \
  --moth-stt ../moth-stt --head-blocks 0 --max-steps 4000 \
  --out runs/E-003/base+0 --device mps
# repeat with 2, 4, 6
```

The final E-003 decision uses the frozen beam+A3 `moth_eval` readout, not the
trainer's greedy dev metric.

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
| S1-dev | 200/900 | in progress | `../moth-cs-dataset/audio/R-S1-dev/S1` |

The CV 27.0 Italian archive is pinned in `data/sources.lock.json` (sha256
`27e1ef0e…`, 10,542,669,746 B). It was extracted to `data/raw/cv_it_27`.

## Gotchas

- **Clean-50 uses the full `cv_it_train` shards**, not a `cv_it_train_clean50`
  shard directory. The config is already correct.
- **Never build the noisy half with `build_mel`.** Use
  `tools/build_noisy_cv.py`, which mixes MUSAN/room tone at the E-024 SNR.
  It was already run; the shards and manifests are ready.
- Do not run `scripts/verify.py` on the v1.0.0 checkout with S1-dev audio: it
  checks fixed v1.0.0 counts. Use it only after creating a new dataset version.
- S1-dev manifest rows must be `split: dev` and point to
  `audio/R-S1-dev/S1/...`; `build_manifest` writes the canonical R-S1 path and
  `split: train`, so patch it as documented.
- The trainer's greedy dev WER is monitoring only; the gate is beam 12 + A3 LM
  (`moth_eval`).
- Checkpoints are gitignored; do not commit audio or checkpoints.

## Docs map

- `README.md` — project goal and current decisions.
- `docs/decisions.md` — D1–D15, including checkpointing (D15).
- `docs/architecture-options.md` — architecture, budget, data and protocol.
- `docs/review-2026-11-17.md` — evidence corrections and P1–P7.
- `experiments/E-001-baseline.md` — baseline reproduction.
- `experiments/E-002-protocol.md` — frozen evaluation/data protocol.
- `experiments/E-002-s1-dev.md` — S1-dev recording and manifest runbook.
- `experiments/E-003-plan.md` — pre-registered arms, gate and commands.
- `training/README.md` — trainer, data build, preflight and resume.
- `data/README.md`, `data/sources.lock.json` — pins and provenance.
