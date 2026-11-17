# E-003 — Architecture screen plan

**Status: preflight only (2026-11-17); no training started.** The data mixture is
not complete yet and S1-dev has not been recorded. This document pre-registers
the screen so the arms cannot be moved after seeing results.

## Question

Under the 50 MB package ceiling, does the extra depth pay for itself? Compare,
from the same Whisper-base initialization and data order:

| Arm | Head blocks | Acoustic params | Package with A3 + VAD | Room |
|---|---:|---:|---:|---:|
| base+0 | 0 | 20,348,417 | 28.50 MB | 21.50 MB |
| base+2 | 2 | 26,652,161 | 34.91 MB | 15.09 MB |
| base+4 | 4 | 32,955,905 | 41.33 MB | 8.67 MB |
| base+6 | 6 | 39,259,649 | 47.74 MB | 2.26 MB |

`base+6` is the primary candidate; `base+4` is also the E-004 alternative when
paired with a larger LM. E-003 decides the acoustic depth, not the LM trade-off.

## Protocol

- **Data:** frozen E-002 E-028 mixture (`training/configs/base6.toml`), same
  manifests and order for every arm; no new S1 data, TTS or pseudo-labels.
- **Steps:** 4,000 per arm, one seed. If the top two arms are within ~1 pp,
  repeat them with a second seed before naming a winner.
- **Optimizer:** AdamW, head/extra LR 3e-4, encoder LR 3e-5 (screened in P2),
  1,500-step warm-up, cosine to 0.05, grad clip 1.0, bf16 autocast.
- **Memory:** per-block activation checkpointing on (D15), 192 audio-s / 32-clip
  cap. E-002 measured 3.68 GB at the 6 × 30 s bucket.
- **Decoder:** fixed for the screen (beam 12, top-k 10, LM weight 0.55, token
  bonus 2.7, A3 LM). Recalibrate per arm only in E-004, never on T.
- **Metrics:** dev-macro WER on MLS-IT + LS-EN; S1-dev WER/ETA; FLEURS-IT and
  T-600 as diagnostics only. Greedy dev WER in the trainer is monitoring, not a
  gate.

## Gate

At least **1.0 pp better dev-macro WER** than the re-scored 0.8.1 reference
**23.0170 %** (`runs/E-002/dev-reference.json`), with a paired bootstrap CI
excluding zero, and **no S1-dev regression outside its CI**. Memory and
throughput must stay within the E-002 envelope. A tie is a tie; do not call a
sub-1 pp gap a winner without a second seed.

## Commands (not run yet)

The trainer takes one config and overrides only the arm:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.preflight --config training/configs/base6.toml \
  --moth-stt ../moth-stt --out runs/E-003/preflight.json

uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/base6.toml \
  --moth-stt ../moth-stt --head-blocks 0 --max-steps 4000 \
  --out runs/E-003/base+0 --device mps
# same with --head-blocks 2 / 4 / 6
```

Each run writes `train.log.jsonl`, `run.json`, checkpoints and greedy dev
metrics. The final dev-macro/S1-dev readout uses the frozen beam+A3 decoder and
`moth_eval`, not the trainer's greedy metric.

## Preflight (2026-11-17)

`runs/E-003/preflight.json` reports `ready: false`:

- ready: MLS train (40,953 clips, 169.9 h), LS train (28,370 clips, 100.0 h),
  MLS dev, LS dev, gold-train (1,413 clips, 0.96 h);
- missing: CV clean-50 and noisy-50 shards; the spliced manifest/shards depend
  on them and are therefore also missing;
- tokenizer sha256 matches `d6e1054e…`;
- train-row guard skipped until all manifests are present;
- E-002 checkpointed memory: 3.68 GB, 4.47 s/step at 6 × 30 s.

## Not done here

- No training run has started.
- S1-dev is still missing; E-003 can rank on dev-macro but the frozen protocol
  requires the S1-dev readout, so the final decision waits for it.
- CV 27.0 must be downloaded manually from Mozilla Data Collective; the
  `cv_it_train_clean50` and `cv_it_train_noisy50` audio is not on disk.
- The final decoder readout and package measurement belong to E-004/E-005.
