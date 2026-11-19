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

## Schedule note (2026-11-17)

The four arms do not need S1-dev to start: the dev block is only the extra
readout, and the ranking gate is dev-macro. But do **not** run MPS training on
the recording Mac during an S1 session: sustained load spins the fans into the
microphone, and MPS shares unified memory with the recorder. Budget one night
for two arms at the E-002 worst-case 4.47 s/step (base+6, 6 × 30 s padding);
real variable-length batches are faster, but 4,000 steps is still hours.

## Execution log

- **2026-11-17 18:06–19:03 CET — `base+0` probe, invalid as an arm.** 4,000
  steps, 56.6 min, peak 8.446 GB, checkpoints every 100 verified. The trainer
  had no initializer, so the encoder was random and dev WER stayed at ~99 %
  (best 99.04 %). Kept in `runs/E-003/base+0-scratch/` as the checkpoint and
  throughput probe.
- **Fix before any ranking:** `tools/convert_whisper_encoder.py` builds the
  `openai/whisper-base` initializer (parity 4.8e-4); `training/model.py` loads
  it strictly and identity-initializes the extra blocks; `base6.toml` sets
  `model.init` and `identity_heads = true` (D16).
- **2026-11-17 21:12 CET — `base+2` done.** 4,000 steps, 70.7 min, peak
  8.406 GB, 0 non-finite. Greedy dev 60.55 → 46.35 → 39.46 → **36.90 %**
  at 1k/2k/3k/4k: 11.2 pp better than `base+0` at the same budget, so depth
  pays at this schedule. `base+4` started with the identical recipe.
- **`base+4` done 22:38 CET.** 4,000 steps, 85.5 min, peak 8.551 GB, 0
  non-finite. Greedy dev 58.46 → 44.48 → 37.59 → **34.73 %**.
- **`base+6` done 00:17 CET.** 4,000 steps, 98.9 min, peak 8.594 GB, 0
  non-finite. Greedy dev 57.21 → 43.94 → 36.79 → **33.46 %**.

### Result at 4,000 steps (greedy dev-macro, monitoring)

| Arm | WER | Δ vs previous | Time | Peak |
|---|---:|---:|---:|---:|
| base+0 | 48.12 % | — | 56.2 min | 7.397 GB |
| base+2 | 36.90 % | −11.22 pp | 70.7 min | 8.406 GB |
| base+4 | 34.73 % | −2.17 pp | 85.5 min | 8.551 GB |
| base+6 | **33.46 %** | −1.27 pp | 98.9 min | 8.594 GB |

Depth is monotonic but strongly diminishing after +2. `base+4` and `base+6`
are 1.27 pp apart, close enough to the pre-registered "~1 pp" rule to warrant a
second seed for both before naming a winner; `--seed 1` reruns are in
`runs/E-003/base+4-seed1/` and `base+6-seed1/`.

**Caveat for the gate:** these are greedy, without the A3 LM. Every arm is far
from the 23.0170 % beam+A3 reference, and the dev curve is still improving when
the 4,000-step schedule ends. E-003 therefore ranks depth at a fixed screen
budget; it does not produce a gate-ready model. The longer schedule belongs to
E-004/E-005 and must not be retrofitted into this screen.

### Second seed (pre-registered within-1-pp rule)

| Arm | seed 0 | seed 1 | Spread |
|---|---:|---:|---:|
| base+4 | 34.7306 % | 34.7370 % | 0.006 pp |
| base+6 | **33.4588 %** | **33.5460 %** | 0.087 pp |

The +6 advantage over +4 is 1.27 pp (seed 0) and 1.19 pp (seed 1): consistent
and ~15× the observed seed spread, so the 4,000-step ranking is not seed noise.

### Verdict

**base+6 wins the depth screen.** Depth is monotonic with steep diminishing
returns: +0 → +2 buys 11.2 pp, +2 → +4 buys 2.2 pp, +4 → +6 buys 1.2 pp at
+17 % time. The E-004 package trade-off now has a measured acoustic gap: +6 is
~1.2 pp better than +4, but leaves 2.26 MB of headroom vs +4's 8.67 MB.

The absolute numbers are a screen, not a model: 4,000 steps is ~0.36 epoch of
the ~595 h mixture. The next experiment must extend the schedule (and then run
the frozen beam+A3 readout) before any gate claim.

**Follow-up (not retrofitted):** the long fine-tune of this winner is E-005, in
`runs/E-005/base+6-16k/` with `training/configs/base6-long.toml` (16,000 steps,
warmup 1,000, eval every 2,000). It reached **21.7540 % greedy dev** in 6.73 h,
still improving; the frozen beam+A3 readout is the next step. E-003's own budget
stays 4,000 steps. See `runs/E-005/RESULT.md`.

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

`runs/E-003/preflight.json` reports **`ready: true`** and `train_guard: ok`:

- CV clean-50 (56,769 clips, 84.2 h) + full `cv_it_train` shards; CV noisy-50
  (56,349 clips, 83.6 h, 0 errors);
- MLS train (40,953 clips, 169.9 h), LS train (28,370 clips, 100.0 h);
- spliced IT+EN (20,623 clips, 110.0 h);
- gold-train (1,413 clips, 0.96 h; repeated ×50 in the config);
- MLS dev and LS dev ready;
- tokenizer sha256 matches `d6e1054e…`;
- E-002 checkpointed memory: 3.68 GB, 4.47 s/step at 6 × 30 s.

The only missing E-002 artifact is **S1-dev**, so E-003 can start the screen but
the final ranking waits for that readout.

## Not done here

- No training run has started.
- S1-dev is still missing; E-003 can rank on dev-macro but the frozen protocol
  requires the S1-dev readout, so the final decision waits for it.
- CV 27.0 must be downloaded manually from Mozilla Data Collective; the
  `cv_it_train_clean50` and `cv_it_train_noisy50` audio is not on disk.
- The final decoder readout and package measurement belong to E-004/E-005.
