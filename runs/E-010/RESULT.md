# E-010 — encoder-LR screen

> Executed 2026-11-19 19:20–22:11 CET on the M5 Pro. Two arms, 4,000 steps each,
> seed 0; only `optimizer.lr_encoder` differs from the E-003 recipe (1e-4 vs
> 3e-5). The trainer's greedy dev-macro is monitoring, as pre-registered; no
> frozen beam readout and no S1-dev selection were used.

## Result

**1e-4 wins both depths. base+6 at 1e-4 is the best arm.**

| Arm | Encoder LR | Greedy dev @4k | Δ vs E-003 | Time | Peak |
|---|---:|---:|---:|---:|---:|
| base+4 | 3e-5 (E-003) | 34.7306 % | — | 85.5 min | 8.551 GB |
| base+4 | **1e-4** | **33.4003 %** | **−1.3303 pp** | 87.7 min | 8.473 GB |
| base+6 | 3e-5 (E-003) | 33.4588 % | — | 98.9 min | 8.594 GB |
| base+6 | **1e-4** | **32.3912 %** | **−1.0677 pp** | 101.8 min | 9.648 GB |

Dev curve (greedy WER, %):

| Arm | 1k | 2k | 3k | 4k |
|---|---:|---:|---:|---:|
| base+4, 3e-5 | 58.46 | 44.48 | 37.59 | 34.73 |
| base+4, 1e-4 | 54.77 | 44.15 | 37.13 | **33.40** |
| base+6, 3e-5 | 57.21 | 43.94 | 36.79 | 33.46 |
| base+6, 1e-4 | 54.70 | 44.14 | 36.78 | **32.39** |

0 non-finite steps in both runs. The 1e-4 arms are ahead at every 1k checkpoint,
so the gain is not a last-step artefact.

## Gate (pre-registered)

- 1e-4 wins an arm if greedy dev-macro improves by ≥ 0.5 pp (≈ 5× the measured
  seed spread) with no non-finite step. **Both arms pass** (−1.33 / −1.07 pp).
- If base+4 at 1e-4 is within 0.2 pp of base+6 at 3e-5 (≤ 33.66 %), base+4
  becomes an E-004 candidate. **Pass**: base+4 at 1e-4 is 33.4003 %, actually
  0.0585 pp *better* than base+6 at 3e-5 (33.4588 %).
- If 1e-4 had been worse by ≥ 0.5 pp, a 5e-5 arm would be added; it is not.

## Decision

- **E-011 uses base+6 with `lr_encoder = 1e-4` and the rest of the E-003 recipe.**
  base+6 at 1e-4 is 1.01 pp better than base+4 at 1e-4, at +6.4 MB and +14 min
  per 4,000 steps.
- **base+4 at 1e-4 is an E-004 depth arm** (6.4 MB freed versus base+6). It is
  competitive at the 4,000-step screen; only a long E-011-style run can decide
  whether the depth still pays at convergence.
- No decoder, readout or package changes. The frozen beam+A3 protocol remains
  the gate for E-011.

## Artifacts

- `runs/E-010/preflight.json` — `ready: true`, train guard ok.
- `runs/E-010/base+4-enc1e-4/` — `metrics.json`, `dev-*.json`, `train.log.jsonl`.
- `runs/E-010/base+6-enc1e-4/` — same.
- Checkpoints are gitignored (`checkpoints/ckpt_best.pt`, `ckpt_last.pt`).

Note: each run's `config.toml` is the shared `base6-enc1e-4.toml` and still says
`head_blocks = 6`; the effective depth is in `run.json` (`head_blocks: 4/6`) and
in the checkpoint's `model_config`.

## Reproduce

```bash
for blocks in 4 6; do
  uv run --frozen --project ../moth-stt --extra training -- \
    python -m training.train \
    --config training/configs/base6-enc1e-4.toml --moth-stt ../moth-stt \
    --head-blocks $blocks --max-steps 4000 \
    --out runs/E-010/base+$blocks-enc1e-4 --device mps
done
```
