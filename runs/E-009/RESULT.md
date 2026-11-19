# E-009 — decoder recalibration: dev-macro gain, S1-dev veto

> Executed 2026-11-19 16:47–17:08 CET on the M5 Pro. Amended rule (E-002 §6.3):
> rank the grid on dev-macro only, then use S1-dev once as a veto against the
> frozen MOTH-50 decoder. No S1-dev grid, no S1-dev ranking.

## Result

**The dev-macro winner does not transfer to the product dev set. The frozen
decoder 0.55 / 2.7 stays.**

| Step | Value |
|---|---|
| Grid | 20 points × 3,939 dev clips, 1,135 s (18.9 min), 8 workers |
| Best dev-macro | **lm_weight 0.45, token_bonus 2.0**: 17.3427 % |
| Frozen | 17.6212 % |
| Dev-macro paired | **−0.2786 pp**, 95 % CI [−0.3821, −0.1718] |
| S1-dev candidate | **14.6384 % WER / 74.4141 % ETA** |
| S1-dev frozen MOTH-50 | 14.0541 % WER / 75.5371 % ETA |
| S1-dev paired WER | **+0.5844 pp**, CI [0.2422, 0.9515] — regression |
| S1-dev paired ETA | **−1.1230 pp**, CI [−1.8408, −0.4764] — regression |
| Veto | **FAIL** (`pass: false`) |

The grid's `adopt: true` only means the dev-macro improvement is significant.
The S1-dev veto overrides it: a 0.58 pp WER increase and a 1.12 pp ETA drop, both
with CIs excluding zero, are exactly the product regression the rule is meant to
catch.

Top dev-macro points:

| LM weight | Token bonus | Dev-macro WER | Δ vs frozen |
|---:|---:|---:|---:|
| **0.45** | **2.0** | **17.3427 %** | **−0.2786 pp** |
| 0.35 | 1.5 | 17.3805 % | −0.2407 pp |
| 0.35 | 2.0 | 17.4017 % | −0.2195 pp |
| 0.45 | 1.5 | 17.4850 % | −0.1362 pp |
| 0.55 | 2.7 (frozen) | 17.6212 % | — |

No other grid point was read on S1-dev. Trying runner-ups until one passes would
turn the veto into S1-dev tuning, which the amendment forbids.

## Decision

- **Keep the frozen 0.55 / 2.7 decoder** for the release and all gate readouts.
- No package, config, hypothesis or WER/ETA artifact changes.
- This is the E-028 lesson again: dev-macro and the product dev set can move in
  opposite directions. The pre-registered veto is doing its job.

## Artifacts

- `runs/E-009/grid.json` — full 20-point grid, eligibility, paired CIs and
  selection.
- `runs/E-009/s1-dev-veto.json` — the one veto readout (`decoder_frozen=false`,
  `gate_eligible=false` by design).
- `runs/E-009/s1-dev-veto.hyp.jsonl` — its hypotheses.
