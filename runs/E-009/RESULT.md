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

## Post-hoc analysis (2026-11-19)

Written after the result; it explains the veto, it does not change the decision.

**The frozen point already sits on the optimum ridge.** For each LM weight the
best token bonus is (dev-macro WER):

| LM weight | Best bonus | Dev-macro | Bonus / weight |
|---:|---:|---:|---:|
| 0.25 | 1.5 (grid edge) | 17.9070 % | 6.0 |
| 0.35 | 1.5 | 17.3805 % | 4.3 |
| 0.45 | 2.0 | 17.3427 % | 4.4 |
| 0.55 | 2.7 (frozen) | 17.6212 % | 4.9 |
| 0.65 | 2.7 | 17.9680 % | 4.2 |

The bonus has to grow with the LM weight because it pays back the LM's
per-token cost (Moth E-022). Along the ridge dev-macro moves by only 0.28 pp,
so the review's expectation that recalibration would be a large lever (it was
6.85 pp in Moth E-022, which started far off the ridge) did not hold for
MOTH-50.

**The lighter LM hurts every S1-dev category**, including pure Italian
(`runs/E-009/eta-breakdown-veto.json`):

| Category | WER frozen → 0.45 / 2.0 | ETA frozen → 0.45 / 2.0 |
|---|---|---|
| phrase | 14.78 → 15.51 % | 64.60 → 63.23 % |
| single_word | 7.09 → 7.40 % | 87.50 → 85.71 % |
| control_it | 8.24 → 8.83 % | — |
| acronym | 22.65 → 22.99 % | 29.20 → 26.55 % |
| brand | 19.62 → 20.14 % | 48.06 → 46.51 % |
| control_en | 16.82 → 17.29 % | 85.41 → 84.71 % |
| switch | 13.22 → 13.83 % | 81.41 → 80.50 % |
| italianized | 11.89 → 12.78 % | 72.32 → 70.54 % |

ETA misses: 33 new, 10 fixed. Because `control_it` degrades too, the effect is
not only A3's ×3 term weighting: S1-dev simply profits more from the LM than
MLS/LibriSpeech do. A plausible reading is domain: A3 is built from Moth's
training-side text (Moth D41), which includes the gold recording scripts from
the same script pipeline as S1-dev, while on read audiobooks the stronger
acoustic model needs less help.

**Consequence.** Dev-macro systematically asks for a lighter LM than the
product set wants, so it is the wrong set to calibrate any LM or decoder
change on. E-017 (`experiments/next-experiments.md`) proposes a mixed
calibration set separate from the S1-dev gate, needed before E-004, E-020 and
E-022. The style overlap between A3's text and S1-dev is also a reason to read
T-600, whose scripts and speakers differ, before trusting LM gains.

## Artifacts

- `runs/E-009/grid.json` — full 20-point grid, eligibility, paired CIs and
  selection.
- `runs/E-009/s1-dev-veto.json` — the one veto readout (`decoder_frozen=false`,
  `gate_eligible=false` by design).
- `runs/E-009/s1-dev-veto.hyp.jsonl` — its hypotheses.
- `runs/E-009/eta-breakdown-veto.json` — post-hoc term-miss comparison
  (`tools/eta_breakdown.py`).
