# E-023 — R1: E-011 with encoder LR 3e-5

> **Status: seed 0 complete (40,000 steps, 15.78 h). Seed 1 not launched.**
> Seed 0 passes the dev-macro, S1-dev and INT8-cost gates; the R-S1-cal
> category-recovery condition (control_en + switch ETA within 0.5 pp of E-005)
> is the only miss. The seed-1 decision is pending a protocol reading.
> Nothing is running.

## Question

P6 re-read E-011 with E-034c and it failed the S1-dev ETA veto while winning
dev-macro. P7 found no evidence that gold over-exposure is the primary cause.
R1 changes **one variable** from E-011: `lr_encoder` 1e-4 → **3e-5**
(`lr_head` stays 3e-4). Everything else is E-011: base+6, 40,000 steps
(~2.97 epochs), EMA 0.9998, gold ×50, same mixture, same seeds.

## Pre-registration and control

- Pre-registration: `experiments/next-experiments.md` §E-023.
- Config: `training/configs/base6-e023-r1.toml`.
- E-017 calibration set is frozen (`manifests/R-S1-cal.jsonl`,
  `data/shards/s1-cal`, E-002 §8).
- Control E-005 + E-034c on R-S1-cal: **13.9212 % WER / 75.6291 % ETA**
  (`runs/E-017/control-e005-e034c.json`; hypotheses in
  `runs/E-017/control-e005-e034c.hyp.jsonl`).
- Control on dev-macro: E-005 + E-034c is **16.9897 %**.

## Seed 0 run

- Started **2026-11-21 16:10:08 +0100**, finished **2026-11-22 07:56**;
  **15.78 h** (56,807 s), 40,000 steps, 0 nonfinite steps.
- Output: `runs/E-023/r1-s0/` (`config.toml`, `run.json`, `train.log.jsonl`,
  `console.log`, checkpoints ignored by git, `preflight.json`).
- Command:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/base6-e023-r1.toml \
  --moth-stt ../moth-stt --out runs/E-023/r1-s0 --device mps --seed 0
```

- Preflight: `ready: true`, train guard ok, 273,714 train clips, 3,939 dev
  clips, tokenizer sha256 `d6e1054e…`; 39,259,649 params.
- Trainer greedy dev (MLS-IT + LS-EN, EMA): **17.0771 %** at 40,000 steps,
  still improving from 17.2898 % at 36,000. E-011 s0 was 18.3373 %, s1
  18.6531 %.
- Peak MPS memory 9.819 GB; final loss 0.4253.
- Resource samples: `runs/E-023/resource-samples.jsonl` (60 s interval).
- `ckpt_best.pt` and `ckpt_last.pt` are the same 40,000-step checkpoint
  (sha256 `5ff95f0e…`); EMA weights, matching the dev evaluation.

## Gate components (seed 0 only)

All readouts are INT8, frozen decoder 0.55 / 2.7, E-034c LM, unless marked
FP32. Differences are candidate − control.

| Gate | Result | Verdict |
|---|---|---|
| 1. dev-macro ≥ 1.0 pp better | **14.3274 %** vs 16.9897, **−2.6623 pp [−2.8611, −2.4666]** | **pass** |
| 2. R-S1-cal pooled ETA not a regression | **75.4967 %** vs 75.6291, −0.1325 pp [−2.6490, +2.3677] | pass on the pooled rule |
| 2b. `control_en` + `switch` within 0.5 pp of E-005 | control_en **87.00** vs 88.33 (**−1.33**); switch **70.83** vs 73.61 (**−2.78**) | **miss** |
| 3. S1-dev veto | **13.3820 % / 75.1465 %** vs 14.9744 / 73.9258; WER −1.5924 pp, ETA +1.2207 pp | **pass** |
| 4. INT8 cost ≤ 0.2 pp | dev-macro +0.0641 pp; cal WER +0.0895 pp, ETA 0.0 | **pass** |

The readout's `gate_eligible` flag is false because the LM is E-034c, not A3;
the paired CIs above are the relevant numbers.

## R-S1-cal detail (seed 0)

| Metric | E-005 + E-034c | R1 s0 |
|---|---:|---:|
| WER | 13.9212 % | **12.8469 %** |
| ETA | 75.6291 % | **75.4967 %** |
| insertion rate | 2.1934 % | — |

| Category ETA | Control | R1 s0 | Δ |
|---|---:|---:|---:|
| single_word | 80.00 | 84.00 | +4.00 |
| phrase | 79.63 | 81.48 | +1.85 |
| brand | 30.77 | 32.69 | +1.92 |
| acronym | 49.02 | 49.02 | 0.00 |
| switch | 73.61 | 70.83 | −2.78 |
| italianized | 66.00 | 70.00 | +4.00 |
| control_en | 88.33 | 87.00 | −1.33 |

Against E-011 s0 on the same set, R1 recovers the ETA: 74.3046 → 75.4967
overall, `control_en` 84.00 → 87.00, `switch` 66.67 → 70.83. H1 works relative
to E-011, but not enough to match E-005 on the two English-heavy categories.

## Seed-1 decision

The pre-registration says seed 1 runs only if seed 0 is within 0.5 pp of E-011
on dev-macro and the E-017 ETA recovery in (b) holds. Seed 0 is **1.04 pp
better** than E-011 on dev-macro (not worse), and it recovers ETA relative to
E-011, but the formal 0.5 pp condition against **E-005** on `control_en` /
`switch` misses. Seed 1 has not been launched; decide explicitly whether to
read the 0.5 pp clause against E-011 or E-005 before spending another ~16 h.

## Gate (seed mean, paired bootstrap 1,000, seed 0)

1. dev-macro **≥ 1.0 pp** better than E-005 + E-034c (16.9897 %) with the CI
   excluding zero.
2. R-S1-cal pooled ETA is not a regression against the control (75.6291 %; CI
   not entirely below zero), and `control_en` + `switch` ETA point estimates
   are within 0.5 pp of E-005.
3. S1-dev veto on the seed mean: WER CI not entirely above zero and ETA CI not
   entirely below zero (E-002 §6.2).
4. INT8 cost ≤ 0.2 pp against the FP32 readout.

The readout's `gate_eligible` flag stays false with the product LM; use the
`paired` CIs, with `--reference-hyp runs/E-017/control-e005-e034c.hyp.jsonl`.

## Next

1. **Seed 0 is done** (40,000 steps, 15.78 h); INT8 export and the FP32/INT8
   readouts are in `runs/E-023/`.
2. **Seed 1 decision pending.** The seed-0 criterion is ambiguous: R1 is
   +1.04 pp better than E-011 on dev-macro and recovers ETA relative to E-011,
   but the 0.5 pp category clause against E-005 misses on `control_en` and
   `switch`. Do not launch seed 1 until that clause is read explicitly.
3. If seed 1 runs, use the same config/seed override, read it the same way, and
   apply the seed-mean gate (E-002 §7). If it does not run, record R1 as a
   one-seed result and decide on R2 under a dated amendment.
4. PASS → R1 is the final-recipe candidate, seed chosen on R-S1-cal, P11 may
   run. FAIL → R2 (gold ×20) under a dated amendment; no P11 without a passing
   recipe.

## Artifacts (seed 0)

- `runs/E-023/r1-s0/` — config, run metadata, train log, dev curves, preflight.
- `runs/E-023/release-s0/` — INT8 graph + E-034c package (gitignored);
  `export-report-s0.json` records the budget and logit gaps.
- `dev-macro-s0.json/.hyp.jsonl` — INT8 dev-macro; `dev-macro-s0-fp32.json`
  and `.hyp.jsonl` — FP32.
- `s1-cal-s0.json/.hyp.jsonl`, `s1-cal-s0-fp32.json/.hyp.jsonl` — R-S1-cal.
- `s1-dev-s0.json/.hyp.jsonl` — S1-dev.
- `s1-cal-e011-s0.json/.hyp.jsonl` — E-011 s0 on R-S1-cal, for the H1 recovery
  comparison only.
