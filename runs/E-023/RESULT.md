# E-023 — R1: E-011 with encoder LR 3e-5

> **Status: authorized 2026-11-21; seed 0 running.** No result yet. Seed 1 is
> conditional on the seed-0 criterion below.

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

- Started **2026-11-21 16:10:08 +0100** on the M5, MPS, PID **48613**
  (`uv` parent 48611).
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
- Early health (step 150): loss 6.4779, 0.77 s/step, MPS 6.5 GB, nonfinite 0.
  At this rate 40,000 steps is roughly 9–10 h plus dev evaluations; revise
  when the run settles.
- Resource samples: `runs/E-023/resource-samples.jsonl` (60 s interval).
- `caffeinate -dimsu -w 48613` (PID 48909) keeps the Mac awake while the
  trainer lives.
- Dev evaluations every 4,000 steps; `ckpt_best.pt` follows the greedy dev
  WER with EMA weights.

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

1. Let seed 0 reach 40,000 steps; check dev evaluations and resource samples.
2. If the seed-0 criterion holds (dev-macro within 0.5 pp of E-011 and the
   calibration ETA recovery), run seed 1; otherwise stop and write the result.
3. Export INT8 and read FP32/INT8 on R-S1-cal and S1-dev; apply the gate.
4. PASS → R1 is the final-recipe candidate, seed chosen on R-S1-cal, P11 may
   run. FAIL → R2 (gold ×20) under a dated amendment; no P11 without a passing
   recipe.
