# E-011 — long schedule with EMA: dev-macro win, S1-dev ETA veto

> Executed 2026-11-19 23:00 → 2026-11-21 07:25 CET on the M5 Pro. Two seeds,
> 40,000 steps each, base+6 with `lr_encoder = 1e-4`, EMA 0.9998. The frozen
> E-002 decoder and A3 LM were applied to E-011 and E-005 identically (FP32 and
> INT8). **E-011 is not adopted: seed 0 fails the S1-dev ETA no-regression rule.**

## Training

| Seed | Steps | Greedy dev (EMA) | Time | Peak | Non-finite |
|---|---:|---:|---:|---:|---:|
| 0 | 40,000 | **18.3373 %** | 16 h 20 min | 9.834 GB | 0 |
| 1 | 40,000 | **18.6531 %** | 16 h 06 min | 9.787 GB | 0 |

Greedy dev curve (%, EMA weights at eval):

| Step | 4k | 8k | 12k | 16k | 20k | 24k | 28k | 32k | 36k | 40k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | 31.29 | 26.11 | 23.80 | 21.97 | 20.84 | 19.99 | 19.16 | 18.73 | 18.54 | **18.34** |
| seed 1 | 31.16 | 26.47 | 23.92 | 22.31 | 21.15 | 20.27 | 19.34 | 19.07 | 18.80 | **18.65** |

The 0.32 pp seed gap is inside the measured 1 pp noise band. EMA and the long
schedule are stable: 0 non-finite steps in both runs.

## Gate — dev-macro PASS

Frozen beam+A3, paired against E-005 decoded by the same tool (FP32 vs FP32,
INT8 vs INT8). Negative is better.

| Seed | Precision | E-011 macro | E-005 macro | Δ | 95 % CI | Pass |
|---|---|---:|---:|---:|---|---|
| 0 | FP32 | **15.8584 %** | 17.5565 % | **−1.6981 pp** | [−1.9221, −1.4858] | ✅ |
| 1 | FP32 | **15.7945 %** | 17.5565 % | **−1.7620 pp** | [−2.0001, −1.5364] | ✅ |
| 0 | INT8 | **15.9099 %** | 17.6212 % | **−1.7114 pp** | [−1.9205, −1.5044] | ✅ |
| 1 | INT8 | **15.9026 %** | 17.6212 % | **−1.7186 pp** | [−1.9530, −1.4888] | ✅ |

Both seeds clear the 1.0 pp bar with CIs excluding zero, in both precisions.

## Gate — S1-dev: seed 0 fails the ETA rule

Same S1-dev manifest, decoder and LM; paired against E-005 at the same precision.

| Seed | Precision | WER | E-005 WER | Δ WER | ETA | E-005 ETA | Δ ETA (95 % CI) | Pass |
|---|---|---:|---:|---:|---:|---:|---|---|
| 0 | FP32 | 13.9080 % | 13.9956 % | −0.0877 pp | 73.1445 % | 75.3418 % | **−2.1973 pp [−3.8280, −0.4390]** | ❌ |
| 1 | FP32 | 13.9810 % | 13.9956 % | −0.0146 pp | 73.9746 % | 75.3418 % | −1.3672 pp [−2.9079, +0.3265] | ✅ |
| 0 | INT8 | 13.9226 % | 14.0541 % | −0.1315 pp | 73.0469 % | 75.5371 % | **−2.4902 pp [−4.1217, −0.6760]** | ❌ |
| 1 | INT8 | 13.9226 % | 14.0541 % | −0.1315 pp | 74.1211 % | 75.5371 % | −1.4160 pp [−2.9381, +0.3264] | ✅ |

- WER is a **tie** on S1-dev: all paired WER CIs include zero.
- Seed 0's ETA drop has a CI entirely below zero in both precisions: a
  pre-registered **regression**.
- Seed 1's ETA CI includes zero: no significant regression.
- The FP32 seed gap is 0.83 pp ETA (inside the 1 pp noise band), the INT8 gap is
  1.07 pp. Either way, selecting the passing seed on S1-dev would turn the veto
  into tuning, which the protocol forbids.

## INT8 cost

| Seed | Macro FP32 | Macro INT8 | Cost | S1-dev ETA FP32 | S1-dev ETA INT8 | Cost |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 15.8584 % | 15.9099 % | **+0.0515 pp** | 73.1445 % | 73.0469 % | −0.0976 pp |
| 1 | 15.7945 % | 15.9026 % | **+0.1081 pp** | 73.9746 % | 74.1211 % | +0.1465 pp |

Both are well inside the ≤ 0.2 pp INT8 budget. Packages are **44,770,432 B**
(5,229,568 B of room) with the 16 kHz VAD and packed A3 LM.

## Verdict

**E-011 is a dev-macro win that does not become a clean product win.**

- The long schedule buys **~1.7 pp** on dev-macro (MLS-IT + LS-EN) on both seeds,
  significantly and reproducibly.
- On S1-dev, WER does not improve significantly, and the ETA (short English
  terms/natural code-switch words such as *it, a, too, on, is, coffee, makes*)
  drops significantly on seed 0.
- One of two seeds fails the frozen **no-regression** rule. A one-seed pass is
  not evidence that the recipe is safe, and the protocol does not allow choosing
  the seed after seeing S1-dev.
- **E-005 remains the shipped/product model.** E-011 stays as an experiment and
  as evidence that dev-macro and the product session can move apart (the E-009
  lesson again, now with a training change).

Possible follow-ups (not run, not pre-registered yet): a third seed to see
whether the ETA drop is systematic; a code-switch/term-focused data arm (E-006)
to recover short English words; an explicit term-aware decoder study. Do not
pick a seed or tune the decoder on S1-dev.

## Artifacts

- Training: `runs/E-011/base+6-e011-s0/`, `...-s1/` (metrics, dev curves, logs;
  checkpoints are gitignored).
- FP32: `readout-s0.json`, `readout-s1.json`, `s1-dev-s0.json`, `s1-dev-s1.json`
  and their `.hyp.jsonl`.
- INT8: `readout-int8-s0.json`, `readout-int8-s1.json`,
  `s1-dev-int8-s0.json`, `s1-dev-int8-s1.json` and their `.hyp.jsonl`.
- Packages: `release-s0/`, `release-s1/` (gitignored) and
  `export-report-s0.json`, `export-report-s1.json`.

## Reproduce

```bash
for seed in 0 1; do
  uv run --frozen --project ../moth-stt --extra training -- \
    python -m training.train --config training/configs/base6-e011.toml \
    --moth-stt ../moth-stt --seed "$seed" \
    --out "runs/E-011/base+6-e011-s$seed" --device mps
done
# FP32 gate readouts
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode macro \
  --checkpoint runs/E-011/base+6-e011-s0/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml \
  --reference-hyp runs/E-005/readout-a3.hyp.jsonl \
  --out runs/E-011/readout-s0.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed \
  --checkpoint runs/E-011/base+6-e011-s0/checkpoints/ckpt_best.pt \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --reference-hyp runs/E-005/s1-dev-a3-fp32.hyp.jsonl \
  --out runs/E-011/s1-dev-s0.json
```

## P6 — re-read with the product LM E-034c (2026-11-21)

`runs/E-011/p6/RESULT.md`. Under D25 and the E-002 §7 pre-registration:
dev-macro seed mean **15.3695 %** vs 16.9897 (−1.6202 pp [−1.8215, −1.4417],
gate PASS); clean S1-dev seed mean **14.4704 % / 72.1680 %** vs
14.9744 / 73.9258, WER −0.5040 (CI includes 0) and **ETA −1.7578 pp
[−3.1844, −0.1422] (gate FAIL)**. **Not adopted; E-005 + E-034c stays.** The ETA
loss is recipe-level (`control_en`, `switch`, `single_word`), not an A3 artifact
and not one bad seed. Next: P7 diagnostics, then P10 R1 (encoder LR 3e-5) under
E-023.
