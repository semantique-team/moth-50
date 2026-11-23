# E-023 — R1: E-011 with encoder LR 3e-5

> **Status: adopted. Both seeds complete; R1 passed the amended gate (E-002
> §9, D26) and the T-600 release guard, and seed 0 is the product as
> `0.9.7-e023-r1` (D27, 2026-11-23).** Seed mean: dev-macro **14.3578 %**
> (−2.6320 pp), S1-dev **13.7911 % / 74.6826 %** (no regression), R-S1-cal
> **12.6679 % / 75.4305 %** (ETA tie, WER improves). Guard on T-600:
> **12.2993 % / 78.1400 %** vs 13.7041 / 76.1266 (−1.4048 pp WER
> [−2.3455, −0.4276], +2.0134 pp ETA [−0.1921, +4.1837]). Package
> **49,203,776 B**; browser exact at **RTFx 28.48**. Nothing is running.

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

## Training runs

| Seed | Run | Steps | Time | Best greedy dev | Final loss | Peak MPS |
|---|---|---:|---:|---:|---:|---:|
| 0 | `runs/E-023/r1-s0/` | 40,000 | 15.78 h | 17.0771 % | 0.4253 | 9.819 GB |
| 1 | `runs/E-023/r1-s1/` | 40,000 | 15.80 h | 17.3026 % | 0.0848 | 9.758 GB |

Both finished with 0 nonfinite steps. E-011 greedy dev was 18.3373 % (s0) and
18.6531 % (s1). The trainer's dev curve is in each run's `dev-*.json`; the
seed-0 curve improves monotonically from 30.8070 % at 4k to 17.0771 % at 40k,
seed 1 from 30.6660 % to 17.3026 %.

Command (same for both, `--seed` differs):

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/base6-e023-r1.toml \
  --moth-stt ../moth-stt --out runs/E-023/r1-s0 --device mps --seed 0
```

Preflight: `ready: true`, train guard ok, 273,714 train clips, 3,939 dev clips,
tokenizer sha256 `d6e1054e…`; 39,259,649 params.

## Seed-mean gate

All readouts are INT8, frozen decoder 0.55 / 2.7, E-034c LM, paired bootstrap
1,000 (seed 0). Differences are candidate − control.

| Gate | Seed mean | Verdict |
|---|---|---|
| 1. dev-macro ≥ 1.0 pp better | **14.3578 %** vs 16.9897, **−2.6320 pp [−2.8253, −2.4515]** | **pass** |
| 2. R-S1-cal pooled ETA not a regression | **75.4305 %** vs 75.6291, −0.1987 pp [−2.2603, +1.8844]; WER **12.6679 %** vs 13.9212, −1.2534 pp [−2.3767, −0.0446] | **pass** |
| 2b. `control_en` + `switch` within 0.5 pp of E-005 | control_en **86.50** vs 88.33 (**−1.83**, CI [−4.97, +1.16]); switch **71.18** vs 73.61 (**−2.43**, CI [−8.21, +3.36]) | **FAIL** |
| 3. S1-dev veto | **13.7911 % / 74.6826 %** vs 14.9744 / 73.9258; WER **−1.1833 pp [−1.7660, −0.5459]**, ETA **+0.7568 pp [−0.5000, +2.1149]** | **pass** |
| 4. INT8 cost ≤ 0.2 pp | dev-macro +0.0232 pp (seed mean); R-S1-cal WER +0.2910 pp, ETA 0.0 | pass on dev-macro; **cal WER cost is 0.291 pp** |

The readout's `gate_eligible` flag is false because the LM is E-034c, not A3;
the paired CIs above are the relevant numbers. Category CIs include zero, as
pre-registered; 2b is a point-estimate clause, and it misses.

## R-S1-cal detail

| Metric | E-005 + E-034c | R1 s0 | R1 s1 | R1 seed mean |
|---|---:|---:|---:|---:|
| WER | 13.9212 % | 12.8469 % | 12.4888 % | **12.6679 %** |
| ETA | 75.6291 % | 75.4967 % | 75.3642 % | **75.4305 %** |
| insertion rate | 2.1934 % | — | — | — |

| Category ETA | Control | R1 s0 | R1 s1 | Seed mean | Δ vs control |
|---|---:|---:|---:|---:|---:|
| single_word | 80.00 | 84.00 | 76.00 | 80.00 | 0.00 |
| phrase | 79.63 | 81.48 | 85.19 | 83.33 | +3.70 |
| brand | 30.77 | 32.69 | 30.77 | 31.73 | +0.96 |
| acronym | 49.02 | 49.02 | 50.98 | 50.00 | +0.98 |
| switch | 73.61 | 70.83 | 71.53 | 71.18 | **−2.43** |
| italianized | 66.00 | 70.00 | 72.00 | 71.00 | +5.00 |
| control_en | 88.33 | 87.00 | 86.00 | 86.50 | **−1.83** |

Against E-011 on the same set, R1 recovers the ETA: E-011 seed mean is
**12.6455 % WER / 74.7682 % ETA**, `control_en` 84.34, `switch` 68.41; R1 is
+0.66 pp ETA, +2.17 pp `control_en`, +2.78 pp `switch` (WER essentially tied,
+0.02 pp). H1 works relative to E-011, but does not match E-005 on the two
English-heavy categories.

## Verdict

**FAIL under the pre-registered gate, solely on 2b.** Gates 1, 2 (pooled), 3
and 4 (on dev-macro) pass, and the result is far stronger than E-011 on
dev-macro and S1-dev. But the 0.5 pp category clause against E-005 misses on
R-S1-cal, and the pre-registration says FAIL → R2 (gold ×20, H2) under a dated
amendment; P11 only after a passing recipe. No product change is adopted.

Two explicit owner options:

1. **Amend E-023** with a dated decision that accepts R1 despite 2b, reading
   the category CIs (both include zero) and the overall ETA tie plus WER/S1-dev
   gains as sufficient. Then choose the seed on R-S1-cal: seed 0 has the better
   ETA (75.4967 vs 75.3642), dev-macro (14.3274 vs 14.3881) and S1-dev
   (13.3820/75.1465 vs 14.2001/74.2188).
2. **Follow the pre-registration:** R1 fails; run R2 (E-011 with gold ×20)
   under a dated amendment and do not start P11.

## Amendment 2026-11-23 (E-002 §9, D26)

**The owner chose option 1.** The amendment is post-hoc and is labelled as
such; it changes clause 2b only and fixes the release guard before any further
readout. The seed is chosen on R-S1-cal alone; the dev-macro and S1-dev numbers
in option 1 are context, not the rule.

- **2b becomes report-only.** It contradicted E-002 §8.4 (category ETA on
  R-S1-cal is report-only) and was below the set's resolution. On R-S1-cal
  `control_en` has 300 English reference words and `switch` 144, so 0.5 pp is
  1.5 and 0.72 words; the seed-mean misses are 5.5 words (265 → 261 / 258) and
  3.5 words (106 → 102 / 103). Same-recipe seed spread on the set is larger
  than the clause (`single_word` R1 84.00 / 76.00, `switch` E-011 66.67 /
  70.14), and the control has one seed.
- **Residual risk:** both seeds lose a little on `control_en` and `switch` on
  R-S1-cal, and `control_en` WER is worse on both (13.67 → 14.67 / 16.00 %). Seen
  before the amendment, not a ranking input: on S1-dev the same categories
  improve on both seeds (`control_en` 84.35 → 84.82 / 85.41, `switch` 80.27 →
  81.63 / 82.31).
- **Gate 4 is read on dev-macro:** INT8 − FP32 +0.0641 / −0.0177 pp (s0 / s1).
  R-S1-cal is reported: WER +0.0895 / +0.4924 pp, ETA 0.00 / −0.13 pp.
- **R2 is withdrawn**; further recipe arms start from R1.
- **Seed 0** by the E-002 §8.3 rule (R-S1-cal ETA 75.4967 vs 75.3642 %, 570 vs
  569 of 755 terms).
- **Release guard (passed; result below):** one declared repeated read of
  T-600 with `release-s0` + E-034c against
  `runs/E-004/t600-e034c-int8.hyp.jsonl`; pass iff pooled WER and ETA show no
  regression with the CI excluding zero (E-002 §6.2); speakers and categories
  report-only. Pass → package `0.9.7-e023-r1`, browser check, FLEURS once
  (report-only). Fail → E-005 + E-034c stays.

```bash
shasum -a 256 runs/E-023/release-s0/model.int8.onnx   # must be 85abd830…
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed --name t600 \
  --onnx runs/E-023/release-s0/model.int8.onnx \
  --lm runs/E-004/lm_e034c_n2.bin --allow-other-lm \
  --manifest ../moth-cs-dataset/manifests/test.jsonl --shards data/shards/t600 \
  --config training/configs/base6-long.toml \
  --reference-hyp runs/E-004/t600-e034c-int8.hyp.jsonl \
  --out runs/E-023/t600-s0.json --hyp-out runs/E-023/t600-s0.hyp.jsonl
```

## Release guard and adoption (2026-11-23)

The guarded graph is `runs/E-023/release-s0/model.int8.onnx`, sha256
**`85abd830…`** (verified before the readout). One INT8 read of T-600, frozen
decoder, E-034c LM, paired against the E-005 + E-034c hypotheses:

| T-600 (600 clips, 3,488 ref words) | E-005 + E-034c | R1 s0 | Δ (95 % CI) |
|---|---:|---:|---|
| WER | 13.7041 % | **12.2993 %** | **−1.4048 pp** [−2.3455, −0.4276] |
| ETA | 76.1266 % | **78.1400 %** | **+2.0134 pp** [−0.1921, +4.1837] |
| insertion rate | — | 0.4014 % | — |

The WER CI excludes zero in the improvement direction and the ETA CI is not
entirely below zero, so the §9 rule **passes**. Report-only, per speaker: S1
**10.06 % / 81.19 %**, S2 **13.47 / 77.62**, S3 **13.37 / 75.77**. Categories:
`single_word` 13.26 / 73.33, `phrase` 18.86 / 60.28, `brand` 7.36 / 87.25,
`acronym` 10.98 / 85.00, `italianized` 3.75 / 96.25, `switch` 16.01 / 79.27,
`control_it` 7.84 / —, `control_en` 15.97 / 85.42. The read was declared
repeated and vetoed only; it did not rank or change the seed, decoder or LM.

Deltas against E-005 + E-034c (report-only, added 2026-11-23 from
`t600-s0.json` and `runs/E-004/t600-e034c-int8.json`): every speaker improves,
WER / ETA S1 −1.98 / +1.79, S2 −0.70 / +1.98, S3 −1.53 / +2.25 pp. Every
category improves except **`switch`** (WER +0.22, ETA **−2.03** pp, 60 clips)
and `control_en` (0.00 / 0.00, 20 clips). `switch` therefore goes the same way
as on R-S1-cal (−2.43) and the opposite way to S1-dev; it is the residual risk
named in E-002 §9 and is reported first by every later arm (D34).

**Package `0.9.7-e023-r1`:** **49,203,776 B**, **796,224 B** of room; graph
`85abd830…`, E-034c packed LM `7a4332f1…` (7,787,184 B), 16 kHz VAD
`7ed98ddb…`. Export report: `export-report-0.9.7.json` (FP32/INT8 top-1
agreement 1.0).

**Browser check:** load **70.1 ms** with every sha256 verified, transcript
exactly "ho spostato il meeting a domani", **RTFx 28.48** (5-run median after 2
warm-ups, 4 threads, isolated, SIMD; single run 16.72, total 140.1 ms).
Measured with the app frontmost: in a background window App Nap throttles the
same page to RTFx ~4.8 (feature 322 ms, inference 456 ms), an environment
effect to note in any re-run.

**FLEURS (report-only, once per language, vs E-034c):** IT **11.8304 %**
(−2.9935 pp [−3.4362, −2.5563], n = 697), EN **28.1955 %**
(−2.0720 pp [−2.7117, −1.4574], n = 525); no ETA. FLEURS never ranks, and
seed 1 was not read.

**Verdict: R1 seed 0 is adopted as the product (D27).** E-005 + E-034c is
retired from the product line and remains the historical reference. R2 stays
withdrawn; no further T-600 read with this seed.

## Artifacts

- `runs/E-023/r1-s{0,1}/` — config, run metadata, train logs, dev curves,
  preflight (seed 0), checkpoints (gitignored).
- `runs/E-023/release-s{0,1}/` — INT8 packages with E-034c (gitignored);
  `export-report-s{0,1}.json` records budget and logit gaps.
- `dev-macro-s{0,1}.json/.hyp.jsonl`, `dev-macro-s{0,1}-fp32.*` — dev-macro.
- `s1-cal-s{0,1}.json/.hyp.jsonl`, `s1-cal-s{0,1}-fp32.*` — R-S1-cal.
- `s1-dev-s{0,1}.json/.hyp.jsonl` — S1-dev.
- `s1-cal-e011-s{0,1}.json/.hyp.jsonl` — E-011 on R-S1-cal, context only.
- `p6/dev-macro-seed-mean.json`, `p6/s1-cal-seed-mean.json` (with categories),
  `p6/s1-dev-seed-mean.json`.
- `t600-s0.json/.hyp.jsonl` — release guard (declared repeated read).
- `release-0.9.7/` (gitignored) + `export-report-0.9.7.json` — product package.
- `browser-run-0.9.7.json` — browser check of the product package.
- `fleurs-{it,en}-s0.json/.hyp.jsonl` — report-only FLEURS vs E-034c.
