# E-022 — Keyterm biasing re-test (oracle upper bound)

> **Status: done 2026-11-23, report-only.** Step 1 passes: the per-clip oracle
> list lifts pooled acronym+brand ETA from **40.78 % to 46.60 % (+5.82 pp)** on
> R-S1-cal, more than the pre-registered 5-point stop rule. Step 2 (oracle + 500
> training-side distractors) peaks at **b = 5.0** with R-S1-cal WER 12.89 →
> **9.40 %**, ETA 75.63 → **85.83 %**, insertion 1.92 → 2.10, `control_it`
> unchanged. On S1-dev the oracle upper bound is **10.47 % WER / 84.96 % ETA**
> vs 13.38 / 75.15 (paired **−2.91 pp WER, +9.81 pp ETA**). **Not a product
> change:** the lists contain the answers; the realistic-glossary/UI decision
> remains open.

## Setup

- Release `0.9.7-e023-r1` (R1 s0 graph + E-034c, packed MOTHLMB2); decoder
  frozen 0.55 / 2.7, beam 12, top-k 10; `segment: false` to match the Python
  cal/S1-dev readouts.
- `tools/keyterm_probe.py` loads the release through Moth's `ReleaseMoth`,
  reads the clip's own `en_spans` (oracle), optionally adds 500 distractors
  from `../moth-stt/data/en_terms.txt` (per clip, excluding that clip's own
  terms), and decodes with `ContextualBias` at bonus / prefix 0.25.
- The oracle contains the answer words. It is a **ceiling**, never a gate and
  never a ranking input. The product runtime already accepts
  `contextualSequences`; this experiment only measures the decoder feature.

## Step 1 — oracle, no distractors

R-S1-cal (400 clips, 2,234 ref words), bonus 1.0 / prefix 0.25:

| Condition | WER | ETA | acronym ETA | brand ETA | acr+brand ETA | control_it WER |
|---|---:|---:|---:|---:|---:|---:|
| no context | 12.89 | 75.63 | 50.98 | 30.77 | **40.78** | 4.81 |
| oracle | **11.46** | **79.74** | 54.90 | 38.46 | **46.60** | 4.81 |

The pooled acronym+brand ETA gain is **+5.82 pp**, above the 5-point stop rule.
The overall WER also falls by 1.43 pp and ETA rises by 4.11 pp. No `control_it`
change.

## Step 2 — oracle + 500 distractors, bonus curve

500 training-side terms per clip, sampled without that clip's own terms; prefix
0.25. All runs report the same `control_it` WER (4.81) unless noted.

| Bonus | WER | ETA | acr+brand ETA | Ins/100w | control_it WER |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 12.04 | 78.15 | 44.66 | 1.92 | 4.81 |
| 1.0 | 11.41 | 79.87 | 46.60 | 1.88 | 4.81 |
| 1.5 | 11.15 | 80.79 | 47.57 | 1.92 | 4.81 |
| 2.0 | 10.83 | 81.59 | 48.54 | 1.92 | 4.81 |
| 2.5 | 10.34 | 83.18 | 49.51 | 2.01 | 4.81 |
| 3.0 | 10.16 | 83.58 | 51.46 | 2.01 | 4.81 |
| 4.0 | 9.71 | 84.90 | 53.40 | 2.06 | 4.81 |
| **5.0** | **9.40** | **85.83** | **54.37** | **2.10** | 4.81 |
| 7.0 | 9.80 | 86.23 | 55.34 | 2.64 | 4.81 |
| 10.0 | 11.15 | 87.55 | 57.28 | 4.39 | 5.15 |

The curve has a knee at **5.0**: it has the lowest WER, a 10.20 pp ETA gain
([7.86, 12.84]) and the lowest insertion rate before the higher-bonus trade.
7.0 buys +0.40 pp ETA for +0.40 pp WER and +0.54 insertion; 10.0 starts
inserting terms on `control_it` (WER 5.15, insertion 4.39). **Recommended
operating point: 5.0.**

## S1-dev oracle upper bound (bonus 5.0)

| Condition | WER | ETA | acronym ETA | brand ETA | switch ETA | italianized ETA | control_it WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| product control | **13.38** | **75.15** | 30.97 | 49.61 | 81.63 | 69.64 | 8.01 |
| oracle + 500, b=5 | **10.47** | **84.96** | 38.94 | 58.14 | 92.29 | 79.46 | 8.24 |

Paired: WER **−2.91 pp** [−3.41, −2.40], ETA **+9.81 pp** [+8.21, +11.50].
The `control_it` insertion rate is unchanged at 0.24 / 100 words; the WER rises
by 0.23 pp, within the category's small sample. The oracle is a ceiling, not a
deployable result.

## Verdict

- **Step 1 passed**; **step 2 gives a strong ceiling** and a clean bonus knee at
  5.0. The feature is worth a product decision, but no model, package or decoder
  default changed.
- The next step, if the product wants this, is a **realistic glossary** (user
  terms, not oracle) and a UI/API decision; the runtime already accepts
  `contextualSequences`. It must never enter a gate readout, because the list
  contains the answers.
- The 500 distractors are training-side terms; the false-insertion check on
  `control_it` is clean at b=5.0.

## Artifacts

- `cal-none/`, `cal-oracle/` — step 1 hypotheses and metrics.
- `cal-distractors-b{0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.0}/` — bonus
  curve.
- `s1-dev-oracle-distractors-b5/` — S1-dev oracle upper bound.
- `tools/keyterm_probe.py` — runner (report-only).

## Reproduce

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/keyterm_probe.py --release runs/E-023/release-0.9.7 \
  --manifest ../moth-cs-dataset/manifests/R-S1-cal.jsonl \
  --audio-root ../moth-cs-dataset --context oracle \
  --bonus 1.0 --prefix-bonus 0.25 --out-dir runs/E-022/cal-oracle
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/keyterm_probe.py --release runs/E-023/release-0.9.7 \
  --manifest ../moth-cs-dataset/manifests/R-S1-cal.jsonl \
  --audio-root ../moth-cs-dataset --context oracle-distractors \
  --distractors 500 --bonus 5.0 --prefix-bonus 0.25 \
  --out-dir runs/E-022/cal-distractors-b5.0
```
