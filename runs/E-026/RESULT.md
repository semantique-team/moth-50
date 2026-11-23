# E-026 (E-022 step 3) — Realistic glossary: cal benefit, S1-dev veto fails, not exposed

> **Status: done 2026-11-23, glossary not exposed (D35).** 12 R-S1-cal decodes
> (A500, B50, B25 × bonus 1/2/3/5) plus the one S1-dev veto at *b\** = 5.0. On
> R-S1-cal the distractor-only list is harmless at every bonus (WER
> **+0.00…+0.09 pp**, CIs never entirely above zero) and partial recall *p* =
> 0.5 at *b\** = 5.0 gives **11.1012 % / 80.9272 %** vs **12.8917 / 75.6291**
> (WER **−1.7905 pp** [−2.5333, −1.1287], ETA **+5.2980 pp** [+3.5573,
> +7.2603]). The S1-dev veto fails: A500 at 5.0 is **13.3382 % / 75.0977 %**
> vs **13.2067 / 75.3418**, WER **+0.1315 pp** [+0.0283, +0.2597] — inside the
> +0.25 pp margin but the paired CI is entirely above zero. **Step 4 fails, so
> the glossary stays unexposed.** No model, LM, package or decoder-default
> change and no browser smoke.

## System and grid

- Release `runs/E-023/release-0.9.7` (R1 seed 0 + E-034c), INT8, frozen decoder
  0.55 / 2.7, beam 12, top-k 10, `segment: false`.
- `ContextualBias`, prefix bonus **0.25**; bonus grid **{1, 2, 3, 5}**.
- Distractor pool: `../moth-stt/data/en_terms.txt` normalized to **541**
  single-word terms. The amendment of 2026-11-23 dropped A2000 and C2000
  because the pool is below the 2,008 (and 1,008) the pre-registration needed;
  the harm gate therefore uses **A500 only**. Grid: 3 conditions × 4 bonuses =
  **12 decodes**, not 20.
- Control: the E-022 no-context hypotheses, re-scored with the vendored
  evaluator: **12.8917 % WER / 75.6291 % ETA**, 1.9248 ins/100 w.
- Every decode used 400/400 clips with no errors; 500 distractors per clip
  (200,000 total). B50 kept 379 of the 742 own terms, B25 kept 206.

## Step 1 — harm (A500)

Paired vs the control on the same audio; the bias is deterministic, so the
point estimate is an exact count and the CI only states what R-S1-cal can
resolve.

| Bonus | WER | WER Δ (pp) | 95 % CI (pp) | Ins/100w | control_it WER |
|---:|---:|---:|---|---:|---:|
| 1.0 | 12.9418 | +0.0448 | [0.0000, +0.1352] | 1.92 | 4.81 |
| 2.0 | 12.9418 | +0.0448 | [0.0000, +0.1352] | 1.92 | 4.81 |
| 3.0 | 12.8917 | 0.0000 | [−0.1343, +0.1340] | 1.92 | 4.81 |
| 5.0 | 12.9812 | +0.0895 | [−0.0900, +0.3562] | 1.97 | 4.81 |

All four bonuses are **safe**: the CI is never entirely above zero (at 1.0 and
2.0 the lower bound is exactly 0, i.e. a tie boundary) and the point estimate is
at most +0.25 pp. The 5.0 decode adds two word errors to the control. `control_it`
WER is unchanged at 4.81 % and `control_en` stays at 14.00 %, so the harm is a
few insertions/substitutions, not a systematic loss on Italian.

## Step 2 — bonus choice

B50 on R-S1-cal is lowest at **5.0** (11.1012 %; 3.0 gives 11.4082, 2.0 gives
11.8174, 1.0 gives 12.2202). No tie; *b\** = 5.0.

## Step 3 — benefit at *b\** = 5.0

| Condition | WER | ETA | WER Δ (pp) | 95 % CI (pp) | ETA Δ (pp) | 95 % CI (pp) |
|---|---:|---:|---:|---|---:|---|
| none | 12.8917 | 75.6291 | — | — | — | — |
| **A500** | 12.9812 | 75.3642 | +0.0895 | [−0.0900, +0.3562] | −0.2649 | [−0.6935, 0.0000] |
| **B50** | **11.1012** | **80.9272** | **−1.7905** | **[−2.5333, −1.1287]** | **+5.2980** | **[+3.5573, +7.2603]** |
| B25 | 11.6831 | 78.8079 | −1.2086 | [−1.8215, −0.6736] | +3.1788 | [+1.8006, +4.7558] |

B50 at 5.0 meets the benefit clause (WER CI entirely below zero **and** ETA CI
above zero). B25 is report-only and moves in the same direction, as expected
from a smaller recall.

## Step 4 — S1-dev veto at *b\** = 5.0

S1-dev (960 clips, 6,845 ref words), A500 vs a same-tool `none` readout:

| Condition | WER | ETA | Ins/100w | acr+brand ETA | control_it WER |
|---|---:|---:|---:|---:|---:|
| none | 13.2067 | 75.3418 | 1.42 | 40.91 | 8.01 |
| A500 | 13.3382 | 75.0977 | 1.42 | 40.50 | 8.24 |
| Δ | **+0.1315 pp** [+0.0283, +0.2597] | −0.2441 pp [−0.5519, 0.0000] | 0.00 | −0.41 pp | +0.23 pp |

The point estimate is inside the **+0.25 pp** margin (about 9 word errors of
6,845), the insertion rate and ETA guard are fine, but the **WER-difference CI
is entirely above zero**: the veto requires *not entirely above zero*, and the
lower bound is +0.028 pp. **Step 4 fails**, exactly as pre-registered. The veto
is not reinterpreted after the fact.

## Report-only breakdown at *b\** = 5.0 (R-S1-cal)

| Condition | WER | ETA | acronym ETA | brand ETA | acr+brand ETA | Ins/100w | control_it WER | control_en WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 12.89 | 75.63 | 50.98 | 30.77 | 40.78 | 1.92 | 4.81 | 14.00 |
| A500 | 12.98 | 75.36 | 50.98 | 30.77 | 40.78 | 1.97 | 4.81 | 14.00 |
| B50 | 11.10 | 80.93 | 60.78 | 34.62 | 47.57 | 2.06 | 4.81 | 12.67 |
| B25 | 11.68 | 78.81 | 56.86 | 30.77 | 43.69 | 1.92 | 4.81 | 12.33 |

The benefit is concentrated where the listed terms are: acronym ETA +9.80 pp
and brand ETA +3.85 pp at B50, while `control_it` is untouched. `control_en`
moves −1.33 pp, consistent with biasing toward the listed English terms.

## Caveat

The 541-term distractor pool and the evaluation inventory share vocabulary:
125 of the 527 unique own terms on R-S1-cal (**23.7 %**) are in the pool. The
lists are therefore closer to the product domain than a real user's glossary.
This does not change the veto, which uses no reference word.

## Verdict

**Steps 1–3 pass; step 4 (S1-dev veto) fails, so the glossary is not exposed
(D35).** R-S1-cal shows that a fixed ~500-term glossary neither hurts nor
dilutes the decoder and that partial recall still buys a real WER/ETA win, but
the only harm guard that is independent of the listed words sees a WER
difference whose CI excludes zero, albeit at +0.13 pp. The pre-registered rule
does not allow exposing the feature on that evidence. No model, LM, package,
browser or decoder default changed; no S1-dev B list was read (it contains part
of the answers), and T-600 / FLEURS / dev-macro were not read.

## Artifacts

- `cal-none.metrics.json` — control, re-scored with the vendored evaluator.
- `cal-{A500,B50,B25}-b{1.0,2.0,3.0,5.0}/` — 12 decodes: `hyp.jsonl`,
  `meta.json`, `metrics.json` (paired `vs` the control, category/speaker
  breakdown).
- `s1-dev-none/`, `s1-dev-A500/` — the veto; `metrics.json` in each.
- `loop.log`, `veto.log` — command output of the two cycles.

## Reproduce

```bash
REL=runs/E-023/release-0.9.7
CAL=../moth-cs-dataset/manifests/R-S1-cal.jsonl
S1=../moth-cs-dataset/manifests/R-S1-dev.jsonl
PROBE="uv run --frozen --project ../moth-stt --extra training -- python tools/keyterm_probe.py --release $REL --audio-root ../moth-cs-dataset --prefix-bonus 0.25"
EVAL="uv run --frozen -- python vendor/moth-stt/moth_eval.py --breakdown category,speaker"

$EVAL --manifest $CAL --hyp runs/E-022/cal-none/hyp.jsonl --out runs/E-026/cal-none.metrics.json
for b in 1.0 2.0 3.0 5.0; do
  $PROBE --manifest $CAL --bonus $b --context distractors --distractors 500 --out-dir runs/E-026/cal-A500-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --recall 0.5 --distractors 500 --out-dir runs/E-026/cal-B50-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --recall 0.25 --distractors 500 --out-dir runs/E-026/cal-B25-b$b
done
for d in runs/E-026/cal-*-b*; do
  $EVAL --manifest $CAL --hyp $d/hyp.jsonl --vs runs/E-026/cal-none.metrics.json --out $d/metrics.json
done

$PROBE --manifest $S1 --context none --out-dir runs/E-026/s1-dev-none
$PROBE --manifest $S1 --bonus 5.0 --context distractors --distractors 500 --out-dir runs/E-026/s1-dev-A500
$EVAL --manifest $S1 --hyp runs/E-026/s1-dev-none/hyp.jsonl --out runs/E-026/s1-dev-none/metrics.json
$EVAL --manifest $S1 --hyp runs/E-026/s1-dev-A500/hyp.jsonl --vs runs/E-026/s1-dev-none/metrics.json --out runs/E-026/s1-dev-A500/metrics.json
```
