# E-026 (E-022 step 3) — Realistic glossary: harm and partial recall

**Pre-registered 2026-11-23, before any E-026 readout.** No training and no
recording. The system is the product `0.9.7-e023-r1` (R1 seed 0 + E-034c).
The result decides whether the product exposes a user glossary; it never
changes the model, the LM, the package, the decoder defaults or a gate readout.

> **Amended 2026-11-23 (distractor pool, before any E-026 result):** the pool
> has 541 terms, so A2000 and C2000 are dropped and the harm gate uses A500
> only. See [Amendment](#amendment-2026-11-23--distractor-pool).

> **Result (2026-11-23):** steps 1–3 pass, the S1-dev veto fails (WER +0.1315 pp
> [+0.0283, +0.2597], CI entirely above zero), so the glossary is **not
> exposed** (D35). `runs/E-026/RESULT.md`.

## Question

E-022 measured only lists that always contain the clip's own English terms: a
ceiling (S1-dev oracle at bonus 5.0: −2.91 pp WER, +9.81 pp ETA; D32). A
product glossary is fixed per user: most sentences contain none of its terms,
and a spoken term may be missing from it. Two questions:

1. **Harm.** When no listed term is spoken, does biasing cost WER?
2. **Partial recall.** When only some of the spoken terms are listed, is there
   still a WER or ETA gain?

## System (fixed)

- Release `runs/E-023/release-0.9.7`: INT8 graph `85abd830…`, packed E-034c
  LM `7a4332f1…`, frozen decoder 0.55 / 2.7, beam 12, top-k 10,
  `segment: false` (as in E-022 and the Python readouts).
- `ContextualBias` with prefix bonus **0.25** in every condition.
- Runner `tools/keyterm_probe.py`, extended for this experiment with
  `--context distractors` and `--recall` (`tests/test_keyterm_probe.py`). The
  E-022 conditions are unchanged: the distractor draw is the same seeded
  sample.
- **Distractors:** normalized single words of `../moth-stt/data/en_terms.txt`,
  sampled per clip with `Random("0:<id>")`, never one of the clip's own terms.
  In condition A they also exclude **every word of the reference**, so the list
  is irrelevant to the sentence. The pool size is recorded in `meta.json`.
- **Recall:** each unique own term is kept with probability *p*, from its own
  stream `Random("0:recall:<id>")`; the distractor draw does not change.
- **Scoring:** the vendored `moth_eval.py --breakdown category,speaker --vs`,
  paired bootstrap 1,000, seed 0; differences are condition − control on the
  same release, tool and set.

## Conditions (R-S1-cal, 400 clips, 2,234 ref words)

| ID | List per clip | Role |
|---|---|---|
| none | empty: the E-022 hypotheses `runs/E-022/cal-none/hyp.jsonl`, re-scored | control |
| **A500** | 500 distractors, no reference word | **harm** |
| **A2000** | 2,000 distractors, no reference word | **harm**, long list |
| **B50** | each own term with *p* = 0.5, plus 500 distractors | **benefit** |
| B25 | *p* = 0.25, plus 500 distractors | report-only |
| C2000 | every own term plus 2,000 distractors | report-only (dilution) |

Bonus grid **{1.0, 2.0, 3.0, 5.0}** for every condition: 20 decodes of 400
clips. If the pool has fewer than 2,008 usable terms, the tool refuses and
A2000 / C2000 run at **1,000** instead; the count is reported.

## Decision rule (fixed before the readout)

1. **Safe bonus.** A bonus *b* is safe iff, for **both** A500 and A2000 on
   R-S1-cal, the WER-difference CI is not entirely above zero **and** the WER
   point estimate rises by at most **+0.25 pp** (≈ 6 word errors). The bias is
   deterministic, so the point estimate is an exact count on the same audio;
   the margin keeps a harm that the cal CI (±~1 pp) cannot resolve from
   passing. Category tables are report-only (E-002 §8.4).
2. **Bonus choice.** Among safe bonuses, pick the one with the lowest B50 WER
   on R-S1-cal; an exact tie goes to the lower bonus. **No safe bonus → stop,
   not exposed.**
3. **Benefit.** At the chosen *b\**, B50 on R-S1-cal: the WER-difference CI is
   entirely below zero and the ETA-difference CI is not entirely below zero,
   **or** the ETA-difference CI is entirely above zero and the WER-difference
   CI is not entirely above zero. **Otherwise → stop, not exposed.**
4. **S1-dev veto, once, at *b\** only.** A500 on S1-dev against a same-tool
   `none` readout of S1-dev: WER-difference CI not entirely above zero, WER
   point estimate at most **+0.25 pp** (≈ 17 word errors of 6,845), ETA
   difference CI not entirely below zero. B is not read on S1-dev: it contains
   part of the answers.
5. **Not read:** T-600, FLEURS and dev-macro. T-600 was the R1 release guard
   and is not re-read with this seed; the question concerns mixed dictation,
   and condition A on S1-dev is the harm guard.

Report for every decode: WER, ETA, insertion rate per 100 words,
acronym + brand ETA, `control_it` and `control_en` WER, and the meta counts
(oracle terms, terms used, distractors used, skipped terms).

## Outcome

- **1–4 pass → a dated decision exposes the glossary as an opt-in runtime
  option**, empty by default: `contextualSequences` with
  `contextualBonus = b*` and `contextualPrefixBonus = 0.25`, documented in
  `vendor/moth-stt-web/README.md` with the evidence boundary (S1's voice,
  single-word terms, lists up to the tested size, synthetic recall). Before the
  decision, one browser smoke: 20 R-S1-cal clips with their B50 lists at *b\**;
  browser and Python transcripts identical, or each difference a numerical
  near-tie of the E-016 class. `tools/browser_probe.mjs` needs a context option
  for this; it is written only if 1–4 pass.
- **Any step fails → the glossary stays unexposed**, and a dated decision
  records which step failed (harm or no benefit).
- Either way: no model, LM, package or decoder-default change, and a glossary
  never enters a gate readout.

**Caveat.** The distractors and, very likely, the evaluation scripts' terms
come from the same training-side term inventory, so the lists are closer to the
product domain than a real user's glossary would be. The report states the
overlap between the B50 lists and the pool.

## Commands

> The calibration loop below is the original one; after the amendment, run the
> loop in [Amendment](#amendment-2026-11-23--distractor-pool) instead (no A2000
> or C2000). The control and S1-dev commands are unchanged.

```bash
REL=runs/E-023/release-0.9.7
CAL=../moth-cs-dataset/manifests/R-S1-cal.jsonl
S1=../moth-cs-dataset/manifests/R-S1-dev.jsonl
PROBE="uv run --frozen --project ../moth-stt --extra training -- python tools/keyterm_probe.py --release $REL --audio-root ../moth-cs-dataset --prefix-bonus 0.25"
EVAL="uv run --frozen -- python vendor/moth-stt/moth_eval.py --breakdown category,speaker"
mkdir -p runs/E-026

# control: the E-022 no-context hypotheses, re-scored with the vendored evaluator
$EVAL --manifest $CAL --hyp runs/E-022/cal-none/hyp.jsonl --out runs/E-026/cal-none.metrics.json

# calibration: 5 conditions x 4 bonuses
for b in 1.0 2.0 3.0 5.0; do
  $PROBE --manifest $CAL --bonus $b --context distractors --distractors 500 --out-dir runs/E-026/cal-A500-b$b
  $PROBE --manifest $CAL --bonus $b --context distractors --distractors 2000 --out-dir runs/E-026/cal-A2000-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --recall 0.5 --distractors 500 --out-dir runs/E-026/cal-B50-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --recall 0.25 --distractors 500 --out-dir runs/E-026/cal-B25-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --distractors 2000 --out-dir runs/E-026/cal-C2000-b$b
done
for d in runs/E-026/cal-*-b*; do
  $EVAL --manifest $CAL --hyp $d/hyp.jsonl --vs runs/E-026/cal-none.metrics.json --out $d/metrics.json
done

# S1-dev veto, only if steps 1-3 pass; B_STAR is the bonus chosen in step 2
$PROBE --manifest $S1 --context none --out-dir runs/E-026/s1-dev-none
$PROBE --manifest $S1 --bonus $B_STAR --context distractors --distractors 500 --out-dir runs/E-026/s1-dev-A500
$EVAL --manifest $S1 --hyp runs/E-026/s1-dev-none/hyp.jsonl --out runs/E-026/s1-dev-none/metrics.json
$EVAL --manifest $S1 --hyp runs/E-026/s1-dev-A500/hyp.jsonl \
  --vs runs/E-026/s1-dev-none/metrics.json --out runs/E-026/s1-dev-A500/metrics.json
```

Cost: about 35 s per R-S1-cal decode on the M5 (E-022: RTFx ~29), so about
15 minutes for the grid and a few minutes for S1-dev. Artifacts go to
`runs/E-026/`; the result is written to `runs/E-026/RESULT.md`.

## Amendment 2026-11-23 — distractor pool

Written after the runner refused A2000, and before any E-026 result was on
`main` or scored. Nothing else in the pre-registration changes.

1. **Fact.** `../moth-stt/data/en_terms.txt` gives **541** normalized
   single-word terms. That is below 2,008, and also below the 1,008 the
   pre-registered 1,000-term fallback needs: the pre-registration assumed a
   larger pool without checking it.
2. **A2000 and C2000 are dropped**, with no replacement pool. Another source
   (common English words, LM-corpus vocabulary) would change what the list
   represents, and choosing one now would be a new design decision, not a
   correction.
3. **Step 1 (safe bonus) uses A500 only**, with the same two clauses: WER CI not
   entirely above zero and WER point estimate at most +0.25 pp. Steps 2–5, B50,
   B25 and the S1-dev veto are unchanged.
4. **What A500 now means.** 500 of 541 terms is about 92 % of the inventory, so
   the per-clip lists overlap almost entirely: A500 is in effect one fixed
   ~500-term glossary that never matches the sentence, which is the product
   case. Each clip keeps a margin of 41 terms for its own exclusions; a
   per-clip refusal (`only N distractors available`) invalidates that decode
   and is reported, not worked around.
5. **Evidence boundary.** If the glossary is exposed, the documentation
   recommends **at most 500 single-word terms**; longer lists are untested.
6. **Grid:** 3 conditions (A500, B50, B25) × 4 bonuses = 12 decodes of
   R-S1-cal. The calibration loop becomes:

```bash
for b in 1.0 2.0 3.0 5.0; do
  $PROBE --manifest $CAL --bonus $b --context distractors --distractors 500 --out-dir runs/E-026/cal-A500-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --recall 0.5 --distractors 500 --out-dir runs/E-026/cal-B50-b$b
  $PROBE --manifest $CAL --bonus $b --context oracle-distractors --recall 0.25 --distractors 500 --out-dir runs/E-026/cal-B25-b$b
done
for d in runs/E-026/cal-*-b*; do
  $EVAL --manifest $CAL --hyp $d/hyp.jsonl --vs runs/E-026/cal-none.metrics.json --out $d/metrics.json
done
```

If A2000 or C2000 output directories were created by the refused runs, delete
them before scoring; they are not part of the experiment.
