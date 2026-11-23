# E-024 (P8) — ×3-term unpruned LM: no effect, not adopted

> **Status: done 2026-11-23, not adopted.** The LM built with A3's ×3 term
> weighting has the same binary size and packs losslessly, but it changes no
> hypothesis on R-S1-cal (**0.0 pp** WER and ETA, byte-identical), and S1-dev
> is a tie (+0.0146 pp WER, −0.0488 pp ETA, 4/960 clips differ). Gate 1 (a
> calibration win) fails, so E-034c stays the product LM. Dev-macro and T-S3
> were not run because the primary gate failed.

## Build

The E-034c corpus was reproduced exactly (185,103 rows, sha256 `569c8233…`)
and rebuilt with one variable, `term_weight = 3` (134 term sentences). The
n-gram **type** set is unchanged:

| Artifact | E-034c (term ×1) | E-024 (term ×3) |
|---|---:|---:|
| MOTHLMB1 binary | 12,303,348 B | 12,303,348 B |
| binary sha256 | `4b122fbe…` | `0e7f0cbf…` |
| packed MOTHLMB2 | 7,787,184 B | 7,787,184 B |
| packed sha256 | `7a4332f1…` | `7c92ddaa…` |
| distinct bigram log-probs | 1,018 | 1,016 |
| distinct trigram log-probs | 878 | 874 |
| `pack_lm.py` | — | **lossless** |

The LM is genuinely different: 21,820 of 265,713 bigram log-probs and 69,673
of 2,278,369 trigram log-probs change (all 1,026 unigrams change). The packed
file is nevertheless the same size, so the 50 MB budget is not a constraint.
Artifacts: `lm-build.json`, `lm-pack.json`, `lm-corpus-clean-report.json`.

## Gate

Control: E-023 R1 seed 0 + E-034c, INT8, frozen decoder 0.55 / 2.7, beam 12,
top-k 10. Candidate − control, paired bootstrap 1,000.

| Gate | Result | Verdict |
|---|---|---|
| 1. R-S1-cal ETA win with WER guard (or WER win with ETA guard) | **12.8469 / 75.4967**, exactly the control; **0.0 pp** WER and ETA, CI [0, 0], byte-identical hypotheses | **FAIL: no win** |
| 2. S1-dev veto | **13.3966 / 75.0977** vs 13.3820 / 75.1465; WER +0.0146 pp [0, +0.0456], ETA −0.0488 pp [−0.1566, 0]; both ties, no regression | pass, but moot |
| 3. dev-macro guard | **not run** — gate 1 failed | — |
| 4. Budget | 7,787,184 B packed, lossless, package unchanged | pass |
| 5. T-S3 report-only | **not run** — gate 1 failed | — |

Only 4 of 960 S1-dev hypotheses change, all one-word variants around terms
(`shoming` → `shomy`, `sinc` → `sing`, `file trail` → `file`, `check` → `che`);
the aggregate is noise. The term weighting changes the LM probabilities but
not the decisions at the frozen decoder settings.

## Verdict

**Not adopted. E-034c remains the product LM.** The ×3-term weighting is not a
lever at 0.55 / 2.7 with this graph and corpus; the A3 recipe's term weighting
does not transfer to the unpruned E-034 LM as a measurable gain. This is a
negative result: keep the simpler ×1 term weighting and move to the next lever
(E-018 teacher bake-off, then E-019/E-020 text/data).

No package, browser or T read was made, so no product readout changed.

## Reproduce

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/clean_lm_corpus.py \
  --corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \
  --scripts ../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \
  --out /tmp/lm-corpus-nos1dev.jsonl \
  --report runs/E-024/lm-corpus-clean-report.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/build_lm_from_corpus.py --corpus /tmp/lm-corpus-nos1dev.jsonl \
  --out runs/E-024/lm_e034c_x3 --term-weight 3 --report runs/E-024/lm-build.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/pack_lm.py --lm runs/E-024/lm_e034c_x3.bin \
  --out runs/E-024/lm_e034c_x3_mothlmb2.bin --bits 10 \
  --report runs/E-024/lm-pack.json
# cal and S1-dev readouts are the commands in experiments/E-024-p8-lm-x3.md
```

## Artifacts

- `lm-corpus-clean-report.json` — 185,103 rows, same hash as E-034c.
- `lm-build.json`, `lm-pack.json` — build and lossless pack.
- `s1-cal.json/.hyp.jsonl` — identical to `runs/E-023/s1-cal-s0.*`.
- `s1-dev.json/.hyp.jsonl` — 4 differing clips, tie.
- `lm_e034c_x3.bin`, `lm_e034c_x3_mothlmb2.bin` — gitignored binaries.
