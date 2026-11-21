# P6 — E-011 re-read with the product LM E-034c

> Executed 2026-11-21 under `experiments/E-002-protocol.md` §7, which was
> committed **before** the readout (D25). **Verdict: FAIL on the S1-dev ETA
> veto; dev-macro passes.** E-011 + E-034c is not adopted; E-005 + E-034c
> remains the product. This is the second look at E-011 under a changed LM, as
> P6 pre-registered.

## Setup

- **Candidate:** E-011 seeds 0 and 1, INT8
  (`runs/E-011/release-s{0,1}/model.int8.onnx`, sha256 `2eb08ec9…` /
  `3b36a893…`).
- **Control:** E-005 INT8, graph byte-identical to the product
  (`runs/E-005/release-a3/model.int8.onnx`, sha256 `14ba2034…e2d6`).
- **LM:** E-034c, MOTHLMB1 sha256 `4b122fbe…`, packed `7a4332f1…` (D25).
- **Decoder:** frozen 0.55 / 2.7, beam 12, top-k 10; INT8.
- **Sets:** dev-macro (MLS-IT 1,248 + LS-EN 2,691, both corpus exclusions) and
  clean S1-dev (960 takes; 0 exact matches and 4.09 % 4-gram share against the
  E-034c corpus).
- **Statistic:** seed mean, per E-002 §7: per-utterance errors and ETA hits
  averaged across seeds, paired utterance bootstrap (1,000, seed 0), each seed
  also reported. Differences are candidate − control; negative WER and positive
  ETA are better.

## Dev-macro (gate a)

| System | Macro WER | Δ vs control (95 % CI) |
|---|---:|---|
| E-005 + E-034c (control) | 16.9897 % | — |
| E-011 s0 | 15.3839 % | −1.6058 pp [−1.8112, −1.3898] |
| E-011 s1 | 15.3551 % | −1.6347 pp [−1.8490, −1.4125] |
| **Seed mean** | **15.3695 %** | **−1.6202 pp [−1.8215, −1.4417]** |

The 1.0 pp gate passes on the mean and on both seeds. IT 20.3579 / 20.3430 %
vs 22.5011; EN 10.4100 / 10.3671 % vs 11.4784.

## S1-dev (gate b)

| System | WER | ETA | Δ WER (95 % CI) | Δ ETA (95 % CI) |
|---|---:|---:|---|---|
| E-005 + E-034c (control) | 14.9744 % | 73.9258 % | — | — |
| E-011 s0 | 14.6238 % | 71.2891 % | −0.3506 [−1.1301, +0.4286] | **−2.6367 [−4.4240, −0.7280]** |
| E-011 s1 | 14.3170 % | 73.0469 % | −0.6574 [−1.4752, +0.1681] | −0.8789 [−2.4890, +0.9814] |
| **Seed mean** | **14.4704 %** | **72.1680 %** | **−0.5040 [−1.1566, +0.1945]** | **−1.7578 [−3.1844, −0.1422]** |

WER improves on both seeds but neither CI excludes zero; the seed-mean WER is a
tie, not a regression. **ETA regresses on the seed mean, and the CI excludes
zero.** Per the §6.2 rule, that is a regression.

Per category (seed 0 / seed 1 / control, ETA %):

| Category | Control | s0 | s1 |
|---|---:|---:|---:|
| `control_en` | 84.35 | 82.00 | 82.59 |
| `switch` | 80.27 | 74.83 | 78.68 |
| `single_word` | 85.71 | 80.36 | 83.93 |
| `phrase` | 61.86 | 61.86 | 64.95 |
| `brand` | 44.19 | 42.64 | 44.19 |
| `acronym` | 27.43 | 26.55 | 25.66 |
| `italianized` | 70.54 | 69.64 | 69.64 |

WER improves where Italian dominates (`control_it` 8.24 → 6.24 / 7.66;
`italianized` 13.22 → 10.55 / 10.85; `phrase` 15.64 → 14.38 / 13.25), while the
ETA categories that depend on accented English and short English terms get
worse. The trade is the same shape as with A3 (review §4), so it is
recipe-level, not an A3 artifact and not one bad seed.

## Verdict

**FAIL.** Gate (a) passes at −1.62 pp, gate (b) fails on ETA: s0 is a clear
regression, s1 is a tie, and the pre-registered seed mean is −1.76 pp with the
CI entirely below zero. E-011 + E-034c is **not adopted**; the product remains
E-005 + E-034c (D25). The final run stays conditional on a recipe fix.

Per E-002 §7: run P7 diagnostics, then **P10 R1** — E-011 with
`lr_encoder = 3e-5`, testing H1 — under a new pre-registration (E-023). R2
(gold ×20, H2) only if R1 does not recover ETA. Do not re-read T-600 for this
candidate; it was already read twice.

## Artifacts

- `s0-dev-macro.json/.hyp.jsonl`, `s1-dev-macro.json/.hyp.jsonl` — dev-macro.
- `dev-macro-seed-mean.json` — per-seed and seed-mean paired macro.
- `s0-s1-dev.json/.hyp.jsonl`, `s1-s1-dev.json/.hyp.jsonl` — clean S1-dev.
- `s1-dev-seed-mean.json` — per-seed and seed-mean paired WER/ETA.

## Reproduce

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode macro \
  --onnx runs/E-011/release-s0/model.int8.onnx \
  --lm runs/E-004/lm_e034c_n2.bin --allow-other-lm \
  --config training/configs/base6-long.toml \
  --reference-hyp runs/E-004/readout-e034c-int8.hyp.jsonl \
  --out runs/E-011/p6/s0-dev-macro.json --hyp-out runs/E-011/p6/s0-dev-macro.hyp.jsonl
# s1: same with release-s1
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed \
  --onnx runs/E-011/release-s0/model.int8.onnx \
  --lm runs/E-004/lm_e034c_n2.bin --allow-other-lm \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --out runs/E-011/p6/s0-s1-dev.json --hyp-out runs/E-011/p6/s0-s1-dev.hyp.jsonl
# s1: same with release-s1
python tools/seed_mean.py --macro \
  --manifest ../moth-stt/data/manifests/mls_it_dev.jsonl \
  --manifest ../moth-stt/data/manifests/ls_en_dev.jsonl \
  --control runs/E-004/readout-e034c-int8.hyp.jsonl \
  --seed runs/E-011/p6/s0-dev-macro.hyp.jsonl \
  --seed runs/E-011/p6/s1-dev-macro.hyp.jsonl \
  --out runs/E-011/p6/dev-macro-seed-mean.json
python tools/seed_mean.py \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl \
  --control runs/E-004/s1-dev-e034c-int8.hyp.jsonl \
  --seed runs/E-011/p6/s0-s1-dev.hyp.jsonl \
  --seed runs/E-011/p6/s1-s1-dev.hyp.jsonl \
  --out runs/E-011/p6/s1-dev-seed-mean.json
```
