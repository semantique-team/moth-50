# Post-E-005 experiment plan

**Status: proposed 2026-11-19; the E-005 addendum, E-009, E-010, E-011, E-012
and E-004 ran on 2026-11-19/21; E-016/P4, E-017 and E-023 ran on 2026-11-21/23,
E-023 R1 is adopted (D27), the web runtime is vendored (D28), E-016b is
adopted (D29) and E-024/P8 ran negative (D30); E-013–E-015 and E-018–E-022 have
not run.**
E-017–E-022 were added after the frontier-model review
(`docs/frontier-stt-2026-11-19.md`, D24); the 2026-11-21 review
(`docs/review-2026-11-21.md`) adds P0–P11 and the S1-dev/LM-corpus overlap
check, all proposed and not pre-registered here. This file pre-registers the
experiments that follow E-005, with their gates, so they cannot be moved after
seeing results. It comes from the 2026-11-19 review of the E-005 model and package
(`docs/architecture-options.md` §11). Experiment IDs continue the MOTH-50
ledger; Moth's own IDs are cited as "Moth E-0NN".

## What the review found

- **The architecture holds.** Whisper-base + 6 identity-initialized blocks + CTC
  + n-gram LM + VAD is the right family for ≤ 50 MB, CPU/browser,
  record-then-transcribe and ~600 h of data. E-005 beat the 0.8.1 dev-macro
  reference by 5.40 pp at INT8, with an INT8 cost of 0.06 pp.
- **E-005 used the wrong LM.** The first readout and package used
  `../moth-stt/release/lm_bpe3.bin`, a stale copy of the pre-A3 LM shipped from
  0.5.0 to 0.8.0 (sha256 `dcf1d59f…`, 5,116,464 B), not the 0.8.1 A3 LM
  (`3be1177b…`, 5,381,992 B) behind the reference. Fixed by
  `tools/release_assets.py` (D17) and re-measured in the addendum below: the
  verdict stands and the A3 numbers are slightly better.
- **The package NOTICE was Moth 0.8.1's** (it names `openai/whisper-tiny`). Fixed
  by `tools/release_notice.py` (D17).
- **E-005 is under-trained.** 16,000 steps are **1.19 epochs** (epoch 0 ended
  after 13,458 batches), not ~1.4; the greedy curve still fell 0.48 pp in the
  last 2,000 steps with the LR already at its floor.
- **The encoder LR was never screened.** Review P2 asked for 3e-5–1e-4; every
  run used 3e-5 for the pretrained encoder and 3e-4 for the new blocks. Part of
  the +0 → +2 depth gain (−11.2 pp) may be "more parameters at the 10× LR".
- **The decoder settings are Moth's.** LM weight 0.55 and token bonus 2.7 were
  calibrated by Moth E-022 for an acoustic model 3.4× smaller; Moth E-022 moved
  T-600 by 6.85 pp with decoder settings alone.
- **The VAD ships an unused 8 kHz branch** (1.04 MB; `runs/probe-2026-11-19/`).

## What the second review found (after the E-005 addendum and E-009)

- **The decoder is not the lever.** The E-009 grid has a ridge (token bonus
  ≈ 4.3–4.9 × LM weight) and the frozen 0.55 / 2.7 sits on it; the best point
  gains only 0.28 pp dev-macro. The review's expectation, based on Moth E-022's
  6.85 pp, did not hold (`runs/E-009/RESULT.md`, "Post-hoc analysis").
- **Dev-macro is the wrong calibration target for the LM.** The lighter LM that
  dev-macro preferred made all eight S1-dev categories worse, `control_it`
  included. Any further LM or decoder change needs a mixed calibration set that
  is not the S1-dev gate: E-017.
- **The remaining term errors are acronyms, brands and elisions.** Acronym ETA
  29.2 %, brand 48.1 %; elided acronyms (`l'sdk`) did not improve over 0.8.1;
  deletions fell from 151 to 53 (`runs/E-005/RESULT.md`, "S1-dev error
  analysis"). Data and LM problems: E-019, E-020, E-021.
- **We do not yet know where MOTH-50 stands on T-600.** Parakeet TDT 0.6B v3
  is at 10.98 % / 83.60 % there (Moth E-017 baselines); 0.8.1 at 24.08 % /
  55.23 %. The owed E-005 product readout comes next.
- **Frontier models help as teachers, data and ceilings, not as
  architectures** (`docs/frontier-stt-2026-11-19.md`, D24): E-018, E-019,
  E-020, E-022 and the E-008 addition. Only permissively licensed models and
  data may feed training.

## Byte budget

| Component | E-005 as shipped | E-005 with A3 (measured) | After E-012 (measured) |
|---|---:|---:|---:|
| `model.int8.onnx` | 40,039,997 | 40,039,997 | 40,039,997 |
| `lm_bpe3.bin` | 5,116,464 (pre-A3) | 5,381,992 (A3) | 3,353,840 (A3, lossless 10-bit) |
| `silero_vad.onnx` | 2,327,524 | 2,327,524 | 1,289,603 (16 kHz-only) |
| vocab, mel filters, config, NOTICE | 86,129 | 86,989 | 86,993 |
| **Total** | **47,570,114** | **47,836,502** | **44,770,433** |
| **Room to 50,000,000 B** | 2,429,886 | **2,163,498** | **5,229,567** |

The A3 10-bit codebook is lossless: 1,018 distinct bigram and 859 distinct
trigram log-probs. The packed file and both readouts are in `runs/E-012/`; the
old ~44.77 MB estimate was within 1 KB of the measured package.

## Order

Do not run MPS training on the recording Mac during an S1 session (fans and
unified memory, `experiments/E-003-plan.md`). CPU readouts are also best run
after the session.

| Order | ID | Question | Needs | Cost | Bytes |
|---:|---|---|---|---|---:|
| 1 | E-005 addendum | A3 readout, A3 release, S1-dev no-regression | S1-dev complete | **done 2026-11-19** | +0.27 MB |
| 2 | E-009 | Decoder recalibration on dev | 1 | **done 2026-11-19: frozen stays** | 0 |
| 3 | E-012 | 16 kHz VAD and lossless LM packing | — | **done 2026-11-19, both gates pass** | −1.04 / −2.03 MB |
| 4 | E-010 | Encoder-LR screen | — | **done 2026-11-19: 1e-4 wins both arms** | 0 |
| 5 | E-011 | ~3 epochs + EMA, 2 seeds | 2, 4 | **done 2026-11-21: dev pass, S1-dev ETA fail on seed 0** | 0 |
| 6 | E-004 | Where the freed bytes go | 3, 5 | **done 2026-11-21: arm B (unpruned LM) adopted** | 49.21 MB |
| 7 | E-005 product readout | FLEURS once, report-only; T-600 already read in E-004 | FLEURS mel shards | ~1 h CPU | 0 |
| 8 | E-017 | Mixed calibration set, separate from the S1-dev gate | owner decision | **done 2026-11-21: 400/400 takes, manifest + shards, E-002 §8** | 0 |
| 9 | E-018 | Teacher bake-off on the gold train set | — | ~1–2 h inference, no training | 0 |
| 10 | E-016 | Browser: 1-thread RTFx, browser-vs-Python WER, vendored runtime | a release | **done 2026-11-23: 1-thread pass, parity explained, runtime vendored (D28); E-016b proposed** | 0 |
| 11 | E-019 | Teacher labels (pseudo-label distillation) | 8, 9 | labelling + fine-tunes | 0 |
| 12 | E-020 | LLM-written mixed text for the LM | 8 | LM builds + readouts | ≤ LM budget |
| 13 | E-021 | Targeted acronym, brand and elision data | product acoustic baseline | recording + fine-tunes | 0 |
| 14 | E-022 | Keyterm biasing re-test (product feature) | 8 | CPU readouts | 0 |
| 15 | E-013 | Extra blocks before `final_ln` | — | ~1.6 h MPS | 0 |
| 16 | E-014 | Conformer-style convolution in the extra blocks | — | ~1.6 h MPS | ≈ 0 |
| 17 | E-015 | 25 Hz extra blocks (P7) | only if E-016 fails | ~1.6 h MPS | 0 |
| 18 | E-006 | Remaining data arms | product baseline | days | 0 |
| 19 | E-008 | External evaluation and ceilings | final model | hours | 0 |
| 20 | E-023 | R1 recipe ablation: E-011 with encoder LR 3e-5 | E-017, a P6 failure | **done 2026-11-23: 2b report-only (D26), T-600 guard passed, adopted (D27)** | 0 |

P8 was run after this table was written: **E-024 (2026-11-23) built the
unpruned E-034 LM with ×3 term weighting, got byte-identical R-S1-cal
hypotheses and a tie on S1-dev, and was not adopted (D30)**. The term weighting
is not a lever at the frozen decoder; E-034c stays.

The FLEURS readout, E-017 and E-018 need no training and can run while the Mac
is otherwise idle; E-016 is engineering and can run in parallel. The full order
for the product decision, including the 2026-11-21 review's P0–P11 (parity,
S1-dev/LM overlap, E-011 re-read with E-034, LM-first D), is
`docs/review-2026-11-21.md` §9. P0–P2 are prerequisites for readouts taken on
the second Mac.

## E-005 addendum — A3 readout, A3 release and S1-dev

**Why:** the erratum above. **Gate:** the E-002 gate, unchanged, now complete:
dev-macro ≥ 1.0 pp better than 23.0170 % with a paired CI excluding zero, **and**
no S1-dev regression as defined in the E-002 amendment of 2026-11-19 (neither a
WER increase nor an ETA drop whose paired 95 % CI excludes zero, against 0.8.1
INT8 + A3 decoded by the same tool).

```bash
python tools/release_assets.py fetch           # A3 LM, 0.8.1 graph, Silero graphs
# 1. dev-macro with A3 (FP32 and INT8)
uv run --frozen --project ../moth-stt --extra training -- python tools/frozen_readout.py \
  --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml --out runs/E-005/readout-a3.json \
  --hyp-out runs/E-005/readout-a3.hyp.jsonl
# 2. the release with A3 and the MOTH-50 NOTICE
uv run --frozen --project ../moth-stt --extra export -- python tools/export_release.py \
  --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml --out-dir runs/E-005/release-a3 --version 0.9.1-e005
uv run --frozen --project ../moth-stt --extra training -- python tools/frozen_readout.py \
  --onnx runs/E-005/release-a3/model.int8.onnx \
  --config training/configs/base6-long.toml --out runs/E-005/readout-a3-int8.json
# 3. S1-dev: 0.8.1 reference, then MOTH-50 FP32 and INT8 against it
uv run --frozen --project ../moth-stt --extra training -- python tools/frozen_readout.py \
  --mode mixed --onnx data/raw/moth-0.8.1/model.int8.onnx \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --out runs/E-005/s1-dev-0.8.1.json --hyp-out runs/E-005/s1-dev-0.8.1.hyp.jsonl
uv run --frozen --project ../moth-stt --extra training -- python tools/frozen_readout.py \
  --mode mixed --onnx runs/E-005/release-a3/model.int8.onnx \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml --reference-hyp runs/E-005/s1-dev-0.8.1.hyp.jsonl \
  --out runs/E-005/s1-dev-int8.json --hyp-out runs/E-005/s1-dev-int8.hyp.jsonl
```

**Result (2026-11-19): gate complete.** Dev macro with A3: **17.5565 % FP32 /
17.6212 % INT8** vs 23.0170 %, paired **−5.4605 / −5.3957 pp** (CIs exclude
zero). S1-dev: 0.8.1 **22.2498 % WER / 62.1094 % ETA**; MOTH-50 FP32
**13.9956 / 75.3418**, INT8 **14.0541 / 75.5371**; paired WER
**−8.2542 / −8.1958 pp**, ETA **+13.2324 / +13.4277 pp**, no regression on
either. The A3 package is **47,836,502 B** (2,163,498 B of room) with the
MOTH-50 NOTICE; the browser transcript is exact at **RTFx 22.3**. Artifacts:
`runs/E-005/readout-a3*.json`, `release-a3/`, `s1-dev-*.json`,
`browser-run-a3.json`.

### E-005 product readout — FLEURS, once (T-600 done in E-004)

`docs/architecture-options.md` §7 pre-registered it for E-005: **T-600 ≤ 22 %
WER and ≥ 58 % ETA, FLEURS-IT ≤ 18 %**. It is a report, read **once**, and it is
never used to select or tune. **T-600 was read in E-004** (A3 13.9908 / 75.3595,
E-034 13.4748 / 76.5101; `runs/E-004/t600-*.json`), against the retained 0.8.1
E-037 hypotheses; any further T read is a declared repeated read. **FLEURS-IT/EN
was read on 2026-11-21 with E-034c** (P3, once per language, report-only):
**IT 14.8238 %** (−5.3383 pp vs 0.8.1, CI excludes 0), **EN 30.2675 %**
(−19.5401 pp); no ETA. FLEURS-IT meets the original ≤ 18 % expectation.
`runs/E-004/fleurs-RESULT.md`.

- **Product package:** `runs/E-004/release-e034c/` (E-005 graph + E-034c LM +
  16 kHz VAD). Read both languages once, `--mode mixed`, no reference; report
  WER and the per-category table.
- **References:** 0.8.1 FLEURS results and the E-018 winner on the same set, as
  diagnostics only.
- **Expectations, not gates:** FLEURS-IT ≤ 18 % from the original E-005
  pre-registration; a miss is recorded, not acted on.
- **Not a selection set.** FLEURS never ranks a model, a decoder or an LM.

```bash
# FLEURS mel shards (audio from Moth's data/raw; re-fetch with its fetch.sh)
for lang in it en; do
  uv run --frozen --project ../moth-stt --extra training -- python -m stt.data.build_mel \
    --manifest ../moth-stt/data/manifests/fleurs_${lang}_test.jsonl \
    --audio-root ../moth-stt/data/raw --out-dir data/shards/fleurs_${lang}_test --workers 8
  uv run --frozen --project ../moth-stt --extra training -- python tools/frozen_readout.py \
    --mode mixed --name fleurs-${lang} --onnx runs/E-004/release-e034/model.int8.onnx \
    --lm ../moth-stt/runs/E-034/build/N2_1_1.bin --allow-other-lm \
    --manifest ../moth-stt/data/manifests/fleurs_${lang}_test.jsonl \
    --shards data/shards/fleurs_${lang}_test --config training/configs/base6-long.toml \
    --out runs/E-004/fleurs-${lang}-e034-int8.json \
    --hyp-out runs/E-004/fleurs-${lang}-e034-int8.hyp.jsonl
done
```

## E-009 — Decoder recalibration (dev-macro selection, S1-dev veto)

- **Hypothesis:** the best (LM weight, token bonus) for base+6 differs from
  (0.55, 2.7); a stronger acoustic model usually wants a lighter LM.
- **Grid:** LM weight {0.25, 0.35, 0.45, 0.55, 0.65} × token bonus {1.5, 2.0,
  2.7, 3.5}; beam 12, top-k 10 and the A3 LM fixed. If the selected point lies on
  the edge of the grid, extend the grid once in that direction.
- **Data:** dev-macro (MLS-IT + LS-EN) for ranking. S1-dev is **not** a grid
  input: it is a one-point veto. Never T, FLEURS or CS-FLEURS.
- **Selection rule** (amended 2026-11-19, before any grid result; implemented in
  `tools/calibrate_decoder.py`): drop points significantly worse than the frozen
  point on dev-macro WER (paired 95 % CI); pick the lowest dev-macro WER (ties:
  closer to frozen); adopt it only if its paired dev-macro WER CI against the
  frozen point excludes zero. Otherwise the frozen settings stay.
- **S1-dev veto:** one full readout of the selected point against the frozen
  MOTH-50 hypotheses on the same model (`runs/E-005/s1-dev-a3-int8.hyp.jsonl`)
  with the same A3 LM. If WER or ETA regresses with a CI excluding zero, keep the
  frozen decoder. The veto is a gate, not a ranking: do **not** decode the full
  grid on S1-dev.
- **Adoption:** the selected pair goes into the release `config.json` and into
  later gate readouts as a dated E-002 amendment; any paired gate that compares
  models uses the **same** decoder for both systems. The published 0.8.1 number
  stays a historical reference in its own settings.
- **Cost:** one acoustic pass and 20 beam decodes on 3,939 dev clips, ~20 min
  with 8 workers on the M5 Pro; the S1-dev veto is ~1 min.

```bash
# 1. rank on dev-macro only (no --s1-dev-*)
uv run --frozen --project ../moth-stt --extra training -- python tools/calibrate_decoder.py \
  --onnx runs/E-005/release-a3/model.int8.onnx --config training/configs/base6-long.toml \
  --workers 8 --out runs/E-009/grid.json
# 2. veto the selected point, once, against the frozen MOTH-50 readout
uv run --frozen --project ../moth-stt --extra training -- python tools/frozen_readout.py \
  --mode mixed --onnx runs/E-005/release-a3/model.int8.onnx \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --experiment-decoder --lm-weight <selected> --token-bonus <selected> \
  --reference-hyp runs/E-005/s1-dev-a3-int8.hyp.jsonl \
  --out runs/E-009/s1-dev-veto.json --hyp-out runs/E-009/s1-dev-veto.hyp.jsonl
```

**Result (2026-11-19):** the grid ran on dev-macro (3,939 clips, 18.9 min with
8 workers). The best point was **0.45 / 2.0**: dev-macro **17.3427 %** vs
17.6212 %, paired **−0.2786 pp**, CI [−0.3821, −0.1718]. The S1-dev veto
**failed**: 14.6384 % WER / 74.4141 % ETA vs the frozen MOTH-50 14.0541 /
75.5371, paired **+0.5844 pp WER** [0.2422, 0.9515] and **−1.1230 pp ETA**
[−1.8408, −0.4764]. The frozen decoder **0.55 / 2.7 stays**; no package or
readout changed. Artifacts: `runs/E-009/grid.json`, `s1-dev-veto.json`,
`runs/E-009/RESULT.md`. Its post-hoc analysis (the ridge, and the lighter LM
hurting every S1-dev category, `control_it` included) is what motivates E-017.

## E-010 — Encoder-LR screen

- **Hypothesis:** an encoder LR of 3e-5 under-adapts the pretrained Whisper-base
  encoder; at 1e-4 the same depth reaches a lower greedy dev-macro WER, and the
  depth gap shrinks.
- **Arms:** `training/configs/base6-enc1e-4.toml` (E-003 recipe, only
  `lr_encoder = 1e-4`) with `--head-blocks 4` and `--head-blocks 6`, 4,000 steps,
  seed 0. Controls: E-003 at 3e-5 — base+4 34.73 / 34.74 %, base+6 33.46 /
  33.55 % (seed spread ≤ 0.09 pp).
- **Gate:** 1e-4 wins for an arm if greedy dev-macro at 4,000 steps improves by
  ≥ 0.5 pp (≈ 5× the measured seed spread) with no non-finite step. If base+4 at
  1e-4 lands within 0.2 pp of base+6 at 3e-5 (≤ 33.66 %), base+4 becomes an E-004
  candidate (6.4 MB freed). If 1e-4 is worse by ≥ 0.5 pp, add a 5e-5 arm before
  E-011; otherwise E-011 keeps 3e-5.

```bash
for blocks in 4 6; do
  uv run --frozen --project ../moth-stt --extra training -- python -m training.train \
    --config training/configs/base6-enc1e-4.toml --moth-stt ../moth-stt \
    --head-blocks $blocks --max-steps 4000 --out runs/E-010/base+$blocks-enc1e-4 --device mps
done
```

**Result (2026-11-19): 1e-4 wins both depths; base+6 at 1e-4 is the best arm.**
Greedy dev-macro at 4,000 steps:

| Arm | 3e-5 (E-003) | 1e-4 (E-010) | Δ |
|---|---:|---:|---:|
| base+4 | 34.7306 % | **33.4003 %** | **−1.3303 pp** |
| base+6 | 33.4588 % | **32.3912 %** | **−1.0677 pp** |

Both pass the ≥ 0.5 pp gate with 0 non-finite steps (87.7 / 101.8 min, peak
8.473 / 9.648 GB). base+4 at 1e-4 is 0.0585 pp *better* than base+6 at 3e-5
(≤ 33.66 %), so it becomes an E-004 depth candidate. **E-011 uses base+6 +
`lr_encoder = 1e-4`**; the config is `training/configs/base6-e011.toml`.
Artifacts: `runs/E-010/`, `runs/E-010/RESULT.md`.

## E-011 — Long schedule with EMA

- **Hypothesis:** ~3 epochs with a weight EMA beat E-005 by ≥ 1.0 pp dev-macro
  with the same decoder.
- **Recipe:** the E-010 winner — **base+6, `lr_encoder = 1e-4`** — 40,000 steps
  (~2.97 epochs at 13,458 steps per epoch), warm-up 1,000, cosine to 0.05,
  `train.ema_decay = 0.9998` (horizon ~5,000 steps; `training/ema.py`), eval every
  4,000 steps, dev sets as in E-005. Two seeds (the final recipe needs 2–3,
  E-002 §3). Config: `training/configs/base6-e011.toml` (preflight ready).
- **Gate:** frozen beam+A3 readout (the frozen E-002 decoder, since E-009 kept
  it, applied to both E-011 and E-005) ≥ 1.0 pp better dev-macro than E-005,
  paired CI excluding zero; no S1-dev regression against E-005; INT8 cost
  ≤ 0.2 pp. Report both seeds; a sub-1 pp difference between seeds is noise, not
  a result.
- **Cost:** ~17 h per seed at the E-005 throughput (6.73 h per 16,000 steps);
  the EMA copy adds ~0.16 GB.
- **Result (2026-11-21): dev-macro PASS, S1-dev ETA fail on seed 0 — not
  adopted.** Both seeds beat E-005 by ~1.7 pp macro, FP32 and INT8, with CIs
  excluding zero: seed 0 **15.8584 %** (−1.6981 pp), seed 1 **15.7945 %**
  (−1.7620 pp) against 17.5565 %. INT8 costs +0.05/+0.11 pp and the packages are
  44,770,432 B. On S1-dev WER is a tie (13.908/13.981 % vs 13.9956 %), but seed 0
  loses **2.1973 pp ETA** (CI [−3.8280, −0.4390]) and fails the no-regression
  rule; seed 1 loses 1.3672 pp with a CI including zero. The ETA errors are
  short English words (*it, a, too, on, is, coffee, makes*), not rare terms.
  One seed passing is not enough to adopt, and choosing the seed on S1-dev would
  be tuning. **E-005 remains the product model.** Artifacts: `runs/E-011/`,
  `runs/E-011/RESULT.md`.
- **P6 (2026-11-21): re-read with E-034c, still not adopted.** Under the E-002
  §7 pre-registration and the seed-mean statistic: dev-macro 15.3695 % vs
  16.9897 (−1.6202 pp, gate passes), clean S1-dev WER −0.5040 pp (CI includes 0)
  but **ETA −1.7578 pp [−3.1844, −0.1422]** (gate fails; s0 −2.64, s1 −0.88).
  The ETA loss is recipe-level, not an A3 artifact. E-005 + E-034c stays; next
  P7, then P10 R1. `runs/E-011/p6/RESULT.md`.

## E-012 — Package bytes: 16 kHz VAD and lossless LM packing

- **(a) VAD.** `tools/probe_vad.py` on S1-dev, T-600 and the 480-clip noise set
  (`data/manifests/noise.jsonl`), then a browser run of a package built with
  `export_release.py --vad 16k` (load, exact transcript of `T/S1/S1_0001.wav`,
  RTFx). **Gate:** `drop_in: true` on all real audio (identical window
  probabilities within 1e-4, gates and segments) and the exact browser
  transcript. Saves 1,037,921 B. The synthetic preflight is already
  bit-identical (`runs/probe-2026-11-19/`).
- **(b) LM.** Implement `MOTHLMB2` (review P5) with a 10-bit **lossless**
  codebook: writer, Python reader, JS reader, validator magic. **Gate:** A3 has
  ≤ 1,024 distinct values per order, the packed file round-trips exactly, and the
  dev-macro + S1-dev hypotheses are identical to `MOTHLMB1`. Saves ~2.02 MB.
- **Result (2026-11-19): both gates PASS (D19, D20).** (a) The probe ran on
  2,040 real clips, 9,268.29 s (S1-dev 960, T 600, noise 480): **bit-identical**
  (max window-probability gap 0.0, 0 flips, 0 gate or segment disagreements),
  `drop_in: true`; a package with the graph loads in 393 ms, transcribes
  `T/S1/S1_0001.wav` **exactly**, and benchmarks at 23.90 RTFx (4 threads).
  (b) A3 has 1,018 distinct bigram and 859 trigram log-probs; the lossless
  10-bit `MOTHLMB2` is **3,353,840 B** (from 5,381,992, −2,028,152) and leaves
  the dev-macro and S1-dev hypothesis files **byte-identical** (macro 17.6212 %,
  S1-dev 14.0541 % WER / 75.5371 % ETA). The final package is **44,770,433 B**
  (5,229,567 B of room); the browser loads it in 413 ms and reaches **RTFx
  24.63**. Both formats default to the new encoding; `--vad full` and
  `--lm-format mothlmb1` remain for reproduction. Artifacts: `runs/E-012/`.
- **Runtime:** `moth-stt` is archived. The `budget_bytes` patch is local commit
  `b0fa11c` and the `MOTHLMB2` JS reader is local commit `0e15442`; (b) needs
  both. MOTH-50 must vendor the web runtime at those local commits (E-016).

## E-004 — Where the freed bytes go (re-scoped)

The original question (base+6 + 5.38 MB LM vs base+4 + 12 MB LM) assumed no
headroom. After E-012 the room is **5,229,567 B**, so the arms at ≤ 50,000,000 B
are:

| Arm | Acoustic | LM | Package (est.) |
|---|---|---|---:|
| A (control) | base+6 | A3, packed | **44.77 MB (measured)** |
| B | base+6 | unpruned Moth E-034 LM, packed (8-bit 7.16 MB / 10-bit 7.80 MB) | ~48.6 / 49.2 MB |
| C | base+7 | A3, packed | ~47.88 MB |
| D | base+4 | 12–14 MB LM | only if E-010 makes base+4 competitive |

Recalibrate the decoder per arm on dev-macro with the E-009 procedure and its
S1-dev veto; decide on S1-dev WER/ETA with dev-macro as a guard; read T once at
the end. Arm C needs an E-011-length run; E-003 (+4 → +6 bought 1.2 pp at 4,000
steps) suggested +7 is likely ≤ 0.6 pp. **After E-011 (2026-11-21): the long
depth+schedule won dev-macro but failed the S1-dev ETA veto, so C and D are not
authorized without a new pre-registration. Start with B (readout-only) against
the E-005 control.** B keeps A3's ×3 term weighting only if the unpruned LM is
rebuilt with it; otherwise it trades terms for perplexity, and S1-dev ETA
decides.
- **Result (2026-11-21): arm B adopted.** The unpruned E-034 N2_1_1 LM
  (12,313,440 B) packs losslessly to **7,793,632 B**; the package is
  **49,210,224 B** (789,776 B of room), graph byte-identical to E-005. Decoder
  calibration kept 0.55 / 2.7. Versus A3: dev-macro **16.9863 %** vs 17.6212 %
  (−0.6349 pp), S1-dev **11.6289 % WER / 79.8340 % ETA** vs 14.0541 / 75.5371
  (−2.4251 / +4.2969 pp), T-600 **13.4748 / 76.5101** vs 13.9908 / 75.3595.
  Browser loads the package and transcribes exactly at RTFx 28.36. C and D were
  not run (E-011 removed their precondition). Artifacts: `runs/E-004/`,
  `runs/E-004/RESULT.md`.
- **Correction (2026-11-21, P2): the S1-dev result is withdrawn.** 900 of the
  960 S1-dev cues are exact sentences of the E-031 corpus: S1-dev was drawn from
  `text/pool.jsonl`, and the pool is a corpus `train_script`
  (`runs/E-004/lm-corpus-overlap.json`). The same N2_1_1 recipe on the corpus
  minus those 900 sentences (**E-034c**, 185,103 rows) gives dev-macro
  **16.9897 %**, S1-dev **14.9744 / 73.9258** and T-600 **13.7041 / 76.1266**.
  Against the A3 recipe rebuilt on the same clean corpus (A3c): −0.6355 pp
  macro, −0.4381 pp WER / +1.1230 pp ETA on S1-dev (CIs exclude zero), −0.4870
  / +1.0550 on T-600 (CIs include zero, repeated read). The honest gain is
  ~0.4–0.5 pp WER / ~1.1 pp ETA, not 2.4 / 4.3. **Decision pending: ship
  E-034c.** Never draw evaluation cues from a file that feeds the LM corpus.
- **Follow-up (proposed, not pre-registered):** `docs/review-2026-11-21.md` —
  base+7 no longer fits next to the E-034 LM (~52.3 MB); D only LM-first (a
  larger LM read on base+6 before any base+4 training); check the S1-dev scripts
  against the LM corpus, since the S1-dev gain is ~5× the T-600 gain.

## E-016 — Browser runtime checks

- **1-thread RTFx** without cross-origin isolation on 5 / 10 / 20 s clips. Gate:
  ≥ 8× (the E-005 target; projected 7–9×). E-005 only measured 4 threads (22.3×).
- **Browser vs Python WER** on S1-dev: the comparison Moth's E-041
  pre-registered and never ran. Gate: identical hypotheses, or every difference
  explained.
- **Runtime ownership:** vendor `moth-stt/web/` into this repository at local
  commits `b0fa11c` (`budget_bytes`) and `0e15442` (`MOTHLMB2` reader), so the
  package no longer depends on a local commit in an archived repository.

**Result (2026-11-23, D28).** 1-thread non-isolated RTFx **10.93× / 19.74× /
21.17×** on 5 / 10 / 20 s (gate pass). Browser vs Python on S1-dev, product
`0.9.7-e023-r1`: Python 13.3820 / 75.1465; browser default **14.0979 / 75.5859**
(675/960 identical, +0.7159 pp WER [+0.1678, +1.2599], no tie); `segment:
false` **13.7327 / 75.5859** (697/960, +0.3506 pp [−0.1255, +0.8160], **tie**);
Python-FFT 16 kHz input 14.5215 / 74.7559 (+1.1395 pp). The systematic gap is
the runtime's default VAD **segmentation**, not resampling; the residual 263/960
differences are numerical near-ties (Moth E-022 measured 70–77 % identity with a
WER tie). Every difference is explained, so the gate passes, but the browser
default costs ~0.4–0.7 pp WER on S1-dev. Runtime vendored at `0e15442` in
`vendor/moth-stt-web/` with a one-file `ort.js` patch; `tools/serve_web.py`,
`tools/browser_probe.mjs` and `tools/browser_parity.py` reproduce the checks.
`runs/E-016/RESULT.md`.

### E-016b — segmentation default (done 2026-11-23, D29)

**Status: adopted.** The proposed blanket auto-off below 30 s was rejected: it
removes segmentation from multi-sentence clips and lost **2.14 pp**
([+0.77, +3.49]) on a 100-group five-sentence diagnostic. The adopted rule
splits at **three or more VAD runs** for a clip that fits one 30 s segment, and
at two or more for longer recordings. On S1-dev the browser goes from **14.0979
%** to **13.7180 %** WER (+0.3360 pp [−0.1312, +0.7948], tie with Python;
ETA 75.6348, tie). The 35 s smoke still yields 8 segments. MLS-IT dev
concatenations (2 long utterances, 3+ runs) stay segmented, so the full-clip
gain there is not captured; this is recorded as a limitation. Patch and
provenance: `vendor/moth-stt-web/PATCH-2026-11-23.diff`,
`vendor/moth-stt-web/PROVENANCE.md`; evidence: `runs/E-016/E-016b.md`,
`runs/E-016/e016b-diagnostics.json`.

## E-013 — Extra blocks before `final_ln`

- **Today:** encoder blocks → `final_ln` → 6 pre-LN extra blocks → linear head.
  No LayerNorm closes the residual stream the head reads.
- **Variant:** insert the extra blocks **before** `final_ln` (block expansion as
  in LLaMA Pro). With identity initialization step 0 is still exactly
  Whisper-base, and the head always reads a normalized stream. 0 bytes, no
  latency change.
- **Screen:** base+6, 4,000 steps, seed 0, E-003 recipe; control E-003 base+6
  (33.46 / 33.55 %). Gate: ≥ 0.5 pp greedy dev-macro. Needs a model option in
  `training/model.py` (forward, checkpointed forward) and an export check.

## E-014 — Conformer-style convolution in the extra blocks

- **Hypothesis:** the extra blocks model local acoustic context inefficiently
  with attention alone. A convolution module per block (pointwise 512 → 1,024 +
  GLU, depthwise k = 15, LayerNorm, Swish, pointwise 512 → 512, ≈ 0.80 M
  parameters) paid for by a narrower FFN (2,048 → 1,280 saves ≈ 0.79 M per
  block) keeps the bytes equal.
- **Screen:** 4,000 steps against E-003 base+6; identity initialization kept by
  zeroing the last pointwise convolution. Gate: ≥ 0.5 pp greedy dev-macro at
  equal or lower INT8 bytes, and 1-thread CPU latency within +10 %.

## E-015 — 25 Hz extra blocks (review P7)

Run only if E-016 measures < 8× at 1 thread, or if training time becomes the
limit. Stride-2 pooling after the encoder; extra blocks and CTC at 25 Hz. First
check CTC feasibility on the training manifests: BPE tokens per output frame
must stay < 0.8 for every clip. Gate: dev-macro within 0.3 pp of the 50 Hz
screen and ≥ 1.4× faster extra blocks.

## E-006 — Data arms (re-scoped)

One variable at a time, each as a fine-tune from the **E-005 acoustic + E-034
LM product baseline** against a control fine-tune of the same length on the
unchanged mixture; judged on S1-dev WER/ETA with dev-macro as a guard, S2/S3/
FLEURS as diagnostics.

1. **VoxPopuli Italian transcribed** (91 h, CC0): the cheapest new real Italian
   speech not yet in the mixture.
2. **Gold repetition:** ×50 (current) vs ×20 plus a gentle gold-only stage at
   Moth E-028's LRs (1e-4 head / 1e-5 encoder): a 39 M model may memorize 1,413
   sentences seen 50 times per epoch.
3. **Word/phrase-level splicing** with forced alignment vs the utterance-level
   splices.
4. **Pseudo-labels:** superseded by E-018 (choose the teacher) and E-019 (label
   and train).
5. **New S1 training sessions** (2–5 h), never the S1-dev session: unscripted
   sessions are E-019 arm B; acronym/brand/elision scripts are E-021.
6. **TTS** as a 10–20 % minority source (permissive voices only): the targeted
   version is E-021 arm 2.

## Frontier-model follow-ups (E-017–E-022)

These come from the review in `docs/frontier-stt-2026-11-19.md` (D24) and from
the S1-dev error analysis. The licence rule applies to all of them: only
permissively licensed models and data feed training; commercial APIs are
evaluation-only; every teacher, corpus and generator is pinned in
`data/sources.lock.json` before use, and anything that enters a release is
attributed in its NOTICE (`tools/release_notice.TRAINING_SOURCES`).

### E-017 — Mixed calibration set (protocol; decide before E-004, E-020, E-022)

- **Problem:** E-009 showed that dev-macro asks for a lighter LM than the
  product set wants, and S1-dev is the gate and the veto, so tuning on it would
  be circular. Every later LM or decoder choice (E-004 per-arm recalibration,
  E-020, E-022) needs a mixed IT–EN set used **only** for calibration.
- **Option A (recommended): a new calibration session**, `R-S1-cal`: 300–400 new
  mixed cues (~15–20 min) from the same script pipeline, category mix as
  S1-dev, deduplicated against T, A, R-S1, R-S2 and the S1-dev scripts,
  recorded by S1 in a separate session with the E-002 runbook. Costs about an
  hour of recording and review; S1-dev keeps its full power as the gate. The
  cues must be new: **not selected from `text/pool.jsonl`**, which is in the
  E-031 LM corpus (P2), and the new set must be checked with
  `tools/lm_corpus_overlap.py` before the first readout.
- **Option B: split S1-dev** into calibration and gate halves (480 / 480,
  stratified by category, fixed by a seeded hash of the cue id before any
  calibration readout). No recording, but the gate's paired CIs widen by about
  √2 (the E-009 veto's WER half-width 0.35 pp → ~0.50 pp). The E-005 and E-009
  results on the full S1-dev stay as recorded.
- **Either way** the calibration set is S1's voice, so it cannot separate
  speaker adaptation from domain fit; T-600 S3 at release time checks that.
- **Adoption:** a dated E-002 amendment (§8) names the set and the rule: rank
  decoder/LM settings on the calibration set; dev-macro is a guard (no
  regression with CI excluding zero); the S1-dev veto (§6.3) stays. Nothing is
  calibrated on the new set before that amendment exists.
- **Status (2026-11-21): option A done and frozen.** 400 cues, 50 per
  category, 755 English-term words, mean 5.58 words; 0 exact/near duplicates
  against T, A, R-S1, R-S2, pool and S1-dev; 0 exact sentences in the E-031 LM
  corpus; 4-gram share 8.75 % (T 9.75 %, clean S1-dev 4.09 %). Reviewed by
  Tancredi, recorded (400/400, 16.95 min, 19 retakes), level-matched to S1-dev
  with one fixed −16.58 dB gain (0 clipped after), built into
  `manifests/R-S1-cal.jsonl` (`split: dev`) and `data/shards/s1-cal` (400
  clips, 1 shard, 0 errors). E-002 §8 names the set and the rule before any
  readout. The control E-005 + E-034c was read once: **13.9212 % WER /
  75.6291 % ETA**; no R1 readout has run.
  `../moth-cs-dataset/text-2026-11-cal/`, `runs/E-017/RESULT.md`.

### E-018 — Teacher bake-off (no training)

- **Question:** which permissively licensed open model gives the best labels for
  mixed IT–EN dictation, and is it far enough ahead of MOTH-50 to teach it?
- **Candidates:** Parakeet TDT 0.6B v3 (CC-BY-4.0), Canary-1B-v2 (CC-BY-4.0),
  Whisper large-v3-turbo (MIT; forced `it` and auto language), Voxtral Mini
  (Apache-2.0). Optionally, on a rented GPU: SOV-Transcribe-1 (Apache-2.0) and
  Nemotron 3.5 Transcribe (OpenMDW-1.1). Pin each by revision and file hash
  first.
- **Data:** the gold train set v1.0.0 (1,413 clips, S1 + S2, 57.6 min): real
  code-switching with reviewed transcripts, and no candidate was trained on it.
  Not S1-dev (the gate), not T (the test set). Optional sanity check: the first
  500 MLS-IT dev clips against MOTH-50's MLS-IT dev WER.
- **Metrics:** WER, ETA, insertion rate, per category, RTFx on the M5, all
  through Moth's normalizer and `vendor/moth-stt/moth_eval.py --breakdown
  category,speaker`. Watch `italianized`: a teacher that writes `crashed` for
  `crashato`, or translates a term, teaches the wrong thing even at low WER.
- **Selection rule:** a teacher qualifies if on the gold train set it reaches
  **WER ≤ 7.0 %, ETA ≥ 85 % and `italianized` ETA ≥ 72.3 %** (half of MOTH-50's
  S1-dev WER, 10 points above its S1-dev ETA, and not below its `italianized`
  ETA; same speaker S1 and same script pipeline, so the comparison is
  approximate). Among qualifying teachers pick the lowest WER; within 0.5 pp,
  the faster one. The runner-up, if it qualifies, is the agreement filter in
  E-019. If none qualifies, E-019 keeps only monolingual Italian sources, where
  any candidate is far ahead of MOTH-50.
- **Tooling (not yet written):** a runner that writes `{id, hyp}` JSONL per
  teacher; Moth's `baselines/parakeet_runner.py` and `whisper_runner.py` are
  the starting point.
- **Cost:** Parakeet ran at RTFx 34 on 4 CPU threads in Moth E-017, so the gold
  set takes ~2 min; the larger local candidates tens of minutes each.

### E-019 — Teacher labels (pseudo-label distillation)

- **Hypothesis:** teacher-labelled speech that MOTH-50 has never heard improves
  S1-dev WER without an ETA regression; monolingual Italian helps dev-macro too.
- **Sources** (each pinned with its licence before use):
  1. **NVIDIA Granary, Italian** (dataset card CC-BY-4.0; YODAS-Granary
     CC-BY-3.0, MOSEL CC-BY-4.0; the audio keeps its source licence: VoxPopuli
     CC0, YouTube-Commons CC-BY). Already pseudo-labelled, no teacher compute.
     Start with the VoxPopuli part; YouTube-sourced audio carries
     uploader-licence risk and needs its own check.
  2. **VoxPopuli Italian, unlabelled** (CC0), labelled by the E-018 teacher.
  3. **Unscripted S1 sessions** (2–5 h): S1 speaks freely about work topics;
     the teacher transcribes. It adds spontaneous speech, which no current
     source has, and costs no scripting or review. Never reads S1-dev or T
     scripts.
- **Filters:** agreement between two teachers (keep a clip if the normalized
  WER between them is ≤ 10 %); 1–30 s; Silero speech fraction > 0.10; at most
  4 words per second (Parakeet hallucinated on 24 noise clips in Moth E-011);
  Moth's training guard (no T, A, S3, FLEURS, dev audio or IDs).
- **Audit before training:** listen to 100 random kept clips and score them by
  hand. If the audited WER exceeds 10 %, tighten the filter and audit again.
- **Arms**, one variable each, fine-tuned from the **product acoustic checkpoint
  (E-005)** against a control fine-tune of the same length on the unchanged
  mixture. E-011 ran but was not adopted, so it is not the parent. A = + Italian
  pseudo-labels (sources 1–2), capped at the MLS-IT share; B = + unscripted S1
  pseudo-labels (source 3).
- **Gate:** S1-dev paired WER better with the CI excluding zero and no ETA
  regression (E-002 §6), dev-macro without a regression whose CI excludes zero.
  A passing source enters the mixture of the next full run.
- **Risks:** arm B trains on the S1-dev speaker, so its S1-dev gain overstates
  the product; T-600 S2/S3 read at release time decide. The student inherits
  the teacher's errors and cannot exceed it on those domains.

### E-020 — LLM-written mixed text for the LM

- **Hypothesis:** an A3-style LM rebuilt with in-domain mixed IT–EN text
  (dictation about work and technology, dense in acronyms, brands, italianized
  verbs and elisions like `l'SDK`) raises S1-dev ETA without hurting WER, at the
  same LM bytes.
- **Generator:** an open-weight LLM whose licence leaves outputs free; never an
  API whose terms restrict outputs. Pin the model, prompts and seeds; commit the
  prompts.
- **Leakage control:** seed prompts only with training-side terms (Moth's
  `data/en_terms.txt`, the gold train scripts), never with S1-dev, T or A
  sentences or their term lists; drop every generated sentence that shares a
  normalized word 4-gram with an S1-dev, T or A script; report the drop counts.
- **Build:** the A3 recipe (Moth E-031/D41: BPE trigram, bigrams and trigrams
  ≥ 2, term sentences ×3) plus the generated text, pruned to A3's size
  (5,381,992 B) — or to the packed budget. Control: A3. Before building,
  subtract every evaluation sentence from the corpus with
  `tools/clean_lm_corpus.py` (P2: S1-dev was 900/960 in the corpus) and keep
  `pool.jsonl` out of future corpora.
- **Calibration:** an E-009-style grid per LM on the E-017 set, then the S1-dev
  veto.
- **Gate:** on S1-dev, ETA better with the CI excluding zero and no WER
  regression, or WER better with the CI excluding zero and no ETA regression;
  acronym and brand ETA reported; dev-macro without a significant regression.
- **Risk:** S1-dev and the gold scripts come from a script pipeline that may be
  LLM-assisted, so an LLM-text LM can look better on S1-dev than on real
  dictation. T-600 at release and, later, unscripted speech (E-019 arm B audio
  with reviewed transcripts) check it.

### E-021 — Targeted acronym, brand and elision data

- **Evidence:** acronym ETA 29.2 %, brand 48.1 %, acronym insertions 4.70 per
  100 words; elided acronyms not improved over 0.8.1 (34 vs 32 misses).
- **Arm 1, recordings:** 300–500 new S1 sentences in a new session, dense in
  acronyms (read with Italian and with English letter names), brands, and
  `l'` / `dell'` / `all'` + acronym; same pipeline and dedupe as E-017.
  Moth D42 is the warning: an S1-only term addendum raised rare-term ETA but
  regressed S3 and FLEURS, so the share stays moderate and T-S3 is read at
  release.
- **Arm 2, TTS:** the same kind of sentences from permissively licensed
  multi-voice TTS (no cloning of S1/S2/S3), at most 10–20 % of the arm's audio.
- **Gate:** a paired readout on the acronym + brand subset of S1-dev
  (`--mode mixed` on a filtered manifest, 226 clips) with ETA better and the CI
  excluding zero, plus no regression on full S1-dev (E-002 §6) or dev-macro.

### E-022 — Keyterm biasing re-test (conditional, product feature)

- **Prior:** negative. Moth E-027's 24-term glossary moved T-600 WER by
  −0.03 pp and left the rare-term rows unchanged.
- **Why re-test:** MOTH-50's term misses are mostly substitutions (335 of 501)
  rather than deletions, so the right term may now be in the beam more often,
  and frontier APIs sell exactly this feature.
- **Step 1, upper bound (report only):** a per-clip oracle list (the clip's own
  English terms) with the decoder's `ContextualBias` at Moth's bonus 1.0 /
  prefix 0.25 on the E-017 set. If acronym + brand ETA does not rise by
  ≥ 5 points, stop: a realistic list can only do worse.
- **Step 2:** oracle terms plus 500 distractors from the training-side term
  list; tune the bonus on the E-017 set; report S1-dev ETA and false insertions
  on `control_it`.
- **Scope:** a user glossary in the product, 0 bytes (`web/ort.js` already takes
  `contextualSequences`). Never part of a gate readout, because the list
  contains the answers.

### E-023 — R1: E-011 with encoder LR 3e-5 (recipe ablation, H1)

**Status: adopted 2026-11-23 (D27); both seeds complete, the pre-registered
gate failed only on 2b, and R1 is accepted by the dated amendment E-002 §9
(D26).** E-017 is done and frozen (`manifests/R-S1-cal.jsonl`,
`data/shards/s1-cal`, E-002 §8), the control E-005 + E-034c is 13.9212 % WER /
75.6291 % ETA on it. Seed mean (`runs/E-023/r1-s{0,1}`, 40,000 steps each,
15.78/15.80 h): dev-macro **14.3578 %** (−2.6320 pp, pass), S1-dev
**13.7911 / 74.6826** (pass), R-S1-cal **12.6679 / 75.4305** (pooled ETA tie;
`control_en` −1.83 and `switch` −2.43 vs E-005 miss the 0.5 pp clause, now
report-only). **Seed 0 is the product**; the T-600 guard passed
(12.2993 / 78.1400 vs 13.7041 / 76.1266). `runs/E-023/RESULT.md`.

- **Why:** P6 re-read E-011 with E-034c and failed on clean-S1-dev ETA
  (−1.7578 pp [−3.1844, −0.1422] on the seed mean) while dev-macro passed
  (−1.6202 pp). P7 found no support for H2 as the primary cause: on the gold
  train set E-005 is already at 0.2018 % WER and E-011 at 0.0831 / 0.1424 %, so
  over-exposure is saturated; the ETA loss is broad substitutions of short,
  common English words. H1 (encoder LR 1e-4 for ~3 epochs) is the first
  untested lever.
- **Arm R1:** the E-011 recipe exactly — base+6, 40,000 steps, EMA 0.9998, gold
  ×50, same mixture — with **one variable: `lr_encoder` 1e-4 → 3e-5**
  (`lr_head` stays 3e-4). Config `training/configs/base6-e023-r1.toml`.
- **Seeds and order:** seed 0 first (~16 h). Run seed 1 only if seed 0 is
  within 0.5 pp of E-011 on dev-macro and the E-017 ETA recovery in (b) holds.
  Two seeds for the gate.
- **Control:** the existing E-011 s0/s1 checkpoints and P6 readouts
  (`runs/E-011/p6/`) decoded with E-034c; no new control training. E-005 +
  E-034c is the product reference.
- **Ranking:** E-017 calibration set only — never S1-dev, never T. Report
  overall WER/ETA and `control_en`, `switch`, `single_word`, the categories
  that failed in P6. Control on R-S1-cal (read 2026-11-21): **13.9212 % WER /
  75.6291 % ETA** (`runs/E-017/control-e005-e034c.json`). The readout's
  `gate_eligible` flag stays false with the product LM; use the `paired` CIs.
- **Gate (seed mean, paired bootstrap 1,000, seed 0):**
  1. dev-macro **≥ 1.0 pp** better than E-005 + E-034c (16.9897 %) with the CI
     excluding zero — i.e. it keeps most of E-011's −1.62 pp.
  2. E-017 pooled ETA is not a regression against E-005 + E-034c
     (control 75.6291 %; CI not entirely below zero), and the `control_en` +
     `switch` ETA point estimates are within 0.5 pp of E-005 (their CIs are
     reported; the category sample is too small for a CI gate).
  3. S1-dev veto on the seed mean: WER CI not entirely above zero and ETA CI
     not entirely below zero (E-002 §6.2).
  4. INT8 cost ≤ 0.2 pp against the FP32 readout.
- **Outcome:** PASS → R1 is the final-recipe candidate, the seed is chosen on
  E-017, and P11 may run. FAIL → R2 (E-011 with gold ×20, H2) under a dated
  amendment; P11 only after a passing recipe.
- **Cost:** ~16 h per seed on the M5; ~16 h for the screen, ~32 h for two.
- **Risk:** 3e-5 may under-train the encoder and lose the dev-macro gain. If it
  does, H1 is not the lever and H3/gold (R2) is next.

**Result (both seeds, 2026-11-23).** 40,000 steps each in 15.78 / 15.80 h;
best greedy dev 17.0771 % (s0) and 17.3026 % (s1), vs E-011 18.3373 / 18.6531.
INT8 + E-034c seed mean: dev-macro **14.3578 %**
(−2.6320 pp [−2.8253, −2.4515] vs E-005, gate 1 pass); S1-dev
**13.7911 % / 74.6826 %** (WER −1.1833 [−1.7660, −0.5459], ETA +0.7568
[−0.5000, +2.1149], veto pass); R-S1-cal **12.6679 % / 75.4305 %** (WER
−1.2534 [−2.3767, −0.0446], ETA −0.1987 [−2.2603, +1.8844], pooled pass).
Against E-011 on the same cal set, R1 recovers ETA 74.7682 → 75.4305,
`control_en` 84.34 → 86.50, `switch` 68.41 → 71.18. The formal 0.5 pp clause
against **E-005** still misses (`control_en` −1.83, `switch` −2.43; both CIs
include zero), so the pre-registered gate **FAILS on 2b**. Gates 1, 2 pooled,
3 and 4 (dev-macro) pass; INT8 cost 0.0232 pp dev-macro, 0.2910 pp cal WER,
0.0 cal ETA. R2 or a dated amendment accepting R1 is the next decision.
`runs/E-023/RESULT.md`.

**Amendment (2026-11-23, E-002 §9, D26; post-hoc).** The owner amended rather
than ran R2. Clause 2b becomes report-only: it contradicted E-002 §8.4, and
0.5 pp was 1.5 of 300 words on `control_en` and 0.72 of 144 on `switch`, below
the seed spread of the set. R1 passes and is the final recipe; R2 as written
is withdrawn. **Seed 0** is chosen on R-S1-cal pooled ETA (75.4967 vs
75.3642 %). The release guard, fixed before the readout, was one declared
repeated read of T-600 against the E-005 + E-034c hypotheses: **12.2993 % /
78.1400 %** vs 13.7041 / 76.1266, WER −1.4048 pp [−2.3455, −0.4276] and ETA
+2.0134 pp [−0.1921, +4.1837]; no regression, so **R1 seed 0 is adopted as
`0.9.7-e023-r1` (D27)**. Speakers and categories are report-only. P11 is no
longer a prerequisite; seed 1 was not read on T-600 or FLEURS.

### E-008 addition — ceilings for the paper

Read T-600 once with the E-018 winner and, evaluation-only, with one or two
frontier APIs, after MOTH-50's final release. Report-only, never selection.
S2/S3 audio goes to a third party only if their consent covers it; otherwise
report S1 only.

## Not planned

- **INT4 weights.** ONNX Runtime Web's WASM `MatMulNBits` dequantizes the
  weights on every call (onnxruntime issue #33795), and 4-bit quality at 40 M
  parameters is unproven. Revisit only if the ceiling still binds after E-004.
- **Whisper-small (whole or truncated), transducer, AED, from-scratch
  Conformer:** unchanged from D8/D11.
- **More English-only data:** Moth E-012 made the mixed set 5.16 points worse.
- **An LLM decoder, or a FastConformer/TDT student trained from scratch**
  (the frontier recipes): they need billions of parameters or thousands of GPU
  hours; see `docs/frontier-stt-2026-11-19.md` §3. Revisit a distilled student
  only if E-019 shows the Whisper-base encoder is the ceiling.
- **Training on commercial-API outputs** (labels, text or LM data): their terms
  generally forbid it, and the model must stay commercially usable (D24).
