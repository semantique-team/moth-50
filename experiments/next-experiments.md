# Post-E-005 experiment plan

**Status: proposed 2026-11-19; the E-005 addendum, E-009, E-012, E-010 and
E-011 ran on 2026-11-19/21, E-013–E-016 have not run.** This file pre-registers
the experiments that follow E-005, with their gates, so they cannot be moved
after seeing results. It comes from the 2026-11-19 review of the E-005 model and package
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
| 6 | E-004 | Where the freed bytes go | 3, 5 | LM builds + readouts | ≤ 50 MB |
| 7 | E-016 | Browser: 1-thread RTFx, browser-vs-Python WER | a release | ~1 h | 0 |
| 8 | E-013 | Extra blocks before `final_ln` | — | ~1.6 h MPS | 0 |
| 9 | E-014 | Conformer-style convolution in the extra blocks | — | ~1.6 h MPS | ≈ 0 |
| 10 | E-015 | 25 Hz extra blocks (P7) | only if E-016 fails | ~1.6 h MPS | 0 |
| 11 | E-006 | Data arms | 5 | days | 0 |
| 12 | E-008 | External evaluation | final model | hours | 0 |

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

The product readout that `docs/architecture-options.md` §7 pre-registered for
E-005 (T-600 ≤ 22 % WER and ≥ 58 % ETA, FLEURS-IT ≤ 18 %) is still **once**, as a
report: build the T and FLEURS mel shards and run `--mode mixed` with the retained
0.8.1 T-600 hypotheses
(`../moth-stt/runs/E-037/readout/paired-1/product/int8/t600.hyp.jsonl`) as the
reference. Never select or tune on it.

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
`runs/E-009/RESULT.md`.

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

## E-016 — Browser runtime checks

- **1-thread RTFx** without cross-origin isolation on 5 / 10 / 20 s clips. Gate:
  ≥ 8× (the E-005 target; projected 7–9×). E-005 only measured 4 threads (22.3×).
- **Browser vs Python WER** on S1-dev: the comparison Moth's E-041
  pre-registered and never ran. Gate: identical hypotheses, or every difference
  explained.
- **Runtime ownership:** vendor `moth-stt/web/` into this repository at local
  commits `b0fa11c` (`budget_bytes`) and `0e15442` (`MOTHLMB2` reader), so the
  package no longer depends on a local commit in an archived repository.

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

One variable at a time, each as a fine-tune from the E-011 checkpoint against a
control fine-tune of the same length on the unchanged mixture; judged on S1-dev
WER/ETA with dev-macro as a guard, S2/S3/FLEURS as diagnostics.

1. **VoxPopuli Italian transcribed** (91 h, CC0): the cheapest new real Italian
   speech not yet in the mixture.
2. **Gold repetition:** ×50 (current) vs ×20 plus a gentle gold-only stage at
   Moth E-028's LRs (1e-4 head / 1e-5 encoder): a 39 M model may memorize 1,413
   sentences seen 50 times per epoch.
3. **Word/phrase-level splicing** with forced alignment vs the utterance-level
   splices.
4. **Pseudo-labels:** Parakeet-TDT-0.6B-v3 (CC-BY-4.0) on unlabelled VoxPopuli
   Italian (CC0), confidence-filtered (review P4).
5. **New S1 training sessions** (2–5 h), never the S1-dev session.
6. **TTS** as a 10–20 % minority source (permissive voices only).

## Not planned

- **INT4 weights.** ONNX Runtime Web's WASM `MatMulNBits` dequantizes the
  weights on every call (onnxruntime issue #33795), and 4-bit quality at 40 M
  parameters is unproven. Revisit only if the ceiling still binds after E-004.
- **Whisper-small (whole or truncated), transducer, AED, from-scratch
  Conformer:** unchanged from D8/D11.
- **More English-only data:** Moth E-012 made the mixed set 5.16 points worse.
