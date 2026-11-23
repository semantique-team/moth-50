# MOTH-50 — scaffold

**Status (2026-11-23): the product is E-023 R1 seed 0 + the E-034c LM (D27).**
R1 is the E-011 recipe with `lr_encoder` 3e-5. Seed mean: dev-macro **14.36 %**,
S1-dev **13.79 % / 74.68 %**, R-S1-cal **12.67 % / 75.43 %**. The T-600 release
guard passed: **12.30 % / 78.14 %** vs **13.70 / 76.13** for E-005 + E-034c
(−1.40 pp WER [−2.35, −0.43], +2.01 pp ETA [−0.19, +4.18]). The package
**`0.9.7-e023-r1`** is **49,203,776 B** (796,224 B of room), graph
`85abd830…`, browser exact at **RTFx 28.48** (4 threads, SIMD, isolated, load
70.1 ms). FLEURS, report-only: **IT 11.83 %**, **EN 28.20 %** (vs 14.82 /
30.27 for E-034c and 20.16 / 49.81 for 0.8.1). **E-005 + E-034c**
(`0.9.6-e034c`, D25) is retired from the product line and remains the
historical reference.

E-034c had replaced E-034 B after P2 found that 900 of the 960 S1-dev cues were
exact sentences of the E-031 LM corpus, so the advertised E-034 B gain (S1-dev
11.63 % / 79.83 %) was mostly memorisation. On the clean corpus **E-034c** was
dev-macro **16.99 %**, S1-dev **14.97 % / 73.93 %**, T-600 **13.70 / 76.13**;
against the A3 recipe rebuilt on the same corpus it was −0.64 pp macro, −0.44 pp
WER / +1.12 pp ETA on S1-dev and −0.49 / +1.06 on T-600 (the honest gain:
~0.4–0.5 pp WER, ~1.1 pp ETA). The E-002 A3 gate remains the historical
reference: dev-macro **17.62 %** (INT8) / **17.56 %** (FP32) against **23.02 %**
for 0.8.1; S1-dev **14.05 % / 14.00 %**, ETA **75.5 / 75.3 %** (inflated by the
same leak; the paired A3-vs-0.8.1 verdict stands). E-012 cut that package to
**44.77 MB** with the 16 kHz VAD and the lossless packed LM. The first E-005
readout used the pre-A3 LM and Moth's NOTICE; the erratum is in
`runs/E-005/RESULT.md` (D17). **E-016/P4 (2026-11-23, D28/D29)** vendored the
web runtime at `0e15442` in `vendor/moth-stt-web/` and measured **10.93× /
19.74× / 21.17×** at 1 thread without cross-origin isolation (gate ≥ 8×). On
S1-dev the upstream browser default was **+0.72 pp** WER vs the Python readout
(VAD segmentation, CI excludes zero), while `segment: false` was a **tie**
(+0.35 pp). **E-016b** adopted a one-file rule: split only at **three or more
VAD runs** for clips that fit one 30 s segment, two or more for longer audio.
The browser S1-dev WER is now **13.7180 %** vs Python 13.3820 % (+0.34 pp
[−0.13, +0.79], tie); a blanket auto-off was rejected because it lost 2.14 pp
on a five-sentence diagnostic. **P8 (E-024)** built the unpruned E-034 LM with
×3 term weighting: byte-identical R-S1-cal hypotheses (12.8469 / 75.4967) and
a tie on S1-dev (13.3966 / 75.0977), so it was **not adopted**; E-034c stays
(D30). **E-018** found no qualifying mixed IT–EN teacher: Parakeet 9.95 % WER
and Whisper/Canary 9.92 / 13.60 %, all below the ETA/italianized gates;
Voxtral was excluded for cost on the 16 GB M5. E-019 therefore stays
monolingual Italian (D31). The plan is `experiments/next-experiments.md`;
E-004, E-009–E-012, E-016–E-018, E-023 and E-024 have run, the rest has not.
Read `STATUS.md` first when resuming.

E-009 (2026-11-19) tested a lighter decoder: 0.45 / 2.0 improved dev-macro by
0.28 pp but regressed S1-dev by 0.58 pp WER and 1.12 pp ETA, so the frozen
0.55 / 2.7 stays. `runs/E-009/RESULT.md`. E-012 (2026-11-19) adopted the
16 kHz-only Silero graph (bit-identical on 2,040 real clips) and the lossless
packed A3 LM (3,353,840 B, −2.03 MB). E-010 (2026-11-19) screened the encoder LR:
**1e-4 beat 3e-5** by 1.33 pp on base+4 and 1.07 pp on base+6 (greedy dev
**32.39 %** at 4,000 steps). E-011 (2026-11-21) ran that recipe long: dev-macro
**15.86 / 15.79 %** on two seeds vs 17.56 % for E-005, but S1-dev WER was a tie
and seed 0 lost **2.20 pp ETA** (CI excluding zero), so **E-011 was not adopted**.
P6 re-read it with E-034c (2026-11-21): dev-macro **−1.62 pp** but clean S1-dev
ETA **−1.76 pp** on the seed mean (CI excludes zero), so the veto holds
(`runs/E-011/p6/RESULT.md`). P7 found gold memorisation saturated in both models
(E-005 0.20 %, E-011 0.08 / 0.14 %) and the ETA loss spread over short English
words, so the first recipe ablation is **E-023 R1** (encoder LR 3e-5),
pre-registered and **authorized 2026-11-21** (`experiments/next-experiments.md`).
Seed mean (both seeds, 40,000 steps, 15.78/15.80 h): dev-macro **14.3578 %**
(−2.63 pp vs E-005), S1-dev **13.7911 / 74.6826**, R-S1-cal **12.6679 /
75.4305**. The cal `control_en`/`switch` 0.5 pp clause missed (CIs include
zero); the dated, post-hoc amendment E-002 §9 (D26, 2026-11-23) makes it
report-only, as §8.4 already required, and accepts R1. **Seed 0** is the product
seed: the pre-registered T-600 release guard passed on 2026-11-23 (12.2993 /
78.1400 vs 13.7041 / 76.1266, no WER or ETA regression) and **D27 adopts it as
`0.9.7-e023-r1`**. E-017 is
done: the 400-cue `R-S1-cal` set is reviewed, recorded (400/400, 16.95 min),
level-matched to S1-dev (−16.58 dB), built into `manifests/R-S1-cal.jsonl`
(`split: dev`) and `data/shards/s1-cal`, and frozen under **E-002 §8**
(`../moth-cs-dataset/text-2026-11-cal/`, `runs/E-017/RESULT.md`). The control
E-005 + E-034c was read once: **13.9212 % WER / 75.6291 % ETA**. E-004
arm B (2026-11-21) then used the E-012 headroom for LM coverage. Its
S1-dev number was withdrawn by P2; E-034c is the corrected candidate. E-011's
S1-dev absolute numbers are inflated by the same leak, but its veto compared
E-011 and E-005 with the same A3 LM. `runs/E-010/RESULT.md`,
`runs/E-011/RESULT.md`, `runs/E-004/RESULT.md`.

The frontier-model review (2026-11-19, `docs/frontier-stt-2026-11-19.md`, D24
proposed) keeps the architecture and adds E-017–E-022: a mixed calibration set,
a teacher bake-off, teacher labels, LLM-written LM text, targeted
acronym/brand data and a keyterm-biasing re-test. T-600 was read once in E-004;
the 2026-11-21 review (`docs/review-2026-11-21.md`) puts E-017 and the E-011
re-read with the product LM first.

**Name:** `MOTH-50` is the model and display name; `moth-50` is the repository
and package path. The previous model is referred to as **Moth-20** (released as
`moth-stt` 0.8.1); its repository name is deliberately unchanged, because it is
published, archived and cited.

## Goal

A **MOTH-50-class** Italian + English code-switching ASR: the full required
runtime package (acoustic INT8 + LM + VAD + metadata) must stay **≤ 50 MB**, run
on CPU and in the browser, and be trained on the 16 GB M5 Pro. The 100 MB tier
was evaluated and rejected. The model must remain commercially usable.

## Decisions (2026-11-17)

- **Architecture**: Whisper-base encoder + 6 extra pre-LN transformer blocks +
  linear CTC over the 1,024-piece BPE + E-031/E-034 LM + Silero VAD. **39.3 M
  acoustic parameters, 47.74 MB initial package** (measured graph size; the
  current product package is **49.20 MB** (`0.9.7-e023-r1`, E-034c LM + 16 kHz
  VAD, 0.80 MB under the ceiling); the A3 package is 44.77 MB).
  Same-budget challenger: base + 4 blocks + a 12 MB LM (~47.94 MB), decided by
  E-004. Memory: E-002 measured 13.37 GB without checkpointing and 3.68 GB with
  per-block checkpointing; checkpointing is the default (D15).
- **No 100 MB tier**; no from-scratch Conformer, transducer or AED backbone.
- **No streaming**: encoder-only CTC, record-then-transcribe, like Moth.
- **Data**: public CC0/CC-BY corpora, 2–5 h of new **S1** recordings, and
  commercially permissive TTS (Piper CC0/CC-BY voices, Kokoro-82M Apache 2.0).
  No voice cloning of S1/S2/S3. CS-FLEURS/SwitchLingua are CC-BY-NC and
  **test-only**, never training.
- **Evaluation**: T-600/T-S3 for comparability, FLEURS-IT/EN, CS-FLEURS-READ as
  an external mixed test. S1-only new data does not create a new speaker; do not
  claim unseen-speaker generalization beyond S2/S3/FLEURS.

## Baseline to beat (Moth 0.8.1, INT8 ONNX)

| Set | n | WER | ETA |
|---|---:|---:|---:|
| Mixed test T, all speakers | 600 | 24.08 | 55.23 |
| T-S3 (held out from training, repeatedly evaluated) | 200 | 24.96 | 51.27 |
| FLEURS-IT test (filtered) | 697 | 20.16 | — |
| FLEURS-EN test (filtered) | 525 | 49.81 | — |

Package 19,706,289 B; INT8 graph 11,910,677 B; 11.6 M acoustic parameters.
Decoder for the reference numbers: beam 12, top-k 10, LM weight 0.55, token
bonus 2.7, E-031 LM. See `../moth-stt/runs/E-031/release-0.8.1.json`.

## Data

Pin the recorded dataset at **`v1.0.0`** and verify it before training:

```bash
python ../moth-cs-dataset/scripts/verify.py
```

- `train` = R-S1 + R-S2 (1,413 clips, 57.6 min): the only recorded training split.
- `test` = T (600 clips, 23.2 min): evaluation only.
- `anchor` = A (180 clips, 7.0 min): speaker-vs-sentence control; never train.
- S3 never contributes to `train`. S3 informed Moth's development and is **not**
  a pristine unseen speaker.

The dataset is 1.46 h: enough for a benchmark and a fine-tuning seed, not for
training a larger model from scratch. The additional data plan is public
corpora, **2–5 h of new S1 recordings** (new sessions; one held-out S1 session
for dev), permissive TTS augmentation, and word/phrase-level splicing. There is
no new speaker: S2, S3 and FLEURS remain the generalization checks. Do not reuse
T as the only test of the new model. Do not train on CC-BY-NC data.

## What to reuse from moth-stt

Vendored in `vendor/moth-stt/` at commit
`f3ecd1b0f4f0c5c044fa637e5fb5dd8a0c798d97`:

- `moth_text.py` — the one text rule (`normalize`), so WER/ETA are comparable.
- `moth_manifest.py` — manifest schema and validation.
- `moth_features.py` — reference log-mel (parity with the browser runtime), with
  its Whisper mel filterbank in `assets/`.
- `moth_eval.py` — WER/ETA, per-category/speaker, bootstrap CI.
- `moth_release.py` — package validation and budget checks (raised to the
  50,000,000-byte ceiling; see `vendor/moth-stt/PROVENANCE.md`).

Do **not** copy `model.py`, `train.py` or `decode.py` blindly: the decoder
depends on `stt.hotwords` and `stt.lm`, and the model/training code is specific
to the 0.8.1 recipe. Copy them only with their dependencies, or import the
pinned checkout. Keep the evaluation semantics identical; that is the point of
the vendored files. The Moth pipeline hard-codes the Whisper-tiny encoder and
the 20 MB budget in several places; `docs/review-2026-11-17.md` §4 lists them.
The SentencePiece tokenizer (`moth_bpe.model`, sha256 `d6e1054e…f2fd`) is not in
the published release; it is pinned at `data/tokenizer/moth_bpe.model`.

## Layout

```
STATUS.md                 session handoff: done, running, next commands
README.md                 this file
pyproject.toml            dependencies, ruff and pytest configuration
data/README.md            dataset pin, splits, restrictions
data/sources.lock.json    pinned source versions, URLs and checksums
data/tokenizer/           pinned Moth 1,024-piece BPE model and provenance
docs/decisions.md         decisions and open questions
docs/architecture-options.md  architecture analysis and planned sequence
docs/review-2026-11-17.md evidence check, measurements and proposals
docs/review-2026-11-21.md post-E-004 review: selection target, C/D and final-run procedure
docs/frontier-stt-2026-11-19.md  frontier STT models: teachers, licences, limits
experiments/README.md     new experiment ledger, starts at E-001
experiments/E-001-baseline.md
experiments/E-002-protocol.md  frozen evaluation protocol and data mixture
experiments/E-002-s1-dev.md   S1-dev recording runbook and manifest patch
experiments/E-003-plan.md     pre-registered architecture screen and its result
experiments/next-experiments.md  post-E-005 plan: E-009–E-023, gates, commands
experiments/E-024-p8-lm-x3.md  P8 ×3-term LM pre-registration and negative result
runs/probe-2026-11-17/    measured INT8 size, CPU latency, activation memory
runs/probe-2026-11-19/    16 kHz Silero graph parity preflight
runs/E-001/               Moth 0.8.1 baseline reproduction
runs/E-002/               real 200-step MPS training probe
runs/E-003/, runs/E-005/  depth screen; long fine-tune, package, browser run
runs/E-023/               R1 recipe, amended gate, guard, product package
runs/E-016/               browser checks: RTFx, parity, vendored runtime smoke
runs/E-024/               P8 ×3-term LM: build, pack, cal and S1-dev readouts
runs/E-018/               teacher bake-off: pins, hyps, metrics, exclusion
tests/                    tests for tools/ and the vendored code
training/                 MOTH-50 trainer port (config, data, model, EMA, loop)
training/configs/         E-003/E-005 configs, the E-010 arm, smoke configs
training/fixtures/        tiny smoke manifests (audio not included)
tools/architecture_budget.py  parameter/size/compute estimator
tools/probe_export.py     real INT8 export size and CPU latency (needs moth-stt)
tools/probe_activations.py  activation memory, CPU or MPS (needs moth-stt)
tools/probe_training.py   200-step MPS training memory/throughput probe
tools/build_noisy_cv.py   noisy Common Voice half (E-024 mixer, MOTH-50 paths)
tools/s1_dev_qc.py        S1-dev session QC: takes, coverage, levels, projection
tools/finalize_s1_dev.py  patch build_manifest output to split:dev + session paths
tools/release_assets.py   pinned A3 LM, 0.8.1 graph and Silero graphs (fetch/check)
tools/frozen_readout.py   E-002 gate readout: beam 12 + A3 LM; dev-macro or a mixed set
tools/serve_web.py        serve a release with the vendored web runtime
tools/browser_probe.mjs   Playwright RTFx / S1-dev browser probe (E-016)
tools/browser_parity.py   browser-vs-Python identity and paired WER/ETA (E-016)
tools/e018_whisper.py     Whisper teacher runner (E-018)
tools/e018_teacher.py     Canary/Voxtral teacher runner (E-018)
tools/e018_summary.py     teacher table and selection rule (E-018)
tools/calibrate_decoder.py  E-009 LM-weight × token-bonus grid on dev
tools/export_release.py   INT8 release folder + MOTH-50 NOTICE + 50 MB package check
tools/release_notice.py   the MOTH-50 package NOTICE
tools/probe_vad.py        Silero graph parity probe (E-012)
tools/moth_lm.py          MOTHLMB2 writer/reader and runtime scorer
tools/pack_lm.py          lossless A3 → MOTHLMB2 packing with an exact round trip
tools/eta_breakdown.py    classify English-term misses of a mixed-set readout
tools/seed_mean.py        seed-mean paired comparison for the E-002 §7 gate
tools/lm_corpus_overlap.py  evaluation scripts vs the LM corpus (P2)
tools/clean_lm_corpus.py  drop evaluation sentences from an LM corpus
tools/build_lm_from_corpus.py  build a MOTHLMB1 LM from a corpus (E-034c, E-020)
tools/probe_lm_pack.py    packed LM format prototype (review P5)
tools/sample_resources.py  CPU/RAM/MPS sampler for long training runs
tools/resource_dashboard.py  self-contained SVG dashboard from logs + samples
vendor/moth-stt/          evaluation code copied from the frozen repo
vendor/moth-stt-web/      web runtime from Moth 0e15442; ort.js patched (E-016b/D29)
LICENSE                   Apache-2.0
NOTICE                    attribution
```

## First steps

0. Read `docs/architecture-options.md` (2026-11-17): the Moth evidence, the
   50 MB / 100 MB candidate configurations, the training-feasibility estimates
   and the planned E-003–E-008 sequence. Then read `docs/review-2026-11-17.md`
   for the corrected evidence and the proposals. `tools/architecture_budget.py`
   reproduces the parameter and package-size arithmetic; the probes in `tools/`
   reproduce the measurements.
1. Confirm the size/compute budget and the target deployment (CPU? browser?).
2. Choose the model family and tokenizer; record the decision in
   `docs/decisions.md`.
3. Run **E-001**: evaluate Moth 0.8.1 on the canonical dataset with the vendored
   evaluator and confirm 24.08 / 55.23 (or explain the difference).
4. Freeze the evaluation protocol **before** training: T for comparability,
   FLEURS, a held-out S1 session, and an external mixed test (CS-FLEURS-READ,
   test-only). There is no new speaker.
5. Scale the data, then train. Keep `runs/` and `EXPERIMENTS.md` with the same
   discipline as `moth-stt`.

## Rules

- No training on `test` or `anchor`; no S3 in `train`.
- No audio or checkpoints in git. The dataset lives in `moth-cs`; checkpoints
  are private/regenerable.
- One logical change per commit; conventional commits; `ruff` and `pytest`.
- The model card must state the evidence boundary (scripted read speech, known
  voices, no spontaneous code-switching, speaker coverage).
- Do not describe a tie as a win; report per-speaker ranges.
