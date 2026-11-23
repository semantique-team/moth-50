# MOTH-50 — session status and handoff (2026-11-23)

This is the first file to read when resuming. It records what is done, what is
running, and the exact next commands. Update it at the end of each session.

- **Remote:** `semantique-team/moth-50`, branch `main`
- **Current product (D27, 2026-11-23): E-023 R1 seed 0 acoustic + E-034c
  LM.** Package **49,203,776 B** (796,224 B of room), version
  `0.9.7-e023-r1`, INT8 graph `85abd830…`, packed E-034c LM `7a4332f1…`,
  16 kHz VAD. The T-600 release guard passed: **12.2993 % / 78.1400 %** vs
  13.7041 / 76.1266 (−1.4048 pp WER [−2.3455, −0.4276], +2.0134 pp ETA
  [−0.1921, +4.1837]). Browser exact at **RTFx 28.48** (4 threads, isolated,
  SIMD, load 70.1 ms). FLEURS once, report-only, vs E-034c: IT **11.8304 %**
  (−2.9935 pp), EN **28.1955 %** (−2.0720 pp). **E-034c** (`0.9.6`, D25) is
  retired from the product line and stays the historical reference: dev-macro
  **16.9897 %**, S1-dev **14.9744 / 73.9258**, T-600 **13.7041 / 76.1266**,
  browser RTFx 29.35. `runs/E-023/RESULT.md`, D27.
- **E-023 R1 both seeds complete (2026-11-23); 2b missed, accepted by
  amendment (E-002 §9, D26), then adopted after the guard (D27).** Seed mean:
  dev-macro **14.3578 %** (−2.6320 pp [−2.8253, −2.4515], pass); S1-dev
  **13.7911 % / 74.6826 %** (WER −1.1833, ETA +0.7568, pass); R-S1-cal
  **12.6679 % / 75.4305 %** (WER −1.2534, ETA tie). On cal, `control_en`
  **86.50** vs 88.33 (−1.83) and `switch` **71.18** vs 73.61 (−2.43) missed
  the 0.5 pp clause; §9 makes it report-only (it contradicted §8.4 and was
  1.5 / 0.72 words). R2 is withdrawn. **Seed 0** (cal ETA rule) is the product
  seed; T-600 guard passed, seed 1 is never read on T or FLEURS.
  `runs/E-023/RESULT.md`.
- **E-016/P4 done (D28, 2026-11-23).** 1-thread non-isolated RTFx **10.93× /
  19.74× / 21.17×** on 5 / 10 / 20 s (gate ≥ 8×). Browser vs Python on S1-dev:
  default **+0.7159 pp** WER [+0.1678, +1.2599] (675/960 identical),
  `segment: false` **+0.3506 pp** [−0.1255, +0.8160] (**tie**, 697/960);
  Python-FFT 16 kHz input is not better, so the cause is the runtime's VAD
  segmentation, not resampling. The residual differences are numerical
  near-ties (Moth E-022: 70–77 % identity, WER tie). The web runtime is
  vendored byte-identical at `0e15442` in `vendor/moth-stt-web/` with
  `tools/serve_web.py`, `tools/browser_probe.mjs` and
  `tools/browser_parity.py`. **Finding:** the browser default costs ~0.4–0.7 pp
  WER on S1-dev; E-016b (auto-disable segmentation for clips that fit one
  30 s segment, then re-read) is proposed, not applied. `runs/E-016/RESULT.md`.
- **E-005 A3 gate (historical):** `base+6`, 16,000 steps, pinned A3 LM. Frozen
  beam+A3 dev-macro **17.5565 % FP32 / 17.6212 % INT8** against the **23.0170 %**
  reference (paired **−5.4605 / −5.3957 pp**, CIs exclude zero). Held-out S1-dev
  (960/960 takes, 51.30 min): **13.9956 / 14.0541 %** vs **22.2498 %** for 0.8.1,
  ETA **75.34 / 75.54 %** vs **62.11 %**, no WER or ETA regression outside its
  CI. A3 package **47,836,502 B (47.84 MB)**, **2.16 MB under** the 50 MB
  ceiling. Browser exact at **RTFx 22.3** (4 threads, SIMD, isolated). The
  S1-dev absolute numbers (and ETA) were inflated by the P2 pool leak; the
  paired A3-vs-0.8.1 comparison stands. `runs/E-005/RESULT.md`.
- **Erratum (2026-11-19, D17):** the first E-005 readout/package used the
  **pre-A3 LM** (`dcf1d59f…`) and Moth 0.8.1's NOTICE. The verdict stood, and
  the final A3 numbers above supersede the old ones. The tools pin A3 by sha256
  (`tools/release_assets.py`).
- **Plan:** `experiments/next-experiments.md` pre-registers E-009–E-022.
  **E-009 kept the frozen decoder; E-012 passed both gates (16 kHz VAD + lossless
  MOTHLMB2, D19/D20); E-010 chose encoder LR 1e-4 (D21); E-011 won dev-macro but
  failed the S1-dev ETA rule on seed 0 and was not adopted (D22); E-004 arm B
  moved the product LM to E-034 N2_1_1 (D23). E-013–E-015 have not run.
  E-017 is done (400-cue `R-S1-cal` reviewed, recorded, level-matched,
  manifest + shards built, E-002 §8 written); E-018–E-022 have not run.
  E-023 R1 is adopted (E-002 §9 + D26, then D27 after the T-600 guard);
  E-016/P4 is done (D28, web runtime vendored).**
  **P0 (cross-machine parity) passed** byte-identical; **P1 merged** the frontier
  branch (D24); **P2 found the S1-dev leak** and built E-034c. No training starts
  without an explicit go-ahead. Next: **P8** (×3-term unpruned LM on R-S1-cal),
  **E-018** (teacher bake-off), the proposed **E-016b** segmentation re-read,
  then E-019–E-022 and E-006.
- **Review 2026-11-21 (proposed):** `docs/review-2026-11-21.md`. dev-macro is
  the wrong ranking target; the LM is the lever and C no longer fits next to the
  E-034 LM; the E-011 ETA loss looks recipe-level. Procedure P0–P11: cross-machine
  parity (**P0 passed**), an S1-dev/LM-corpus overlap check (**P2 found 900/960
  exact matches**), E-011 re-read with the product LM (**P6 failed**), D only
  LM-first, one recipe ablation (**E-023 R1 passed and was adopted**).
- **E-012 package (historical A3):** **44,770,433 B** (5,229,567 B of room) with
  the 16 kHz VAD and the packed A3 LM; dev-macro and S1-dev hypotheses are
  byte-identical to the MOTHLMB1 runs. Browser: exact transcript, RTFx 24.63.
  Superseded by E-004 B (49.21 MB); its S1-dev evidence is withdrawn by P2. `runs/E-012/`.
- **E-011:** two seeds, dev-macro **15.8584 / 15.7945 %** vs E-005 17.5565 %
  (−1.70/−1.76 pp, CIs exclude zero), but S1-dev ETA **73.1445 / 73.9746 %** vs
  75.3418 %; seed 0's **−2.20 pp** CI excludes zero (seed 1 −1.37 pp includes
  zero). **Not adopted**; E-011 was superseded by R1 (D26/D27) and its S1-dev
  absolute numbers were inflated by the P2 leak, but the paired E-011-vs-E-005
  comparison used the same A3 LM. `runs/E-011/RESULT.md`.
- **Tests:** 121 passed in the training env, 109 passed + 4 skipped (torch,
  `stt.lm` parity and soundfile) in a plain venv; `ruff check` and
  `ruff format --check` clean.
- **Tokenizer:** pinned at `data/tokenizer/moth_bpe.model`
  (sha256 `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`).

## What is done

| Item | State | Where |
|---|---|---|
| E-001 baseline reproduction | done: 24.0826 WER / 55.2253 ETA, matches E-038 | `runs/E-001/`, `experiments/E-001-baseline.md` |
| E-002 protocol | frozen, §6 amendment 2026-11-19 | `experiments/E-002-protocol.md` |
| E-002 memory probe | 200 steps: 13.368 GB no-ckpt / 3.680 GB ckpt; no non-finite | `runs/E-002/RESULT.md` |
| E-002 dev reference | 23.0170 % macro (MLS-IT 28.1040, LS-EN 17.9299) | `runs/E-002/dev-reference.json` |
| Tokenizer pin (D14) | vendored and tested | `data/tokenizer/`, `docs/decisions.md` |
| S1-dev script | 960 cues, all reviewed: 900 short-form + 60 long (0901–0960) | `../moth-cs-dataset/text-2026-11-dev/` |
| S1-dev recording | **960/960 WAV, 51.30 min**, mean 3.21 s, no hard QC errors | `../moth-cs-dataset/audio/R-S1-dev/S1/` |
| S1-dev QC | 960/960, 0 errors, 2 advisory short takes (563, 886) | `runs/E-002/s1-dev-qc.json` |
| S1-dev manifest | 960 rows, `split: dev`, `audio/R-S1-dev/` | `../moth-cs-dataset/manifests/R-S1-dev.jsonl` |
| S1-dev mel shards | 960 clips, 1 shard, 0 errors | `data/shards/s1-dev/` |
| Public data | downloaded, verified, shards built | `data/raw/`, `data/shards/` |
| Trainer port | base+6, pretrained init, identity heads, checkpointing, resume | `training/`, `docs/decisions.md` D16 |
| E-003 preflight | **ready: true**, train guard ok | `runs/E-003/preflight.json` |
| E-003 base+0 | **done**: greedy dev 48.12 %, 56.2 min, peak 7.397 GB | `runs/E-003/base+0/` |
| E-003 base+2 | **done**: greedy dev 36.90 %, 70.7 min, peak 8.406 GB | `runs/E-003/base+2/` |
| E-003 base+4 | **done**: 34.73 % / 34.74 % (2 seeds), ~85 min, peak 9.525 GB | `runs/E-003/base+4*/` |
| E-003 base+6 | **done, winner**: 33.46 % / 33.55 % (2 seeds), ~99 min, peak 8.594 GB | `runs/E-003/base+6*/` |
| E-005 base+6 long | **done**: 16,000 steps, greedy dev 21.7540 %, 6.73 h, peak 8.604 GB | `runs/E-005/base+6-16k/` |
| E-005 A3 readout | **gate PASS**: 17.5565 % FP32 / 17.6212 % INT8 vs 23.0170 | `runs/E-005/readout-a3*.json` |
| E-005 A3 INT8 export | **done**: 47,836,502 B (47.84 MB), 2.16 MB headroom | `runs/E-005/release-a3/`, `export-report-a3.json` |
| E-005 A3 browser run | **done**: exact transcript, RTFx 22.3 (4 threads, SIMD, isolated) | `runs/E-005/browser-run-a3.json` |
| E-009 decoder grid | **done, no adoption**: dev-macro 0.45/2.0 17.3427 % (−0.2786 pp), but S1-dev veto +0.5844 pp WER / −1.1230 pp ETA; frozen 0.55/2.7 stays | `runs/E-009/` |
| **S1-dev no-regression** | **PASS**: 13.9956 % FP32 / 14.0541 % INT8 vs 22.2498; ETA +13.23/+13.43 pp | `runs/E-005/s1-dev-a3-*.json` |
| Release asset pins (D17) | A3 LM, 0.8.1 graph, Silero graphs pinned and fetched by sha256 | `tools/release_assets.py`, `data/sources.lock.json` |
| S1-dev readout tool | `frozen_readout.py --mode mixed`: WER/ETA + paired no-regression vs 0.8.1 | `tools/frozen_readout.py`, E-002 §6 |
| Post-E-005 tooling | decoder grid (E-009), EMA, E-010 config, VAD probe, LM packing (E-012), term-error breakdown | `experiments/next-experiments.md` |
| VAD 16 kHz preflight | bit-identical on 23 synthetic clips, −1,037,921 B | `runs/probe-2026-11-19/` |
| S1-dev term-error analysis | 501 misses (0.8.1: 776); deletions 151 → 53; acronyms 29.2 % / brands 48.1 % ETA; elided acronyms not improved | `runs/E-005/RESULT.md`, `runs/E-005/eta-breakdown.json`, `tools/eta_breakdown.py` |
| Frontier-model review (D24, proposed) | teachers, LM text and ceilings, not architectures; E-017–E-022 pre-registered | `docs/frontier-stt-2026-11-19.md` |
| E-012(a) VAD 16 kHz | **done, default**: 2,040 real clips bit-identical; browser exact, RTFx 23.90; 46,798,588 B package | `runs/E-012/` |
| E-012(b) LM packing | **done, default**: A3 → 3,353,840 B lossless; dev/S1-dev hypotheses byte-identical; browser RTFx 24.63 | `tools/moth_lm.py`, `tools/pack_lm.py`, `runs/E-012/` |
| E-012 final package | **44,770,433 B**, 5,229,567 B of room; base+7 now fits (est. 47.88 MB) | `runs/E-012/export-report-mothlmb2.json` |
| E-010 base+4, 1e-4 | **done**: greedy dev 33.4003 % vs 34.7306 at 3e-5 (−1.33 pp) | `runs/E-010/base+4-enc1e-4/` |
| E-010 base+6, 1e-4 | **done, winner**: 32.3912 % vs 33.4588 at 3e-5 (−1.07 pp) | `runs/E-010/base+6-enc1e-4/`, `runs/E-010/RESULT.md` |
| E-011 config | preflight ready: base+6, 1e-4, 40k steps, EMA 0.9998, 2 seeds | `training/configs/base6-e011.toml` |
| E-011 training | **done**: seed 0 18.3373 %, seed 1 18.6531 % greedy; 0 non-finite | `runs/E-011/base+6-e011-s0/`, `...-s1/` |
| E-011 gate | **dev PASS, not adopted**: FP32 15.8584/15.7945 vs 17.5565; S1-dev seed 0 ETA −2.20 pp (CI excludes 0) | `runs/E-011/RESULT.md` |
| E-004 arm B | **adopted 2026-11-21; S1-dev evidence withdrawn (P2)**: E-034 N2_1_1 packed lossless; T-600 13.4748/76.5101; S1-dev 11.6289/79.8340 was 900/960 cue leakage | `runs/E-004/RESULT.md` |
| E-004 package | **49,210,224 B**, 789,776 B of room; graph byte-identical to E-005; browser RTFx 28.36; S1-dev evidence withdrawn (P2) | `runs/E-004/export-report-e034.json` |
| P0 cross-machine parity | **passed**: A3 INT8 S1-dev hypotheses byte-identical (`d43af295…`) to E-005; WER/ETA 14.0541 / 75.5371 | `runs/E-004/parity-a3-s1-dev.*` |
| P1 frontier branch | **merged**: E-017–E-022, `eta_breakdown` + tests, D19 renumbered D24, order table rebased | `docs/frontier-stt-2026-11-19.md`, `experiments/next-experiments.md` |
| P2 LM-corpus overlap | **leakage**: 900/960 S1-dev exact matches (pool.jsonl is a corpus `train_script`); S1-dev 4-grams 61.16 % vs T 9.75 %; 60 long cues clean | `runs/E-004/lm-corpus-overlap*.json` |
| E-034c clean rebuild | **done, product (D25)**: corpus 185,103 rows; dev-macro 16.9897; S1-dev 14.9744/73.9258; T-600 13.7041/76.1266; packed 7,787,184 B | `runs/E-004/lm-build-e034c.json`, `runs/E-004/RESULT.md` |
| E-034c package/browser | **done**: 49,203,774 B (796,226 B room), graph `14ba2034…`, load 68.7 ms, exact transcript, RTFx 29.35 | `runs/E-004/export-report-e034c.json`, `runs/E-004/browser-run-e034c.json` |
| P6 E-011 + E-034c | **FAIL on S1-dev ETA**: dev-macro −1.6202 pp (pass); S1-dev WER −0.5040 (tie), ETA −1.7578 [−3.1844, −0.1422]; s0/eta −2.64, s1 −0.88; not adopted | `runs/E-011/p6/RESULT.md` |
| P7 E-011 diagnostics | **done**: gold WER E-005 0.2018 %, s0 0.0831, s1 0.1424 (H2 saturated); ETA misses +54 / +18, mostly short-English substitutions; D2 blocked on E-017 | `runs/E-011/p7/RESULT.md` |
| E-023 R1 | **adopted (D27)**: both seeds complete; 2b report-only by E-002 §9 (D26); seed mean dev-macro 14.3578 % (−2.63 pp), S1-dev 13.7911/74.6826, cal 12.6679/75.4305; seed 0 chosen on cal ETA | `runs/E-023/RESULT.md`, `runs/E-023/p6/` |
| E-023 T-600 guard | **PASS**: 12.2993 % / 78.1400 % vs 13.7041 / 76.1266; WER −1.4048 pp [−2.3455, −0.4276], ETA +2.0134 pp [−0.1921, +4.1837]; declared repeated read, speakers/categories report-only | `runs/E-023/t600-s0.json/.hyp.jsonl` |
| E-023 package/browser | **done, product**: 49,203,776 B (796,224 B room), graph `85abd830…`, load 70.1 ms, exact transcript, RTFx 28.48 (4 threads) | `runs/E-023/export-report-0.9.7.json`, `runs/E-023/browser-run-0.9.7.json` |
| E-023 FLEURS | **done, report-only**: IT **11.8304 %** (−2.9935 pp vs E-034c), EN **28.1955 %** (−2.0720 pp); no ETA | `runs/E-023/fleurs-{it,en}-s0.*` |
| E-016/P4 browser | **done (D28)**: 1-thread non-isolated 10.93/19.74/21.17× (pass); S1-dev default +0.72 pp WER (no tie), `segment:false` +0.35 pp (tie); runtime vendored at `0e15442` | `runs/E-016/RESULT.md`, `vendor/moth-stt-web/`, `tools/serve_web.py` |
| P3 FLEURS | **done, report-only**: IT **14.8238 %** (−5.3383 pp vs 0.8.1, CI excludes 0), EN **30.2675 %** (−19.5401 pp); FLEURS-IT meets ≤18 % | `runs/E-004/fleurs-RESULT.md` |
| E-017 R-S1-cal | **done, frozen**: 400 cues reviewed; 400/400 takes, 16.95 min, 19 retakes; raw +16.58 dB vs S1-dev, fixed −16.58 dB copies, 0 clipped after; manifest `split: dev`, 1 shard 0 errors; E-002 §8 written; control E-005 + E-034c **13.9212 % WER / 75.6291 % ETA**; no R1 readout | `../moth-cs-dataset/text-2026-11-cal/`, `manifests/R-S1-cal.jsonl`, `data/shards/s1-cal`, `runs/E-017/` |

The pre-A3 artifacts (`readout.json`, `readout-int8.json`, `release/`,
`browser-run.json`) are kept for the erratum; they are **not** the final numbers.

## Next steps: product adopted; no training without go-ahead

**The product is E-023 R1 seed 0 + E-034c, version `0.9.7-e023-r1` (D27).**
The T-600 guard passed on 2026-11-23 (12.2993 / 78.1400 vs 13.7041 / 76.1266),
the package is 49,203,776 B with graph `85abd830…`, the browser transcript is
exact at RTFx 28.48, and FLEURS is IT 11.8304 % / EN 28.1955 % (report-only).
E-005 + E-034c (`0.9.6-e034c`, D25) is retired from the product line. Nothing
is running. **Do not re-read T-600 or FLEURS with this seed.** E-016/P4 is
done (D28): the web runtime is vendored, 1-thread RTFx passes, and the browser
parity is explained (default segmentation costs ~0.4–0.7 pp WER on S1-dev).

1. **E-016b — segmentation default (proposed, no training).** E-016 measured
   the browser default at **+0.72 pp** WER vs Python on S1-dev (CI excludes
   zero); `segment: false` is a **tie** (+0.35 pp). Decide whether to
   auto-disable segmentation when the clip fits one 30 s segment, keep it for
   longer inputs, and re-read S1-dev report-only. This is a runtime decision,
   not a new model.
2. **P8 — LM coverage on the product.** Build the unpruned E-034 LM with the
   ×3-term weighting and rank it on **R-S1-cal** (never S1-dev, never T), then
   the S1-dev veto. No training; the same `pack_lm.py` / `frozen_readout.py`
   path as E-034c. A win is a new dated LM decision, not a product change by
   default.
3. **E-018 — teacher bake-off** on the gold train set v1.0.0 (1,413 clips):
   Parakeet TDT 0.6B v3, Canary-1B-v2, Whisper large-v3-turbo, Voxtral Mini;
   selection rule in `experiments/next-experiments.md` §E-018. Runs on the
   second Mac; no training.
4. **E-019 → E-022, then E-006:** teacher labels, LLM-written LM text, targeted
   acronym/brand/elision data, keyterm-biasing re-test, then the data arms.
   Each has its own pre-registration and gate in `experiments/next-experiments.md`.
5. **Optional, needs an explicit training go-ahead:** E-013 (extra blocks
   before `final_ln`), E-014 (convolution in the extra blocks), E-015 (25 Hz,
   only if the 1-thread gate had failed — it passed at 21.17× on 20 s), and a
   third R1 seed with per-eval checkpoints (P11 is no longer a prerequisite).

`R-S1-cal` and S1-dev stay as they are: no new recordings are planned. New
cues never come from `text/pool.jsonl`, and future LM corpora must exclude the
pool.

The adopted readouts and the guard command are in `runs/E-023/RESULT.md`
(`t600-s0.json/.hyp.jsonl`, `browser-run-0.9.7.json`,
`fleurs-{it,en}-s0.*`). The A3 package/readout (`runs/E-005/`) remains the
historical E-002 gate reference; do not overwrite it.

## Data inventory

| Source | Clips | Hours | Path |
|---|---:|---:|---|
| CV clean (full `cv_it_train`) | 113,118 | 167.31 | `data/shards/cv_it_train` |
| CV clean-50 manifest | 56,769 | 84.22 | `../moth-stt/data/manifests/cv_it_train_clean50.jsonl` |
| CV noisy-50 | 56,349 | 83.33 | `data/shards/cv_it_noisy50` |
| MLS Italian train | 40,953 | 169.89 | `data/shards/mls_it_train` |
| MLS Italian dev | 1,248 | 5.18 | `data/shards/mls_it_dev` |
| LibriSpeech train-clean-100 | 28,370 | 100.00 | `data/shards/ls_en_train` |
| LibriSpeech dev-clean | 2,691 | 5.31 | `data/shards/ls_en_dev` |
| Spliced IT+EN | 20,623 | 109.90 | `data/shards/spliced_it_en` |
| Gold v1.0.0 train | 1,413 | 0.96 | `data/shards/gold-train` |
| MUSAN noise set | 480 | 1.33 | `data/raw/noise`, `data/manifests/noise.jsonl` |
| S1-dev | **960** | **0.855** | `data/shards/s1-dev`, `../moth-cs-dataset/audio/R-S1-dev/S1` |

The CV 27.0 Italian archive is pinned in `data/sources.lock.json` (sha256
`27e1ef0e…`, 10,542,669,746 B). It was extracted to `data/raw/cv_it_27`.

## Gotchas

- **Clean-50 uses the full `cv_it_train` shards**, not a `cv_it_train_clean50`
  shard directory. The config is already correct.
- **Never build the noisy half with `build_mel`.** Use
  `tools/build_noisy_cv.py`, which mixes MUSAN/room tone at the E-024 SNR.
- Do not run `scripts/verify.py` on the v1.0.0 checkout with S1-dev audio: it
  checks fixed v1.0.0 counts. Use it only after a new dataset version.
- S1-dev manifest rows are `split: dev` and point to
  `audio/R-S1-dev/S1/...`; `build_manifest` writes the canonical R-S1 path and
  `split: train`, so `tools/finalize_s1_dev.py` must patch them. The final
  manifest is committed in `../moth-cs-dataset`.
- **S1-dev is held out** from acoustic training and decoder tuning, but its
  **scripts were in the LM corpus** until 2026-11-21: S1-dev was drawn from
  `../moth-cs-dataset/text/pool.jsonl`, and the pool is a `train_script` of
  E-031 (900/960 exact matches, P2). Never draw evaluation cues from a file that
  feeds the LM corpus, and exclude the pool from future corpora
  (`tools/lm_corpus_overlap.py`, `tools/clean_lm_corpus.py`). Do not turn S1-dev
  into a calibration set without amending E-002 first (see E-009 above).
- S1-dev levels: 960 takes, 51.30 min, peak −17.6 dBFS mean (max −3.8), drift
  1.6 dB, no hard errors. Two advisory warnings: `S1_0563.wav` 1.49 s and
  `S1_0886.wav` 1.27 s, both four-word cues. `S1_0726` was re-recorded after a
  0.0 dBFS peak; the final max is −3.8 dBFS.
- **`R-S1-cal` is the E-017 calibration set** (400 reviewed cues, 16.95 min).
  Raw takes are in `audio/R-S1-cal/`; the manifest points at
  `audio/R-S1-cal-norm/`, a fixed −16.58 dB copy that matches S1-dev (median
  RMS −34.34 dBFS, mel offset −0.014, 0 clips). It is `split: dev` only as a
  trainer guard: never train on it, never use it for the S1-dev veto, and never
  rank on S1-dev. The rule is E-002 §8.
- **Every training run needs `model.init`.** `base6.toml` points at
  `data/raw/whisper-base-encoder.pt` (gitignored; rebuild with
  `tools/convert_whisper_encoder.py`). A random-init run is invalid: measured
  99 % dev WER after 4,000 steps. The extra blocks are identity-initialized
  (D16), so base+0/+2/+4/+6 start as the same function.
- **The web runtime reads `config.budget_bytes`.** The vendored
  `vendor/moth-stt-web/ort.js` uses it when present (default 20,000,000, so
  0.8.1 is unchanged); the MOTH-50 release declares 50,000,000. The patch is
  upstream local commit `b0fa11c`; the vendored copy at `0e15442` includes it.
  Serve a release with `python tools/serve_web.py --release
  runs/E-023/release-0.9.7`; do not depend on the archived `../moth-stt`
  checkout (E-016/D28).
- **The 16 kHz VAD is the export default (D19).** `export_release.py` ships
  `silero_vad_16k_op15.onnx` as `silero_vad.onnx` (1,289,603 B); `--vad full`
  remains only for 0.8.1-shaped packages. Parity is measured on 2,040 real clips
  (S1-dev, T, noise) and in the browser (`runs/E-012/`).
- **MOTHLMB2 is the LM default (D20).** `pack_lm.py` and both readout/export
  defaults use the lossless 10-bit codebook; `--lm-format mothlmb1` reproduces
  the old encoding. Never adopt a lossy codebook without a WER readout: A3 fits
  10 bits (1,018/859 distinct values), other LMs may not, and `pack_lm.py`
  refuses rather than approximating.
- **The MOTHLMB2 JS reader is vendored.** `vendor/moth-stt-web/lm.js` is
  byte-identical to upstream `0e15442` (on top of `b0fa11c`) and is covered by
  `vendor/moth-stt-web/PROVENANCE.md`; the browser package no longer depends on
  the archived `../moth-stt` checkout. Install the npm dependencies once with
  `npm ci --prefix vendor/moth-stt-web`.
- **Never take the LM from `../moth-stt/release/`.** That gitignored copy is the
  pre-A3 LM (`dcf1d59f…`). The readout, calibration and export tools resolve the
  LM through `tools/release_assets.py` (A3, `3be1177b…`) and refuse anything
  else without `--allow-other-lm`.
- **E-009 kept the frozen decoder.** The dev-macro grid preferred 0.45 / 2.0
  (−0.28 pp), but S1-dev regressed (+0.58 pp WER, −1.12 pp ETA), so 0.55 / 2.7
  stays. Do not adopt a decoder on dev-macro without the S1-dev veto.
- **The gate readout is `tools/frozen_readout.py`.** It loads `ckpt_best.pt`
  directly, or the INT8 graph with `--onnx`; `--mode mixed` reads one mixed set
  (S1-dev) and `--reference-hyp` adds the paired no-regression verdict. FP32 and
  INT8 differ by 0.06 pp macro on the A3 readout.
- **Measured training (2026-11-17/19):** 4,000 steps at 56.2–99.5 min
  (`base+0`…`base+6`), peak 7.397–9.525 GB; 16,000 steps at 6.73 h, peak
  8.604 GB; 0 non-finite. The E-002 4.47 s/step was the padded worst case.
- **4,000 steps is ~0.30 epoch; 16,000 steps is 1.19 epochs.** E-005's epoch 0
  ended after 13,458 batches (the earlier ~11.2k steps/epoch estimate assumed
  full 192 s batches). The E-003 numbers are a depth screen, not the ceiling.
- Checkpoints every 100 steps (`ckpt_last.pt`); resume with
  `--resume runs/<run>/checkpoints/ckpt_last.pt`.
- The trainer's greedy dev WER is monitoring only; the gate is beam 12 + A3 LM.
- Checkpoints and release binaries are gitignored; do not commit audio.

## Docs map

- `README.md` — project goal and current decisions.
- `docs/decisions.md` — D1–D28, including checkpointing (D15), init (D16),
  release asset pins (D17), the post-E-005 plan (D18), the 16 kHz VAD and
  MOTHLMB2 (D19/D20), encoder LR (D21), E-011 (D22), E-004 B (D23), frontier
  models as teachers (D24, proposed), E-034c (D25), E-023 R1 by amendment
  (D26), R1 seed 0 as the product (D27) and the vendored web runtime / E-016
  (D28).
- `docs/architecture-options.md` — architecture, budget, data and protocol.
- `docs/frontier-stt-2026-11-19.md` — what the frontier STT models can and
  cannot give MOTH-50; licences; teachers.
- `docs/review-2026-11-17.md` — evidence corrections and P1–P7.
- `docs/review-2026-11-21.md` — post-E-004 review: selection target, E-004 B
  check, E-011 diagnosis, C/D and final-run procedure (P0–P11).
- `experiments/E-001-baseline.md` — baseline reproduction.
- `experiments/E-002-protocol.md` — frozen evaluation/data protocol.
- `experiments/E-002-s1-dev.md` — S1-dev recording and manifest runbook.
- `experiments/E-003-plan.md` — arms, screen result and E-005 follow-up.
- `experiments/next-experiments.md` — post-E-005 plan: E-005 addendum and
  FLEURS, E-009–E-022, re-scoped E-004/E-006, gates and commands.
- `runs/E-005/RESULT.md` — long fine-tune, A3 readout, S1-dev, term analysis,
  package, browser.
- `runs/E-010/RESULT.md` — encoder-LR screen; base+6 at 1e-4 wins.
- `runs/E-011/RESULT.md` — long schedule + EMA; dev win, S1-dev ETA fail, not adopted.
- `runs/E-011/p6/RESULT.md` — P6 re-read with E-034c: dev pass, S1-dev ETA veto holds.
- `runs/E-011/p7/RESULT.md` — P7 diagnostics: gold saturation, ETA-miss classes, R1 order.
- `runs/E-004/fleurs-RESULT.md` — P3 FLEURS-IT/EN product readout vs 0.8.1.
- `runs/E-017/RESULT.md` — R-S1-cal: review, recording, level match, manifest, shards, power caveat.
- `runs/E-023/RESULT.md` — R1 seeds, amended gate, T-600 guard, package, browser, FLEURS.
- `runs/E-016/RESULT.md` — 1-thread RTFx, browser-vs-Python parity, segmentation finding, vendored runtime.
- `runs/E-004/RESULT.md` — arm B, P0 parity, P2 leakage, E-034c; 49.21 MB package.
- `runs/probe-2026-11-19/` — 16 kHz Silero graph parity preflight.
- `runs/E-012/` — E-012 both gates: VAD parity, MOTHLMB2 packing, packages and
  browser runs.
- `training/README.md` — trainer, data build, initialization, preflight, resume.
- `data/README.md`, `data/sources.lock.json` — pins and provenance.
