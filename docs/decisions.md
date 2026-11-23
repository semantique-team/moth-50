# Decisions — MOTH-50

Initial decisions for the new model. Update this file when reality diverges; do
not silently change the evaluation semantics or the data pin.

## D1 — Separate repository and paper

`MOTH-50` is a new project, not a branch of `moth-stt`. The previous model is
referred to as **Moth-20** (released as `moth-stt` 0.8.1); it is frozen, and its
41 experiments and 86 MB history are the baseline evidence, not the starting
point. The model, the training recipe and the paper are new.

## D2 — Pin the dataset, do not fork it

Use `semantique-team/moth-cs` at `v1.0.0`. Record the revision and the manifest
hashes in every run. If the dataset changes, release a new dataset version; do
not edit v1.0.0 in place.

## D3 — Keep the evaluation semantics identical

Vendor `moth_text.normalize`, `moth_manifest`, `moth_eval` and
`moth_features` from `moth-stt@f3ecd1b`. WER/ETA, per-speaker reporting and
bootstrap CIs must be computed the same way as in the Moth paper, or the
comparison is not meaningful. If a change is necessary, document it and
recompute the baseline with both definitions.

## D4 — Evaluation protocol (confirmed 2026-11-17)

T is reused and public; it is not a clean test for a model trained after 2026.
Before training, freeze the evaluation protocol: T-600/T-S3 for comparability,
FLEURS-IT/EN, a held-out **new S1 session** as a dev check, and an external
mixed test (CS-FLEURS-READ, test-only; 0.51 h, one speaker). There is no new
speaker available, so do not describe any result as unseen-speaker
generalization beyond S2/S3/FLEURS; S3 informed Moth's development and is not
pristine. Keep any newly recorded test private until the first paper is
submitted; public external tests are used as published. The frozen implementation
is `experiments/E-002-protocol.md`.

## D5 — Size and deployment budget (confirmed 2026-11-17)

**≤ 50 MB for the full required runtime package**: acoustic INT8 graph + LM + VAD
+ metadata, following Moth's accounting. This is a hard product constraint, not
a licence to add parameters without measuring. The target runtime is CPU and
browser WASM, with the Moth browser path as the reference deployment cost. The
**100 MB tier is rejected**; `docs/architecture-options.md` keeps that analysis
as rejected evidence, not as a plan.

## D6 — Data scale

1.46 h of gold code-switching is a benchmark and a fine-tuning seed. The larger
model needs public IT/EN corpora for scale plus **2–5 h of new S1 recordings**
(D9). The data mixture, filtering and dev split are part of the experiment
record.

## D7 — No old runs

Start `EXPERIMENTS.md` at E-001. Reference Moth's E-xxx IDs only as external
evidence; do not renumber or copy its runs.

## D8 — Architecture (accepted 2026-11-17)

`docs/architecture-options.md` analyses the Moth record against the 50 MB
budget. The 50 MB target changes the conclusion of the 20 MB project: Whisper-
base is no longer a last resort, because its ~21 MB INT8 graph leaves room for
the LM and VAD.

- **Primary**: Whisper-base encoder + 6 extra pre-LN transformer blocks + linear
  CTC over the 1,024-piece BPE + E-031 LM + Silero VAD. **39.3 M acoustic
  parameters, 47.74 MB package** (measured graph size, 2.26 MB under the
  ceiling; `runs/probe-2026-11-17`). Full fine-tune from base, 2–4 epochs.
- **Same-budget challenger**: Whisper-base + 4 extra blocks + a 12 MB LM
  (~47.94 MB). Decide by experiment at equal package size, not by prior.
- **100 MB tier**: rejected (D5). Whisper-small and the deep-base 100 MB
  variants are retained in `docs/architecture-options.md` only as rejected
  evidence.
- **Do not start with**: a from-scratch Conformer, a transducer/AED backbone,
  BPE 4,096, QAT, InterCTC, contextual bias, or more English-only data.

Status: accepted. The architecture screen is E-003; the package-budget
trade-off is E-004.

## D9 — Data priority (accepted 2026-11-17)

The only new real recordings will be **S1**, because that is the available
speaker. Record 2–5 h of new sentences, including term-rich and dictation-style
material, in new sessions. Keep one new S1 session as a dev set; never train on
T or A. New sessions need a written consent/addendum covering `moth-50` and
commercial use; the v1.0.0 consent covers the published dataset, not future
recordings. More S1 audio can improve domain and term coverage, but it does
**not** create a new-speaker test: S2, S3 and FLEURS remain the generalization
checks, and the model card must not claim unseen-speaker generalization beyond
them.

**Do not clone S1/S2/S3 voices**: the dataset consent asks that the voices not
be used to train synthetic voices. For synthetic data, use generic TTS voices
with commercially permissive licences, such as Piper `it_IT-paola-medium`
(dataset CC0, fine-tuned from a CC BY 4.0 Hi-Fi TTS base: attribute),
`it_IT-serena-high` (CC BY 4.0, trained on synthetic Qwen3-TTS data), and
public-domain English voices, or Kokoro-82M (Apache 2.0, Italian voices
`if_sara`/`im_nicola`; its training data includes synthetic audio from closed
commercial TTS providers; validate quality). The provenance of each voice is in
`docs/architecture-options.md` §8.2. TTS data
is augmentation, not a substitute for real mixed speech, and synthetic voices
never enter the test set.

If TTS is used, keep it as a minority source (e.g. 10–20 % of mixed exposure),
with a separate manifest, and measure it against the real S1 data. CS-FLEURS and
SwitchLingua are CC-BY-NC: **test-only** for a commercial release, never
training. Word/phrase-level splicing of permissive corpora remains the other
data lever.

Status: accepted. See `docs/architecture-options.md` §8.

## D10 — Commercial data policy (confirmed 2026-11-17)

The model must remain commercially usable. Training data must be CC0, CC-BY,
public domain, recorded with consent, or synthetic from a commercially
permissive TTS model/voice. No CC-BY-NC/ND training data. NC external test sets
(CS-FLEURS, SwitchLingua) may be used for evaluation only if the model card
states it; they must not be mixed into training. SwitchLingua's access agreement
also forbids commercial applications without written permission: ask its
authors before evaluating the commercial release on it, or report it only in the
research paper. The release NOTICE must list every source and voice licence.

## D11 — Record-then-transcribe only (confirmed 2026-11-17)

No streaming requirement. The model stays encoder-only CTC with an external
LM and VAD, like Moth. Transducer and autoregressive-decoder backbones are out
of scope for this release; revisit only if streaming becomes a product
requirement.

## D12 — Name and scope (confirmed 2026-11-17)

The model and display name are **`MOTH-50`**; the repository and package path
are **`moth-50`**. It is a 50 MB-class Italian–English code-switching recognizer.
"50" is the full-package ceiling, not a parameter count. The previous model is
referred to as **Moth-20** (released as `moth-stt` 0.8.1); its repository name is
deliberately unchanged. The historical `moth-large` folder name is retired.

## D13 — Architecture review (2026-11-17)

`docs/review-2026-11-17.md` confirms D8 and corrects the cited Moth evidence in
`docs/architecture-options.md`. Accepted with the review, because they are
measurements or fixes rather than choices:

- The package numbers are the measured ones: base+6 is 39.95 MB INT8 and 47.74 MB
  in total; base+7 (50.95 MB) does not fit with the A3 LM in its current
  `MOTHLMB1` format.
- The 0.8.1 acoustic model (E-028) scored 31.01 greedy on Moth's dev; the 30.86
  previously cited as the E-003 baseline is E-014.
- Moth's seed variance was never measured (one seed per configuration).
- The vendored release validator accepts the 50,000,000-byte ceiling and the
  `moth-50` name.

Proposed, not yet decided (P1–P7 in the review): a staged adaptation schedule
from Whisper-base, identity initialization of the extra blocks,
Parakeet-TDT-0.6B-v3 pseudo-labelling as an E-006 arm, a tighter LM binary
format, the evaluation-hygiene items (including which artifact and protocol
define the E-003 reference, whether the gate fits a ~4,000-step screen, and how
many seeds MOTH-50 runs), and 25 Hz extra blocks as the fallback before base+4.
P1 (activation checkpointing) is accepted in D15.

## D14 — Tokenizer pin (2026-11-17)

The exact Moth 0.8.1 tokenizer is vendored at `data/tokenizer/moth_bpe.model`
(14,327 B, sha256 `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`,
1,024 pieces). The published release ships only `vocab.json`; the LM is indexed
by these piece IDs, so the model is pinned and tested, not rebuilt. Provenance
and attribution are in `data/tokenizer/README.md` and `NOTICE`. A new tokenizer
is a different model and a new experiment, not a configuration change.

## D15 — Memory plan for the first base run (2026-11-17)

E-002 measured base+6 on the M5 Pro with real `moth-cs` mels, the pinned
tokenizer and 200 MPS steps at the 6 × 30 s padded bucket:

- no checkpointing: **13.368 GB** peak, 3.461 s/step (51.31 padded audio-s/s);
- per-block checkpointing: **3.680 GB** peak, 4.473 s/step (40.08 padded
  audio-s/s);
- no non-finite losses; memory flat across the run.

**Use per-block activation checkpointing from the first base run** (P1): it
leaves >12 GB of headroom for the data pipeline and macOS at 1.29× the step
time. Keep the 192 audio-s batch cap. Fall back to a 128 s batch or base+4 only
if a later real run exceeds 12.5 GB; this probe did not need the fallback.
Throughput is worst-case padded and is not the final training estimate.

## D16 — Pretrained initialization for E-003 (2026-11-17)

The trainer port had no initializer: `build_model` created the encoder from
random weights. The first `base+0` run exposed this after 4,000 steps — dev WER
stayed at ~99 %, CTC loss flat at ~6.1 (≈ ln 1024), i.e. the model only learned
the output prior. That run is kept as a checkpoint/throughput probe
(`runs/E-003/base+0-scratch/`), not as an E-003 result. Moth's own evidence says
the same: pretrained initialization is essential at this data scale
(`docs/architecture-options.md` §5.1).

From E-003 on:

- every arm loads the same converted **`openai/whisper-base`** encoder
  (`tools/convert_whisper_encoder.py`, HF revision
  `e37978b90ca9030d5170a5c07aadb050351a65bb`, state sha256
  `fa339cfe…e0dd`, parity 4.8e-4 with the position table matched);
- the extra blocks start as **exact identity** (`out` and `mlp2` zeroed), so at
  step 0 base+0/+2/+4/+6 compute the same function and depth is the only
  difference;
- Moth's fixed sinusoidal positions are kept (the learned `embed_positions`
  table is skipped, exactly as in Moth 0.8.1).

The identity initialization resolves the open question below (review P3).

## D17 — Pinned release assets and a MOTH-50 NOTICE (2026-11-19)

The E-005 readout and package used a stale `../moth-stt/release/lm_bpe3.bin`:
the pre-A3 LM (`dcf1d59f…`, 5,116,464 B), not the 0.8.1 A3 LM (`3be1177b…`,
5,381,992 B) behind the E-002 reference. The package also shipped Moth 0.8.1's
NOTICE, which names `openai/whisper-tiny`. Accepted as fixes, not choices:

- `tools/release_assets.py` pins the A3 LM and the 0.8.1 INT8 graph (Hugging Face
  `semantique-team/moth-stt` revision `1257327e…`) and the Silero graphs (commit
  `bfdc019`) by size and sha256, fetches them into `data/raw/`, and every readout
  and export resolves the LM through it. Another LM needs `--allow-other-lm` and
  is recorded as not comparable.
- `tools/release_notice.py` writes the package NOTICE from the run's encoder
  preset and revision, the extra blocks, the Silero commit, the reused Moth LM
  and tokenizer, and the MOTH-50 training sources in `data/sources.lock.json`.
- The E-005 dev-macro verdict stands (A3 vs pre-A3 was −0.097 pp in Moth E-031;
  the margin is 5.35 pp). Measured on 2026-11-19: dev macro **17.5565 % FP32 /
  17.6212 % INT8** (paired −5.4605 / −5.3957 pp), S1-dev **13.9956 / 14.0541 %**
  vs **22.2498 %** with no regression, A3 package **47,836,502 B** (2.16 MB of
  room). E-002 §6 records the amendment; `runs/E-005/RESULT.md` has the tables.

## D18 — Post-E-005 plan and tooling (proposed 2026-11-19)

`experiments/next-experiments.md` pre-registers E-009–E-016 and re-scopes E-004
and E-006 after the review of E-005 (`docs/architecture-options.md` §11). The
tooling exists; the E-005 addendum and E-009 ran on 2026-11-19, E-010–E-016
have not:

- `tools/frozen_readout.py --mode mixed`: S1-dev WER/ETA and the no-regression
  verdict (E-002 §6);
- `tools/calibrate_decoder.py`: the E-009 decoder grid, ranked on dev-macro only
  (amended 2026-11-19); S1-dev is a separate one-point veto in the readout.
  E-009 ran: 0.45 / 2.0 won dev-macro but was rejected by the veto, so the
  frozen 0.55 / 2.7 stays;
- `train.ema_decay` (`training/ema.py`), off by default; tools choose weights
  with `--weights`;
- `training/configs/base6-enc1e-4.toml`: the E-010 encoder-LR arm;
- `tools/probe_vad.py` and `export_release.py --vad 16k`: E-012(a). The default
  VAD stays the full graph until E-012 passes on real audio and in the browser.

Status: proposed. Each experiment's gate is in that file.

## D19 — 16 kHz-only Silero VAD as the export default (2026-11-19)

E-012(a) passed its pre-registered gate. `tools/probe_vad.py` compared the
shipped 8+16 kHz graph (2,327,524 B) with `silero_vad_16k_op15.onnx`
(1,289,603 B) through Moth's own VAD path on 2,040 real clips, 9,268.29 s:
S1-dev (960), T (600) and the 480-clip noise set. Outputs were **bit-identical**
(max window-probability gap 0.0, 0 window flips, 0 gate or segment
disagreements), so `drop_in` is true.

A package built with the 16 kHz graph loads in the browser in 393 ms,
transcribes `T/S1/S1_0001.wav` exactly, and benchmarks at RTFx 23.90 (4 threads,
median of 5) versus 22.29 for the E-005 A3 package. It totals **46,798,588 B**
(3,201,412 B of room): −1,037,914 B versus E-005, of which 1,037,921 B is the
VAD and +7 B is the longer version string in `config.json`.

`export_release.py --vad` now defaults to `16k`; `--vad full` remains for
0.8.1-shaped packages. This covers (a) only; (b) is D20. Artifacts:
`runs/E-012/`.

## D20 — Lossless MOTHLMB2 LM and its adoption (2026-11-19)

E-012(b) passed its pre-registered gate. The A3 LM has **1,018 distinct bigram
and 859 distinct trigram log-probs**, so a 10-bit codebook is lossless. The new
`MOTHLMB2` format (`tools/moth_lm.py`, `tools/pack_lm.py`) stores an int16
codebook per order, 11-bit tokens, bit-packed entries, an 11-bit `prev1` array
and minimal-width trigram row offsets. A3 packs from 5,381,992 to **3,353,840 B**
(sha256 `6b57bb41…d821`, ratio 0.62316), saving **2,028,152 B**.

The encoding is value-preserving, not approximating: every array and header
field round-trips exactly; 225,818 Python score queries and 980 JS cases are
identical; the dev-macro and S1-dev hypothesis files are byte-identical to the
`MOTHLMB1` runs (macro **17.6212 %**, S1-dev **14.0541 % WER / 75.5371 % ETA**).
The final package (packed LM + 16 kHz VAD) is **44,770,433 B**, with
**5,229,567 B** of room. The browser loads it in 413 ms, transcribes
`T/S1/S1_0001.wav` exactly and benchmarks at **RTFx 24.63** (4 threads).

Adopted:

- `export_release.py --lm-format` and `frozen_readout.py --lm-format` default to
  `mothlmb2`; `--lm-format mothlmb1` remains for reproduction.
- The release validator accepts `MOTHLMB1` or `MOTHLMB2` (packaging-only change,
  recorded in `vendor/moth-stt/PROVENANCE.md`).
- The JS reader lives in the archived Moth checkout as local commit `0e15442`
  (with `b0fa11c` for `budget_bytes`); vendoring it is E-016.
- `tools/architecture_budget.py` now uses the packed LM and 16 kHz VAD, so
  base+7 is estimated at 47.88 MB — E-004's depth arm fits.

Artifacts: `runs/E-012/` (`RESULT.md`, `lm-pack.json`, `js-lm-parity.json`,
`readout-dev-mothlmb2.json`, `s1-dev-mothlmb2.json`,
`export-report-mothlmb2.json`, `browser-run-mothlmb2.json`).

## D21 — Encoder LR 1e-4 for E-011 (2026-11-19)

E-010 screened the review P2 range on the E-003 recipe. At 4,000 steps and seed
0, `lr_encoder = 1e-4` beat `3e-5` at both depths:

| Arm | 3e-5 (E-003) | 1e-4 (E-010) | Δ |
|---|---:|---:|---:|
| base+4 | 34.7306 % | **33.4003 %** | −1.3303 pp |
| base+6 | 33.4588 % | **32.3912 %** | −1.0677 pp |

Both runs had 0 non-finite steps. The gain is present at every 1k dev checkpoint,
not only at the end. base+4 at 1e-4 is 0.0585 pp better than base+6 at 3e-5, so
it becomes an E-004 depth candidate at 6.4 MB less acoustic weight; base+6 at
1e-4 remains the best arm and is the **E-011 recipe**.

Accepted as the optimizer choice for E-011; the frozen decoder, the evaluation
protocol and the package format are unchanged. `training/configs/base6-e011.toml`
holds the long recipe (40,000 steps, warm-up 1,000, EMA 0.9998, 2 seeds).
Artifacts: `runs/E-010/RESULT.md`.

## D22 — E-011 long schedule not adopted (2026-11-21)

E-011 trained base+6 with `lr_encoder = 1e-4`, 40,000 steps and EMA 0.9998 on
two seeds. It is a **dev-macro win but not a product win**:

| Seed | FP32 macro vs E-005 | S1-dev WER | S1-dev ETA | Verdict |
|---|---:|---:|---:|---|
| 0 | **15.8584 %** (−1.6981 pp) | 13.9080 % (tie) | **73.1445 %** (−2.1973 pp, CI excludes 0) | **fail ETA** |
| 1 | **15.7945 %** (−1.7620 pp) | 13.9810 % (tie) | 73.9746 % (−1.3672 pp, CI includes 0) | pass |

The INT8 readouts repeat the pattern: macro **15.9099 / 15.9026 %** vs 17.6212 %
(−1.71/−1.72 pp, cost +0.05/+0.11 pp), S1-dev ETA **73.0469 / 74.1211 %**; seed 0
still fails. Packages are 44,770,432 B.

The one-of-two failure is not resolved by the seed difference (0.83 pp ETA FP32,
1.07 pp INT8; the protocol treats sub-1 pp as noise) and the protocol forbids
selecting the passing seed on S1-dev. The ETA errors are common short English
words, not rare terms. **E-005 remains the product model; E-011 is not adopted.**
The next step is E-004 (budget arms) with E-005 as the control, plus a
pre-registered code-switch/term arm if the ETA loss is to be chased. Artifacts:
`runs/E-011/RESULT.md`.

**P6 (2026-11-21).** Re-read with the product LM E-034c under the E-002 §7
pre-registration: dev-macro seed mean 15.3695 % vs 16.9897 (−1.6202 pp, CI
excludes zero), but clean S1-dev ETA −1.7578 pp [−3.1844, −0.1422] on the seed
mean (s0 −2.64, s1 −0.88). **The veto holds; E-011 stays out and E-005 + E-034c
remains the product.** P7 then found gold memorisation saturated in both models
(E-005 0.20 %, E-011 0.08/0.14 %), so H2 is not the main cause; the ETA loss is
broad short-English substitutions. **E-023 R1** (encoder LR 3e-5) is
pre-registered and not authorized; it needs E-017 first.
`runs/E-011/p6/RESULT.md`, `runs/E-011/p7/RESULT.md`.

## D23 — E-004 arm B: the product LM becomes E-034 N2_1_1 (2026-11-21)

The 5.23 MB freed by E-012 go to **LM coverage**, not more depth. The unpruned
Moth E-034 N2_1_1 LM (all bigrams/trigrams, term weight ×1) was rebuilt from the
frozen E-031 corpus and packed losslessly as `MOTHLMB2`: 12,313,440 →
**7,793,632 B** (sha256 `86aa7cbf…`, 1,015/873 distinct values). The package
with the unchanged E-005 acoustic graph and the 16 kHz VAD is **49,210,224 B**
(789,776 B of room); the graph is byte-identical to E-005.

Decoder calibration kept **0.55 / 2.7** (best 0.45/2.0 was 0.06 pp better, not
significant). Arm B versus the A3 control:

| Set | A3 | E-034 N2_1_1 | Δ |
|---|---:|---:|---:|
| Dev macro | 17.6212 % | **16.9863 %** | **−0.6349 pp** (CI excludes 0) |
| S1-dev | 14.0541 % / 75.5371 % | **11.6289 % / 79.8340 %** | **−2.4251 pp WER / +4.2969 pp ETA** |
| T-600 | 13.9908 % / 75.3595 % | **13.4748 % / 76.5101 %** | −0.5161 pp / +1.1505 pp |

The browser loads the 49.21 MB package in 70 ms, transcribes the exact T clip
and benchmarks at **RTFx 28.36** (4 threads). **Adopted as the product package,
version `0.9.5-e004-lm`.** The acoustic checkpoint, decoder 0.55/2.7 and VAD are
unchanged. The historical A3 package remains for E-002/E-005 reproduction.

**Erratum 2026-11-21 (P2, `runs/E-004/RESULT.md`).** The S1-dev row above is
**withdrawn**. 900 of the 960 S1-dev cues are exact sentences of the E-031 LM
corpus: S1-dev was drawn from `text/pool.jsonl`, and the pool is a `train_script`
of the corpus. Dedupe at script time checked T, A, R-S1 and R-S2, not the LM.
The unpruned-LM gain was mostly memorisation of those sentences. The same
N2_1_1 recipe rebuilt without the 900 sentences (**E-034c**, 185,103 corpus
rows, packed LM 7,787,184 B) gives:

| Set | A3c (clean A3 recipe) | E-034c (clean N2_1_1) | Δ |
|---|---:|---:|---:|
| Dev macro | 17.6253 % | **16.9897 %** | **−0.6355 pp** (CI excludes 0) |
| S1-dev | 15.4127 % / 72.8027 % | **14.9744 % / 73.9258 %** | **−0.4381 pp WER / +1.1230 pp ETA** (CIs exclude 0) |
| T-600 (repeated read) | 14.1915 % / 75.0719 % | **13.7041 % / 76.1266 %** | −0.4870 pp / +1.0550 pp (CIs include 0) |

A3's own S1-dev number was inflated by the same leak (A3c vs A3: +1.3590 pp WER
/ −2.7340 pp ETA); every historical S1-dev absolute number is affected, while
paired A3-vs-A3 comparisons are not. The honest arm-B gain is ~0.4 pp WER /
~1.1 pp ETA on S1-dev and ~0.5 pp / ~1.1 pp on T-600, not 2.4/4.3.
**Decision (D25, same day):** ship **E-034c**. E-034 B remains as history; its
S1-dev numbers must not be quoted.

E-004 arms C (base+7) and D (base+4 long) were not run: their precondition was
that E-011 showed depth still pays on the product, and it did not. A ×3
term-weighted unpruned LM is a possible later arm, not a prerequisite. Artifacts:
`runs/E-004/RESULT.md`.

## D24 — Frontier STT models as teachers, data and ceilings (proposed 2026-11-19)

The top of the Artificial Analysis speech-to-text leaderboard
(`docs/frontier-stt-2026-11-19.md`) is English-only, 0.6 B to 34 B parameters,
and mostly LLM-decoder models. None of it fits a ≤ 50 MB browser package, and
its scores are not comparable with MOTH-50's mixed IT–EN numbers.

Proposed:

- **No architecture change.** Whisper-base + extra blocks + CTC + n-gram LM
  stays (D8, D11, §11 of `docs/architecture-options.md`). LLM decoders and
  from-scratch FastConformer/TDT students stay out.
- **Frontier models enter as teachers, text generators and ceilings:** E-018
  (teacher bake-off on the gold train set), E-019 (pseudo-label distillation:
  Granary Italian, VoxPopuli, unscripted S1 sessions), E-020 (LLM-written mixed
  text for the LM), E-022 (keyterm biasing, conditional), E-008 (ceilings).
- **Licence rule.** Only permissively licensed models and data feed training
  (CC-BY, MIT, Apache-2.0, OpenMDW-1.1, CC0); attribution goes into the NOTICE.
  Commercial APIs are evaluation-only, because their terms generally forbid
  training competing models on their outputs. Sending S2/S3 audio to a third
  party needs their consent.
- **A mixed calibration set** (E-017), separate from the S1-dev gate, must be
  chosen before any further LM or decoder change: E-009 showed that dev-macro
  calibrates the LM in the wrong direction for the product.
- **The E-005 product readout:** T-600 ran in E-004 (and again for E-034c,
  declared repeated); FLEURS ran in P3 (IT **14.82 %**, EN **30.27 %**, P3
  report-only). Parakeet is at 10.98 % / 83.60 % on T-600.

Evidence used: the 2026-11-19 leaderboard snapshot; Moth E-017 baselines (T-600
and T-S3 for Parakeet, Whistle, Whisper-tiny), Moth E-027 (contextual bias
negative), Moth D31 (unpruned LM ties on the held-out speaker), Moth D42 (S1-only
term addenda regress S3); the S1-dev error analysis in `runs/E-005/RESULT.md`
(`tools/eta_breakdown.py`) and the E-009 post-hoc analysis.

Status: proposed. The owner decides E-017 (option A or B) before E-004, E-020
or E-022 run.

## D25 — E-034c is the product LM (2026-11-21)

E-034 B is retired from the product line: its S1-dev evidence was leaked (D23
erratum, P2). The product LM becomes **E-034c**, the same N2_1_1 recipe built on
the E-031 corpus minus the 900 S1-dev sentences: 185,103 rows; MOTHLMB1
12,303,348 B (sha256 `4b122fbe…`), packed MOTHLMB2 **7,787,184 B** (sha256
`7a4332f1…`). Package `0.9.6-e034c`: **49,203,774 B**, **796,226 B** of room;
graph byte-identical to E-005 (`14ba2034…`); decoder 0.55 / 2.7, beam 12,
top-k 10 and the 16 kHz VAD unchanged.

Browser check: load **68.7 ms** with every sha256 verified, transcript exactly
"ho spostato il meeting a domani", **RTFx 29.35** (4 threads, isolated, SIMD;
single run 15.69). Readouts: dev-macro **16.9897 %** (−0.6355 pp vs A3c, CI
excludes 0), S1-dev **14.9744 % / 73.9258 %** (−0.4381 pp WER / +1.1230 pp ETA
vs A3c, CIs exclude 0), T-600 **13.7041 / 76.1266** (−0.4870 / +1.0550 vs A3c,
CIs include 0; declared repeated read). FLEURS (P3, report-only, once per
language): IT **14.8238 %** (−5.3383 pp vs 0.8.1, CI excludes 0), EN **30.2675 %**
(−19.5401 pp), no ETA. E-034 B remains as history. P6 compared E-011 against
E-005 + E-034c and failed on S1-dev ETA (D22 note), so **E-005 + E-034c remains
the product**. Artifacts: `runs/E-004/export-report-e034c.json`,
`runs/E-004/browser-run-e034c.json`, `runs/E-004/release-e034c/`,
`runs/E-004/fleurs-RESULT.md`.

## D26 — E-023 R1 accepted by amendment; seed 0 is the product candidate (2026-11-23)

E-023 R1 (the E-011 recipe with `lr_encoder` 3e-5: base+6, 40,000 steps, EMA
0.9998, gold ×50) failed its pre-registered gate only on clause 2b, the
R-S1-cal `control_en` / `switch` ETA within 0.5 pp of E-005. Seed mean against
E-005 + E-034c, INT8, frozen decoder:

| Set | E-005 + E-034c | R1 seed mean | Δ |
|---|---:|---:|---|
| Dev macro | 16.9897 % | **14.3578 %** | **−2.6320 pp** (CI excludes 0) |
| S1-dev | 14.9744 % / 73.9258 % | **13.7911 % / 74.6826 %** | WER **−1.1833 pp** (CI excludes 0), ETA +0.7568 pp |
| R-S1-cal | 13.9212 % / 75.6291 % | **12.6679 % / 75.4305 %** | WER **−1.2534 pp** (CI excludes 0), ETA −0.1987 pp |
| R-S1-cal `control_en` / `switch` ETA | 88.33 / 73.61 | 86.50 / 71.18 | −1.83 / −2.43 pp (CIs include 0) |

**Decision (owner, 2026-11-23): amend rather than run R2** (E-002 §9). Clause
2b becomes report-only because it contradicted §8.4 (category ETA on R-S1-cal
is report-only) and was below the set's resolution: 0.5 pp is 1.5 words of 300
on `control_en` and 0.72 words of 144 on `switch`, while the misses are 5.5
and 3.5 words and the same-recipe seed spread on that set reaches 3.5–8 pp. The
amendment is post-hoc and says so. The residual risk is recorded: on R-S1-cal
both seeds lose a little on S1's English, which the data neither show nor
exclude.

- **R1 is the final recipe.** R2 as pre-registered (back to 1e-4 with gold
  ×20) is withdrawn; P7 found H2 saturated and R1 identified the encoder LR.
- **Seed 0** is chosen by the §8.3 rule (R-S1-cal pooled ETA 75.4967 vs
  75.3642 %); seed 1 is not read on T or FLEURS.
- **Adoption is conditional** on the E-002 §9 release guard: one declared
  repeated read of T-600 for seed 0, paired against the E-005 + E-034c
  hypotheses, no WER or ETA regression with the CI excluding zero (§6.2);
  speakers and categories are report-only. Pass → package `0.9.7-e023-r1`
  (same 49.20 MB layout, graph `85abd830…`), browser check, FLEURS once.
  Fail → E-005 + E-034c stays the product. **The guard passed on 2026-11-23;
  see D27.**
- **P11 is no longer a prerequisite**; a third seed with per-eval checkpoints
  stays optional.

Status: accepted as the product candidate; the E-002 §9 release guard passed
on 2026-11-23 and **D27 adopts R1 seed 0 as `0.9.7-e023-r1`**. Artifacts:
`runs/E-023/RESULT.md`, `experiments/E-002-protocol.md` §9.

## D27 — E-023 R1 seed 0 is the product: 0.9.7-e023-r1 (2026-11-23)

The E-002 §9 release guard passed. One declared repeated read of T-600 with
R1 seed 0 + E-034c, INT8, frozen decoder 0.55 / 2.7, beam 12, top-k 10,
paired against the E-005 + E-034c hypotheses:

| T-600 | E-005 + E-034c | R1 s0 | Δ (95 % CI) |
|---|---:|---:|---|
| WER | 13.7041 % | **12.2993 %** | **−1.4048 pp** [−2.3455, −0.4276] |
| ETA | 76.1266 % | **78.1400 %** | **+2.0134 pp** [−0.1921, +4.1837] |

The WER CI excludes zero in the improvement direction; the ETA CI is not
entirely below zero. Both §9 pass conditions hold, so **R1 seed 0 is adopted**.
Report-only: S1 **10.06 % / 81.19 %** ETA, S2 **13.47 / 77.62**, S3
**13.37 / 75.77**; `switch` 16.01 / 79.27, `control_en` 15.97 / 85.42,
`brand` 7.36 / 87.25, `acronym` 10.98 / 85.00, `italianized` 3.75 / 96.25;
insertion rate 0.40 %. T-600 vetoed only; it did not rank or change the seed,
decoder or LM.

The product package is **`0.9.7-e023-r1`**: **49,203,776 B**, **796,224 B** of
room, INT8 graph `85abd830…` (the guarded graph), E-034c packed LM
`7a4332f1…` and the 16 kHz VAD `7ed98ddb…`. Browser check: load **70.1 ms**
with every sha256 verified, transcript exactly "ho spostato il meeting a
domani", **RTFx 28.48** (5-run median, 4 threads, isolated, SIMD; single run
16.72). The benchmark must run with the app frontmost: App Nap throttles the
same page to RTFx ~4.8, which is an environment effect, not a package change.

FLEURS was read once per language, report-only, against the E-034c product
hypotheses: **IT 11.8304 %** (−2.9935 pp [−3.4362, −2.5563], n = 697) and
**EN 28.1955 %** (−2.0720 pp [−2.7117, −1.4574], n = 525); no ETA on FLEURS.
FLEURS never ranks. Seed 1 was not read on T-600 or FLEURS.

- **E-005 + E-034c is retired from the product line** and remains the
  historical reference (`0.9.6-e034c`, D25).
- **R1 is the final recipe**; seed 0 is the product seed. R2 stays withdrawn.
- No further read of T-600 with this seed: the guard was the declared repeated
  read.
- Next, no training is implied: P4/E-016 (1-thread RTFx and vendoring the web
  runtime), P8 (×3-term unpruned LM on R-S1-cal), E-018 (teacher bake-off),
  then E-019–E-022 and E-006. E-013–E-015 still need a training go-ahead.

Artifacts: `runs/E-023/t600-s0.json/.hyp.jsonl`,
`runs/E-023/export-report-0.9.7.json`,
`runs/E-023/browser-run-0.9.7.json`, `runs/E-023/fleurs-{it,en}-s0.*`,
`runs/E-023/RESULT.md`.

## D28 — Web runtime vendored; E-016 browser checks (2026-11-23)

The archive dependency is closed. `vendor/moth-stt-web/` holds 12 files from
`semantique-team/moth-stt` at
`0e15442ecb85dffcb4edd68e9d9ece18af6f35df` (on top of `b0fa11c`),
Apache-2.0, with per-file sha256 in `PROVENANCE.md`; eleven are byte-identical
and `ort.js` carries the D29 segmentation patch. `node_modules/` is not
vendored (`npm ci --prefix vendor/moth-stt-web`); upstream tests/acceptance are
not vendored because they depend on Moth's private runs and fixtures. The new
`tools/serve_web.py` serves any release without copying the runtime;
`tools/browser_probe.mjs` drives RTFx and S1-dev runs and
`tools/browser_parity.py` scores them against a Python hypothesis file.

Measurements on `0.9.7-e023-r1`, Chromium 159.0.8154.6, 4 threads isolated
unless stated:

- **1-thread non-isolated RTFx: 10.93× / 19.74× / 21.17×** on 5 / 10 / 20 s
  (gate ≥ 8×) — pass, consistent with Moth E-022's 20.2× at 20 s.
- **Browser vs Python on S1-dev:** Python 13.3820 % / 75.1465 %; browser
  default 14.0979 / 75.5859 (675/960 identical, **+0.7159 pp**
  [+0.1678, +1.2599], no tie); browser `segment: false` **13.7327 / 75.5859**
  (697/960, **+0.3506 pp** [−0.1255, +0.8160], **tie**); browser on
  Python-FFT 16 kHz PCM 14.5215 / 74.7559 (+1.1395 pp, no tie).
- **Explanation:** the runtime default VAD segmentation is the systematic term;
  disabling it recovers a tie. Resampling is not the cause (FFT input is
  slightly worse). The residual 263/960 differences are numerical near-ties
  from re-derived mel, cached mel and WASM int8 vs native, the class Moth E-022
  measured at 70–77 % identity with a WER tie at the shipped decode. ETA is a
  tie in every variant.
- **Finding, then fixed by D29:** the browser default cost ~0.4–0.7 pp WER on
  S1-dev versus the Python readout. E-016b tested the proposed blanket
  auto-off, rejected it (it removes segmentation from multi-sentence clips),
  and adopted a narrower rule: split at three or more VAD runs for clips that
  fit one 30 s segment, two or more for longer recordings. The browser S1-dev
  WER is now 13.7180 % (tie with Python).

Next remains P8 (×3-term unpruned LM on R-S1-cal) and E-018 (teacher bake-off);
no training is implied. Artifacts: `runs/E-016/`, `tools/serve_web.py`,
`tools/browser_probe.mjs`, `tools/browser_parity.py`.

## D29 — Segmentation splits at three or more runs (E-016b, 2026-11-23)

**Post-hoc decision**, written after the E-016 measurements and before any
further model readout. It changes the vendored web runtime only; the package,
model, LM and decoder are untouched.

- **What was rejected.** The E-016b sketch said "auto-disable segmentation when
  the clip fits one 30 s segment". That removes segmentation from every
  multi-sentence clip under 30 s. A 100-group five-sentence concatenation
  (T-600, **diagnostic only**, never a selection input) went from 12.13 % to
  14.26 % WER when segmentation was disabled (+2.14 pp [+0.77, +3.49]); the
  runtime comment already recorded a 9.4 WER-point cost for five sentences read
  as one stream. The blanket rule is rejected.
- **Adopted rule.** Split when the VAD finds at least **three** speech runs for
  a clip that fits one 30 s segment; keep two runs for longer recordings so
  every span fits the model. A single run always keeps the leading silence the
  model expects. An explicit `segment: true/false` still overrides.
  `minimumRuns = pcm.length > maxSamples ? 2 : 3`.
- **S1-dev (960 clips, browser vs Python 13.3820 / 75.1465):** upstream default
  14.0979 / 75.5859; `segment: false` 13.7327 / 75.5859 (tie); **new default
  13.7180 / 75.6348**, +0.3360 pp [−0.1312, +0.7948] (**tie**) and ETA
  +0.49 pp (tie). By VAD runs: 882 clips have one (identical), 73 have two
  (upstream 15.60 % vs full 13.21 %), 5 have three or four (11.31 % vs
  11.90 %). The rule takes the better side of each group.
- **Multi-sentence diagnostics.** MLS-IT dev concatenations (100 groups, 200
  clips, 21.4–28.4 s): segmented 20.672 %, full 18.572 % (−2.10 pp
  [−3.36, −0.93]), **new default 20.672 %** — unchanged, so the rule does not
  capture that full-clip gain (two long utterances still produce 3+ VAD runs),
  but does not regress it. The five-sentence T-600 diagnostic stays at
  12.125 % (segmented). Gold-train concatenations are a tie (0.484 / 0.588 %)
  and not discriminative. A 35 s clip still yields 8 segments.
- **Patch.** One file: `vendor/moth-stt-web/ort.js`, sha256
  `96d099b547b05faac970e278b74485c7efd48b7082898e287877c68faa24072c`,
  diff `vendor/moth-stt-web/PATCH-2026-11-23.diff`; the other eleven files are
  byte-identical to upstream `0e15442`.
- **Caveat.** The rule is a heuristic: two runs are treated as one utterance
  with a pause. The MLS-IT two-long-utterance case shows the limit; a future
  E-016c could decide by utterance length or a two-pass score, but it is not
  adopted here. The S1-dev read is report-only and the model package is
  unchanged.

Artifacts: `runs/E-016/E-016b.md`, `runs/E-016/e016b-diagnostics.json`,
`runs/E-016/s1-dev-browser-minruns3.json`,
`runs/E-016/browser-vs-python.json`.

## D30 — P8 ×3-term LM rejected: no effect (2026-11-23)

The unpruned E-034 N2_1_1 LM rebuilt with A3's ×3 term weighting (E-024, P8)
is **not adopted**. The corpus and recipe are identical to E-034c except
`term_weight = 3`; the n-gram type set and binary size are unchanged
(12,303,348 B MOTHLMB1, 7,787,184 B packed, lossless). The probabilities do
change (21,820 bigram and 69,673 trigram log-probs), but on R-S1-cal the
hypotheses are **byte-identical** to the control: 12.8469 % / 75.4967 %, 0.0 pp
WER and ETA. S1-dev is a tie (13.3966 / 75.0977 vs 13.3820 / 75.1465; WER
+0.0146 pp [0, +0.0456], ETA −0.0488 pp [−0.1566, 0], 4/960 clips differ).
Gate 1 (calibration win) failed, so dev-macro and T-S3 were not run and no
package, browser or T read changed. E-034c remains the product LM; the term
weighting is not a lever at the frozen 0.55 / 2.7 decoder. Next: E-018
(teacher bake-off), then E-019/E-020 text and data. Artifacts:
`experiments/E-024-p8-lm-x3.md`, `runs/E-024/RESULT.md`.

## D31 — E-018: no mixed IT–EN teacher; E-019 stays monolingual Italian (2026-11-23)

The teacher bake-off on the gold train set (1,413 clips, 57.6 min) found **no
qualifying mixed IT–EN teacher** under the pre-registered rule (WER ≤ 7.0 %,
ETA ≥ 85 %, `italianized` ETA ≥ 72.3 %):

| Candidate | WER | ETA | italianized | RTFx |
|---|---:|---:|---:|---:|
| Parakeet TDT 0.6B v3 | 9.95 % | 79.47 % | 62.71 % | 25.48 |
| Whisper large-v3-turbo, forced it | 9.92 % | 80.12 % | 70.76 % | 2.73 |
| Whisper large-v3-turbo, auto | 10.99 % | 82.60 % | 69.92 % | 1.51 |
| Canary-1B-v2, forced it | 13.60 % | 68.62 % | 61.44 % | 14.08 |

Voxtral Mini 3B is pinned and licensed (Apache-2.0) but not feasible on the
16 GB M5: one 2.62 s clip at RTFx 0.009 projects to >100 h for the set; it is
excluded on cost, not accuracy. Models are pinned by revision and file hash in
`data/sources.lock.json`; the runners are `tools/e018_whisper.py`,
`tools/e018_teacher.py` and `tools/e018_summary.py`.

The failure is structural, not a threshold artefact: Parakeet is the best on
English-only (`control_en` 2.8 % WER) but weak on Italianized terms; Whisper
and Canary are good on Italian-only but translate English-only clips
(`control_en` 54.5 % and 102.2 %; Whisper auto fixes that to 7.5 % at the cost
of the Italian side). A teacher that writes Italian for an English cue teaches
the wrong mapping even when its Italian WER is good.

- **No teacher selected; no runner-up** for the E-019 agreement filter.
- **E-019 stays monolingual Italian** (Granary Italian, VoxPopuli Italian),
  where the candidates are far ahead of MOTH-50; it does not label mixed
  audio.
- A language-routed ensemble (Parakeet for English, Whisper/Canary for Italian)
  is recorded as an option but **not adopted**; it is outside the pre-registered
  E-018 rule and needs its own experiment.
- No product change. Artifacts: `runs/E-018/RESULT.md`,
  `runs/E-018/summary.json`, `experiments/next-experiments.md` §E-018.

## D32 — E-022 keyterm biasing: positive oracle ceiling, not shipped (2026-11-23)

The keyterm re-test is **positive as an oracle upper bound** but is not a
product change. Step 1 on R-S1-cal (400 clips, product release + E-034c,
`segment: false`, `ContextualBias` bonus 1.0 / prefix 0.25): pooled
acronym+brand ETA **40.78 → 46.60 % (+5.82 pp)**, WER 12.89 → 11.46, ETA
75.63 → 79.74. This clears the 5-point stop rule.

Step 2 adds 500 training-side distractors per clip and tunes the bonus. The
knee is **5.0**: WER **9.40 %**, ETA **85.83 %**, acronym+brand ETA **54.37 %**,
insertion 2.10/100w, `control_it` unchanged at 4.81 %. The S1-dev oracle upper
bound at b=5 is **10.47 % WER / 84.96 % ETA** vs 13.38 / 75.15 (paired
−2.91 pp WER, +9.81 pp ETA); `control_it` insertion stays 0.24/100w. The list
contains the answer, so this is a ceiling, never a gate or ranking input.

- **Decision: no shipped glossary and no decoder/config change.** The product
  runtime already accepts `contextualSequences` (0 bytes), but a realistic
  user-glossary test and UI/API decision are separate work; the oracle does not
  prove a real list will help.
- Do not use E-022 in a gate readout; the list contains the answers.
- Artifacts: `runs/E-022/RESULT.md`, `tools/keyterm_probe.py`,
  `experiments/next-experiments.md` §E-022.

## D33 — E-020 LLM-written LM text: no effect, not adopted (2026-11-23)

E-025 added 3,521 filtered sentences (Qwen2.5-1.5B-Instruct, Apache-2.0,
pinned `989aa798…`) to the E-034c corpus: 6,000 generated, 249 dropped for
4-gram overlap with S1-dev/T/A, 1,385 for missing seed terms, 139 duplicates,
plus length/category drops. The combined corpus is 188,624 rows; the LM is
12,375,336 B MOTHLMB1 and packs losslessly to **7,833,032 B** (package
49,249,624 B, 750,376 B room).

On **R-S1-cal** it scores **12.8917 % / 75.6291 %** vs **12.8469 / 75.4967**
for E-034c: WER +0.0448 pp [−0.0900, +0.1807], ETA +0.1325 pp [0, +0.4178],
both **ties**, 7/400 clips differ. Gate 1 (a calibration win) fails, so
S1-dev, dev-macro and T were **not** read; **E-034c stays the product LM**.
A 1.9 % corpus addition (≈90k tokens) is too small to move the unpruned
2.3M-trigram LM at the frozen decoder, or the generator's distribution is too
far from the product. A larger generation or a stronger generator is a new
experiment with its own pre-registration; no package/browser/T read changed.
Artifacts: `experiments/E-025-e020-llm-text.md`,
`experiments/E-025-prompts.json`, `runs/E-025/RESULT.md`,
`tools/e020_generate.py`, `tools/e020_filter.py`, `tools/e020_combine.py`.

## Open questions

- Exact S1 recording plan: how many hours, which sessions, and which held-out
  session is the dev set? S1-dev is recorded (960 takes); open are the E-017
  calibration session (option A), the unscripted sessions (E-019 arm B) and the
  acronym/brand sessions (E-021).
- TTS mixture weight and voice selection; how much synthetic data survives the
  real-data readout without hurting S2/S3/FLEURS?
- Does the headroom go to a larger LM, a deeper model, or nothing? With A3 it was
  ~2.16 MB; after E-012 (16 kHz VAD, lossless packed A3) **5.23 MB measured**.
  E-004 B spent it on LM coverage (D23) and leaves 0.79 MB. A further LM
  (E-020) or depth (C, not authorized) needs a new pre-registration, ranked on
  the E-017 set.
- Which teacher, if any, is good enough on mixed IT–EN dictation (E-018), and
  does Granary's YouTube-sourced Italian audio pass a licence review?
- ~~Should the extra blocks be identity-initialized (review P3), randomly
  initialized as in Moth E-006, or copied from the last Whisper-base block?~~
  Resolved in D16: identity, for every E-003 arm.
- Which adaptation schedule from Whisper-base (review P2): frozen-encoder warm-up
  first, and which encoder LR? The encoder LR is E-010.
- Is a later 25 Hz / conv-adapter variant worth testing once the 50 MB baseline
  exists? E-015 (25 Hz, only if the 1-thread browser gate fails) and E-014
  (convolution in the extra blocks).
- Which NC external test set is acceptable for the paper, and how is it labelled
  in the model card?
