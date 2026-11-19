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

## Open questions

- Exact S1 recording plan: how many hours, which sessions, and which held-out
  session is the dev set?
- TTS mixture weight and voice selection; how much synthetic data survives the
  real-data readout without hurting S2/S3/FLEURS?
- Does the headroom go to a larger LM, a deeper model, or nothing? With A3 it is
  ~2.16 MB; after E-012 (16 kHz VAD, lossless packed A3) ~5.2 MB. E-004
  (re-scoped) decides.
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
