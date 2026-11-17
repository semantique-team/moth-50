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
submitted; public external tests are used as published.

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
  (~47.95 MB). Decide by experiment at equal package size, not by prior.
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
  in total; base+7 (50.95 MB) does not fit.
- The E-003 gate is relative to the 0.8.1 acoustic model re-measured under the
  frozen protocol (E-028 was 31.01 greedy on Moth's dev, not 30.86).
- Moth's seed variance is unknown; MOTH-50 measures its own.
- The vendored release validator accepts the 50,000,000-byte ceiling and the
  `moth-50` name.

Proposed, not yet decided (P1–P7 in the review): activation checkpointing from
the first base run, a staged adaptation schedule from Whisper-base, identity
initialization of the extra blocks, Parakeet-TDT-0.6B-v3 pseudo-labelling as an
E-006 arm, a tighter LM binary format, the evaluation-hygiene items, and 25 Hz
extra blocks as the fallback before base+4.

## Open questions

- Exact S1 recording plan: how many hours, which sessions, and which held-out
  session is the dev set?
- TTS mixture weight and voice selection; how much synthetic data survives the
  real-data readout without hurting S2/S3/FLEURS?
- Does the 2.26 MB headroom in the primary package go to a larger trigram, a
  more tightly packed LM (review P5), or nothing?
- Should the extra blocks be identity-initialized (review P3), randomly
  initialized as in Moth E-006, or copied from the last Whisper-base block?
- Which adaptation schedule from Whisper-base (review P2): frozen-encoder warm-up
  first, and which encoder LR?
- Does base+6 fit in 16 GB with activation checkpointing (review P1, E-002
  probe)?
- Where is the pinned Moth tokenizer (`moth_bpe.model`, sha256 `d6e1054e…f2fd`)
  stored for this repository?
- Is a later 25 Hz / conv-adapter variant worth testing once the 50 MB baseline
  exists?
- Which NC external test set is acceptable for the paper, and how is it labelled
  in the model card?
