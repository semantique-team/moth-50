# MOTH-50 — scaffold

**Status: local repository, initialized 2026-11-17 on `main`, no remote yet.**
It was created 2026-11-16 as `moth-large` and renamed `MOTH-50` (repo
`moth-50`) on 2026-11-17. The model does not exist yet. Moth 0.8.1 is frozen in
`../moth-stt`; this directory is the starting point for a larger
Italian–English code-switching speech recognizer and its own paper.

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
  linear CTC over the 1,024-piece BPE + E-031 LM + Silero VAD. **39.3 M acoustic
  parameters, 47.74 MB package** (measured graph size; 2.26 MB under the
  ceiling). Same-budget challenger: base + 4 blocks + a 12 MB LM (~47.94 MB),
  decided by E-004. Memory: E-002 measured 13.37 GB without checkpointing and
  3.68 GB with per-block checkpointing; checkpointing is the default (D15).
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
README.md                 this file
pyproject.toml            dependencies, ruff and pytest configuration
data/README.md            dataset pin, splits, restrictions
data/sources.lock.json    pinned source versions, URLs and checksums
data/tokenizer/           pinned Moth 1,024-piece BPE model and provenance
docs/decisions.md         decisions and open questions
docs/architecture-options.md  architecture analysis and planned sequence
docs/review-2026-11-17.md evidence check, measurements and proposals
experiments/README.md     new experiment ledger, starts at E-001
experiments/E-001-baseline.md
experiments/E-002-protocol.md  frozen evaluation protocol and data mixture
experiments/E-002-s1-dev.md   S1-dev recording runbook and manifest patch
runs/probe-2026-11-17/    measured INT8 size, CPU latency, activation memory
runs/E-001/               Moth 0.8.1 baseline reproduction
runs/E-002/               real 200-step MPS training probe
tests/                    tests for tools/ and the vendored code
training/                 MOTH-50 trainer port (config, data, model, loop)
training/configs/         base+6 E-003 config and the 20-step smoke config
training/fixtures/        tiny smoke manifests (audio not included)
tools/architecture_budget.py  parameter/size/compute estimator
tools/probe_export.py     real INT8 export size and CPU latency (needs moth-stt)
tools/probe_activations.py  activation memory, CPU or MPS (needs moth-stt)
tools/probe_training.py   200-step MPS training memory/throughput probe
tools/probe_lm_pack.py    packed LM format prototype (review P5)
vendor/moth-stt/          evaluation code copied from the frozen repo
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
