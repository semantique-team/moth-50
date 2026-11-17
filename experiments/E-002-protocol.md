# E-002 — Evaluation protocol and data mixture

**Status: protocol frozen 2026-11-17; memory gate passed 2026-11-17.** The new
S1 dev session still has to be recorded, so E-003 cannot be ranked until that
artifact exists. This document is the pre-registered protocol for E-003 onward;
change it only with a dated amendment, and re-run the affected baseline with
both definitions.

It implements D4, D9, D10 and the review's P6 (evaluation hygiene), and it fixes
the memory gate from P1 before any multi-hour run.

## 1. Set roles

| Set | Role | Selection / restriction |
|---|---|---|
| **T-600 / T-S3** | comparability with Moth 0.8.1 | reused and public; report as reused in every table; never train |
| **FLEURS-IT / FLEURS-EN** | unseen-speaker read speech | filtered manifests, same normalizer; evaluation only |
| **S1-dev** | primary product dev and checkpoint selection | new S1 session, 30–60 min, recorded after 2026-11-17 under a `moth-50` consent addendum; never train |
| **S2 / S3** | diagnostic only | S3 never trains but informed Moth development; not pristine; do not use for selection |
| **anchor A** | speaker-vs-sentence analysis | reuses 60 T sentences; not an independent dev set; never train |
| **CS-FLEURS-READ IT–EN** | external mixed test | test-only, 176 utterances / 0.51 h / one speaker; report a CI and no speaker range |
| **SwitchLingua** | optional external test | only with the authors' written permission for a commercial model; otherwise paper-only |

**Hard rules:** no T, A, S3 or FLEURS data in training; no CC-BY-NC data in
training; no TTS or synthetic audio in any test set. Do not describe results as
unseen-speaker generalization beyond S2/S3/FLEURS.

**Blocking prerequisite:** the new S1-dev session does not exist in
`moth-cs@v1.0.0` (only `train`, `test`, `anchor`). Until it is recorded and
verified, E-003 results are provisional and must not be used for a paper claim.
Do not substitute T, A, S3 or a held-out slice of R-S1 as the primary dev.

## 2. Data mixture for the E-003 architecture screen

The screen keeps the data fixed across arms so that architecture is the only
variable. It uses the Moth E-028 mixture, re-fetched at pinned revisions and
rebuilt into mel shards before the first arm:

- Common Voice IT 27 clean and noisy-50 manifests (`cv_it_train_clean50`,
  `cv_it_train_noisy50`; CC0);
- MLS Italian train (CC-BY 4.0);
- LibriSpeech EN train-clean-100 (CC-BY 4.0);
- the E-014 utterance-level spliced IT+EN manifest (derived from the above);
- the recorded `moth-cs@v1.0.0` gold train split (R-S1 + R-S2, 1,413 clips,
  57.6 min), weighted as in E-028 (base gold ×50, term addenda ×20).

No new S1 recordings, TTS or pseudo-labels enter the E-003 screen. Those are
E-006 variables. Each run records the manifest hashes, source revisions and
hours per source in `runs/E-003/config.yaml`; a mixture change is a new
experiment, not a tweak inside the screen.

The exact public-corpus revisions, URLs and available checksums are pinned in
`data/sources.lock.json`; the download and mel-shard build are still pending.
The manifests must pass `moth_manifest` validation and the dataset
`../moth-cs-dataset/scripts/verify.py` before use. The tokenizer is pinned at
`data/tokenizer/moth_bpe.model` (sha256 `d6e1054e…f2fd`); do not rebuild it.

## 3. Metrics and selection

Scoring always uses the vendored `moth_text.normalize` and `moth_eval.py`:

- WER/ETA, per-utterance bootstrap 1,000 resamples, seed 0, 95 % CI;
- pooled and per-speaker tables; three speakers are a range, not a variance
  estimate;
- package size, INT8 cost and browser RTFx in the same result table when the
  artifact exists.

**Two dev readouts, different jobs:**

1. **Dev-macro (screen):** Moth's MLS-IT + LS-EN dev, macro WER. It is the
   comparability metric for E-003. The measured 0.8.1 reference is **23.0170 %**
   macro (MLS-IT 28.1040, LS-EN 17.9299) from the retained E-037 INT8 + A3
   hypotheses re-scored under this protocol (`runs/E-002/dev-reference.json`);
   do not use the stale 30.86 (E-014) or the acoustic-only 31.01 (E-028).
2. **Dev-S1 (product and checkpoint selection):** the new S1 session, pooled
   WER and ETA. A candidate that improves dev-macro but regresses dev-S1 does
   not win; E-028 showed dev and product can move in opposite directions.

T-600, FLEURS and CS-FLEURS are test sets: never use them for checkpoint
selection, decoder tuning or arm ranking.

**E-003 gate (pre-registered):** at least **1.0 pp better dev-macro WER** than
the re-scored 0.8.1 reference (**23.0170 %**), with a paired bootstrap CI
excluding zero, and no dev-S1 regression outside its CI. Keep the decoder fixed
during the screen (beam 12, top-k 10, LM weight 0.55, token bonus 2.7, A3 LM);
recalibrate per arm on dev only in E-004, never on T.

**Seeds:** one seed per arm in E-003. If the top two arms are within ~1 pp,
repeat them with a second seed before naming a winner; the final recipe gets
2–3 seeds. Moth never measured seed variance, so a sub-1 pp gap is not a win.

## 4. Memory and throughput gate (P1)

Before E-003, run the real 200-step training probe on the M5 Pro with base+6,
the 6 × 30 s bucket and the real data pipeline, **with and without per-block
activation checkpointing**:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/probe_training.py --moth-stt ../moth-stt \
  --manifest ../moth-cs-dataset/manifests/train.jsonl \
  --audio-root ../moth-cs-dataset \
  --tokenizer data/tokenizer/moth_bpe.model \
  --steps 200 --batch-clips 6 --device mps --out runs/E-002/train-probe.json

# same, with per-block checkpointing
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/probe_training.py --moth-stt ../moth-stt \
  --manifest ../moth-cs-dataset/manifests/train.jsonl \
  --audio-root ../moth-cs-dataset \
  --tokenizer data/tokenizer/moth_bpe.model \
  --steps 200 --batch-clips 6 --device mps --checkpointing \
  --out runs/E-002/train-probe-checkpointing.json
```

- **Pass without checkpointing:** peak MPS driver memory ≤ 14.5 GB and no
  non-finite loss; keep the 192 audio-s cap.
- **Preferred:** checkpointing from the first base run, peak ≤ 12.5 GB; if it
  exceeds 12.5 GB, cut the batch to 128 audio-s or fall back to base+4.
- Record step time, padded audio-s/s and peak memory; the one-step activation
  probe (13.322 GB / 3.671 GB) is not a substitute for the 200-step run.

**Result (2026-11-17):** both configurations passed. Without checkpointing,
peak **13.368 GB**, median 3.461 s/step, 51.31 padded audio-s/s; with
checkpointing, peak **3.680 GB**, median 4.473 s/step, 40.08 padded audio-s/s;
no non-finite losses in 200 steps. Use checkpointing from the first base run
(D15). Full record: `runs/E-002/RESULT.md`.

The probe is a memory/stability test, not a model-quality result: it uses random
initialization and real mels/texts, no augmentation and no checkpoint saving.

## 5. Artifacts and change control

- Every run: `runs/<id>/` with config, manifest hashes, metrics and logs.
- E-002 artifacts: this protocol, the tokenizer pin, the 200-step probe and its
  result.
- A protocol change gets a dated amendment in this file and a re-scored
  baseline; do not silently move a gate after seeing a result.
