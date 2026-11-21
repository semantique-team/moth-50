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

## 6. Amendment 2026-11-19 (written before any S1-dev result exists)

These make the frozen rules executable; they do not move a gate.

1. **The LM is the pinned A3 file.** "A3 LM" means `lm_bpe3.bin` with sha256
   `3be1177b0ca9f18343efb8d00f0d96365226595c9de32d992720aa4e0b0f16f1`
   (5,381,992 B), fetched and checked by `tools/release_assets.py`. The E-005
   readout used the pre-A3 file `dcf1d59f…` by mistake (erratum in
   `runs/E-005/RESULT.md`); its dev-macro verdict stands, its A3 label does not.
   The readout and export tools now refuse any other LM unless
   `--allow-other-lm` is given, which marks the result as not comparable.
2. **"No dev-S1 regression outside its CI"** is computed by
   `tools/frozen_readout.py --mode mixed`: the candidate and the reference are
   decoded on the same S1-dev manifest with the same frozen decoder and the A3
   LM; the reference is the Moth 0.8.1 INT8 graph (`7bbfad7e…`, pinned in
   `data/sources.lock.json`). Moth's paired utterance bootstrap (1,000
   resamples, seed 0) gives the 95 % CI of (candidate − reference) for WER and
   for ETA. A **regression** is a WER difference whose CI lies entirely above
   zero, or an ETA difference whose CI lies entirely below zero. A tie is not a
   regression.
3. **Decoder recalibration** (E-009), amended 2026-11-19 **before any grid
   result**: the grid is ranked on **dev-macro only** (MLS-IT + LS-EN). S1-dev is
   not used for ranking; it is used once as a **veto** on the selected point,
   against the frozen E-002 decoder on the same MOTH-50 model, with the
   no-regression rule of point 2. A point that regresses S1-dev WER or ETA outside
   its CI is not adopted, regardless of its dev-macro gain. T, FLEURS and
   CS-FLEURS are never used. If a new decoder setting is adopted, it applies to
   later MOTH-50 readouts through a further dated amendment; any paired gate that
   compares models decodes **both** systems with the same decoder. The published
   0.8.1 number keeps its own settings as a historical reference. First
   application (2026-11-19): 0.45 / 2.0 won dev-macro but failed the S1-dev veto
   (+0.58 pp WER, −1.12 pp ETA), so 0.55 / 2.7 stays.
4. **The A3 LM may be encoded as MOTHLMB2** (amended 2026-11-19, after
   E-012(b)). `MOTHLMB2` is a lossless encoding of the same A3 content, not a
   different LM: 1,018/859 distinct log-probs fit the 10-bit codebook, every
   array and score is identical, and the dev-macro and S1-dev hypotheses are
   byte-identical. "A3 LM" therefore identifies the content (source sha256
   `3be1177b…`), while the tools record `lm_format` and the packed sha256. A
   paired gate still decodes both systems with the same encoding (or relies on
   the byte-identical proof).
5. **E-004 arm B changes the product LM** (amended 2026-11-21, after its
   pre-registered readout). The shipped decoder LM becomes the **E-034 N2_1_1
   unpruned LM**, packed lossless as `MOTHLMB2` (7,793,632 B, sha256
   `86aa7cbf…`), not A3. The A3 reference and its 1 pp gate remain historical and
   are not recomputed. E-004's own rule was dev-macro as a guard, one S1-dev
   veto and T read once; arm B passed: dev macro **16.9863 %** vs 17.6212 %
   (−0.6349 pp), S1-dev **11.6289 % WER / 79.8340 % ETA** vs 14.0541 / 75.5371
   (no regression), T-600 **13.4748 / 76.5101** vs 13.9908 / 75.3595. Any later
   paired model comparison decodes **both** systems with the same LM (A3 for
   historical A3 results, E-034 for the product line).

   **Erratum 2026-11-21 (P2).** The S1-dev figures in this point are withdrawn:
   900 of the 960 S1-dev cues are exact sentences of the E-031 LM corpus
   (`text/pool.jsonl` is a `train_script`, and S1-dev was drawn from that pool).
   The rule stands, the evidence does not. The corrected build (E-034c) is
   S1-dev **14.9744 / 73.9258** vs **15.4127 / 72.8027** for the A3 recipe
   rebuilt on the same clean corpus, and dev-macro **16.9897 %** vs 17.6253 %
   (−0.6355 pp). T-600 was read again for E-034c and is a declared repeated
   read. `runs/E-004/RESULT.md`. The product LM is **E-034c** (D25).

## 7. Amendment 2026-11-21b — P6 pre-registration (written before the readout)

1. **Product LM.** The product line decodes with **E-034c** (D25): the E-034
   N2_1_1 recipe rebuilt on the clean corpus, MOTHLMB1 sha256 `4b122fbe…`,
   packed MOTHLMB2 `7a4332f1…`.
2. **Question.** E-011 was judged with A3. With E-034c on both sides, does it
   keep its dev-macro gain without regressing the product set?
3. **Systems.** Candidate: E-011 seeds 0 and 1, INT8 graphs
   `runs/E-011/release-s{0,1}/model.int8.onnx`. Control: E-005 INT8 graph
   `runs/E-005/release-a3/model.int8.onnx`, byte-identical to the product graph.
   Both decode with E-034c, the frozen decoder 0.55 / 2.7, beam 12, top-k 10.
4. **Sets.** Dev-macro (MLS-IT + LS-EN dev; both are corpus exclusions) and
   S1-dev, which is clean for E-034c (0 exact matches, 4.09 % 4-gram share).
   This is a declared second read of E-011 under a changed LM; the gate is
   fixed here, before the numbers exist.
5. **Seed statistic — the gate is the seed mean.** For each seed and system,
   score per utterance (`sub+del+ins`, `ref_words`, `en_hit`, `en_ref`).
   Average the per-utterance counts across the two seeds:
   `(errors_s0 + errors_s1) / 2` and `(en_hit_s0 + en_hit_s1) / 2`, with the
   same reference counts. Bootstrap 1,000 resamples, seed 0, over utterances,
   for the WER and ETA differences against the control. Report each seed's own
   point estimate and CI; **the verdict uses the mean only**, and the seed is
   never chosen on S1-dev. This replaces the earlier P6 wording "no S1-dev
   regression on both seeds": with 2 seeds × 2 one-sided tests, a null candidate
   fails at least one about 10 % of the time (review 2026-11-21 §5), and the
   seed mean is the statistic P11 proposed.
6. **Gate.** PASS iff (a) the seed-mean dev-macro WER is at least **1.0 pp**
   better than the control (E-005 + E-034c, 16.9897 %) with the paired 95 % CI
   excluding zero, and (b) on S1-dev the seed-mean WER difference is not
   entirely above zero and the seed-mean ETA difference is not entirely below
   zero, per §6.2. T-600 is **not** part of the gate: it has been read twice
   (B and E-034c); S2/S3 are diagnostics only.
7. **Outcome.** If PASS: E-011 + E-034c is the product candidate; the shipped
   seed is chosen on dev-macro; the final run becomes optional. If FAIL: P7
   diagnostics, then P10 (R1, encoder LR 3e-5) under a new pre-registration.

Artifacts go to `runs/E-011/p6/`.
