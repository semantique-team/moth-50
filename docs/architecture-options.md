# Architecture options for MOTH-50

**Status: accepted 2026-11-17.** This document does not train anything and does
not change the Moth 0.8.1 baseline. It turns the Moth evidence into the chosen
50 MB architecture and a first experiment sequence. The 100 MB tier was
evaluated and rejected (D5); its analysis is retained below only as rejected
evidence. Decisions are recorded in `docs/decisions.md`.

## 1. The decision in one paragraph

The cheapest credible step is **not** a new architecture family. It is the same
proven recipe — a pretrained Whisper encoder, extra pre-LN transformer blocks, a
linear CTC head over the 1,024-piece IT+EN BPE, an n-gram LM in shallow fusion
and a VAD — scaled from Whisper-tiny to **Whisper-base**. A `base` encoder plus
six extra blocks is **39.3 M acoustic parameters**, about **40 MB int8**, and
about **48 MB** once the 5.4 MB LM and 2.3 MB VAD are included. That fits the
50 MB ceiling and should still train in roughly 16–25 hours for 2–3 epochs on
the 16 GB M5 Pro, if the memory probe passes. The same budget also admits a
second configuration — `base` plus four blocks with a 12 MB LM — which is the
direct acoustic-vs-LM trade-off to measure. The 100 MB tier was evaluated
(Whisper-small, 87.8 M acoustic parameters, ~98 MB package) and **rejected** on
2026-11-17: it is roughly 2–3× slower than the 50 MB candidate and much closer
to the 16 GB memory limit. The target is 50 MB, full package.

The bigger caveat is data, not parameters. Moth's own record says the mixed
data — 50.2 minutes of recorded IT–EN speech, oversampled — moved the held-out
speaker by 6.1 WER points when a second speaker was added (E-017), while adding
200 h of English-only LibriSpeech made the mixed set 5.16 points worse (E-012).
A 39 M model trained on the same mixture will not automatically beat 0.8.1 by a
large margin. Budget for data work in parallel with the architecture work.

## 2. What Moth already established

The paper is a record of 41 experiments, twelve of them rejected. The relevant
lessons for a larger model are these.

### 2.1 What worked

| Evidence | Measured effect | Lesson for MOTH-50 |
|---|---|---|
| E-004 → E-005: frozen Whisper-tiny encoder → full fine-tune on 437.7 h | dev macro WER **75.77 → 33.77** | Pretrained initialization is essential at this data scale. Do not train from scratch first. |
| E-006: +2 transformer blocks (8.03 → 11.58 M) | dev **33.77 → 31.27**; T (S1, 200) **−5.85**; FLEURS-IT **−3.30** | Depth helps at 20 MB. The same lever should work at 50 MB. |
| E-014: 110 h utterance-level IT+EN splicing | FLEURS-IT **−2.27**, FLEURS-EN **−0.95**; T/ETA ties | Splicing is a real generalization gain, but not an ETA fix. |
| E-017: R-S1 + R-S2 gold, 50× | T-600 **32.77 / 42.86**; S3 (never trained) **−6.13 WER** | Real mixed speech generalizes across speakers. This is the strongest data lever in the record. |
| E-022: decoder recalibration | T-600 **32.77 → 25.92**; T-S3 **35.43 → 26.66** | Calibrate the decoder before judging the acoustic model. Costs zero bytes. |
| E-024: noise augmentation on half of CV-IT | **−11.54** WER at 10 dB; gains at 15/10/5 dB | Keep the noise recipe. It is cheap and robust. |
| E-028: gentle gold fine-tune | product gains with dev +0.20; shipped in 0.8.0/0.8.1 | A larger model should also use a gentle fine-tune, not a high-LR shock. |
| E-031: LM corpus rebuild + term weighting | rare ETA **28.4 → 46.9 %**; T-600 24.08; +265 kB | LM data treatment can matter more than another acoustic block. |
| E-037: paired fp32/INT8 | dev **+0.16 pp**, T-600 +0.49 pp (tie) | INT8 is the right deployment format. Measure it, don't fear it. |
| E-041: browser vs Python runner | T-600 **+0.72 pp** (tie); S3 +1.53 pp; median RTFx **42×** | Measure the browser artifact; the never-trained voice pays the most. |

### 2.2 What failed, and should not be repeated first

| Evidence | Result | Why it matters |
|---|---|---|
| E-012: +200 h LibriSpeech EN (53/47 IT/EN) | FLEURS-EN −5.4, but T **+5.16** and ETA flat | More English alone shifts the model away from Italian and does not fix terms. |
| E-013: BPE 4,096 | FLEURS-IT **+1.9**, T flat, ETA flat, +1.2 M params | Vocabulary size is not the ETA bottleneck; the acoustic model is. |
| E-015/E-032: aggressive gold-term addenda | rare terms up, but natural terms/FLEURS/T-S3 down; gates failed | Do not over-weight a term list; it trades broad vocabulary for targeted hits. |
| E-019: Whistle LoRA | S3 **+2.30** WER; FLEURS/CV significantly worse; 28 MB export | The gold effect is not architecture-independent; a decoder-based external model is not a shortcut. |
| E-023/E-034: larger unpruned LM | dev −0.6 to −0.65, but 12–13 MB LM binary and 26 MB package | Under 20 MB the LM could not grow. Under 50 MB it can; re-test, don't assume. |
| E-027: decode-time contextual bias | rare ETA unchanged, T-600 tie | A glossary alone did not move the rare terms. |
| E-029: quantization-aware training | no publishable candidate | INT8 PTQ is adequate; QAT is not the missing lever. |
| E-033: intermediate CTC | dev **+0.0249 pp** (tie) | No reason to add it to the first larger model. |
| E-036/E-040: conservative KD fine-tune | rare shortfall reproducible, silence not; arm gap < seed noise | Distillation from a weak teacher did not help. A strong teacher is a different experiment. |

### 2.3 The size inversion

Moth's 20 MB ceiling made Whisper-base a "last resort": the base encoder plus a
CTC head is ~20.35 M parameters, ~21 MB int8, and would have left no room for
the LM and VAD (`../moth-stt/docs/02-architecture.md`,
`docs/10-optimization-review.md`). At **50 MB** that constraint disappears.
Whisper-base is no longer the fallback; it is the natural first step. This is
the main architectural consequence of the new budget.

## 3. Budget accounting

Assumptions, stated explicitly so they can be challenged:

- The 50 MB figure is the **sum of required runtime assets**, as in Moth: acoustic
  int8 graph + LM + VAD + mel filterbank/vocab/config/NOTICE. It is not the
  download-compressed size and not the runtime footprint.
- INT8 dynamic quantization costs about **1.03 bytes/parameter** in the Moth
  export (11.575 M → 11.91 MB). The estimates below use that ratio; actual export
  must be measured.
- The current LM is **5.38 MB** (E-031 A3, trigram, sparse int16 CSR) and Silero
  VAD is **2.33 MB**. Miscellaneous files are ~0.08 MB.
- The 100 MB tier is rejected (D5); it is retained only as comparison evidence.

The parameter counts below are exact for the stated shapes (conv stem + blocks +
linear CTC head, V=1,024); they exclude buffers, optimizer state and the LM.

| Candidate | Encoder / extra blocks | Acoustic params | INT8 graph | + 5.4 MB LM + 2.3 MB VAD | Compute vs Moth |
|---|---|---:|---:|---:|---:|
| Moth 0.8.1 | tiny 4 + 2 | 11.58 M | 11.9 MB | 19.7 MB | 1.0× |
| base | base 6 + 0 | 20.35 M | 21.0 MB | 28.8 MB | 1.8× |
| base + 2 | base 6 + 2 | 26.65 M | 27.5 MB | 35.3 MB | 2.4× |
| base + 4 | base 6 + 4 | 32.96 M | 33.9 MB | 41.8 MB | 3.0× |
| **base + 6** | base 6 + 6 | **39.26 M** | **40.4 MB** | **48.2 MB** | **3.6×** |
| base + 8 | base 6 + 8 | 45.56 M | 46.9 MB | 54.7 MB | 4.2× |
| base + 12 | base 6 + 12 | 58.17 M | 59.9 MB | 67.7 MB | 5.3× |
| small | small 12 + 0 | 87.79 M | 90.4 MB | 98.2 MB | 8.0× |
| small + 2 | small 12 + 2 | 101.96 M | 105.0 MB | 112.8 MB | 9.3× |

Rows above 50 MB (`base + 8`, `base + 12`, `small`, `small + 2`) are retained
only as rejected 100 MB-tier evidence; they are not planned. The 50 MB table is
the design space.

Same-budget alternatives:

| Package | Acoustic | LM | Total | Comment |
|---|---:|---:|---:|---|
| base + 6 + A3 LM | 40.4 MB | 5.4 MB | 48.2 MB | Maximum acoustic capacity under 50 MB. |
| base + 4 + 12 MB LM | 33.9 MB | 12.0 MB | 48.4 MB | Trades 2 blocks for a 2.2× larger LM. |
| base + 2 + 20 MB LM | 27.5 MB | 20.0 MB | 49.9 MB | LM-heavy; only if the LM proves to be the bottleneck. |
| base + 12 + 12 MB LM | 59.9 MB | 12.0 MB | 74.3 MB | Rejected 100 MB-tier deep model; no Whisper-small. |
| small + A3 LM | 90.4 MB | 5.4 MB | 98.2 MB | Rejected 100 MB-tier; maximum pretrained capacity. |

The base+6 and base+4+12LM rows are the two serious 50 MB candidates. They have
almost the same package size and differ only in where the bytes go. That is a
clean experiment, not a guess.

4-bit quantization is not a shortcut to a larger model under 50 MB: Whisper-
small at 4 bits is still ~44 MB of weights before the LM and VAD, and ONNX/
browser 4-bit support and quality at this scale are unproven. The 50 MB target
is an INT8 target; if a larger model is wanted, distillation/pruning is the
honest route, not a lower bit width.

## 4. Training and inference feasibility on the M5 Pro / 16 GB

Moth's measured throughput is the anchor: E-028 trained 11.58 M parameters at
**342 audio-seconds per wall-second** and ~6.4 GB peak memory on an M5 with MPS
bf16. The estimates below scale that measurement by FLOPs, then apply a 10 %
pessimism factor. They are planning numbers; E-002 must measure them.

| Candidate | Est. audio-s/s | Est. hours per 13.3k-step epoch | Est. peak memory (batch 192 s) | Est. browser RTFx (Moth 42×) |
|---|---:|---:|---:|---:|
| base | 173 | 4.1 | 7–8 GB | ~22× |
| base + 2 | 130 | 5.4 | 8–9 GB | ~17× |
| base + 4 | 104 | 6.8 | 9–10 GB | ~13× |
| **base + 6** | **86** | **8.2** | **10–12 GB** | **~11×** |
| base + 12 (rejected 100 MB) | 58 | 12.2 | 12–14 GB | ~8× |
| small (rejected 100 MB) | 39 | 18.2 | 14–16+ GB | ~5× |

A "13.3k-step epoch" is the E-028 schedule (192 audio-seconds per step, one pass
over the 589 h exposure mixture). A realistic fine-tune from Whisper-base is 2–4
epochs: **16–33 h for base+6**, **14–27 h for base+4**. The M5 Pro may be somewhat
faster than the M5 in the paper, but do not plan on it.

Memory is the real gate. E-028 peaked at 6.4 GB; base+6 is 3.6× the FLOPs and
roughly 2–3× the activation footprint. 10–12 GB is plausible, but macOS memory
pressure can kill a run (Moth lost time to this twice). Do a 200-step probe with
the real batch and `empty_cache` settings before committing to a multi-hour run.
If base+6 exceeds ~12.5 GB, drop to batch 128 s or to base+4; do not fight the
machine.

Inference is not the constraint. A 39 M int8 CTC model at 50 Hz should still run
at roughly 10× real time in the browser, which is plenty for record-then-
transcribe. The rejected Whisper-small tier would have been ~5×; it is out of
scope (D5).

## 5. Architecture families: the honest comparison

### 5.1 Encoder-only CTC on a pretrained Whisper encoder — **recommended**

This is Moth's architecture, scaled. It is non-autoregressive, has no language
token, exports cleanly to ONNX, and is robust with a VAD. The pretrained encoder
gives the model a strong Italian/English acoustic prior; the extra blocks add
task capacity; the n-gram LM fixes the frame-independence errors. The main
weakness is the frame-independent CTC assumption, which is exactly what the LM
and the mixed data compensate for. Moth's record shows this recipe reaches
24.08 WER at 19.7 MB; the question is how much of the remaining gap is capacity
versus data.

### 5.2 Whisper-base encoder + CTC — **the scale-up**

Whisper-base is 20.35 M parameters including a 1,024-piece CTC head; Whisper-
small is 87.79 M. Both are multilingual and pretrained on 680 k h. Whisper-base
zero-shot already reaches **18.27 % FLEURS-IT** (E-001 addendum), better than
Moth 0.8.1's 20.16 % after its full fine-tune. That is the strongest single
argument that a fine-tuned base should beat 0.8.1. Whisper-small was the 100 MB
tier and is out of scope (D5); it is retained only as evidence.

Risk: the base encoder was trained for a decoder's cross-attention, not for CTC.
Moth's tiny encoder transferred well, so this is a low-to-moderate risk, not a
research question. The first probe should fine-tune the whole encoder; freezing
it or using LoRA is not supported by the low-resource evidence (the Pashto
Whisper study in Moth's `docs/10` found full fine-tuning far better than frozen
or LoRA for a shallow encoder).

### 5.3 Conformer / Zipformer from scratch — **not first**

A 30–40 M Conformer-CTC could fit 50 MB and would be more FLOP-efficient per
parameter (convolution + attention, 25–50 Hz subsampling). It is also the
highest-risk option: it needs 1,000+ h and many GPU-days to approach a pretrained
Whisper encoder, and Moth has no evidence that a from-scratch model is competitive
at 438 h. The Oído 13 M Conformer result is encouraging but it is a different
data regime. The same caveat applies to Mamba/SSM or hybrid convolution-attention
encoders: interesting, but there is no pretrained IT+EN checkpoint at this size
and the data budget is the same. Keep this as a research bet only if the paper's
contribution is architectural, not a product upgrade.

### 5.4 Transducer (RNN-T / TDT) — **not for this product**

A transducer can beat CTC at the same size and supports streaming, but it adds a
predictor, a joint network, alignment handling and a more complex decoder. Moth
is record-then-transcribe; streaming is not a requirement in the scaffold. The
export path to ONNX/browser is harder, and the 50 MB budget would have to cover
the predictor too. Revisit only if streaming becomes a hard requirement.

### 5.5 Attention encoder–decoder (Whisper-style) — **not for the first model**

An AED can model language better and drop the external LM, but the decoder is
autoregressive: slower, larger, prone to silence hallucination, and it reintroduces
the language-token/utterance-level decision that Moth deliberately removed. At
50 MB, a decoder would eat 10–20 M parameters that could be encoder capacity or
LM bytes. A small decoder for N-best rescoring is a later experiment, not the
backbone.

### 5.6 Distillation / pseudo-labeling — **the highest-upside data experiment**

A strong teacher (Whisper-small/large, Parakeet) can label unlabeled Italian/
English audio or generate N-best targets for a 30–40 M CTC student. This is the
only option that could add hundreds of hours without new recordings. It is also
the most expensive pipeline: teacher inference, filtering, alignment, and a
student that must not inherit the teacher's hallucinations. Moth's E-036 KD from
its own 11.6 M model did not help; that is evidence about a weak teacher, not
about distillation in general. Do this after the base+CTC baseline exists.

### 5.7 Product-key memory / MoE — **a later capacity trick**

Duyu-small (19.3 M active / 49.3 M stored, Conformer + product-key memory + CTC)
is a precedent for a ~50 MB code-switching model. Stored capacity without
proportional compute is attractive, but it is a new implementation, a new
training recipe and a new export path. Only consider it if the dense base+6
saturates and the paper needs a capacity story.

### 5.8 Ensemble of two Moth-sized models — **a fallback, not a plan**

Two 11.6 M models fit in ~24 MB of acoustic weights, leaving room for a 15 MB LM.
Logit averaging can recover 1–2 points, but inference doubles and the result is
still two small models. A single 39 M model is the better use of the same bytes.

### 5.9 If the goal is architectural novelty

The recommendation above optimizes for quality per byte and per training day. If
the paper's contribution is a new architecture, the calculus changes: a from-
scratch Conformer/Zipformer or a product-key/MoE CTC at 30–40 M is a legitimate
research bet, but it needs (a) a much larger data budget (1,000 h+), (b) a
controlled scaled-Whisper baseline, and (c) one to two weeks of training. Run it
as a side experiment after the baseline, not instead of it; otherwise a negative
result is uninterpretable.

## 6. Recommendation

### 6.1 Primary 50 MB configuration

- **Encoder**: Whisper-base, width 512, 6 pretrained blocks, 8 heads, FFN 2,048,
  conv stem stride 2 (50 Hz), fixed sinusoids.
- **Extra capacity**: 6 pre-LN transformer blocks at width 512, randomly
  initialized (or initialized by duplicating the last base block; test in E-002).
- **Head**: linear 512 → 1,025, blank + 1,024 BPE. Keep the Moth tokenizer for
  comparability; do not rebuild a 4,096-piece vocabulary.
- **LM**: E-031 A3 trigram first (5.38 MB); re-tune the LM budget only after the
  acoustic model is measured.
- **VAD**: Silero VAD v5 (2.33 MB).
- **Package**: ~48.2 MB estimated; hard ceiling 50 MB. The ~1.8 MB headroom
  can fund a modest LM expansion or a small GRU LM; do not spend it before the
  acoustic model is measured.
- **Training**: full fine-tune, 2–4 epochs, batch 192 audio-s / 32 clips (reduce
  to 128 s if memory demands), AdamW, bf16 autocast, fp32 CTC, encoder LR
  5e-6–1e-5, head/extra-block LR 1e-4, cosine with 500–1,000 warm-up, grad clip
  1.0, SpecAugment + tempo + E-024 noise, dropout 0.05–0.1, EMA or checkpoint
  averaging, best by dev macro WER. Use gradient checkpointing if the 200-step
  probe is above ~12 GB. Prefer 2 epochs first; Moth's 8-epoch run was for a tiny
  encoder, not a pretrained base.
- **Expected planning range** (not a promise): T-600 **19–22 % WER**, ETA
  **58–64 %**, FLEURS-IT **13–17 %**, browser RTFx **~10×**. The ranges are
  anchored on E-006's +2-block gain and the base zero-shot FLEURS number; they
  must be replaced by measurements.

### 6.2 Same-budget alternative to measure, not assume

`base + 4` with a **12 MB LM** is the other 48 MB configuration. It gives up two
acoustic blocks (about 1.5–2.5 dev points in Moth's E-006 analogy) for a 2.2×
larger LM (about 0.6 dev points in E-034, plus unknown ETA/rare-term gains). The
right decision is an experiment at equal package size, not a prior. If the
product metric is dominated by rare English terms, the LM-heavy variant may win;
if it is dominated by Italian fluency, the acoustic-heavy variant will.

### 6.3 Rejected 100 MB tier (out of scope)

The 100 MB tier was rejected on 2026-11-17 (D5). The two evaluated options were
**Whisper-small + CTC** (87.8 M, ~98 MB) and **base + 12 + 12 MB LM** (58.2 M,
~74 MB). The small model has the most pretrained capacity; the deep base is
cheaper to train and leaves room for a larger LM. Neither is planned. A 100 MB
model is not automatically 2× better; at 438 h + 1.46 h mixed, the data may not
support 88 M parameters without overfitting. The small-model package is only ~2
MB under 100 MB, so it was already a tight fit. Stay at 50 MB; if the 50 MB model
plateaus, revisit the data plan before revisiting the size ceiling.

### 6.4 Do not repeat

- From-scratch Conformer as the first model.
- Transducer or AED as the backbone.
- BPE 4,096.
- More English-only data without balancing Italian and mixed data.
- QAT, InterCTC, contextual bias, conservative KD, Whistle LoRA as early levers.
- A 100 MB model: the tier is rejected (D5).

## 7. Planned experiment sequence

The existing ledger starts at E-001 (reproduce Moth 0.8.1) and E-002 (freeze the
new protocol and data mixture). The architecture work slots in after that.

| ID | Question | Gate / decision |
|---|---|---|
| E-001 | Reproduce Moth 0.8.1 on the canonical dataset with the vendored evaluator | 24.08 / 55.23 within the published interval |
| E-002 | Freeze the evaluation protocol and the data mixture; build manifests | S1 new sessions + external test; no T in training; commercial-only training data |
| E-003 | **Architecture screen**: base vs base+2 vs base+4 vs base+6, short runs on the same mixture | At least 1.0 pp dev macro better than Moth's 30.86 with a CI excluding zero; memory/throughput within budget |
| E-004 | **Package-budget trade-off**: base+6 + 5.4 MB LM vs base+4 + 12 MB LM | Product WER/ETA/rare-term readout at equal package size; pick one |
| E-005 | **Full fine-tune** of the winner, 2–4 epochs, int8 export, browser run | T-600 ≤ 22 WER / ≥ 58 ETA, FLEURS-IT ≤ 18, package ≤ 50 MB, RTFx ≥ 8× (confirm targets before running) |
| E-006 | **Data scale**: S1 recordings, TTS augmentation, word/phrase-level splicing, more Italian | One variable at a time; S1 held-out session + T-S2/S3/FLEURS readouts; no T training; no voice cloning |
| E-007 | **Optional architecture**: 25 Hz extra blocks, conv adapters, dropout/EMA, frame-level LID auxiliary | Only if E-005 leaves a clear bottleneck; pre-register the expected gain |
| E-008 | **External evaluation**: CS-FLEURS-READ IT–EN (test-only), FLEURS, per-speaker ranges | No NC data in training; report reused T and the no-new-speaker boundary |

For E-003, do not run four full 8-epoch trainings. Run each arm for ~4,000 steps
from the same initialization and data order, evaluate dev macro plus FLEURS-IT
and T, and only the winner gets a full run. At the estimated throughputs, the
four-arm screen is about 6–8 h of training plus evaluations; the 200-step memory
probe before it is minutes. Moth's E-006 and E-012 show that short-run rankings
can differ from product rankings, so keep T and FLEURS in the screen even though
T is reused.

The Moth model and converter are already shape-generic: the primary candidate is
`EncoderConfig(width=512, n_layers=6, n_heads=8, ffn_dim=2048)` with
`MothSTT(vocab_size=1024, head_blocks=6)`. The conversion script maps HF encoder
keys by name and shape; it needs a `--model openai/whisper-base` / config flag,
not a rewrite. The same `build_mel.py`, normalizer, tokenizer and release
validator can be reused, but the new repository should copy them deliberately
and re-run the parity tests rather than importing the frozen training code.

There is no 100 MB experiment in this ledger: the tier is rejected (D5). E-008 is
an evaluation experiment, not a capacity experiment.

## 8. Data: the other half of the architecture decision

### 8.1 What is available now

| Source | Hours | Clips | Use | Licence |
|---|---:|---:|---|---|
| Common Voice IT 27 | 167.8 | 113,118 | train | CC0 |
| MLS Italian | 169.9 | 40,953 | train | CC-BY 4.0 |
| LibriSpeech EN train-clean-100 | 100.0 | 28,370 | train | CC-BY 4.0 |
| Spliced IT+EN | 110.0 | 21,205 | train (derived) | derived |
| R-S1 + R-S2 gold (v1.0.0) | 0.96 (57.6 min) | 1,413 | train; Moth's base 50.2 min at 50× | recorded, consent |
| T-600 | 0.39 (23.2 min) | 600 | test only | recorded, consent |
| A (anchor) | 0.12 (7.0 min) | 180 | analysis only | recorded, consent |
| **Total exposure in Moth** | **589.55** | — | — | — |

The 41.9 h of gold exposure in Moth is the 50.2 min base subset repeated 50×;
v1.0.0 adds the 150 term-addendum clips (7.4 min), used at lower repetition in
E-028/E-032. The 1,413-clip, 57.6-min figure is the dataset total, not the
E-017 training subset.

The public corpora are re-downloadable but not currently on disk in the
moth-stt checkout (raw audio and mel shards were deleted at archive time). The
MOTH-50 data build must re-fetch them at pinned revisions and rebuild mel
shards; budget a day for the pipeline, not just the download. The recorded
dataset is pinned at `semantique-team/moth-cs` `v1.0.0` and verified by
`../moth-cs-dataset/scripts/verify.py`.

### 8.2 What could be added, in order of expected value

1. **New S1 recordings, 2–5 h, in new sessions.** S1 is the only available
   speaker. Record new sentences, term-rich dictation and note-taking material,
   and keep one new S1 session as a dev set. New sessions need a written
   consent/addendum covering `moth-50` and commercial use. This can improve
   domain/term coverage and S1 performance, but it **does not create a
   new-speaker test**: S2, S3 and FLEURS remain the generalization checks. Do
   **not** clone S1/S2/S3 voices: the dataset consent asks that the voices not
   be used to train synthetic voices.
2. **TTS-generated code-switched speech, 20–100 h.** Use only commercially
   permissive voices: Piper `it_IT-paola-medium` (CC0) or `it_IT-serena-high`
   (CC BY 4.0) for Italian, public-domain English voices such as
   `en_US-ljspeech-high` for embedded terms, or Kokoro-82M (Apache 2.0, Italian
   voices `if_sara`/`im_nicola`; validate the quality first). Synthesize Italian
   sentences with English terms, or concatenate Italian and English TTS spans at
   word boundaries. Keep TTS as a minority source (e.g. 10–20 % of mixed
   exposure), in a separate manifest, and never in the test set. No S1/S2/S3
   voice cloning.
3. **Word/phrase-level splicing, 100–300 h derived.** The current E-014 splicer
   concatenates whole utterances, producing clause-level switches. A forced-
   alignment splicer that inserts an English word or phrase at a natural Italian
   boundary would look much more like the product use case. It reuses existing
   permissive corpora and is a pure data-engineering win. Measure it against
   E-014's utterance-level splicing.
4. **More monolingual Italian/English, balanced.** Use full MLS-IT, CV-IT,
   LibriSpeech train-360, Common Voice EN, TED-LIUM. Keep Italian dominant
   (do not repeat E-012's 53/47 mistake); measure the mixture on the mixed
   readout, not just dev macro.
5. **Pseudo-labeling with a strong teacher.** Label unlabeled Italian/English
   audio with Whisper-small/large or Parakeet, filter by confidence, train the
   CTC student. High upside, high pipeline cost; do it after the base+CTC
   baseline.
6. **External code-switching corpora for evaluation, test-only.** CS-FLEURS has
   an Italian–English read test (~1.2 h) and a synthetic XTTS subset;
   SwitchLingua has an Italian subset (~6–7 h) but is gated. Both are
   **CC-BY-NC**: for this commercial model they are **evaluation only**, never
   training. The ELRA Italian–English mobile corpus (122 h, 213 speakers) is
   commercial (€25,200) and not an option for an open model.

### 8.3 Evaluation

T-600 is reused and public. It is a comparability set, not a clean test for a
model trained in 2027. The new protocol should have:

- **T-600 / T-S3**: same semantics as Moth, for direct comparison; label it
  reused in every table.
- **FLEURS-IT / FLEURS-EN**: public unseen-speaker read speech; keep the
  filtered manifests and the same normalizer.
- **A new S1 dev session + S2/S3**: with S1-only recordings there is no new real
  speaker. Report the S1, S2 and S3 ranges separately; S3 never contributed
  training audio but informed Moth's development, so it is not pristine.
- **An external mixed test**: CS-FLEURS-READ Italian–English (real voices,
  test-only) and/or SwitchLingua Italian (test-only, licence permitting). Report
  the domain difference honestly: read/generated, not spontaneous.
- **Per-speaker ranges, bootstrap CIs, seed replicates** for the final recipe.
  Moth's seed range was 0.46 pp; do not call a smaller difference a win.
- **Package size, INT8 cost, browser RTFx** in the same table as WER/ETA.

Do not claim unseen-speaker generalization beyond S2/S3/FLEURS. The S1-only data
plan buys domain and term coverage, not a new speaker population.

## 9. Confirmed decisions (2026-11-17)

1. **50 MB is the full package**: acoustic INT8 + LM + VAD + metadata. The
   100 MB tier is rejected; `base + 8` is not admissible because its full
   package would be ~55 MB.
2. **The model must remain commercially usable.** Training data must be CC0,
   CC-BY, public domain, recorded with consent, or synthetic from a commercially
   permissive TTS model/voice. CS-FLEURS and SwitchLingua are CC-BY-NC:
   **evaluation only**, never training. The NOTICE must list every source and
   voice licence.
3. **New real recordings are S1-only, 2–5 h, new sessions.** No new speaker is
   available, so the model card must not claim unseen-speaker generalization
   beyond S2/S3/FLEURS. Do not clone S1/S2/S3 voices; use generic permissive TTS
   voices for synthetic data.
4. **No streaming.** The model stays encoder-only CTC with an external LM and
   VAD, record-then-transcribe.
5. **The repository and model are named `moth-50`.** "50" is the package ceiling,
   not a parameter count.

Remaining open questions: the exact S1 recording plan and dev session, the TTS
mixture weight/voice selection, whether the ~1.8 MB headroom funds a larger LM
or a small GRU LM, how to initialize the extra blocks, and whether a later 25 Hz
or conv-adapter variant is worth testing.

## 10. Bottom line

- **Chosen model**: Whisper-base + 6 extra transformer blocks + CTC, 1,024 BPE,
  E-031 LM, Silero VAD; **39.3 M acoustic parameters, ~48.2 MB full package**.
- **Same-budget challenger**: Whisper-base + 4 blocks + 12 MB LM (~48.4 MB);
  decide by E-004, not by intuition.
- **No 100 MB tier**: rejected (D5). The analysis is retained only as evidence.
- **Most likely bottleneck**: mixed data, not parameters. The data plan is
  2–5 h of new S1 recordings plus permissive TTS and word/phrase-level splicing;
  S1-only data buys domain/term coverage, not a new speaker population.
- **Most important discipline**: commercial-only training data, Moth's evaluation
  semantics, pre-registered gates, per-speaker and seed variation, and the
  shipped INT8/browser artifact — not only the fp32 checkpoint.
