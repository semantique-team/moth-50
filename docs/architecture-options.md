# Architecture options for MOTH-50

**Status: accepted 2026-11-17; numbers corrected by the 2026-11-17 review.** This
document does not train anything and does not change the Moth 0.8.1 baseline. It
turns the Moth evidence into the chosen 50 MB architecture and a first experiment
sequence. The 100 MB tier was evaluated and rejected (D5); its analysis is
retained below only as rejected evidence. Decisions are recorded in
`docs/decisions.md`. The review (`docs/review-2026-11-17.md`) checked every cited
Moth number against the Moth records and measured the real INT8 size, CPU latency
and activation memory of the candidates (`runs/probe-2026-11-17/`); the
corrections are applied in place below.

## 1. The decision in one paragraph

The cheapest credible step is **not** a new architecture family. It is the same
proven recipe — a pretrained Whisper encoder, extra pre-LN transformer blocks, a
linear CTC head over the 1,024-piece IT+EN BPE, an n-gram LM in shallow fusion
and a VAD — scaled from Whisper-tiny to **Whisper-base**. A `base` encoder plus
six extra blocks is **39.3 M acoustic parameters**, **39.95 MB int8** (measured
export), and **47.74 MB** once the 5.38 MB LM and 2.33 MB VAD are included. That
fits the 50 MB ceiling with 2.26 MB to spare. It should train in roughly 14–28
hours for 2–3 epochs on the 16 GB M5 Pro, but memory is tight: the measured
activations project the long-clip batches to 12–21 GB, so it may need activation
checkpointing (proposal P1; §4). The same budget also admits a second
configuration — `base` plus four blocks with a 12 MB LM — which is the direct
acoustic-vs-LM trade-off to measure. The 100 MB tier was evaluated
(Whisper-small, 87.8 M acoustic parameters, ~98 MB package) and **rejected** on
2026-11-17: it is roughly 2–3× slower than the 50 MB candidate and much closer
to the 16 GB memory limit. The target is 50 MB, full package.

The bigger caveat is data, not parameters. Moth's own record says the mixed
data — 50.2 minutes of recorded IT–EN speech, oversampled — moved the held-out
speaker by 6.1 WER points when a second speaker was added (E-017, which also
changed the gold amount, text and step count), while adding 200 h of
English-only LibriSpeech made the mixed set 5.16 points worse (E-012).
A 39 M model trained on the same mixture will not automatically beat 0.8.1 by a
large margin. Budget for data work in parallel with the architecture work.

## 2. What Moth already established

The paper is a record of 41 experiments, twelve of them rejected. The relevant
lessons for a larger model are these.

### 2.1 What worked

| Evidence | Measured effect | Lesson for MOTH-50 |
|---|---|---|
| E-004 → E-005: frozen Whisper-tiny encoder (MLS-IT only, padded-30 s arm B) → full fine-tune on 437.7 h IT+EN | MLS-IT dev WER **70.79 → 38.08** (IT+EN macro 33.77) | Pretrained initialization is essential at this data scale. Do not train from scratch first. (The often-quoted 75.77 → 33.77 compares arm A's MLS-only dev with the bilingual macro; Moth's ledger says not to.) |
| E-006: +2 transformer blocks (8.03 → 11.58 M) | dev **33.77 → 31.27**; T (S1, 200) **−5.85**; FLEURS-IT **−3.30** | Depth helps at 20 MB. The same lever should work at 50 MB. Caveat: E-006 also added 4 epochs, changed the LR and fixed augmentation; it is not a capacity-only estimate, and ETA rose 16.4 → 20.0 (higher is better: an improvement). |
| E-014: 110 h utterance-level IT+EN splicing | FLEURS-IT **−2.27**, FLEURS-EN **−0.95**; T/ETA ties | Splicing is a real generalization gain, but not an ETA fix. |
| E-017: R-S1 + R-S2 gold, 50× | T-600 **32.77 / 42.86**; S3 (never trained) **−6.13 WER** | Real mixed speech generalizes across speakers. This is the strongest data lever in the record. |
| E-022: decoder recalibration | T-600 **32.77 → 25.92**; T-S3 **35.43 → 26.66** | Calibrate the decoder before judging the acoustic model. Costs zero bytes. |
| E-024: noise augmentation on half of CV-IT | **−11.54** WER at 10 dB; gains at 15/10/5 dB | Keep the noise recipe. It is cheap and robust. |
| E-028: gentle gold fine-tune | product gains with dev +0.20; shipped in 0.8.0/0.8.1 | The gold stage of a larger model should also be gentle, not a high-LR shock. E-028 started from an already CTC-trained checkpoint, so its LRs are not the adaptation LRs from Whisper (review P2). |
| E-031: LM corpus rebuild + term weighting | rare ETA **28.4 → 46.9 %**; T-600 24.08; +265 kB | LM data treatment can matter more than another acoustic block. |
| E-037: paired fp32/INT8 | dev **+0.16 pp** (significant), FLEURS-EN +0.45 (significant), T-600 +0.49 pp (tie) | INT8 is the right deployment format, with a small real cost. Measure it on the deeper model, don't fear it. |
| E-031 browser acceptance (0.8.1) | median RTFx **46.7×** (4 threads) / **25.1×** (1 thread) on one 3 s fixture | Runtime acceptance, not a WER benchmark. E-041 (browser vs Python WER for 0.8.1) was pre-registered but **never run**; the only measurement of that gap is E-022 on 0.6.0 (−0.09 on 200 clips, a tie). Measure it again for MOTH-50. |

### 2.2 What failed, and should not be repeated first

| Evidence | Result | Why it matters |
|---|---|---|
| E-012: +200 h LibriSpeech EN (53/47 IT/EN) | FLEURS-EN −5.4, but T **+5.16** and ETA flat | More English alone shifts the model away from Italian and does not fix terms. |
| E-013: BPE 4,096 | FLEURS-IT **+2.19** (greedy vs greedy; the quoted +1.9 compared beam with greedy), T flat, ETA flat, +1.2 M params | Vocabulary size is not the ETA bottleneck; the acoustic model is. |
| E-015/E-032: aggressive gold-term addenda | rare terms up, but natural terms/FLEURS/T-S3 down; gates failed | Do not over-weight a term list; it trades broad vocabulary for targeted hits. |
| E-019: Whistle LoRA | S3 **+2.30** WER (tie); FLEURS/CV significantly worse; 29.35 MB export | The gold effect did not transfer with this LoRA recipe; Moth rejected the recipe, not the architecture question, which stays untested. A decoder-based external model is not a shortcut. |
| E-023/E-034: larger unpruned LM | dev −0.6 to −0.65, but 12–13 MB LM binary and 26 MB package | Under 20 MB the LM could not grow. Under 50 MB it can; re-test, don't assume. |
| E-027: decode-time contextual bias | rare ETA unchanged, T-600 tie | A glossary alone did not move the rare terms. |
| E-029: quantization-aware training | no publishable candidate | INT8 PTQ is adequate; QAT is not the missing lever. |
| E-033: intermediate CTC | dev **+0.0249 pp** (tie) | No reason to add it to the first larger model. |
| E-036: conservative KD fine-tune (teacher = E-028 itself) | rare 39–41/81 vs a 42 target; `x` emitted on 120/120 silence clips (one repeated signal) | Distillation from the same-size model did not help. A strong teacher is a different experiment. E-040 (seed replication of E-036) was pre-registered but never run. |

### 2.3 The size inversion

Moth's 20 MB ceiling made Whisper-base a "last resort": the base encoder plus a
CTC head is ~20.35 M parameters, 20.5–23.5 MB int8 by Moth's arithmetic (20.70 MB
measured, `runs/probe-2026-11-17`), and would have left no room for the LM and
VAD (Moth repository `docs/02-architecture.md` §7 and
`docs/10-optimization-review.md`; these docs are in the development repository,
not in the published `moth-stt` release folder). At **50 MB** that constraint
disappears. Whisper-base is no longer the fallback; it is the natural first step.
This is the main architectural consequence of the new budget.

## 3. Budget accounting

Assumptions, stated explicitly so they can be challenged:

- The 50 MB figure is the **sum of required runtime assets**, as in Moth: acoustic
  int8 graph + LM + VAD + mel filterbank/vocab/config/NOTICE. It is not the
  download-compressed size and not the runtime footprint.
- INT8 dynamic quantization costs **1.026 bytes/parameter at width 384 and 1.0175
  at width 512**, measured with Moth's own export recipe on random-weight graphs
  (`runs/probe-2026-11-17`; the tiny+2 control is within 34 kB of the shipped
  11,910,677 B graph). The first version of this table assumed 1.03 everywhere.
- The current LM is **5.38 MB** (E-031 A3, trigram, sparse int16 CSR) and Silero
  VAD is **2.33 MB**. Miscellaneous files (vocab, mel filters, config, NOTICE) are
  0.086 MB. MB means 10⁶ bytes; the ceiling is 50,000,000 bytes.
- The 100 MB tier is rejected (D5); it is retained only as comparison evidence.

The parameter counts below are exact for the stated shapes (conv stem + blocks +
linear CTC head, V=1,024); they exclude buffers, optimizer state and the LM. The
"measured" rows come from `runs/probe-2026-11-17`; `tools/architecture_budget.py
--all` reproduces them within ~0.01 MB with the measured bytes-per-parameter
ratios.

| Candidate | Encoder / extra blocks | Acoustic params | INT8 graph | + 5.38 MB LM + 2.33 MB VAD + misc | Compute vs Moth |
|---|---|---:|---:|---:|---:|
| Moth 0.8.1 (shipped) | tiny 4 + 2 | 11.58 M | 11.91 MB | 19.71 MB | 1.0× |
| base | base 6 + 0 | 20.35 M | 20.70 MB (measured) | 28.49 MB | 1.8× |
| base + 2 | base 6 + 2 | 26.65 M | 27.11 MB (measured) | 34.91 MB | 2.4× |
| base + 4 | base 6 + 4 | 32.96 M | 33.53 MB (measured) | 41.33 MB | 3.0× |
| **base + 6** | base 6 + 6 | **39.26 M** | **39.95 MB (measured)** | **47.74 MB** | **3.6×** |
| base + 7 | base 6 + 7 | 42.41 M | 43.15 MB (measured) | 50.95 MB ✗ | 3.9× |
| base + 8 | base 6 + 8 | 45.56 M | 46.4 MB | 54.2 MB | 4.1× |
| base + 12 | base 6 + 12 | 58.17 M | 59.2 MB | 67.0 MB | 5.3× |
| small | small 12 + 0 | 87.79 M | 90.4 MB (1.03 est.) | 98.2 MB | 8.0× |
| small + 2 | small 12 + 2 | 101.96 M | 105.0 MB (1.03 est.) | 112.8 MB | 9.3× |

Rows above 50 MB (`base + 7`, `base + 8`, `base + 12`, `small`, `small + 2`) are
not admissible; the last four are retained only as rejected 100 MB-tier
evidence. **base + 6 is the deepest model that fits with the A3 LM in its current
`MOTHLMB1` format** and Silero: base + 7 is 0.95 MB over. With a packed LM (P5),
base + 7 + A3 would be 48.93 MB if A3 packs losslessly — an option to keep, not
a plan. The 50 MB table is the design space.

Same-budget alternatives:

| Package | Acoustic | LM | Total | Comment |
|---|---:|---:|---:|---|
| base + 6 + A3 LM | 39.95 MB | 5.38 MB | 47.74 MB | Maximum acoustic capacity under 50 MB; room for an LM of up to 7.64 MB. |
| base + 4 + 12 MB LM | 33.53 MB | 12.0 MB | 47.94 MB | Trades 2 blocks for a 2.2× larger LM; room for an LM of up to 14.06 MB, so the unpruned E-034 LM (12.31 MB) fits as is. |
| base + 2 + 20 MB LM | 27.11 MB | 20.0 MB | 49.53 MB | LM-heavy; only if the LM proves to be the bottleneck. |
| base + 12 + 12 MB LM | 59.2 MB | 12.0 MB | 73.6 MB | Rejected 100 MB-tier deep model; no Whisper-small. |
| small + A3 LM | 90.4 MB | 5.38 MB | 98.2 MB | Rejected 100 MB-tier; maximum pretrained capacity. |

The base+6 and base+4+12LM rows are the two serious 50 MB candidates. They have
almost the same package size and differ only in where the bytes go. That is a
clean experiment, not a guess. One caveat: the Moth LM stores 4 bytes per n-gram
(uint16 token + int16 log-prob) although a token needs 11 bits. A packed format
with an 8-bit log-prob codebook (prototyped in `runs/probe-2026-11-17` §4) puts
the unpruned E-034 LM at 7.16 MB, inside base+6's 7.64 MB of LM room, for a
49.52 MB package. That LM's only measured gain with the current decoder is dev
−0.644 pp (E-034), and it drops A3's ×3 term weighting. If the packing costs no
WER and the LM also helps T, E-004 becomes a free gain rather than a trade-off
(review proposal P5); measure both before assuming it.

4-bit quantization is not a shortcut to a larger model under 50 MB: Whisper-
small at 4 bits is still ~44 MB of weights before the LM and VAD, and ONNX/
browser 4-bit support and quality at this scale are unproven. The 50 MB target
is an INT8 target; if a larger model is wanted, distillation/pruning is the
honest route, not a lower bit width.

## 4. Training and inference feasibility on the M5 Pro / 16 GB

Moth's measured throughput is the anchor: E-028 trained 11.58 M parameters at
**≈333 audio-seconds per wall-second** (time-weighted mean of its training log:
590.8 h in 6,393 s) on an M5 with MPS bf16, and peaked at **7,496 MiB (7.86 GB)**
of MPS driver memory at step 7,950. (The 6.45 GB quoted earlier is E-017's peak.)
Throughput is scaled by block FLOPs with a 10 % pessimism factor
(`tools/architecture_budget.py`); memory comes from the measured activation
footprint (`runs/probe-2026-11-17`). They are planning numbers; E-002 must
measure them on the M5 Pro.

| Candidate | Est. audio-s/s | Est. h per epoch (plain / checkpointed) | Activations kept, 6 × 30 s batch (measured) | Est. peak without checkpointing | Est. browser RTFx (1 / 4 threads) |
|---|---:|---:|---:|---:|---:|
| base | 168 | 3.5 / 4.7 | 4.92 GB | 9–10.5 GB | ~13–17× / ~30× |
| base + 2 | 126 | 4.7 / 6.2 | 6.52 GB | 10–14 GB | ~10–13× / ~23× |
| base + 4 | 101 | 5.8 / 7.8 | 8.13 GB | 11–17 GB | ~8.5–10× / ~19× |
| **base + 6** | **84** | **7.0 / 9.4** | **9.73 GB** | **12–21 GB** | **~7–9× / ~15–17×** |

An E-028 epoch is 13,283 steps over 590.8 h of audio (the 589.55 h mixture plus
the term addenda), i.e. **≈160 audio-seconds per step on average**; 192 s is the
batch cap, not the mean. A realistic fine-tune from Whisper-base is 2–4 epochs:
**14–28 h for base+6** without activation checkpointing and **19–37 h** with it;
**16–31 h for base+4** with it. The M5 Pro may be somewhat faster than the M5 in
the paper, but do not plan on it. The rejected 100 MB rows are omitted: they were
not measured.

Memory is the real gate, and it is tighter than the first estimate (10–12 GB),
which assumed an activation share that was never measured. base+6 keeps **2.6×**
the activations of tiny+2 (9.73 vs 3.69 GB for a 6 × 30 s batch), about 80 % of
them attention matrices: Moth's `_Block` materialises the `T × T` scores and
keeps an fp32 softmax copy for backward. Projected onto E-028's 7.86 GB peak,
base+6 needs **12–21 GB**. E-002 has now measured it (2026-11-17) with a real
200-step MPS run on the padded 6 × 30 s bucket, real `moth-cs` mels and the
pinned tokenizer:

- **without checkpointing:** 13.368 GB peak, 3.461 s/step, stable over 200
  steps, no non-finite losses;
- **with per-block activation checkpointing:** 3.680 GB peak, 4.473 s/step.

Both fit the 16 GB machine. **Checkpointing is accepted from the first base run**
(D15): it leaves >12 GB for the data pipeline and macOS at 1.29× the step time.
Keep the 192 audio-s batch cap; the throughput figures are worst-case padded,
not a training-time estimate. The other P1 actions remain:

- cap batches by Σ T² (or cut clips to ≤ 20 s), not only by audio-seconds;
- check whether `scaled_dot_product_attention` on MPS avoids the fp32 copy;
- keep the 128 s batch / base+4 fallback only if a later real run exceeds
  12.5 GB; E-002 did not need it.

Full record: `runs/E-002/RESULT.md`; protocol:
`experiments/E-002-protocol.md`.

Inference is not the main constraint, but it is closer than it looks. On ONNX
Runtime CPU, base+6 is ≈2.7–3.0× slower than tiny+2 at the same clip length
(1 thread, 5–30 s). The export has a dynamic time axis, so the browser projection
must match the clip length: Moth E-022 measured the tiny+2 architecture in the
browser at 24.3 / 22.9 / 20.2× (1 thread; 5 / 10 / 20 s), which gives **≈7–9×
for base+6** (≈7–7.6× at 20 s), and 0.8.1's 4-thread 3 s fixture (46.7×) gives
≈15–17×. That is enough for record-then-transcribe, but at 1 thread it misses
Moth's ≥ 10× browser bar and sits at the E-005 gate of ≥ 8×. Features, VAD and
the decoder do not slow down with model size, so the real end-to-end slowdown is
somewhat smaller. The rejected Whisper-small tier would have been ~5× or less;
it is out of scope (D5).

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
argument that a fine-tuned base should beat 0.8.1. Caveat: that number is the
full autoregressive Whisper-base (287 MB fp32, forced `it`, greedy, 30 s padding),
not the encoder behind a CTC head, and FLEURS may overlap Whisper's pretraining.
Whisper-small was the 100 MB tier and is out of scope (D5); it is retained only
as evidence.

Risk: the base encoder was trained for a decoder's cross-attention, not for CTC.
Moth's tiny encoder transferred well, so this is a low-to-moderate risk, not a
research question. The first probe should fine-tune the whole encoder; freezing
it or using LoRA is not supported by the low-resource evidence (the Pashto
Whisper study in Moth's `docs/10` found full fine-tuning far better than frozen
or LoRA for a shallow encoder). Moth reached its full fine-tune in two stages:
E-004 trained the new head on a frozen encoder, then E-005 trained everything
with head/encoder LR 1e-3/1e-4 and a 1,500-step warm-up (review proposal P2).

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
about distillation in general. Do this after the base+CTC baseline exists. A
strong, commercially usable teacher is already measured in Moth's baselines:
Parakeet-TDT-0.6B-v3 (CC-BY-4.0) scores **10.98 % WER / 83.6 % ETA on T-600** and
3.65 % on FLEURS-IT (INT8 sherpa-onnx export, greedy), against 0.8.1's 24.08 /
55.23 and 20.16. Its vocabulary and
TDT decoder differ from the student, so the route is sequence-level
pseudo-labelling, not logit KD; the review (P4) proposes it as an E-006 data arm.

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
- **Extra capacity**: 6 pre-LN transformer blocks at width 512. Proposed by the
  review (P3): identity initialization — zero the `out` and `mlp2` weights and
  biases so each new block starts as the identity and step 0 is exactly
  Whisper-base. Alternatives: random initialization (Moth E-006) or a copy of the
  last base block; decide in E-003.
- **Head**: linear 512 → 1,025, blank + 1,024 BPE. Keep the Moth tokenizer for
  comparability; do not rebuild a 4,096-piece vocabulary.
- **LM**: E-031 A3 trigram first (5.38 MB); re-tune the LM budget only after the
  acoustic model is measured.
- **VAD**: Silero VAD v5 (2.33 MB).
- **Package**: 47.74 MB (measured graph size, random weights); hard ceiling
  50,000,000 B. The 2.26 MB headroom (LM room 7.64 MB) can fund a larger or more
  tightly packed LM; do not spend it before the acoustic model is measured.
- **Training** (revised by the review, P2; confirm before E-003): follow Moth's
  E-004 → E-005 → E-028 path instead of starting with E-028's gentle LRs, which
  were tuned for a checkpoint that was already CTC-trained.
  1. Short warm-up of the head and extra blocks on the frozen encoder.
  2. Full fine-tune, 2–4 epochs, batch 192 audio-s / 32 clips (reduce to 128 s if
     memory demands), AdamW, bf16 autocast, fp32 CTC, head/extra-block LR
     3e-4–1e-3 and encoder LR screened in 3e-5–1e-4 (E-005 used 1e-3/1e-4 for
     tiny), cosine with 1,000–1,500 warm-up, grad clip 1.0, SpecAugment + safe
     tempo + E-024 noise, dropout 0.05–0.1, EMA or checkpoint averaging.
  3. Gentle gold fine-tune at E-028's 1e-4 head / 1e-5 encoder.

  Per-block activation checkpointing from the start is proposed (P1, §4). Select
  checkpoints on dev macro **and** the held-out S1 mixed session: E-028 had a
  worse dev and a better product. Prefer 2 epochs first; Moth's 8-epoch run was
  for a tiny encoder, not a pretrained base.
- **Expected planning range** (not a promise): T-600 **19–22 % WER**, ETA
  **58–64 %**, FLEURS-IT **13–17 %**, browser RTFx **~7–9× (1 thread) to
  ~15–17× (4 threads)**. The ranges are anchored on E-006's +2-block gain and the
  base zero-shot FLEURS number; they must be replaced by measurements.

### 6.2 Same-budget alternative to measure, not assume

`base + 4` with a **12 MB LM** is the other ~48 MB configuration. It gives up two
acoustic blocks (about 1.5–2.5 dev points in Moth's E-006 analogy, which also
changed epochs, LR and augmentation) for a 2.2× larger LM (about 0.6 dev points
in E-034, plus unknown ETA/rare-term gains). base+4 leaves 14.06 MB for the LM, so
the unpruned E-034 LM (12.31 MB) fits as is. The right decision is an experiment
at equal package size, not a prior. Recalibrate the decoder (LM weight, token
bonus) separately for each arm on dev, never on T: E-022 moved T-600 by 6.85 pp
with decoder settings alone. If the product metric is dominated by rare English
terms, the LM-heavy variant may win; if it is dominated by Italian fluency, the
acoustic-heavy variant will.

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
| E-003 | **Architecture screen**: base vs base+2 vs base+4 vs base+6, short runs on the same mixture | At least 1.0 pp dev-macro better than the re-scored 0.8.1 INT8 + A3 reference (**23.0170 %** macro; `runs/E-002/dev-reference.json`), with a CI excluding zero; no dev-S1 regression outside its CI; memory/throughput within budget. The earlier 30.86 (E-014) and 31.01 greedy (E-028) are different protocols |
| E-004 | **Package-budget trade-off**: base+6 + 5.38 MB LM vs base+4 + 12 MB LM | Product WER/ETA/rare-term readout at equal package size, decoder recalibrated per arm on dev; pick one |
| E-005 | **Full fine-tune** of the winner, 2–4 epochs, int8 export, browser run | T-600 ≤ 22 WER / ≥ 58 ETA, FLEURS-IT ≤ 18, package ≤ 50 MB, RTFx ≥ 8× (base+6 is projected at ≈7–9× at 1 thread; confirm targets before running) |
| E-006 | **Data scale**: S1 recordings, TTS augmentation, word/phrase-level splicing, more Italian | One variable at a time; S1 held-out session + T-S2/S3/FLEURS readouts; no T training; no voice cloning |
| E-007 | **Optional architecture**: 25 Hz extra blocks, conv adapters, dropout/EMA, frame-level LID auxiliary | Only if E-005 leaves a clear bottleneck; pre-register the expected gain |
| E-008 | **External evaluation**: CS-FLEURS-READ IT–EN (test-only), FLEURS, per-speaker ranges | No NC data in training; report reused T and the no-new-speaker boundary |

For E-003, do not run four full 8-epoch trainings. Run each arm for ~4,000 steps
from the same initialization and data order, evaluate dev macro plus FLEURS-IT
and T, and only the winner gets a full run. At the estimated throughputs, the
four-arm screen is about 6–9 h of training plus evaluations (the upper end with
checkpointing); the memory probe before it is minutes. Moth's E-006 and E-012
show that short-run rankings can differ from product rankings, so keep T and
FLEURS in the screen even though T is reused. A short screen also favours
shallower models when the new blocks start random; identity initialization (P3)
reduces that bias. Moth never measured seed variance, so a gap under ~1 pp
between the top two arms is not a ranking until a second seed confirms it.

The Moth **model** is shape-generic: the primary candidate is
`EncoderConfig(width=512, n_layers=6, n_heads=8, ffn_dim=2048)` with
`MothSTT(vocab_size=1024, encoder_config=..., head_blocks=6)`, and the probes in
`runs/probe-2026-11-17` built and exported it unchanged. The **pipeline** around
it is not: training, evaluation, export and conversion build `MothSTT` or
`WhisperTinyEncoder` with the default tiny `EncoderConfig()`, the training recipe
records `asdict(EncoderConfig())` (so a base run would be logged as tiny without
any error), and the 20,000,000-byte budget is hard-coded in the Python tools and
in `web/ort.js`. The list with line numbers is in `docs/review-2026-11-17.md` §4.
None of it is a rewrite, but every site must take the encoder configuration from
the run config. The same `build_mel.py`, normalizer, tokenizer and release
validator can be reused, but the new repository should copy them deliberately
and re-run the parity tests rather than importing the frozen training code.

There is no 100 MB experiment in this ledger: the tier is rejected (D5). E-008 is
an evaluation experiment, not a capacity experiment.

E-009–E-016 and the re-scoped E-004/E-006 were added after E-005; see §11 and
`experiments/next-experiments.md`. E-017–E-022 followed the frontier-model
review; see §12.

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

The public corpora are re-downloadable but not on disk in the `moth-stt-dev`
checkout reviewed on 2026-11-17: raw MLS/LS audio was deleted after the
2026-10-19 shard build, and the shards, checkpoints and the SentencePiece model
`data/tokenizer/moth_bpe.model` are gitignored. The MOTH-50 data build must
re-fetch the corpora at pinned revisions and rebuild mel shards; budget a day for
the pipeline, not just the download. **The tokenizer is pinned**: the exact model
with sha256 `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`
(Moth E-031/E-034/E-035 `build.json`) is vendored at
`data/tokenizer/moth_bpe.model`; do not rebuild it. The recorded dataset is
pinned at
`semantique-team/moth-cs` `v1.0.0` and verified by
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
   permissive voices, and record each voice's provenance (licences checked
   2026-11-17, `docs/review-2026-11-17.md` §7):
   - Piper `it_IT-paola-medium`: dataset CC0, but fine-tuned from the
     `hifi_tts_92` base model trained on Hi-Fi TTS (CC BY 4.0) — attribute it;
   - Piper `it_IT-serena-high`: CC BY 4.0, but trained only on the synthetic
     `committa/serena-synthetic-it-27h` (Qwen3-TTS cloning a preset voice;
     credit Marco Baccarani) — check accent quality;
   - public-domain English voices such as `en_US-ljspeech-high` for embedded terms;
   - Kokoro-82M (Apache 2.0; Italian voices `if_sara`/`im_nicola`, grade C, 10–100
     min of data each). Its card says the training data includes synthetic audio
     from closed commercial TTS providers: a provenance risk to record. Validate
     the quality first.

   Avoid voices without a stated licence (e.g. Piper `it_IT-riccardo-x_low`,
   "see URL"). Synthesize Italian
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
   audio with Parakeet-TDT-0.6B-v3 (CC-BY-4.0; T-600 10.98 % WER / 83.6 % ETA in
   Moth's baselines) or Whisper-small/large, filter by confidence, train the CTC
   student. High upside, high pipeline cost. The review (P4) proposes it as an
   E-006 arm rather than a post-baseline afterthought, because data is the
   expected bottleneck. The unlabeled audio must itself be commercially usable,
   and the teacher's licence must be attributed in the NOTICE. Candidates (to be
   re-checked when used): VoxPopuli Italian unlabelled audio (4.6 k h in v1,
   21.9 k h in v2; CC0, credit the European Parliament; its released models and
   code are CC BY-NC and must not be used), the Italian Granary manifests
   (VoxPopuli and YouTube-Commons sources; CC-BY-4.0 labels), YODAS (CC-BY-3.0,
   per-video attribution) and, for English, the CC-BY configurations of People's
   Speech (not the `_sa` ones). VoxPopuli also has 91 h of transcribed Italian
   (CC0) that is not in Moth's mixture.
6. **External code-switching corpora for evaluation, test-only.**
   - CS-FLEURS (`byan/cs-fleurs`, arXiv:2509.14161) has an Italian–English read
     test with a real voice, but it is small: **176 utterances, 0.51 h, one
     speaker**. Its synthetic XTTS-v2 Italian–English sets are 9.8 h (train) and
     3.0 h (test1).
   - SwitchLingua (`Shelton1013/SwitchLingua_audio`, arXiv:2506.00087) has 294
     recorded Italian clips; per-language hours are not published (80.2 h and
     174 speakers in total). Access is gated behind a signed agreement that
     limits use to research and non-commercial purposes and forbids commercial
     applications without written permission.
   - Both are **CC-BY-NC**: for this commercial model they are **evaluation
     only**, never training; for SwitchLingua, ask the authors before using it to
     evaluate the commercial release, or report it only in the research paper.
   - ELRA-S0228_108 "Italian English Speech Recognition Corpus (Mobile)" (122 h,
     213 speakers) is **monolingual English spoken by Italians**, not
     code-switched. It costs €25,200 under both its non-commercial and commercial
     (VAR) licences; whether an open-weights release is allowed depends on the
     VAR terms. Not an option for now.

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
- **An external mixed test**: CS-FLEURS-READ Italian–English (real voice,
  test-only; only 176 utterances, 0.51 h and one speaker, so report a CI and no
  speaker range) and/or SwitchLingua Italian (test-only, only with the authors'
  permission for a commercial model). Report the domain difference honestly:
  read/generated, not spontaneous.
- **Per-speaker ranges, bootstrap CIs, seed replicates** for the final recipe.
  Moth never measured seed variance (one seed per configuration; the E-040
  replication was not run), so MOTH-50 must measure it — at least 2–3 seeds for
  the winning recipe — before calling a difference under ~1 pp a win.
- **Package size, INT8 cost, browser RTFx** in the same table as WER/ETA.

Do not claim unseen-speaker generalization beyond S2/S3/FLEURS. The S1-only data
plan buys domain and term coverage, not a new speaker population.

## 9. Confirmed decisions (2026-11-17)

1. **50 MB is the full package**: acoustic INT8 + LM + VAD + metadata, 50,000,000
   bytes. The 100 MB tier is rejected; `base + 7` (50.95 MB measured) and
   `base + 8` (~54 MB) are not admissible with the A3 LM and Silero VAD.
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
mixture weight/voice selection, whether the 2.26 MB headroom funds a larger or
more tightly packed LM, how to initialize the extra blocks (the review proposes
identity initialization), the adaptation schedule from Whisper-base, and whether
a later 25 Hz or conv-adapter variant is worth testing.

## 10. Bottom line

- **Chosen model**: Whisper-base + 6 extra transformer blocks + CTC, 1,024 BPE,
  E-031 LM, Silero VAD; **39.3 M acoustic parameters, 47.74 MB full package**
  (measured graph size), 2.26 MB under the ceiling.
- **Same-budget challenger**: Whisper-base + 4 blocks + 12 MB LM (~47.94 MB);
  decide by E-004, not by intuition.
- **No 100 MB tier**: rejected (D5). The analysis is retained only as evidence.
- **Training risk**: memory. Without activation checkpointing base+6 is projected
  at 12–21 GB on a 16 GB machine; checkpointing from the first base run is
  proposed (P1), and E-002 must measure it.
- **Most likely bottleneck**: mixed data, not parameters. The data plan is
  2–5 h of new S1 recordings plus permissive TTS and word/phrase-level splicing;
  S1-only data buys domain/term coverage, not a new speaker population.
- **Most important discipline**: commercial-only training data, Moth's evaluation
  semantics, pre-registered gates, per-speaker and seed variation, and the
  shipped INT8/browser artifact — not only the fp32 checkpoint.

## 11. Post-E-005 review (2026-11-19)

E-005 trained the primary configuration: base+6, 16,000 steps, frozen beam+A3
dev-macro 17.56 % FP32 / 17.62 % INT8 against the 23.02 % 0.8.1 reference, a
47.84 MB package, and 14.00 / 14.05 % WER on the held-out S1-dev session
(22.25 % reference) with no regression; the browser run is 22.3× at 4 threads
(`runs/E-005/RESULT.md`). This section records the review of that result; the
experiments it implies are pre-registered in `experiments/next-experiments.md`.

**Verdict: the architecture is confirmed.** Nothing measured so far argues for a
different family under 50 MB. The pretrained encoder is essential (a random
encoder stayed at 99 % WER), identity-initialized depth paid monotonically
(48.12 → 36.90 → 34.73 → 33.46 % greedy at 4,000 steps), INT8 costs 0.06 pp, and
the package runs in the browser. The remaining gains are in the training
schedule, the decoder, the byte allocation and the data, roughly in that order of
cost.

**Corrections.** The first E-005 readout and package used the pre-A3 LM, not A3,
and shipped Moth 0.8.1's NOTICE (D17); 16,000 steps are 1.19 epochs, not ~1.4.
The A3 re-measurement is slightly better than the first numbers, and S1-dev
passes both the WER and ETA no-regression rules.

**Where the next gains are, cheapest first:**

1. **Decoder (0 bytes, E-009) — measured, frozen stays.** The dev-macro grid
   preferred 0.45 / 2.0 (−0.28 pp), but the S1-dev veto rejected it (+0.58 pp
   WER, −1.12 pp ETA); the frozen 0.55 / 2.7 stays. This is the E-028
   dev/product divergence again.
2. **Schedule (0 bytes, E-010/E-011).** The encoder LR was never screened (3e-5
   only), the model stopped at 1.19 epochs while still improving, and there is no
   EMA or checkpoint averaging.
3. **Bytes (E-012, then E-004).** The 16 kHz-only Silero graph is bit-identical
   on the synthetic preflight and 1.04 MB smaller; lossless packing of A3 saves
   ~2.0 MB more. That turns ~2.2 MB of room into ~5.2 MB, enough for the unpruned
   LM or a seventh block. Depth has diminishing returns (+4 → +6 bought 1.2 pp at
   4,000 steps), so the LM — which also carries the English terms — is the first
   candidate, decided on S1-dev ETA.
4. **Small architecture variants (E-013/E-014/E-015).** Extra blocks before
   `final_ln` (no closing LayerNorm today), a Conformer-style convolution module
   in the extra blocks at equal bytes, and 25 Hz extra blocks only if the 1-thread
   browser gate fails.
5. **Data (E-006).** Still the largest lever: VoxPopuli Italian (91 h, CC0, not
   in the mixture), gold-repetition and gentle-gold-stage checks, word-level
   splicing, Parakeet pseudo-labels. Re-scoped in §12 as teacher labels
   (E-018/E-019), LM text (E-020) and targeted data (E-021).

**Still rejected:** INT4 weights (WASM `MatMulNBits` dequantizes per call and the
quality at 40 M is unproven), Whisper-small, transducer/AED backbones and
from-scratch encoders.

## 12. Frontier models and the S1-dev errors (2026-11-19)

Reviewed in `docs/frontier-stt-2026-11-19.md` and decided as proposed D24. In
short:

- **The leaderboard does not move the architecture.** Its sets are English, its
  smallest model (Parakeet TDT 0.6B v3) is ~15× MOTH-50's parameters, and the
  top models' strength is an LLM decoder that cannot ship in 50 MB.
- **What transfers is their knowledge**, through permissively licensed weights
  and data: teacher labels (E-018, E-019), LLM-written mixed text for the LM
  (E-020), already pseudo-labelled corpora (Granary), and ceilings on our sets
  (E-008). Commercial APIs are evaluation-only.
- **The S1-dev errors point the same way.** Term deletions fell from 151 to 53,
  but acronyms (ETA 29.2 %), brands (48.1 %) and elided acronyms (`l'sdk`, no
  gain over 0.8.1) remain: data and LM problems (E-019–E-021), not schedule
  problems (`runs/E-005/RESULT.md`, "S1-dev error analysis").
- **Decoder calibration needs a mixed set that is not the gate** (E-017):
  E-009's dev-macro winner made every S1-dev category worse.
- **The owed T-600 readout comes first**: Parakeet is at 10.98 % / 83.60 % on
  T-600, MOTH-50 has not been read there.

Experiments and gates: `experiments/next-experiments.md` (E-017–E-022, E-008
addition).
