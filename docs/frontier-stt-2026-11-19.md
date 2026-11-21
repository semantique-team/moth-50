# Frontier STT models and MOTH-50 (2026-11-19)

**Question.** Can the models at the top of the Artificial Analysis speech-to-text
leaderboard (<https://artificialanalysis.ai/speech-to-text/non-streaming>) make
MOTH-50 better, given the ≤ 50 MB package and the Italian–English
code-switching task?

**Answer.** Not as architectures: none of them fits, and their accuracy comes
from what does not fit. They are useful as **teachers** (labels for audio we
cannot transcribe by hand), as **text generators** for the decoder LM, as a
source of already pseudo-labelled corpora, and as **ceilings** on our own test
sets. Only models and data whose licences allow training a commercial model
may feed training; commercial APIs are evaluation-only. The experiments are
pre-registered in `experiments/next-experiments.md` (E-017–E-022, E-008
addition) and summarized in D24.

## 1. What the leaderboard measures

AA-WER v2.2 is a duration-weighted WER over ~8 h of **English** audio:
AA-AgentTalk (50 %), VoxPopuli-Cleaned-AA (25 %), Earnings22-Cleaned-AA (25 %).
It measures neither Italian nor code-switching, and none of its sets resembles
dictated mixed IT–EN sentences with English terms. Its numbers are therefore
not comparable with MOTH-50's (14.05 % WER / 75.54 % ETA on S1-dev).

Snapshot read on 2026-11-19 (values as listed; some models appear through
several providers with different numbers):

| Model | AA-WER | Weights | Size | Notes |
|---|---:|---|---|---|
| Fun-Realtime-ASR-preview (Alibaba) | 1.7 % | proprietary | — | top of the board |
| Scribe v3 (ElevenLabs) | 1.9 % | proprietary | — | |
| MAI-Transcribe-2 (Microsoft) | 2.0 % | proprietary | — | |
| Voxtral Small Transcribe 3 (Mistral) | 2.1 % | API | — | |
| SOV-Transcribe-1 (Salesforce) | 2.3 % | Apache-2.0 | ~27 B MoE, ~4.6 B active (reported) | best open weights; Italian supported |
| Nemotron 3.5 Transcribe (NVIDIA) | 2.8–3.4 % | OpenMDW-1.1 | 34 B MoE, ~4.3 B active | 40 languages incl. Italian |
| Voxtral Small / Mini (Mistral) | 2.8 / 3.8 % | Apache-2.0 | 24 B / 3 B-class | Italian supported |
| Whisper large-v3 / -turbo | 4.1–5.7 % | MIT | 1.55 B / 0.8 B | provider-dependent (one provider lists 10.1 %) |
| Parakeet TDT 0.6B v3 (NVIDIA) | 4.5 % | CC-BY-4.0 | 0.6 B (670 MB INT8) | fastest on the board (674×); 25 EU languages |

For scale: MOTH-50 is **39.3 M parameters and a 47.84 MB package** including
the LM and the VAD. The smallest model on the board is ~15× larger in parameters
and ~14× larger as an INT8 file, without an LM or a VAD.

## 2. What we already know on our own data

Moth measured three external systems on the canonical T-600 set
(`../moth-stt/docs/13-current-model-and-paper-readiness.md`, Moth E-017
baselines; same normalizer and scorer as MOTH-50):

| System | T-600 WER | T-600 ETA | T-S3 (unseen speaker) | Size |
|---|---:|---:|---|---:|
| Parakeet TDT 0.6B v3, INT8, greedy | **10.98 %** | **83.60 %** | 10.82 / 83.94 | 670 MB |
| Whistle (forced IT) | 29.42 % | 49.28 % | 28.36 / 52.11 | 16.9 MB |
| Whisper-tiny, forced IT, beam 5 | 50.72 % | 36.05 % | 46.51 / 38.03 | 149 MB fp32 |
| Moth 0.8.1 INT8 + A3 (E-001) | 24.08 % | 55.23 % | — | 19.71 MB |
| **MOTH-50 E-005 INT8 + A3** | **not measured** | **not measured** | — | 47.84 MB |

On S1-dev (not T), MOTH-50 is at 14.05 % / 75.54 % against 0.8.1's 22.25 % /
62.11 %. If T-600 behaves like S1-dev for MOTH-50 as it did for 0.8.1, MOTH-50
is ~3–5 WER points and ~8–15 ETA points behind Parakeet. That is the gap a
teacher can close, and why the E-005 T-600 readout (still owed, once) comes
first: it fixes where we are against an external model on the product set.

Also on record from Moth:

- **Contextual bias did not help** (Moth E-027): a 24-term glossary moved T-600
  WER by −0.03 pp and left the 47 rare-term rows unchanged. Keyterm prompting,
  which several frontier APIs sell, is not a free lever here (E-022).
- **The unpruned LM** (12.31 MB) was 0.65 dev points better and won T-600
  (−0.77 pp) but **tied on the held-out speaker** (Moth D31). E-004 arm B is
  that LM, packed.
- Parakeet ran at **RTFx 34 on 4 CPU threads** on the M5 (sherpa-onnx INT8,
  600 clips in 41 s), so labelling hundreds of hours on the Mac is feasible.

## 3. What transfers and what does not

| Idea from the frontier models | For MOTH-50 | Where |
|---|---|---|
| LLM decoder (Voxtral, SOV, Nemotron, Canary-Qwen, Gemini, GPT) | **No.** Billions of parameters; it is what makes them good at context and terms, and what does not fit in a browser package | — |
| FastConformer 8× subsampling + TDT decoder (Parakeet) | **No, as an architecture.** It needs a new encoder pretrained on thousands of hours with GPUs; a from-scratch encoder is rejected (D8/D11). The cheap analogue, fewer frames per second in the extra blocks, is E-015 | E-015 |
| Their transcripts as training labels (pseudo-labels, sequence-level distillation) | **Yes**, from permissively licensed open weights only | E-018, E-019 |
| Corpora they already pseudo-labelled (NVIDIA Granary) | **Yes**, licence permitting, per source | E-019 |
| LLM-written text for the n-gram LM | **Yes**, with an open LLM whose licence leaves outputs free | E-020 |
| Keyterm / custom-vocabulary prompting | **Maybe.** The decoder already has `ContextualBias`; Moth E-027 was negative, but the stronger MOTH-50 acoustic model may keep more terms in the beam | E-022 |
| Their accuracy on our sets | **Yes**, as ceilings for the paper and for teacher choice | E-008, E-018 |

## 4. Licences: who may feed training

The model must stay commercially usable (D5), so a teacher's labels or an LLM's
text can enter training only if the teacher's licence allows it.

| Candidate | Licence | Size | Italian | Runs on the 16 GB Mac |
|---|---|---|---|---|
| Parakeet TDT 0.6B v3 | CC-BY-4.0 | 0.6 B | yes | yes (Moth ran it on CPU) |
| Canary-1B-v2 | CC-BY-4.0 | 1 B | yes | likely |
| Whisper large-v3 / -turbo | MIT | 1.55 B / 0.8 B | yes | yes |
| Voxtral Mini (3 B class) | Apache-2.0 | 3 B LLM + encoder | yes | quantized; bf16 is tight |
| SOV-Transcribe-1 | Apache-2.0 | ~27 B MoE | yes | **no**: rented GPU |
| Nemotron 3.5 Transcribe | OpenMDW-1.1 (no restriction on outputs) | 34 B MoE | yes | **no**: rented GPU |

Rules:

- **CC-BY teachers and corpora need attribution.** When pseudo-labels enter a
  release, the NOTICE must name the teacher and every pseudo-labelled corpus:
  add them to `data/sources.lock.json` and to
  `tools/release_notice.TRAINING_SOURCES`.
- **Commercial APIs are evaluation-only.** Most terms of service forbid using
  outputs to develop competing models (OpenAI's do so explicitly); check each
  provider before any use. Their transcripts never enter training, text
  generation or the LM.
- **Sending recordings to an API is a data transfer.** S1 is the project owner;
  S2/S3 audio may go to a third party only if their consent covers it.
- **Pin before use.** Every teacher is pinned by revision and file hash in
  `data/sources.lock.json` before it labels anything, like the A3 LM (D17).

## 5. Realistic expectations

MOTH-50 at 39 M parameters will not approach 2–5 %: those numbers are English,
and they come from models 15× to more than 800× larger, most with LLM decoders. What is plausible
is closing part of the gap to the best permissive teacher on the product sets.
With good pseudo-labels and an in-domain LM, ~10–11 % on S1-dev is a reasonable
target; it is an expectation, not a gate. Each experiment has its own gate in
`experiments/next-experiments.md`.

## 6. What the S1-dev errors say

`tools/eta_breakdown.py` on the E-005 S1-dev readouts (`runs/E-005/RESULT.md`,
"S1-dev error analysis") shows where the remaining term errors are:

- **Acronyms and brands** are the weakest categories (ETA 29.2 % and 48.1 %;
  acronym insertions 4.7 per 100 words).
- **Elided acronyms** (`l'sdk`, `l'hr`, `l'ebitda`) did not improve at all over
  0.8.1 (34 vs 32 misses).
- **Function words** inside English spans fell from 140 to 59 misses; what is
  left is mostly homophones (`too` → `to`), with little to gain.

These are data and LM problems, not schedule problems: E-011 (longer training)
should lower WER but is unlikely to fix them. They are what E-019 (teachers),
E-020 (LM text) and E-021 (targeted data) aim at.
