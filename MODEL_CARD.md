# MOTH-50 — model card

**Model:** MOTH-50, version **`0.9.8-e023-r1`** (a NOTICE-only repackaging of
`0.9.7-e023-r1` with byte-identical runtime assets: see *Licence and
attribution* and D37)
**Date:** 2026-11-24, revised 2026-11-26
**Status:** final product of the MOTH-50 line. E-027 (TTS fine-tune) tied and was
not adopted (D36); no further training arm is planned.
**Licence:** Apache-2.0 for the repository, model code and weights. Repository
attributions are in `NOTICE`; the release package carries its own NOTICE,
written by `tools/release_notice.py` for the files it ships.

## What it is

An Italian + English code-switching ASR model that fits the **full runtime
package in ≤ 50 MB** and runs record-then-transcribe on CPU and in the browser.
It is an **encoder-only CTC** model, not a streaming or an LLM-decoder system.

| Component | Size / setting |
|---|---|
| Acoustic model | Whisper-base encoder (6 layers, 512 width, 8 heads, 2048 FFN) + 6 pre-LN transformer blocks + linear CTC |
| Acoustic parameters | 39,259,649 |
| Tokenizer | 1,024-piece BPE, sha256 `d6e1054e…` |
| Language model | E-034c, packed MOTHLMB2, 7,787,184 B, sha256 `7a4332f1…` |
| VAD | Silero 16 kHz, 1,289,603 B |
| Decoder | beam 12, top-k 10, LM weight 0.55, token bonus 2.7 |
| INT8 graph | 40,039,997 B, sha256 `85abd830…` |
| Release package | **49,204,097 B** (795,903 B under the 50,000,000 B ceiling) |

Input is 16 kHz mono audio; the runtime applies the 16 kHz VAD. The package is
self-contained: acoustic INT8 graph, LM, VAD, mel filters, vocab and config.

## Intended use

- Italian speech with English terms, acronyms, brands and code-switched
  phrases, **read or dictated** in a quiet-to-moderate environment.
- Offline/batch transcription, desktop CPU and the in-browser runtime.
- Users who need a small, commercially usable package and can tolerate
  record-then-transcribe latency (not live streaming).
- **Not** for safety-critical, medical, legal or financial decisions without a
  human review; not a speaker-identification or emotion system.

## Training data

The product is E-023 R1 seed 0, trained from the pretrained Whisper-base
encoder on the R1 mixture (per epoch, before repeats):

| Source | Clips | Hours | Licence |
|---|---:|---:|---|
| Common Voice IT 27, clean-50 | 56,769 | 84.22 | CC0 |
| Common Voice IT 27, noisy-50 | 56,349 | 83.56 | CC0; mixed with MUSAN noise (CC-BY-4.0) and room tone |
| MLS Italian train | 40,953 | 169.89 | CC-BY-4.0 |
| LibriSpeech train-clean-100 | 28,370 | 100.00 | CC-BY-4.0 |
| Spliced IT+EN | 20,623 | 110.00 | derived from the above |
| Gold v1.0.0, repeated ×50 | 1,413 | 0.96 (×50 = 48.0) | CC-BY-4.0 |

Training: 40,000 steps, encoder LR 3e-5, head LR 3e-4, EMA 0.9998, bf16,
activation checkpointing, on an M5 Pro (16 GB). **S1-dev and R-S1-cal are held
out**; no S1-dev or R-S1-cal audio was trained on. **No TTS audio is in the
product**: E-027 rendered four permissive voices but the arm tied and was not
adopted (D36). The NOTICE admits permissive TTS as a data class; that is a
policy allowance, not a component of `0.9.8`.

## Evaluation

All numbers use the frozen decoder above and the E-034c LM, INT8, unless stated.
Paired CIs are 95 % utterance bootstrap, 1,000 resamples.

| Set | n | WER | ETA | Notes |
|---|---:|---:|---:|---|
| dev-macro (MLS-IT + LS-EN dev) | 3,939 | **14.3274 %** | — | IT 19.5016 / EN 9.1533 |
| S1-dev (960 held-out S1 takes) | 960 | **13.3820 %** | **75.1465 %** | S1 voice |
| R-S1-cal (400 calibration cues) | 400 | **12.8469 %** | **75.4967 %** | S1 voice, ranking set |
| T-600 (S1+S2+S3, read) | 600 | **12.2993 %** | **78.1400 %** | repeated read, declared |
| T-600, S1 | 200 | 10.0602 % | 81.1940 % | |
| T-600, S2 | 200 | 13.4666 % | 77.6204 % | |
| T-600, S3 (never trained) | 200 | 13.3731 % | 75.7746 % | repeatedly evaluated |
| FLEURS-IT test (filtered) | 697 | **11.8304 %** | — | once, report-only |
| FLEURS-EN test (filtered) | 525 | **28.1955 %** | — | once, report-only |
| CS-FLEURS-READ IT–EN | 176 | **34.8085 %** [32.89–36.81] | — | once, external, one speaker |

The T-600 number is a **declared repeated read** (the set was read in E-004 and
E-023); S3 is held out from training but has been evaluated repeatedly and is
not a fresh holdout. FLEURS and CS-FLEURS were read once, report-only, and were
never used to select anything. CS-FLEURS-READ is CC-BY-NC and test-only.

Browser: the release transcript is exact on the smoke clip at **RTFx 28.48**
(4 threads, isolated, SIMD, load 79.7 ms); 1-thread non-isolated throughput is
10.93× / 19.74× / 21.17× at 5 / 10 / 20 s. The browser WER on S1-dev is
**13.7180 %** vs Python 13.3820 % (**+0.34 pp** [−0.13, +0.79], a tie) with the
min-runs-3 segmentation rule.

## Evidence boundary

- **Scripted, read speech.** Training and the S1/S2/S3 evaluation are read
  cues, not spontaneous conversation. The model is not validated on:
  spontaneous or overlapping speech, far-field/noisy meeting audio, strong
  non-Italian accents, children, or unseen dialects.
- **Known voices.** S1, S2 and S3 are the voices in the recording programme.
  S3 was never trained on, but the model was selected and tuned with S1 and S2
  in the loop. Do **not** claim unseen-speaker generalisation beyond the
  external FLEURS/CS-FLEURS numbers.
- **One external mixed speaker.** CS-FLEURS-READ has one speaker; it is a
  single point, not a speaker range. FLEURS-IT/EN are external read sets with
  no ETA annotation.
- **Repeated T-600.** T-600 has been read more than once; treat it as a
  comparability guard, not an untouched test.
- **Code-switching is read, not spontaneous.** English terms are Italian-
  accented and embedded in Italian sentences. The model does not detect or
  label languages.

## Known weaknesses

- **English terms remain the error class.** On T-600 `switch` ETA is 79.27 %
  and `control_en` 85.42 %; acronym ETA 85.00 %, brand 87.25 %. E-023 improved
  S1-dev but `switch` did not improve on T-600 (−2.03 pp ETA, 60 clips) or
  R-S1-cal (−2.43 pp) and is carried as residual risk (D34).
- **Numbers and digits** are normalised with an Italian `lang_hint`; English-
  read numbers are scored as Italian words. This is a text-normalisation limit,
  visible on CS-FLEURS (27/176 rows).
- **Short utterances and insertions**: S1-dev insertion rate is part of the
  published readout; very short four-word cues are the hardest.
- **Browser vs Python** is a tie but not identical; VAD segmentation, not
  resampling, is the residual cause (E-016b min-runs-3).
- **INT8 only in the product.** The product's FP32 dev-macro is 14.2633 %
  against 14.3274 % INT8 (0.064 pp, CI includes zero); the E-027 TTS seeds
  show 0.05–0.08 pp. The INT8 cost is small and not significant.
- **No contextual biasing / glossary.** E-026's realistic glossary was measured
  and not exposed because the S1-dev veto failed (D35). There is no user
  glossary or keyterm injection in `0.9.8`.

## Provenance and reproducibility

- Acoustic: E-023 R1 **seed 0**, `ckpt_best.pt` at step 40,000, EMA weights
  (sha256 `5ff95f0e…`); seed 0 was chosen by the E-002 §8.3 rule (R-S1-cal
  pooled ETA). Seed 1 was never read on T-600 or FLEURS.
- LM: E-034c clean rebuild (D25), packed lossless MOTHLMB2.
- VAD: 16 kHz Silero (D19); tokenizer pinned (D14).
- Package/browser: `runs/E-023/export-report-0.9.8.json`,
  `runs/E-023/browser-run-0.9.8.json` (the `0.9.7` reports record the same
  runtime bytes).
- Readouts: `runs/E-023/RESULT.md`, `runs/E-004/fleurs-RESULT.md`,
  `runs/E-008/RESULT.md`, `runs/E-017/RESULT.md`.
- Training code and configs: `training/`, `training/configs/base6-e023-r1.toml`.
- Web runtime: `vendor/moth-stt-web/` (provenance and patch documented there).

## Licence and attribution

Code, weights and the release package are Apache-2.0. The release NOTICE
(`tools/release_notice.py`) carries the Whisper and Silero VAD MIT notices,
the Moth tokenizer and LM-corpus sources, and the training datasets: moth-cs
(CC-BY-4.0), Common Voice IT (CC0), MLS Italian and LibriSpeech (CC-BY-4.0),
MUSAN (CC-BY-4.0) and the derived splices. **CS-FLEURS and SwitchLingua
are CC-BY-NC and are evaluation-only**; they are not in training and not
redistributed. No voice cloning of S1/S2/S3 was used; no commercial-API output
was used for labels, text or LM data.

**Erratum (2026-11-25, D37; fixed in `0.9.8-e023-r1`).** The NOTICE inside
`0.9.7-e023-r1` said the decoder LM is "reused unchanged from Moth 0.8.1" and
quoted the unpacked sha256 `4b122fbe…`. The shipped LM is E-034c: MOTH-50's
rebuild of Moth's E-034 recipe on Moth's frozen E-031 corpus, packed as
`7a4332f1…`. The listed sources were the right ones; the description and the
hash were not. The fixed NOTICE ships in `0.9.8-e023-r1`, a repackaging whose
graph, LM, VAD, vocabulary and mel filters are byte-identical to `0.9.7`
(every sha256 checked by `tools/renotice_release.py`); every number in this
card applies to both.
