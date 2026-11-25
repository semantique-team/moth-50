# E-027 — TTS fine-tune (E-021 arm 2): a tie on calibration, not adopted

**Result 2026-11-24.** Four 8,000-step fine-tunes from R1 seed 0 (two seeds per
arm) were rendered, trained, exported and read once with the frozen decoder and
the E-034c LM. The TTS arm ties the same-length control on R-S1-cal (**gate 1
fails**) and is **+0.150 pp WER** [+0.038, +0.269] against the product on
dev-macro (**gate 4 fails**). Per the pre-registered stop rule E-027 is **not
adopted and the model line closes**: the product stays `0.9.7-e023-r1` (D27),
and the next work is E-008 (CS-FLEURS-READ once) plus the model card (D36).

Pre-registration: `experiments/E-027-e021-tts.md`. Everything below follows the
commands and the gates written there before any rendering.

## TTS data

- **Script.** 3,522 E-025 sentences → **3,411** kept; 107 dropped for digits, 4
  for a word 4-gram shared with S1-dev, T, A or R-S1-cal; no length or duplicate
  drops. 2,816 rows carry English; 8,439 English pieces. `runs/E-027/tts-script.jsonl`,
  `tts-script-report.json`.
- **Voices.** Pinned and sha256-verified: Kokoro-82M v1.0 `if_sara` and
  `im_nicola` (it and mixed G2P), Piper `it_IT-paola-medium` and
  `it_IT-serena-medium` (it). All Apache-2.0 / MIT with the recorded data
  attributions; nothing ships.
- **Render.** **19,276 clips, 20.827 h, 0 QC drops** across six voice/mode
  strata (`sara-it`, `sara-mix`, `nicola-it`, `nicola-mix`, `paola-it`,
  `serena-it`). Versions: kokoro 0.9.4, misaki 0.9.4, piper-tts 1.8.0,
  onnxruntime 1.31.0, torch 2.14.1, soundfile 0.14.0.
  `data/manifests/tts_e027.jsonl`, `runs/E-027/render-report.json`.
- **Listening audit.** 60 seeded clips, 10 per stratum. One fail:
  `E027-03390-nicola-it` (unintelligible/wrong words). One fail is allowed per
  stratum, so **no stratum was dropped**. `runs/E-027/audit.jsonl`,
  `audit-report.json`, `audit.html`, `audit-listening.md`.
- **Share.** `repeat = 3`, i.e. **9.49 %** of the epoch audio (0.10 / 0.90 ×
  595.702 h / 20.827 h), committed before the first step
  (`runs/E-027/share.json`, `base6-e027-tts.toml`). Mel shards: 19,276 clips,
  1 file, 20.83 h, 0 errors.

## Training

All four runs: base+6, 8,000 steps, `lr_head` 1e-4 / `lr_encoder` 1e-5, EMA
0.999, bf16, checkpointing, from R1 s0 `ckpt_best.pt` (EMA weights, sha256
`5ff95f0e…`, step 40,000), fresh optimizer/schedule. **0 non-finite steps** in
all four. The control/TTS configs differ only by the TTS source.

| Run | Greedy dev (monitoring) | Peak GB | Wall |
|---|---:|---:|---:|
| control s0 | 17.18 % | 8.759 | 3.30 h |
| control s1 | 17.26 % | 8.737 | 0.67 h (resumed from step 6,500 after a power loss) |
| tts s0 | 17.45 % | 9.719 | 3.14 h |
| tts s1 | 17.46 % | 9.713 | 3.14 h |

The readout model is `ckpt_last.pt` at step 8,000 with EMA weights; greedy
`ckpt_best` selection was monitoring only.

## Readout (INT8, frozen 0.55 / 2.7, beam 12, top-k 10, E-034c LM)

Per-run, percentage WER / ETA:

| Run | R-S1-cal (400) | S1-dev (960) | dev-macro (IT / EN) |
|---|---:|---:|---:|
| control s0 | 12.8021 / 75.8940 | 13.0168 / 75.2930 | 14.2898 (19.3630 / 9.2167) |
| control s1 | 13.2498 / 75.4967 | 13.7180 / 75.3418 | 14.3278 (19.4743 / 9.1813) |
| tts s0 | 12.7574 / 75.4967 | 13.4551 / 74.7559 | 14.4716 (19.6649 / 9.2782) |
| tts s1 | 12.2650 / 76.6887 | 13.4843 / 75.0488 | 14.4834 (19.6624 / 9.3043) |
| **control seed mean** | **13.0260 / 75.6954** | **13.3674 / 75.3177** | **14.3088** |
| **tts seed mean** | **12.5112 / 76.0927** | **13.4697 / 74.9023** | **14.4775** |
| product (E-023 s0) | 12.8469 / 75.4967 | 13.3820 / 75.1465 | 14.3274 |

FP32 dev-macro for the INT8 cost: tts s0 **14.4199 %** (INT8 − FP32 = +0.0517
pp), tts s1 **14.4057 %** (+0.0777 pp); seed mean **+0.065 pp**.

## Gate (pre-registered; A = TTS, C = control, P = product R1 s0)

Differences are A − reference; pp with paired 95 % CI, 1,000 bootstrap
resamples, seed 0.

| # | Comparison | WER | ETA | Verdict |
|---:|---|---:|---:|---|
| 1 | cal A vs C | **−0.515 [−1.367, +0.271]** | **+0.397 [−1.223, +2.115]** | **FAIL (tie)** |
| 2 | cal A vs P | −0.336 [−1.136, +0.447] | +0.596 [−1.104, +2.371] | pass (no loss) |
| 3 | S1-dev A vs P | +0.088 [−0.378, +0.515] | −0.244 [−1.265, +0.719] | pass (veto holds) |
| 4 | dev-macro A vs P | **+0.150 [+0.038, +0.269]** | — | **FAIL** |
| 5 | INT8 cost, dev-macro | +0.065 pp | — | pass (≤ 0.2 pp) |

Gate 1 requires a WER CI entirely below zero **or** an ETA CI entirely above
zero; neither holds. Per-seed on cal, A vs C: s0 −0.269 [−1.292, +0.695] WER /
−0.199 [−2.454, +1.997] ETA; s1 −0.761 [−1.664, +0.111] / +0.993 [−0.733,
+2.802]. The dev-macro loss is consistent across seeds: +0.144 [+0.005, +0.293]
and +0.156 [+0.022, +0.289]. Against the product it is on **both** languages
(seed means: IT 19.6637 vs 19.5016, **+0.16 pp**; EN 9.2913 vs 9.1533, **+0.14
pp**); against the control it is mostly Italian (IT +0.25 pp, EN +0.09 pp).
*Corrected 2026-11-25: the first version said "mostly Italian (+0.30 pp IT,
+0.06 pp EN)", which is seed 0 of the arm against seed 0 of the control, not
the gate comparison.*

**Report-only, C vs P** (the effect of the fine-tune alone): cal +0.179
[−0.564, +0.916] WER / +0.199 [−1.262, +1.743] ETA; S1-dev −0.015 [−0.408,
+0.355] / +0.171 [−0.689, +0.945]; dev-macro −0.019 [−0.130, +0.087]. The
same-length fine-tune is neutral, so there is no fallback to adopt.

**Report-first categories** (seed means, pp):

| Category | cal A vs C WER / ETA | S1-dev A vs P WER / ETA |
|---|---:|---:|
| acronym | **−2.29 / +10.78** | −0.42 / +2.65 |
| brand | **−2.32 / +4.81** | +0.43 / +0.00 |
| single_word | −3.26 / +9.00 | +0.55 / −1.79 |
| `switch` | **+2.92 / −5.56** | +1.34 / −2.83 |
| `control_en` | +1.17 / −0.67 | −0.47 / +0.35 |
| `control_it` | +0.00 / +0.00 | +0.29 / +0.00 |
| phrase | −0.73 / −0.46 | −0.17 / +0.17 |
| italianized | +0.00 / +2.00 | −1.34 / +2.68 |

The TTS data moved the intended targets on cal (acronyms, brands, single
words) but gave back the gain on `switch`; the overall WER and ETA are ties.
Category-level numbers are report-only (E-002 §8.4); the post-hoc look below
shows the `switch` loss repeats on S1-dev.

## Decision

**Not adopted; the model line closes (D36).** The product stays E-023 R1 seed 0
+ E-034c (`0.9.7-e023-r1`): no package, browser, decoder or LM change, and
**T-600 and FLEURS were not read** (not triggered). The TTS renders, shards and
four training runs stay as evidence; nothing is shipped.

The stop rule was written before rendering: a tie on gate 1 ends the model
line. The result also illustrates why: TTS can move the targeted terms in the
right direction, but on this calibration size it does not move the sentence
WER, and it costs a small, CI-firm dev-macro regression. The next work is
**E-008** (CS-FLEURS-READ once, the final external number) and the **model
card**; E-006 arm 1 (VoxPopuli) and E-019 arm A remain possible but are not
planned.

## Post-hoc look: why `switch` got worse (report-only, added 2026-11-25)

Read after the verdict from the committed hypotheses and gate files; it changes
nothing above and is not a rescue.

**1. The `switch` loss is firm and repeats on S1-dev.** Seed means, pp, 95 % CI:

| Set, comparison | `switch` WER | `switch` ETA |
|---|---:|---:|
| R-S1-cal, A vs C (50 clips) | +2.92 [+0.51, +5.71] | −5.56 [−10.26, −1.44] |
| S1-dev, A vs P (128 clips) | +1.34 [−0.08, +2.82] | −2.83 [−5.63, −0.11] |

Both seeds move the same way on cal. The single-term categories move the
other way on cal: `single_word` −3.26 [−6.08, −0.86] / +9.00 [+3.00, +16.00],
`acronym` −2.29 [−5.02, +0.19] / +10.78 [+2.94, +20.59], `brand` −2.32 [−4.21,
−0.39] / +4.81 [−2.88, +12.50]. With eight categories some intervals exclude
zero by chance; the pattern across both sets is the finding, not one interval.

**2. Which terms moved.** Term errors made by both TTS seeds and by neither
control seed ("new"), against errors made by both control seeds and by neither
TTS seed ("fixed"): R-S1-cal 14 fixed / 9 new, S1-dev 19 fixed / 20 new.

- Fixed on cal: `sql`, `csv`, `gif`, `tcp`, `http`, `https`, `token`, `update`,
  `mouse`, `samsung`, `uniqlo`; on S1-dev also `casio`, `paramount`,
  `security`, `badge`, `onboardato`, `uploadato`, but also common words
  (`our`, `let's`, `after`, `near`).
- New (cal and S1-dev together): common English words inside phrases, often turned into an
  Italian look-alike: `keep me` → "ki mi", `ask me` → "askmi", `you` → "io",
  `sense` → "sends", `eat` → "need", `date` → "ofdate"/"todate", `weird` →
  "word", `happy`, `small`, `boat`.

**3. A likely cause: the `it` mode also read whole English sentences.** The
pre-registered `it` mode sends the entire sentence through the Italian G2P,
`control_en` rows and long English runs included. In the rendered manifest:

| Row content | `it` mode | `mix` mode |
|---|---:|---:|
| English sentence (`control_en`) | 1,668 clips, 1.69 h | 834 clips, 0.89 h |
| mixed, an English run of ≥ 3 words | 336 clips, 0.47 h | 168 clips, 0.26 h |
| mixed, English runs of 1–2 words | 9,260 clips, 9.75 h | 4,630 clips, 5.63 h |
| Italian only | 2,380 clips, 2.14 h | — |

About **2.2 h, 10 % of the TTS audio, pairs English sentences and phrases with
Italian-phonemized speech**, nearly twice their English-phonemized renderings
(1.15 h). An Italian reading is realistic for a technical term and wrong for
"keep me" or "you", which matches the shape of the term lists. The audit rule
("an Italian accent on English terms is not a failure") could not catch it:
`E027-04030-sara-it`, a fully English sentence read with the Italian G2P,
passed, and `E027-03390-nicola-it`, the same kind of row, was the one fail.

This is a hypothesis from one experiment, and the cause is a design choice of
the pre-registration, not an execution error. A redesign would keep the Italian
G2P for isolated terms only and render every English run of two or more words,
and every `control_en` row, with `en-us`; it would still have to remove the
dev-macro cost, which against the product is on both languages. It is **not
planned** (D36) and would need its own pre-registration and go-ahead.

## Artifacts

- `runs/E-027/` — `tts-script.jsonl` + report, `render-report.json`,
  `audit.jsonl`/`audit-report.json`/`audit.html`/`audit-listening.md`,
  `share.json`, `build-mel.log`, `preflight-{control,tts}.json`, `dry-{control,tts}.log`,
  the four run directories (`console.log`, `train.log.jsonl`, metrics,
  checkpoints), `export-{run}.log`, `release-{run}/`, the 14 readout
  `*.json`/`*.hyp.jsonl`, and `gate/*.json`.
- `data/manifests/tts_e027.jsonl`, `data/raw/tts_e027/` (gitignored audio),
  `data/shards/tts_e027/` (gitignored).
- Configs: `training/configs/base6-e027-control.toml`,
  `training/configs/base6-e027-tts.toml`.
- `tools/e027_tts_script.py`, `e027_tts_render.py`, `e027_audit.py`,
  `e027_share.py`; `tests/test_e027_tts.py`.
