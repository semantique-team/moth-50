# E-005 — RESULT: long fine-tune of the E-003 winner

> Executed 2026-11-18 07:43–14:27 CET on the M5 Pro. `base+6`, 16,000 optimizer
> steps, pretrained Whisper-base encoder, identity-initialized extra blocks,
> frozen E-002 data mixture. The frozen beam+LM readout was run on 2026-11-19.

## Erratum (2026-11-19): the readout and the package used the pre-A3 LM

Every "A3" below is wrong. `tools/frozen_readout.py` and
`tools/export_release.py` loaded `../moth-stt/release/lm_bpe3.bin`, a stale,
gitignored copy of the LM that Moth shipped from 0.5.0 to 0.8.0 (sha256
`dcf1d59f…`, 5,116,464 B). The 0.8.1 A3 LM, which decoded the 23.0170 %
reference, is `3be1177b…` (5,381,992 B; `data/sources.lock.json`). The package
also shipped Moth 0.8.1's NOTICE byte for byte (sha256 `6ca7672f…`), which names
`openai/whisper-tiny` as the encoder.

- **The gate verdict stands.** A3 against the pre-A3 LM was −0.097 pp on Moth's
  dev (Moth E-031: 22.9463 → 22.8494 %), and the comparison here handicapped
  MOTH-50, not the reference; the margin is 5.35 pp.
- **The numbers are not A3 numbers.** They are MOTH-50 + pre-A3 LM against
  0.8.1 + A3. The package lacks A3's ×3 term weighting (rare-term ETA 28.4 →
  46.9 % in Moth E-031), so it must not be used for S1-dev, T-600 or ETA claims.
- **With A3 the package is ~47.84 MB** (2.16 MB of room, not 2.43), and its
  NOTICE must be the MOTH-50 one.
- **16,000 steps are 1.19 epochs**, not ~1.4: epoch 0 ended after 13,458
  batches (`base+6-16k/train.log.jsonl`).

Fixed in the tools (D17): the LM is pinned by sha256 and fetched by
`tools/release_assets.py`, any other LM needs `--allow-other-lm`, and
`tools/release_notice.py` writes the MOTH-50 NOTICE. The A3 readout, the A3
release and the S1-dev readout are the E-005 addendum in
`experiments/next-experiments.md`.

## Verdict

Two results, both on the same checkpoint (`ckpt_best.pt`, step 16,000):

- the long schedule cut greedy dev-macro WER from 33.46 % (E-003 `base+6`,
  4,000 steps) to **21.7540 %**, still improving at the end;
- the frozen beam+A3 readout gives **17.5816 % dev-macro** against the
  **23.0170 %** reference: a paired improvement of **−5.4354 pp**, 95 % CI
  **[−5.6898, −5.1637]**, which meets the E-002 dev-macro gate.

## Greedy curve (monitoring)

| Step | Greedy dev-macro WER |
|---:|---:|
| 2,000 | 42.37 % |
| 4,000 | 35.54 % |
| 6,000 | 30.73 % |
| 8,000 | 27.21 % |
| 10,000 | 24.95 % |
| 12,000 | 22.92 % |
| 14,000 | 22.23 % |
| **16,000** | **21.75 %** |

The last 2,000 steps bought 0.48 pp, so the curve is flattening but not flat.

## Frozen readout (E-002 gate)

Decoder: beam 12, top-k 10, lm_weight 0.55, token_bonus 2.7, `release/lm_bpe3.bin`
(sha256 `dcf1d59f…`: the pre-A3 LM, see the erratum). Same dev manifests and same reference hypotheses as
`runs/E-002/dev-reference.json`; 1,000 bootstrap resamples, seed 0.

| Set | n | Ours WER | 95 % CI | 0.8.1 reference |
|---|---:|---:|---|---:|
| MLS-IT dev | 1,248 | **23.3771 %** | [22.6762, 24.0675] | 28.1040 % |
| LS-EN dev | 2,691 | **11.7861 %** | [11.4463, 12.1617] | 17.9299 % |
| **Macro** | 3,939 | **17.5816 %** | — | **23.0170 %** |

Paired macro bootstrap: point **−5.4354 pp**, CI **[−5.6898, −5.1637]**,
`tie: false`. Gate rule (≥ 1.0 pp better and CI excluding zero): **pass**.

## INT8 release package

Built with `tools/export_release.py` using Moth's 0.8.1 recipe (fp32 ONNX →
dynamic per-channel QInt8), plus the A3 LM, the Silero VAD and the frozen decode
settings.

| File | Bytes |
|---|---:|
| `model.int8.onnx` | 40,039,997 |
| `lm_bpe3.bin` | 5,116,464 (pre-A3; A3 is 5,381,992) |
| `silero_vad.onnx` | 2,327,524 |
| `mel_filters.bin` | 64,320 |
| `vocab.json` | 17,193 |
| `config.json` + `NOTICE` | 4,587 |
| **Total** | **47,570,085** |

Budget 50,000,000 B → **2,429,915 B (2.43 MB) remaining**. The vendored
`moth_release.validate_release` accepts the folder and verifies every hash.

Quality with the same beam+A3 readout:

| Set | FP32 | INT8 | Δ |
|---|---:|---:|---:|
| MLS-IT dev | 23.3771 % | 23.4910 % | +0.1139 pp |
| LS-EN dev | 11.7861 % | 11.8346 % | +0.0485 pp |
| **Macro** | **17.5816 %** | **17.6628 %** | **+0.0812 pp** |

Against the 0.8.1 reference (INT8, same decoder): paired **−5.3542 pp**, 95 % CI
**[−5.6166, −5.0735]** — the gate still passes, now at like-for-like precision.

Export fidelity: the FP32 graph's max logit gap is 4.0e-3 with top-1 agreement
1.0; the INT8 graph's max is 31.1 with top-1 agreement **1.0**. The large max is
a confidence outlier on a single frame, not a decoding failure — the WER cost is
0.08 pp. It is reported for transparency, not used as a quality claim.

## Browser run

The 47.57 MB package loaded in the in-app browser (Chromium, WASM, 4 threads,
cross-origin isolated, SIMD) and transcribed the T-600 clip
`audio/T/S1/S1_0001.wav` exactly: **"ho spostato il meeting a domani"**.

| Measurement | Value |
|---|---|
| Release load | 63 ms (cached) |
| Cross-origin isolated / threads / SIMD | true / 4 / true |
| Single 3.99 s clip | 229 ms total, RTFx **17.4** |
| Benchmark (2 warm-ups + 5 runs) | median 175.6 ms, RTFx **22.7** (inference 44.9) |
| VAD / features / inference / decode (median) | 12.4 / 57.7 / 88.9 / 14.6 ms |

One compatibility change was required: `web/ort.js` now reads an optional
`config.budget_bytes` (default 20,000,000, so the 0.8.1 release is unchanged);
this release declares 50,000,000. Without it the page rejects the package at the
old 20 MB ceiling. Evidence: `runs/E-005/browser-run.json`.

## Run

- Model: 39,259,649 params, base+6, `model.init` = converted
  `openai/whisper-base` (`data/raw/whisper-base-encoder.pt`).
- Schedule: 16,000 steps, warmup 1,000, cosine to 0.05; AdamW head 3e-4 /
  encoder 3e-5; 192 audio-s / 32-clip cap; per-block checkpointing.
- Data: the frozen E-002 mixture (273,714 manifest rows, ~595 h with the gold
  ×50 repeat).
- Wall time **6.73 h**, peak MPS memory **8.604 GB**, 0 non-finite steps.
- Checkpoints every 100 steps; `ckpt_best.pt` is step 16,000.
- Readout: **449.5 s** FP32 and **~7.5 min** INT8 on CPU; artifacts
  `runs/E-005/readout.json`, `readout-int8.json`, `release/`.

## Caveats

- The browser run used a whole clip through the demo page; it is not a
  microphone or VAD-segmentation test, and the readout does not apply the VAD.
- **S1-dev is still incomplete** (300/960 clips), so the gate's "no S1-dev
  regression outside its CI" half is pending. The dev-macro half is met on both
  FP32 and INT8.
- Dev was used for checkpoint selection (`ckpt_best`), as the protocol allows.
- No hotwords, no per-set decoder tuning; the frozen settings are unchanged.

## Next

0. The E-005 addendum (`experiments/next-experiments.md`): A3 readout, A3
   release with the MOTH-50 NOTICE, S1-dev no-regression readout.
1. Finish S1-dev, then run its readout (same decoder) and the no-regression
   comparison.
2. E-004 package/LM trade-off (base+6 + 5.38 MB LM vs base+4 + 12 MB LM) can now
   use the measured +6 acoustic advantage.
3. If the product adopts MOTH-50, carry the `budget_bytes` runtime change in
   the MOTH-50 tree: `moth-stt` is archived and read-only, so the local commit
   there (`b0fa11c`) cannot be pushed.
