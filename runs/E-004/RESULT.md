# E-004 — where the freed bytes go: arm B wins

> Executed 2026-11-21 on the M5 Pro. Decode-only except the LM build: no
> acoustic training, no new data, no weights change. **Arm B (unpruned E-034 LM,
> lossless packed) beats A3 on dev-macro, S1-dev and T-600 and fits at 49.21 MB.
> It is adopted as the product package; the acoustic model remains E-005.**

## Arms

The E-012 package left 5,229,567 B of room. The pre-registered arms were:

| Arm | Acoustic | LM | Package | Status |
|---|---|---|---:|---|
| A (control) | base+6 E-005 | A3, packed | 44,770,433 B | measured |
| **B** | base+6 E-005 | **E-034 N2_1_1, unpruned, packed** | **49,210,224 B** | **adopted** |
| C | base+7 | A3, packed | ~47.88 MB est. | not run |
| D | base+4 long | large LM | ~46–49 MB est. | not run |

C and D were conditional: C needed an E-011-length run only if depth still paid,
and E-011 failed the S1-dev ETA veto; D needed base+4 to be competitive at a
long schedule, which was never shown. They are not authorized by this result.

## Arm B build

- Rebuilt the frozen E-031 LM corpus from its receipt (sha256 `c8d94192…`).
- Built E-034 `N2_1_1`: all bigrams, all trigrams, term weight ×1; 12,313,440 B
  MOTHLMB1.
- **The E-034 LM has 1,015 distinct bigram and 873 distinct trigram values**, so
  the 10-bit `MOTHLMB2` is **lossless**: 7,793,632 B, sha256 `86aa7cbf…`.
- Package: 40,039,997 (graph, byte-identical to E-005) + 7,793,632 (LM) +
  1,289,603 (16 kHz VAD) + 86,992 (misc) = **49,210,224 B**, **789,776 B** of
  room.

Decoder calibration on dev-macro (E-009 grid, 20 points) selected the frozen
**0.55 / 2.7**: the best point 0.45 / 2.0 reached 16.908 % versus 16.968 % for
the frozen point, but the 0.06 pp gain is not significant (`adopt: false`).

## Results

### Dev-macro (guard)

| LM | Macro | Δ vs A3 (95 % CI) |
|---|---:|---|
| A3 | 17.6212 % | — |
| **E-034 N2_1_1 packed** | **16.9863 %** | **−0.6349 pp [−0.7609, −0.5056]** |

The gain is smaller than the E-002 1 pp gate, but E-004's pre-registered guard is
dev-macro, not the 1 pp gate; the product sets decide.

### S1-dev (veto)

| LM | WER | ETA | Δ WER (95 % CI) | Δ ETA (95 % CI) | Pass |
|---|---:|---:|---|---|---|
| A3 | 14.0541 % | 75.5371 % | — | — | — |
| **E-034 N2_1_1 packed** | **11.6289 %** | **79.8340 %** | **−2.4251 pp** | **+4.2969 pp** | ✅ |

Both improvements are far outside the noise; no WER or ETA regression. The
per-utterance paired CIs exclude zero in the favourable direction.

### T-600 (read once, final report)

T-600 mel shards were built for this readout (600 clips, 0.39 h).

| LM | WER | ETA | Δ vs A3 (paired) |
|---|---:|---:|---|
| A3 | 13.9908 % | 75.3595 % | — |
| **E-034 N2_1_1 packed** | **13.4748 %** | **76.5101 %** | **−0.5161 pp WER / +1.1505 pp ETA** |

For context, A3 on T-600 is already **−10.09 pp WER / +20.13 pp ETA** versus the
retained 0.8.1 hypotheses; arm B adds another half point of WER and one point of
ETA.

### Browser

The 49.21 MB package loads and verifies every hash in the existing runtime:

| Measure | Result |
|---|---|
| Load | **70 ms** |
| Transcript `T/S1/S1_0001.wav` | **"ho spostato il meeting a domani"** — exact |
| Single run | 251.6 ms, RTFx 15.86 |
| Benchmark median (5 runs) | 140.7 ms, **RTFx 28.36** |

## Decision

- **Adopt arm B.** The product package is E-005 acoustic + E-034 N2_1_1 packed +
  16 kHz VAD, version `0.9.5-e004-lm`, **49,210,224 B**. The acoustic checkpoint,
  the decoder 0.55 / 2.7 and the VAD are unchanged.
- The A3 LM remains the historical E-002 gate reference and the package for
  reproducing earlier results. E-004 B is a different, pre-registered LM arm, so
  its readout is `gate_eligible: false` by design; it was judged on the E-004
  rule (dev-macro guard + S1-dev WER/ETA veto + T once), not the A3 1 pp gate.
- This answers E-004: the freed bytes go to **LM coverage**, not more depth. C
  (base+7) and D (base+4 long) are not run; E-011's product veto removed their
  precondition.
- A term-weighted ×3 version of the unpruned LM is not tested; the ×1 arm already
  improves ETA by 4.30 pp, so it is a possible later arm, not a prerequisite.

## Artifacts

- `calib-e034.json` — 20-point decoder grid with the new LM.
- `lm-pack-e034.json`, `lm_e034_n2_mothlmb2.bin` (LM binary gitignored).
- `readout-e034-int8.json/.hyp.jsonl` — dev-macro vs E-005 A3.
- `s1-dev-e034-int8.json/.hyp.jsonl` — S1-dev veto.
- `t600-a3-int8.json/.hyp.jsonl`, `t600-e034-int8.json/.hyp.jsonl` — T-600 once.
- `export-report-e034.json`, `release-e034/` (package gitignored).
- `browser-run-e034.json` — load, exact transcript, RTFx.

## Reproduce

```bash
cd ../moth-stt
uv run --frozen --project . --extra training -- \
  python runs/E-031/reproduce_corpus.py --out runs/E-031/frozen/lm-corpus.jsonl
uv run --frozen --project . --extra training -- python runs/E-034/build_variants.py
cd ../moth-50
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/pack_lm.py --lm ../moth-stt/runs/E-034/build/N2_1_1.bin \
  --out runs/E-004/lm_e034_n2_mothlmb2.bin --bits 10 \
  --report runs/E-004/lm-pack-e034.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/calibrate_decoder.py \
  --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml \
  --lm ../moth-stt/runs/E-034/build/N2_1_1.bin --allow-other-lm \
  --workers 8 --out runs/E-004/calib-e034.json
```
