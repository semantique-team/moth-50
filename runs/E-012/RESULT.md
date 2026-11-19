# E-012 — package bytes: 16 kHz VAD and lossless LM packing

> Executed 2026-11-19 on the M5 Pro. **Both gates pass**: (a) the 16 kHz-only
> Silero graph is bit-identical on real audio and in the browser; (b) the lossless
> `MOTHLMB2` LM leaves every hypothesis byte-identical. No training, no model or
> decoder change. Runtime vendoring remains E-016.

## Gates (pre-registered)

- **(a)** `drop_in: true` on real audio — identical window probabilities within
  1e-4, gates and segments — **and** a browser run of a package built with
  `export_release.py --vad 16k`: load, exact transcript of
  `T/S1/S1_0001.wav`, RTFx.
- **(b)** A3 has ≤ 1,024 distinct log-probs per order, the packed file
  round-trips exactly, and the dev-macro + S1-dev hypotheses are identical to
  `MOTHLMB1`.

See `experiments/next-experiments.md` §E-012.

## (a) 16 kHz-only Silero — PASS

`tools/probe_vad.py` ran Moth's own VAD path (`stt.vad`: 512 + 64-sample
windows, state carry, 0.5 window threshold, 0.10 clip gate, `segment_bounds`)
with both graphs on 2,040 real clips, 9,268.29 s (2.57 h):

| Set | Clips |
|---|---:|
| S1-dev | 960 |
| T | 600 |
| noise (silence, room tone, MUSAN, music) | 480 |

| Measure | Result |
|---|---|
| Max window-probability gap | **0.0** (bit-identical) |
| Window flips at 0.5 | 0 |
| Clip-gate disagreements | 0 |
| Segment-bound disagreements | 0 |
| **drop_in** | **true** |
| CPU time | 17.10 s reference / 17.76 s candidate |

A VAD-only package (46,798,588 B) then loaded in 393 ms and transcribed
`T/S1/S1_0001.wav` exactly at RTFx 23.90 (4 threads, 5-run median), versus 22.29
for E-005. Artifacts: `vad-parity.json`, `browser-run-vad16k.json`,
`export-report-vad16k.json`.

## (b) Lossless `MOTHLMB2` — PASS

### Format

`MOTHLMB1` spends 4 B per n-gram entry (uint16 token + int16 log-prob) and 8 B
per trigram context (uint32 key + uint32 offset). A3 stores only **1,018
distinct bigram and 859 distinct trigram log-probs**, so every value fits in a
10-bit codebook. `MOTHLMB2` (`tools/moth_lm.py`) stores:

- an int16 codebook per order (2 × 2,048 B) — the exact values, not fitted;
- each entry as `token (11 bits) | code (10 bits)`;
- the trigram keys as a uint32 `prev2` offset table + an 11-bit `prev1` array;
- trigram row offsets at the minimal width (20 bits for A3).

Scores are computed as `int(codebook[code]) * log_scale`, the same arithmetic as
`stt.lm.BinaryBPELM`; the format has its own magic, so the validator and both
readers dispatch on it.

| Measure | A3 |
|---|---:|
| `MOTHLMB1` | 5,381,992 B |
| `MOTHLMB2`, 10-bit lossless | **3,353,840 B** |
| Saved | **2,028,152 B** (ratio 0.62316) |
| `MOTHLMB2` sha256 | `6b57bb41…d821` |

### Verification

- **Round trip exact**: `tools/pack_lm.py` decodes the packed file and compares
  every array (tokens, int16 log-probs, row offsets, keys, unigram) and the
  header values; the size formula matches the written file.
- **Python scores**: 225,818 queries (stored trigram entries, bigram hits,
  random contexts and fallbacks) gave **exactly** the same values as
  `BinaryBPELM`.
- **JS scores**: the `lm.js` reader (local Moth commit `0e15442`) gave 0
  mismatches on 980 cases, loading the 3.35 MB file in 33.9 ms; `MOTHLMB1`
  still scores 0 mismatches on the same fixture.
- **Dev-macro**: WER **23.4415 % IT / 11.8010 % EN**, macro **17.6212 %** —
  identical to E-005 INT8. The hypothesis file is **byte-identical**
  (`80b60da6…42b4`).
- **S1-dev**: WER **14.0541 %**, ETA **75.5371 %** — identical to E-005 INT8.
  The hypothesis file is **byte-identical** (`d43af295…cb29`).

Artifacts: `lm-pack.json`, `lm_scores_mothlmb2.json`, `js-lm-parity.json`,
`readout-dev-mothlmb2.json`, `s1-dev-mothlmb2.json` and their `.hyp.jsonl`
files (the packed LM itself is gitignored; regenerate it with `pack_lm.py`).

## Final package — 44,770,433 B

`export_release.py` with the packed LM, the 16 kHz VAD, A3 and the frozen
decoder 0.55 / 2.7:

| File | Bytes |
|---|---:|
| `model.int8.onnx` | 40,039,997 |
| `lm_bpe3.bin` (MOTHLMB2 A3) | 3,353,840 |
| `silero_vad.onnx` (16 kHz) | 1,289,603 |
| `mel_filters.bin` | 64,320 |
| `vocab.json` | 17,193 |
| `config.json` | 1,360 |
| `NOTICE` | 4,120 |
| **Total** | **44,770,433** |
| Room to 50,000,000 | **5,229,567** |

Versus E-005 (47,836,502 B): **−3,066,069 B**. Versus the VAD-only package
(46,798,588 B): **−2,028,155 B** (2,028,152 B of LM plus a 3-byte shorter version
string). Every asset hash is verified; INT8 top-1 agreement is 1.0.
Artifact: `export-report-mothlmb2.json` (`release-lm2/` is gitignored).

With the packed A3 and the 16 kHz VAD, the planning tool puts **base+7** at
47.88 MB (2.12 MB of room), so E-004's depth arm is now inside the budget.

## Browser — PASS

Served the packed package with `moth-stt/web/` at local commit `0e15442`
(MOTHLMB2 reader) plus the `budget_bytes` patch `b0fa11c`, behind COOP/COEP on
127.0.0.1.

| Measure | Result |
|---|---|
| Release load | **413 ms** |
| Isolation / threads / SIMD | true / 4 / true |
| Transcript `S1_0001.wav` | **"ho spostato il meeting a domani"** — exact |
| Single run (3.989 s audio) | 277.77 ms total, **RTFx 14.36** |
| Benchmark median (2 warm-ups + 5 runs) | 162.00 ms total, **RTFx 24.63** |
| Decode median | 9.97 ms (VAD-only package: 14.56 ms) |
| E-005 A3, same method | RTFx 22.29 |

The first run pays WASM warm-up; the benchmark median is the comparable number.
This measures 4 threads only — the 1-thread gate is E-016.

Artifact: `browser-run-mothlmb2.json`.

## Decision

- **E-012 passes.** `export_release.py --lm-format` and
  `frozen_readout.py --lm-format` now default to `mothlmb2`; `--vad` already
  defaults to `16k` (D19). Both old encodings stay selectable for reproduction.
  Recorded as D20.
- The validator accepts both magics; `vendor/moth-stt/PROVENANCE.md` records the
  packaging-only change.
- **Runtime vendoring is still E-016.** The JS reader is a local commit in the
  archived Moth repository; `web/` must be copied into MOTH-50 before the
  package stops depending on that checkout.
- No readout or hypothesis changes: the LM encoding is lossless and the VAD only
  gates audio; both parity probes show identity on real data.

## Reproduce

```bash
# (a) VAD parity and package
python tools/release_assets.py fetch vad vad-16k
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/probe_vad.py --moth-stt ../moth-stt \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --audio-root ../moth-cs-dataset \
  --manifest ../moth-cs-dataset/manifests/test.jsonl --audio-root ../moth-cs-dataset \
  --manifest data/manifests/noise.jsonl --audio-root data/raw/noise \
  --out runs/E-012/vad-parity.json
# (b) pack A3 and verify the round trip
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/pack_lm.py --lm data/raw/moth-0.8.1/lm_bpe3.bin \
  --out runs/E-012/lm_a3_mothlmb2.bin --bits 10 --report runs/E-012/lm-pack.json
# (b) hypotheses with the packed LM
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode macro \
  --onnx runs/E-005/release-a3/model.int8.onnx \
  --config training/configs/base6-long.toml --lm-format mothlmb2 \
  --out runs/E-012/readout-dev-mothlmb2.json \
  --hyp-out runs/E-012/readout-dev-mothlmb2.hyp.jsonl
# final package (both defaults)
uv run --frozen --project ../moth-stt --extra export -- \
  python tools/export_release.py \
  --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml --out-dir runs/E-012/release-lm2 \
  --version 0.9.3-e012-lm2
```
