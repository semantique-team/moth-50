# E-012(a) — 16 kHz-only Silero VAD: real audio and browser

> Executed 2026-11-19 on the M5 Pro. This is **E-012(a) only**. E-012(b), the
> lossless `MOTHLMB2` LM packing, has not run; the runtime vendoring is E-016.
> No training, no model or decoder change.

## Gate (pre-registered)

`drop_in: true` on real audio — identical window probabilities within 1e-4,
gates and segments — **and** a browser run of a package built with
`export_release.py --vad 16k`: load, exact transcript of
`T/S1/S1_0001.wav`, RTFx. See `experiments/next-experiments.md` §E-012.

## Real-audio parity — PASS

`tools/probe_vad.py` ran Moth's own VAD path (`stt.vad`: 512 + 64-sample
windows, state carry, 0.5 window threshold, 0.10 clip gate, `segment_bounds`)
with both graphs on 2,040 clips, 9,268.29 s (2.57 h):

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

Artifact: `runs/E-012/vad-parity.json`.

## Package — 46,798,588 B

`tools/export_release.py --vad 16k` from the E-005 `ckpt_best.pt` (step 16,000),
A3 LM, FR-16k config, frozen decoder 0.55 / 2.7:

| File | Bytes |
|---|---:|
| `model.int8.onnx` | 40,039,997 |
| `lm_bpe3.bin` (A3) | 5,381,992 |
| `silero_vad.onnx` (16 kHz) | 1,289,603 |
| `mel_filters.bin` | 64,320 |
| `vocab.json` | 17,193 |
| `config.json` | 1,363 |
| `NOTICE` | 4,120 |
| **Total** | **46,798,588** |
| Room to 50,000,000 | **3,201,412** |

Versus the E-005 A3 package (47,836,502 B): **−1,037,914 B**. The VAD itself
saves 1,037,921 B; `config.json` grew 7 B because the version string is longer.
Every asset hash is verified; INT8 top-1 agreement is 1.0 (the acoustic graph is
unchanged). Artifacts: `runs/E-012/export-report-vad16k.json` (`release-16k/` is
gitignored).

## Browser — PASS

Served the package with the archived `moth-stt/web/` runtime (including the
local `budget_bytes` patch) behind COOP/COEP on 127.0.0.1.

| Measure | Result |
|---|---|
| Release load | **393 ms** |
| Isolation / threads / SIMD | true / 4 / true |
| Transcript `S1_0001.wav` | **"ho spostato il meeting a domani"** — exact |
| Single run (3.989 s audio) | 278.66 ms total, **RTFx 14.32** |
| Benchmark median (2 warm-ups + 5 runs) | 166.90 ms total, **RTFx 23.90** |
| E-005 A3, same method | RTFx 22.29 |

The first run pays WASM warm-up; the benchmark median is the comparable number.
This measures 4 threads only — the 1-thread gate is E-016.

Artifact: `runs/E-012/browser-run-vad16k.json`.

## Decision

- **E-012(a) passes; the 16 kHz-only graph becomes the export default**
  (`export_release.py --vad` default `16k`; `--vad full` stays available for
  reproducing 0.8.1-shaped packages). Recorded as D19.
- E-012 is **not** complete: (b) lossless LM packing and the runtime vendor
  (E-016) remain. The ~2.02 MB packing estimate is still unproven.
- No readout or hypothesis changes: the VAD only gates which audio reaches the
  model, and the parity probe shows identical gates and segments on all 2,040
  real clips.

## Reproduce

```bash
python tools/release_assets.py fetch vad vad-16k
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/probe_vad.py --moth-stt ../moth-stt \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --audio-root ../moth-cs-dataset \
  --manifest ../moth-cs-dataset/manifests/test.jsonl --audio-root ../moth-cs-dataset \
  --manifest data/manifests/noise.jsonl --audio-root data/raw/noise \
  --out runs/E-012/vad-parity.json
uv run --frozen --project ../moth-stt --extra export -- \
  python tools/export_release.py \
  --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml --out-dir runs/E-012/release-16k \
  --version 0.9.2-e012-vad16k --vad 16k
```
