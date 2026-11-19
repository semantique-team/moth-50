# Probe 2026-11-19 — Silero 16 kHz-only graph parity (E-012 preflight)

**Status: measured on synthetic audio only.** This is not E-012: it checks that
the candidate is worth the real run, and that `tools/probe_vad.py` works. It
trains nothing and changes no release.

- **Question:** the shipped `silero_vad.onnx` (2,327,524 B, sha256 `1a153a22…`)
  holds the 8 kHz and 16 kHz branches behind an `If`; MOTH-50 only feeds 16 kHz.
  Is `silero_vad_16k_op15.onnx` from the same Silero commit (`bfdc019`, "add v6.2
  model"; 1,289,603 B, sha256 `7ed98ddb…`) a drop-in replacement?
- **Interface:** identical — inputs `input` [batch, samples], `state` [2, batch,
  128], `sr` int64; outputs `output` [batch, 1], `stateN`. The 0.8.1 web runtime
  and `stt.vad` feed exactly these names, so no runtime change is needed.
- **Audio:** 16 Windows SAPI TTS sentences (12 mixed IT–EN, 4 English), a 46 s
  concatenation of the 12 mixed sentences with 0.8 s pauses (clean and with
  −35 dBFS white noise), and five 10 s non-speech clips (digital silence, white
  noise at −30/−15 dBFS, brown noise, 100 Hz hum): 23 clips, 190.7 s, 5,969
  windows.
- **Method:** Moth's own VAD path (`stt.vad.speech_probabilities`, 512 + 64
  samples, state carry; 0.5 window threshold; 0.10 clip gate; `segment_bounds`)
  run with both graphs on every clip.

| Measure | Result |
|---|---|
| Max window-probability gap | **0.0** (bit-identical outputs) |
| Window flips at 0.5 | 0 |
| Clip-gate disagreements | 0 (speech 18/18 gated in, non-speech 5/5 gated out) |
| Segment-bound disagreements | 0 |
| Bytes saved | **1,037,921 B** (2,327,524 → 1,289,603) |
| CPU time, 190.7 s of audio | 0.85 s vs 1.00 s (one run; noise) |

Raw data: `vad-parity-synthetic.json`. Machine: Windows 11, Intel Core Ultra 7
268V, onnxruntime 1.31.0, Moth `stt.vad` from `moth-stt-dev` at `d5ae605`.

## What it does not show

Synthetic speech and generated noise are not the product audio. E-012 must
repeat the probe on S1-dev, T-600 and the 480-clip MUSAN/room-tone noise set,
then run the browser with the 16 kHz graph (load, `T/S1/S1_0001.wav`, RTFx)
before `export_release.py --vad 16k` becomes the default. See
`experiments/next-experiments.md`.
