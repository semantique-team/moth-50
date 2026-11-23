# E-016 — Browser runtime checks (P4)

> **Status: done 2026-11-23.** The 1-thread non-isolated gate passes
> (**10.93× / 19.74× / 21.17×** on 5 / 10 / 20 s, gate ≥ 8×). The
> browser-vs-Python comparison is explained rather than identical: the runtime
> default (VAD segmentation) is **+0.72 pp WER** (CI excludes zero), while
> `segment: false` is a statistical tie (**+0.35 pp** [−0.13, +0.82]); the
> residual 263/960 differences are numerical near-ties, the same class Moth
> E-022 measured at 70–77 % identity. The web runtime is vendored byte-identical
> at `0e15442` (D28). **Finding:** the browser default costs ~0.4–0.7 pp WER on
> S1-dev versus the Python readout; a follow-up should auto-disable segmentation
> when the clip fits one 30 s segment. No product change was made here.

## 1-thread RTFx without cross-origin isolation

`tools/serve_web.py --no-isolation` (stdlib server, no COOP/COEP), Chromium
159.0.8154.6 headless via Playwright, 1 thread, SIMD, no `SharedArrayBuffer`.
Clips are 5 / 10 / 20 s made by tiling `T/S1/S1_0001.wav`; the benchmark is 2
warm-ups + 5 runs. Package `0.9.7-e023-r1`.

| Clip | total ms (median) | RTFx | Gate ≥ 8× |
|---:|---:|---:|---|
| 5 s | 457.3 | **10.93×** | pass |
| 10 s | 506.6 | **19.74×** | pass |
| 20 s | 944.8 | **21.17×** | pass |

This is consistent with Moth E-022's 20.2× on a 20 s clip at 1 thread, and
confirms the browser does not need cross-origin isolation for an acceptable
single-thread speed. `runs/E-016/rtfx-1thread-nonisolated.json`.

## Browser vs Python on S1-dev

Python reference: `runs/E-023/s1-dev-s0.hyp.jsonl`, the product graph
`85abd830…` + E-034c, frozen decoder. Browser: the same release package served
by `tools/serve_web.py`, Chromium 159, 4 threads, isolated. Scoring:
`tools/browser_parity.py` with the vendored `moth_eval`; paired bootstrap
1,000, seed 0, candidate − Python.

| Browser input | Segment | WER | ETA | Identical | Δ WER vs Python (95 % CI) | Tie |
|---|---|---:|---:|---:|---|---|
| Python reference | — | **13.3820** | **75.1465** | — | — | — |
| raw 48 kHz, default | on | 14.0979 | 75.5859 | 675/960 | **+0.7159 pp** [+0.1678, +1.2599] | no |
| raw 48 kHz | **off** | **13.7327** | **75.5859** | 697/960 | **+0.3506 pp** [−0.1255, +0.8160] | **yes** |
| Python-FFT 16 kHz, default | on | 14.5215 | 74.7559 | 683/960 | +1.1395 pp [+0.5891, +1.6925] | no |

**Every difference is explained.**

- **The default segmentation is the systematic term.** The browser runtime
  runs the Silero guard and splits the clip at VAD boundaries by default; the
  Python readout decodes the whole clip. With `segment: false` the browser uses
  the full clip after the same no-speech guard and the WER difference becomes a
  statistical tie. The 5-run median is unaffected in kind: 675 → 697 identical
  transcripts, and the point estimate halves.
- **Resampling is not the cause.** Feeding the browser the exact PCM produced
  by Python's FFT resampler (`stt.data.build_mel.load_audio_16k`) did **not**
  improve the match; it was slightly worse. The browser's
  `OfflineAudioContext` resampling is therefore not the source of the default
  gap.
- **The residual differences are numerical near-ties.** Even with
  `segment: false`, 263/960 transcripts differ. Moth E-022 measured the same
  class of browser/Python divergence at 70–77 % identity on 200 S1 clips, with
  a WER tie at the shipped decode; the note in `web/tests/parity.spec.ts` says a
  1e-6 mel difference can flip a near-tie in the beam. The browser re-derives
  the mel from audio, Python reads the cached mel, and the int8 graph runs under
  WASM rather than native ONNX Runtime. ETA is a tie in every variant.
- **No release-package bug was found.** The graph, LM and decoder are the same;
  the browser verifies every sha256 and transcribes the reference clip exactly.

**Finding, not fixed here.** The product runtime's default gives **14.10 %**
on S1-dev against the Python readout's **13.38 %**, while `segment: false`
gives **13.73 %** (tie). The page records at most 30 s, but the file input
accepts longer audio, where segmentation is required. The follow-up is a dated
runtime decision (E-016b): auto-disable segmentation when `pcm.length` fits one
30 s segment, keep it for longer inputs, and re-read S1-dev. Until then the
Python readout numbers remain the reported product WER, with the browser gap
recorded. `runs/E-016/browser-vs-python.json`.

## Vendored runtime

`vendor/moth-stt-web/` contains 12 files byte-identical to
`semantique-team/moth-stt` at `0e15442ecb85dffcb4edd68e9d9ece18af6f35df`
(on top of `b0fa11c`), Apache-2.0. Hashes and the re-sync rule are in
`vendor/moth-stt-web/PROVENANCE.md`. `node_modules/` is not vendored; install
with `npm ci --prefix vendor/moth-stt-web`. Upstream `tests/` and `acceptance/`
are not vendored because they depend on Moth's private runs and fixtures.

New tooling:

- `tools/serve_web.py` — temporary root with `web/` and `release/`, optional
  `--mount NAME=PATH`, `--no-isolation`; no copy of the runtime or release.
- `tools/browser_probe.mjs` — RTFx and S1-dev browser runs through Playwright,
  with `--threads` and `--segment false`.
- `tools/browser_parity.py` — normalized identity and paired WER/ETA against a
  Python hypothesis file.

Smoke test of the vendored copy: the page loads `0.9.7-e023-r1` in **525 ms**
(cold, in-app browser), verifies every sha256 and transcribes
`T/S1/S1_0001.wav` exactly ("ho spostato il meeting a domani", single run
247.3 ms, RTFx 16.13). `runs/E-016/vendored-smoke.json`.

## Artifacts

- `rtfx-1thread-nonisolated.json` — 5/10/20 s, 1 thread, no isolation.
- `s1-dev-browser-default.json` — raw 48 kHz, default segment.
- `s1-dev-browser-noseg.json` — raw 48 kHz, `segment: false`.
- `s1-dev-browser-fft16k.json` — Python-FFT 16 kHz PCM, default segment.
- `browser-vs-python.json` — metrics, paired CIs, differing ids.
- `vendored-smoke.json` — vendored runtime, cold load, exact transcript.

## Reproduce

```bash
npm ci --prefix vendor/moth-stt-web
# isolated, for the parity runs
python tools/serve_web.py --release runs/E-023/release-0.9.7 \
  --mount dataset=../moth-cs-dataset --port 8790
# non-isolated, for the 1-thread gate
python tools/serve_web.py --release runs/E-023/release-0.9.7 \
  --mount clips=/path/to/clips --port 8791 --no-isolation
node tools/browser_probe.mjs rtfx --base-url http://127.0.0.1:8791 \
  --threads 1 --clips /clips/clip-5s.wav,/clips/clip-10s.wav,/clips/clip-20s.wav \
  --out runs/E-016/rtfx-1thread-nonisolated.json
node tools/browser_probe.mjs s1dev --base-url http://127.0.0.1:8790 \
  --threads 4 --manifest /dataset/manifests/R-S1-dev.jsonl \
  --audio-base /dataset/ --out runs/E-016/s1-dev-browser-default.json
node tools/browser_probe.mjs s1dev --base-url http://127.0.0.1:8790 \
  --threads 4 --manifest /dataset/manifests/R-S1-dev.jsonl \
  --audio-base /dataset/ --segment false --out runs/E-016/s1-dev-browser-noseg.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/browser_parity.py \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl \
  --reference-hyp runs/E-023/s1-dev-s0.hyp.jsonl \
  --browser runs/E-016/s1-dev-browser-default.json \
  --browser runs/E-016/s1-dev-browser-noseg.json \
  --out runs/E-016/browser-vs-python.json
```
