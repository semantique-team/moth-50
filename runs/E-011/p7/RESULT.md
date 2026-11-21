# P7 — E-011 diagnostics after the P6 failure

> Executed 2026-11-21, readout-only. The question is which of E-011's three
> changes (H1 encoder LR, H2 gold over-exposure, H3 length/EMA) explains the
> S1-dev ETA loss that P6 confirmed with E-034c. Nothing here selects a model;
> it orders P10.

## D1 — gold-train memorisation (H2)

Decode the gold train set (`../moth-cs-dataset/manifests/train.jsonl`,
`data/shards/gold-train`, 1,413 clips / 57.6 min) with E-005 and both E-011
seeds, all with E-034c, INT8, frozen decoder. This is training data, so there is
no protocol issue; it is a probe of acoustic over-exposure, not a test.

| System | Gold WER | 95 % CI | Gold ETA |
|---|---:|---|---:|
| E-005 | **0.2018 %** | [0.0118, 0.4808] | 99.8017 % |
| E-011 s0 | **0.0831 %** | [0.0000, 0.2382] | 99.9008 % |
| E-011 s1 | **0.1424 %** | [0.0000, 0.4016] | 99.8513 % |

**H2's signature is absent.** E-005 was already at 0.20 % on gold after ~59
exposures; E-011's ~149 exposures take it to 0.08–0.14 %, a marginal change on
an almost-saturated set. Gold over-exposure is real (both models are near
perfect on training audio) but it does not explain a 1.8 pp ETA regression on
S1-dev. Caveat: the gold scripts (R-S1/R-S2) are also `train_script` rows of the
E-031 LM corpus, so the LM has their text; the comparison is still paired and
identical on both sides.

## D2 — encoder LR at equal steps (H1)

**Not run.** It needs the E-017 calibration set; S1-dev is the veto set and T
has already been read twice. It becomes available after E-017 (option A or B)
and is the natural companion to R1.

## D3 — what the ETA misses are (report-only)

`tools/eta_breakdown.py` on the P6 S1-dev readouts, against E-005 + E-034c
(2,048 English reference words):

| | Control | s0 | s1 |
|---|---:|---:|---:|
| misses | 534 | **588** | **552** |
| deleted | 57 | 68 | 62 |
| substituted | 361 | 387 | 381 |
| truncated | 77 | 85 | 65 |
| glued | 37 | 46 | 41 |
| apostrophe | 2 | 2 | 3 |
| function words missed | 68 | **85** | 73 |
| elided / no-vowel | 34 / 23 | 34 / 25 | 34 / 25 |
| newly wrong / newly right vs control | — | 177 / 123 | 143 / 125 |

Top missed terms: `a` (7 on s0), `it` (6), `on` (5), `too` (5), `is` (5),
`coffee` (4), `her`, `not`, `makes`, `lunch`, `up`, `of`, `check`, `that`.
The loss is a broad increase in substitutions of **short, common English
words**, not rare terms, elisions or deletions; s0 adds deletions (+11), s1
actually improves truncations (−12). There is no single failure mode to patch.

## Conclusion

P7 does not support H2 as the primary cause and cannot separate H1 from H3 with
readouts alone. The ETA loss is concentrated in short English words, the
category pattern is stable across seeds, and the gold probe is saturated in both
models. **P10 R1 (encoder LR 3e-5, one variable against E-011) is the first
ablation; R2 (gold ×20) only if R1 fails.** D2 and any calibration ranking wait
for E-017.

## Artifacts

- `gold-e005.json/.hyp.jsonl`, `gold-s0.*`, `gold-s1.*` — D1.
- `eta-breakdown-s0.json`, `eta-breakdown-s1.json` — D3, with the control
  comparison embedded.

## Reproduce

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed --name gold-train \
  --onnx runs/E-005/release-a3/model.int8.onnx \
  --lm runs/E-004/lm_e034c_n2.bin --allow-other-lm \
  --manifest ../moth-cs-dataset/manifests/train.jsonl --shards data/shards/gold-train \
  --config training/configs/base6-long.toml \
  --out runs/E-011/p7/gold-e005.json --hyp-out runs/E-011/p7/gold-e005.hyp.jsonl
# s0/s1: same with runs/E-011/release-s{0,1}/model.int8.onnx
python tools/eta_breakdown.py runs/E-011/p6/s0-s1-dev.json \
  --compare runs/E-004/s1-dev-e034c-int8.json \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl \
  --out runs/E-011/p7/eta-breakdown-s0.json
# s1: same with s1-s1-dev.json
```
