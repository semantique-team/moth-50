# P3 — FLEURS readout of the product package

> Executed 2026-11-21, report-only, **once per language**, with the E-034c
> product package (D25). No gate, no selection: FLEURS is a diagnostic set.

## Setup

- **Product:** E-034c package (`runs/E-004/release-e034c`), graph `14ba2034…`,
  LM `7a4332f1…`, decoder 0.55 / 2.7, beam 12, top-k 10, INT8.
- **Reference:** Moth 0.8.1 INT8 + A3 (`data/raw/moth-0.8.1`), same decoder,
  decoded on the same shards. Its numbers reproduce E-037 exactly
  (20.1621 / 49.8077), so the manifests and shards match.
- **Sets:** FLEURS-IT test, 697 clips / 2.74 h; FLEURS-EN test, 525 clips /
  1.39 h. Both are `exclusion` inputs of the E-031 LM corpus, so no text leak.
- **ETA:** not defined on FLEURS (no `en_spans`); WER only.

## Results

| Set | 0.8.1 INT8 + A3 | **E-034c** | Δ (95 % CI) |
|---|---:|---:|---|
| FLEURS-IT | 20.1621 % | **14.8238 %** | **−5.3383 pp [−6.0004, −4.7105]** |
| FLEURS-EN | 49.8077 % | **30.2675 %** | **−19.5401 pp [−20.7161, −18.3123]** |

## Reading

- **FLEURS-IT meets the original E-005 expectation (≤ 18 %)** with 3.2 pp of
  margin, and is 5.34 pp better than 0.8.1 on unseen-speaker Italian read
  speech. This is the clean generalization number for Italian.
- **FLEURS-EN is 30.27 %**, 19.54 pp better than 0.8.1 but still weak in
  absolute terms. The mixture is Italian-centric (Italian is the majority;
  English is LibriSpeech-100 plus spliced IT+EN), and the product target is
  mixed dictation, not English-only read speech. Do not quote 30 % as the
  product's English dictation number.
- The gap to Parakeet (Italian FLEURS ≈ 3 %) is the headroom E-018/E-019 are
  meant to close, not a gate.
- FLEURS is read speech; S1-dev/T remain the mixed product sets. No model,
  decoder or LM change follows from this readout.

## Artifacts

- `fleurs-it-e034c-int8.json/.hyp.jsonl`, `fleurs-en-e034c-int8.*` — product.
- `fleurs-it-081-int8.json/.hyp.jsonl`, `fleurs-en-081-int8.*` — reference.
- `fleurs-it-e034c-vs-081.json`, `fleurs-en-e034c-vs-081.json` — paired
  `moth_eval` comparisons.

## Reproduce

```bash
# mel shards from the FLEURS tarballs (test split)
tar -xzf data/raw/fleurs/data/it_it/audio/test.tar.gz -C /tmp/fleurs/fleurs/it_it/audio
tar -xzf data/raw/fleurs/data/en_us/audio/test.tar.gz -C /tmp/fleurs/fleurs/en_us/audio
for lang in it en; do
  uv run --frozen --project ../moth-stt --extra training -- \
    python -m stt.data.build_mel \
    --manifest ../moth-stt/data/manifests/fleurs_${lang}_test.jsonl \
    --audio-root /tmp/fleurs --out-dir data/shards/fleurs_${lang}_test --workers 8
  uv run --frozen --project ../moth-stt --extra training -- \
    python tools/frozen_readout.py --mode mixed --name fleurs-${lang} \
    --onnx runs/E-004/release-e034c/model.int8.onnx \
    --lm runs/E-004/lm_e034c_n2.bin --allow-other-lm \
    --manifest ../moth-stt/data/manifests/fleurs_${lang}_test.jsonl \
    --shards data/shards/fleurs_${lang}_test --config training/configs/base6-long.toml \
    --out runs/E-004/fleurs-${lang}-e034c-int8.json
done
# reference: replace the --onnx with data/raw/moth-0.8.1/model.int8.onnx and
# drop --lm/--allow-other-lm; compare the hypothesis files with moth_eval --vs.
```
