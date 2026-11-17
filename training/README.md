# MOTH-50 trainer

This is the first port of the Moth training path. It reuses Moth's model,
tokenizer, mel cache, collate, duration batching and evaluator, but owns the
loop and the MOTH-50 settings:

- `EncoderConfig` is explicit (`base` by default); a base run cannot silently
  become Whisper-tiny;
- per-block activation checkpointing is on by default (D15);
- 192 audio-s / 32-clip batch cap;
- AdamW, warmup + cosine, grad clip, bf16 autocast;
- JSONL logs, MPS memory, checkpoints, resume and a greedy dev readout;
- the greedy readout is **monitoring only**. The E-003 gate uses the frozen
  beam+A3 protocol in `experiments/E-002-protocol.md`.

## Environment

The trainer needs Moth's training environment (torch, sentencepiece, soundfile,
pyyaml). From the repository root:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/smoke.toml \
  --moth-stt ../moth-stt --out runs/trainer-smoke --device mps
```

## Smoke test (local, 30 clips)

The committed `training/fixtures/` contain 20 train and 10 dev rows from the
recorded dataset. Build their mel shards once:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel --manifest training/fixtures/smoke-train.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/smoke-train --workers 2
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel --manifest training/fixtures/smoke-dev.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/smoke-dev --workers 2
```

Then run the 20-step smoke test. It builds base+6 (39.26 M params), trains,
evaluates greedily, saves `ckpt_last.pt`/`ckpt_best.pt` and writes
`train.log.jsonl` + `metrics.json`:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/smoke.toml \
  --moth-stt ../moth-stt --out runs/trainer-smoke --device mps
```

On the M5 Pro this takes seconds and peaks below 1 GB because the smoke clips
are short; it is a wiring test, not a quality result.

## Real-shard smoke (MLS)

After the MLS shards and LS dev shard exist, `training/configs/mls-smoke.toml`
runs the same 20-step base+6 check on real long-form data:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/mls-smoke.toml \
  --moth-stt ../moth-stt --out runs/training-mls-smoke --device mps
```

Measured on the M5 Pro: 20 steps at ~192 audio-s per batch, **5.1–6.2 GB** peak
with checkpointing, **1.5–2.0 s/step**, no non-finite losses. The greedy dev WER
is meaningless after 20 steps; this test only proves that the shard loader,
model, CTC targets, optimizer, checkpoints and dev loop work on real clips.

## Full E-003 data build

The real config is `training/configs/base6.toml`. It expects the mel shards in
`data/shards/` and the source audio under `data/raw/`. The exact versions and
checksums are in `data/sources.lock.json`.

1. Extract the archives so the manifest paths resolve:

```bash
mkdir -p data/raw/mls_it data/raw/librispeech data/raw/cv_it_27
tar -xzf data/raw/mls/mls_italian.tar.gz -C data/raw/mls_it
tar -xzf data/raw/librispeech/train-clean-100.tar.gz -C data/raw/librispeech
tar -xzf data/raw/librispeech/dev-clean.tar.gz -C data/raw/librispeech
# Common Voice 27: extract/rename the Italian release so that
# data/raw/cv_it_27/it/clips/... and data/raw/cv_it_27/it/*.tsv exist.
# MUSAN is only needed if you rebuild the noise recipes from raw audio.
```

2. Build the base shards (CPU; use `--workers` for parallel mel extraction):

```bash
for spec in \
  "cv_it_train_clean50:data/raw" \
  "cv_it_train_noisy50:data/raw" \
  "mls_it_train:data/raw" \
  "ls_en_train:data/raw" \
  "mls_it_dev:data/raw" \
  "ls_en_dev:data/raw"
do
  name=${spec%%:*}; root=${spec#*:}
  uv run --frozen --project ../moth-stt --extra training -- \
    python -m stt.data.build_mel \
    --manifest "../moth-stt/data/manifests/${name}.jsonl" \
    --audio-root "$root" --out-dir "data/shards/${name}" --workers 8
done

uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel \
  --manifest ../moth-cs-dataset/manifests/train.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/gold-train --workers 8
```

3. Build the spliced set **after** the CV/MLS/LS shards (it concatenates cached
mels and regenerates the manifest and report):

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_spliced \
  --it-manifest ../moth-stt/data/manifests/cv_it_train_clean50.jsonl,../moth-stt/data/manifests/mls_it_train.jsonl \
  --it-shards data/shards/cv_it_train_clean50,data/shards/mls_it_train \
  --en-manifest ../moth-stt/data/manifests/ls_en_train.jsonl \
  --en-shards data/shards/ls_en_train \
  --hours 110 --seed 0 --out-dir data/shards/spliced_it_en \
  --manifest data/manifests/spliced_it_en.jsonl \
  --report data/manifests/spliced_it_en_report.json
```

4. Check the config and run:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/base6.toml \
  --moth-stt ../moth-stt --out runs/E-003/base6 --device mps --dry-run
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.train --config training/configs/base6.toml \
  --moth-stt ../moth-stt --out runs/E-003/base6 --device mps
```

## Resume and checkpoints

- `--resume runs/.../checkpoints/ckpt_last.pt` restores model, optimizer, epoch,
  batch position and best dev WER.
- `--max-steps N` overrides `train.max_steps` for a short probe.
- `checkpoint_every_steps` writes `ckpt_last.pt`; a better greedy dev WER writes
  `ckpt_best.pt`.
- Checkpoints are private and gitignored (`runs/**/checkpoints/`).

## What is deliberately not here yet

- The beam + A3 LM decoder and the frozen `moth_eval` readout: use the E-002
  protocol for the real gate.
- The browser export/latency path.
- Distillation, TTS and pseudo-labelling; those are E-006.
- A production data loader for the final 2–5 h S1 set; the current loader reads
  Moth mel shards by design.

If a run uses a different mixture or encoder, create a new dated config and
record it in the experiment directory; do not edit `base6.toml` in place after
E-003 starts.
