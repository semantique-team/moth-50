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

## Pretrained initialization

E-003 must start from the pretrained Whisper-base encoder: at this data scale a
random encoder only learns the output prior (measured: 4,000 steps, 99 % dev
WER). `data/raw` is gitignored, so build the initializer once:

```bash
cd ../moth-stt
uv run --frozen --project . -- \
  python ../moth-50/tools/convert_whisper_encoder.py \
  --model openai/whisper-base \
  --out ../moth-50/data/raw/whisper-base-encoder.pt
```

It reuses Moth's `stt.convert_whisper` name map, checks parity against the HF
eager encoder (positions matched, gate 1e-3) and records the HF revision and
state sha256. `base6.toml` sets `model.init` to that file and
`identity_heads = true`: the extra blocks' `out`/`mlp2` are zeroed, so every
arm is the same function at step 0.

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

## E-003 preflight

`training/preflight.py` checks every manifest/shard, the tokenizer hash, Moth's
train-row guard, the four parameter/package budgets and the E-002 memory
reading, without training:

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m training.preflight --config training/configs/base6.toml \
  --moth-stt ../moth-stt --out runs/E-003/preflight.json
```

It reports missing sources as `ready: false`; add `--strict` to make that an
error. The E-003 arms use the same config with `--head-blocks 0/2/4/6
--max-steps 4000`; see `experiments/E-003-plan.md`.

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
  "cv_it_train:data/raw" \
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

The clean-50 manifest uses the full `data/shards/cv_it_train` shards. The
noisy half is **not** a plain `build_mel` job: it must be built with
`tools/build_noisy_cv.py` (the E-024 mixer), which mixes MUSAN noise and room
tone at a deterministic per-clip SNR:

```bash
# check sizes/SNR first (no build)
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/build_noisy_cv.py --check

# build the noisy half (~2 h, 8 workers) and write the clean/noisy manifests
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/build_noisy_cv.py --workers 8
```

Do not run `build_mel` on `cv_it_train_noisy50.jsonl`, or the "noisy" half will
be clean.

3. Build the spliced set **after** the CV/MLS/LS shards (it concatenates cached
mels and regenerates the manifest and report):

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_spliced \
  --it-manifest ../moth-stt/data/manifests/cv_it_train_clean50.jsonl,../moth-stt/data/manifests/mls_it_train.jsonl \
  --it-shards data/shards/cv_it_train,data/shards/mls_it_train \
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

Before E-003, add the held-out S1-dev source to `base6.toml` as documented in
`experiments/E-002-s1-dev.md`; E-003 cannot be ranked without it.

## Resume and checkpoints

- `--resume runs/.../checkpoints/ckpt_last.pt` restores model, optimizer, epoch,
  batch position and best dev WER.
- `--max-steps N` overrides `train.max_steps` for a short probe.
- `checkpoint_every_steps` writes `ckpt_last.pt`; a better greedy dev WER writes
  `ckpt_best.pt`.
- Checkpoints are private and gitignored (`runs/**/checkpoints/`).

## Weight EMA (optional)

`train.ema_decay` (default 0 = off, as in E-003/E-005) keeps an exponential
moving average of the weights with a step warm-up (`training/ema.py`). When it
is on, the greedy dev evaluation and `ckpt_best.pt` use the averaged weights,
and every checkpoint stores them as `ema_model` next to the raw `model` weights;
`--resume` continues from the raw weights and the stored average. The readout,
calibration and export tools take `--weights auto|model|ema` (`auto` = EMA when
present). E-011 plans `ema_decay = 0.9998`.

## What is deliberately not here yet

- The browser export/latency path beyond `tools/export_release.py`; the web
  runtime still lives in the archived `moth-stt` checkout (E-016).
- Distillation, TTS and pseudo-labelling; those are E-006.
- A production data loader for the final 2–5 h S1 set; the current loader reads
  Moth mel shards by design.

The beam + A3 readouts are `tools/frozen_readout.py` (dev-macro gate and
`--mode mixed` for S1-dev) and `tools/calibrate_decoder.py` (E-009); the A3 LM is
fetched and checked by `tools/release_assets.py`.

If a run uses a different mixture or encoder, create a new dated config and
record it in the experiment directory; do not edit `base6.toml` in place after
E-003 starts.
