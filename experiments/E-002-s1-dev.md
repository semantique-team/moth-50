# E-002 — S1-dev recording runbook

**Goal:** one new S1 session, 45–60 min, held out from training and used as the
product/checkpoint-selection dev set in E-003. It is **not** a new speaker and
does not become a test set.

## Naming

Use a session-specific tree; do not edit the published `moth-cs@v1.0.0`
manifests or audio.

| Item | Path |
|---|---|
| scripts | `../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl` |
| audio | `../moth-cs-dataset/audio/R-S1-dev/S1/S1_0001.wav` |
| session log | `../moth-cs-dataset/audio/R-S1-dev/S1/session.log.jsonl` |
| manifest | `../moth-cs-dataset/manifests/R-S1-dev.jsonl` |
| mel shards | `data/shards/s1-dev` |

## 1. Consent

Sign a **new addendum** covering `moth-50` and commercial use. The v1.0.0
template (`docs/consent-template.md`) says `moth-stt` and “no direct commercial
use”; it is not sufficient for this release. Keep the signed form offline.

## 2. Script

The recorder has no `dev` kind, so use `--kind R --speaker S1` with a separate
scripts directory and audio root. Prepare 600–1,000 new mixed IT–EN sentences
(45–60 min at a mix of short and 8–20 s dictation-style material), disjoint from
T, A, R-S1 and R-S2. Do not reuse `text/R-S1.jsonl`.

The generator's duplicate index is per scripts directory. Either run the
candidate check against the published scripts explicitly, or copy the published
`T.jsonl`, `A.jsonl`, `R-S1.jsonl`, `R-S2.jsonl` and `pool.jsonl` into the new
directory for the dedupe pass and keep them out of the recording cue file. A
simple exact/near-duplicate check against the published scripts is enough
before recording; review every sentence by hand.

```bash
cd ../moth-cs-dataset
mkdir -p text-2026-11-dev

python -m scripts.generate_scripts import --kind R --speaker S1 \
  --in candidates-dev.jsonl --scripts-dir text-2026-11-dev
python -m scripts.generate_scripts stats --scripts-dir text-2026-11-dev
python -m scripts.generate_scripts review --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev
python -m scripts.generate_scripts mark-reviewed --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --reviewer <name>
```

## 3. Record

First a dry run, then the real session:

```bash
uv run --no-project --with numpy --with soundfile --with num2words \
  python -m scripts.record --speaker S1 --kind R \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev --dry-run

uv run --no-project --with numpy --with soundfile --with num2words \
  python -m scripts.record --speaker S1 --kind R \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev
```

Keys: **Enter** = record/stop, **p** = play back, **r** = redo, **s** = skip,
**q** = quit. Use 48 kHz mono 16-bit, 1–30 s per take, peaks around −12/−6 dBFS,
no clipping. Keep `session.log.jsonl`.

## 4. Build the dev manifest

```bash
python -m scripts.build_manifest build --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev \
  --out manifests/R-S1-dev.jsonl
```

`build_manifest` validates duration, silence and clipping, but writes the
canonical `audio/R-S1/...` path and `split: train`. Patch the fragment so it is a
dev set with the real audio paths:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("manifests/R-S1-dev.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    row["split"] = "dev"
    row["audio"] = row["audio"].replace("audio/R-S1/", "audio/R-S1-dev/", 1)
path.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
PY
```

Do **not** run `scripts/verify.py` on the v1.0.0 checkout with the new audio:
it checks the fixed v1.0.0 counts and will fail on the extra files. Run it only
after merging the fragment into a new dataset version with updated expected
counts.

## 5. Build the dev mel shards

```bash
cd ../moth-50
uv run --frozen --project ../moth-stt --extra training -- \
  python -m stt.data.build_mel \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl \
  --audio-root ../moth-cs-dataset --out-dir data/shards/s1-dev --workers 8
```

Then add the dev source to `training/configs/base6.toml`:

```toml
[[data.dev]]
manifest = "../../moth-cs-dataset/manifests/R-S1-dev.jsonl"
shards = "../../data/shards/s1-dev"
```

The trainer's dev loader never augments and never trains on these rows. The
manifest is also excluded from the train sources by construction.

## How many sentences?

| Target | At ~2.8 s/sentence | At ~10 s dictation | Mixed recommendation |
|---|---:|---:|---:|
| 45 min dev | ~960 | ~270 | ~600 (~300 short + 300 long) |
| 60 min dev | ~1,280 | ~360 | ~800 (~400 short + 400 long) |

The current R-S1 v1.0.0 is 741 clips, 34.9 min, mean 2.83 s. A dev session of
600–800 mixed sentences is the sensible first target.
