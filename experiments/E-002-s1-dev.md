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

### Session check (run at any point)

```bash
uv run --no-project --with numpy --with soundfile -- \
  python ../moth-50/tools/s1_dev_qc.py \
  --scripts text-2026-11-dev/R-S1.jsonl --audio audio/R-S1-dev/S1 \
  --json ../moth-50/runs/E-002/s1-dev-qc.json
```

It reports takes vs cues, holes inside the session, duration and level
statistics, the projected full-session minutes from the measured s/word, and
the script's long-form count. It exits 1 only on a real audio error (outside
1–30 s, silent, clipped > 1 %).

## 4. Build the dev manifest

```bash
python -m scripts.build_manifest build --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --audio-dir audio/R-S1-dev \
  --out manifests/R-S1-dev.jsonl
```

`build_manifest` validates duration, silence and clipping, but writes the
canonical `audio/R-S1/...` path and `split: train`. Pass the fragment through
the finalizer, which rewrites both and verifies that every audio file exists:

```bash
uv run --no-project -- python ../moth-50/tools/finalize_s1_dev.py \
  --fragment manifests/R-S1-dev.jsonl --dataset-root . --check   # then drop --check
```

It is idempotent (a second run reports "already finalized"), refuses mixed
fragments, and fails on a duplicate id, a missing file, or a duration outside
1–30 s.

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

## Measured so far (2026-11-17, 200/960 takes)

| Quantity | Value |
|---|---|
| Clips | 200 / 960 |
| Audio | 9.53 min, mean 2.86 s, range 1.76–4.59 s |
| Projected total | ~58.6 min at the measured 0.501 s/word (conservative for long takes) |
| Levels | peak −19.0 dBFS mean (−23.7…−15.0), RMS −35.2 dBFS, 0 clipped, drift 0.1 dB |
| Script | 900 short cues (5.7 words) + 60 long cues (24–38 words, 8–17 s), 60 over 8 s |
| Categories | short block balanced 112–113; long block 30 phrase, 16 switch, 8 control_it, 6 control_en |
| Pipeline | manifest → finalize → mel shards → trainer dev loader: dry-run green |

The level matches the existing S1 training audio (peak −19.0 dBFS mean on
`audio/R-S1/S1`), so the gain is consistent with the corpus and should not be
changed mid-session; the feature path has no loudness normalization.

**Decision (2026-11-17): option (b).** The 900 short cues alone land at
~43 min with no 8–20 s coverage, and the pool has nothing longer than 12 words.
Sixty long dictation cues were generated (`deepseek-v4.1-flash`, same candidate
format) and appended as `R-S1-DEV-0901`–`R-S1-DEV-0960`:

- 30 `phrase` (Italian sentence + 2–3 word English phrase),
- 16 `switch` (English clause inside an Italian sentence),
- 8 `control_it` (Italian only), 6 `control_en` (English only),
- 24–38 words each, ~8–17 s estimated, ~1778 words total (~15 min).

They are **`reviewed: false`**. Review and mark them before recording index
901+:

```bash
python -m scripts.generate_scripts review --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --limit 60
python -m scripts.generate_scripts mark-reviewed --kind R --speaker S1 \
  --scripts-dir text-2026-11-dev --reviewer <name>
```

The script is now 960 cues, inside the accepted 600–1000 range. Appending at
901+ is safe because the recorder and `build_manifest` join by position. If the
finished session lands above 60 min, skip a few of the longest `control_en`
cues; the protocol window is 30–60 min.
