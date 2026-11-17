# Data for MOTH-50

## Pinned dataset

The recorded Italian–English code-switching data is
**`semantique-team/moth-cs`, version `v1.0.0`**, prepared on 2026-11-16.

```python
from datasets import load_dataset

ds = load_dataset("semantique-team/moth-cs", revision="v1.0.0")
```

Offline, use the sibling checkout:

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={
        "train": "../moth-cs-dataset/hf/data/train-00000-of-00001.parquet",
        "test": "../moth-cs-dataset/hf/data/test-00000-of-00001.parquet",
        "anchor": "../moth-cs-dataset/hf/data/anchor-00000-of-00001.parquet",
    },
)
```

Before using it, run the integrity check:

```bash
python ../moth-cs-dataset/scripts/verify.py
```

It checks counts, unique ids, SHA-256, 48 kHz mono 16-bit WAV, durations, split
disjointness and `en_spans`.

## Tokenizer

The Moth 0.8.1 tokenizer is vendored at `data/tokenizer/moth_bpe.model` and is
**pinned**, not rebuilt: 14,327 B, sha256
`d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`, 1,024
SentencePiece BPE pieces, CTC mapping `blank = 0`, `ctc_id = sp_id + 1`. The
published `moth-stt` release ships only `vocab.json`; the LM is indexed by these
piece IDs. See `data/tokenizer/README.md` and `tests/test_tokenizer.py`.

## Splits

| Split | Content | Use |
|---|---|---|
| `train` | R-S1 + R-S2, 1,413 clips, 57.6 min | recorded training data only |
| `test` | T, 600 clips, 23.2 min | evaluation only |
| `anchor` | A, 180 clips, 7.0 min | speaker-vs-sentence analysis; never train |

- **S3 never contributes to `train`.**
- `anchor` reuses 60 T sentences by design; it is not an independent dev set.
- T and R have no normalised text overlap.
- The audio is 48 kHz; resample to 16 kHz for most ASR front ends. Use
  `moth_features.py` if you need the exact Moth log-mel.

## Scale

1.46 h is a benchmark and a fine-tuning seed. For MOTH-50, plan:

1. public Italian/English corpora (Common Voice, MLS, LibriSpeech, FLEURS
   evaluation only) re-fetched at pinned versions;
2. **2–5 h of new S1 recordings** in new sessions, with one held-out S1 session
   as dev. New sessions need a written consent/addendum covering `moth-50` and
   commercial use. There is no new speaker available; S2, S3 and FLEURS remain
   the generalization checks;
3. commercially permissive TTS augmentation (Piper `it_IT-paola-medium`: dataset
   CC0, fine-tuned from a CC BY 4.0 Hi-Fi TTS base; `it_IT-serena-high`: CC BY
   4.0, trained on synthetic Qwen3-TTS data; public-domain English voices;
   Kokoro-82M Apache 2.0 with `if_sara`/`im_nicola`, whose training data includes
   synthetic audio from closed TTS providers; provenance in
   `docs/architecture-options.md` §8.2). Do not clone S1/S2/S3 voices. Keep
   TTS as a minority source (10–20 % of mixed exposure) in a separate manifest;
4. word/phrase-level splicing of permissive corpora, measured against Moth's
   utterance-level splicing;
5. a documented mixture and a dev split that never touches T.

Do not train on T, A or S3 audio. Do not use FLEURS train for training (Moth
used FLEURS evaluation only); keep that rule unless this project deliberately
changes it and documents the change. **No CC-BY-NC data in training**:
CS-FLEURS and SwitchLingua are evaluation-only for this commercial release.

## Provenance

`../moth-cs-dataset/releases/v1.0.0.json` records the counts, hashes, consent
and the source commit of the frozen model repository. Cite the dataset with
`../moth-cs-dataset/CITATION.cff`.
