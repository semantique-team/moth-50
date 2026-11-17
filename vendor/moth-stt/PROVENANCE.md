# Vendored Moth evaluation code

Source: `semantique-team/moth-stt` at commit
`f3ecd1b0f4f0c5c044fa637e5fb5dd8a0c798d97` (2026-11-15), Apache-2.0.

| File | Original | Change |
|---|---|---|
| `moth_text.py` | `stt/text.py` | none |
| `moth_manifest.py` | `stt/manifest.py` | `from stt.text` → `from moth_text` |
| `moth_features.py` | `stt/features.py` | none |
| `moth_eval.py` | `stt/eval.py` | `stt.text` imports → `moth_text` |
| `moth_release.py` | `stt/release.py` | budget cap and default 25,000,000 / 20,000,000 → 50,000,000 (D5); config `name` may be `moth-50` as well as `moth-stt` (2026-11-17) |
| `assets/mel_filters.bin` | `stt/assets/mel_filters.bin` | none; copied from the `moth-stt-dev` checkout at `d5ae605` (a different repository from the pinned commit), so the tie to Moth is the sha256 `4f2701b1d287d74a0dc9871026e9519d98cb76426615f2539b0d151a0ae4ec2e`, identical to the Moth 0.8.1 release asset (2026-11-17) |

The import changes are mechanical; the scoring logic is unchanged. Do not edit
these files to make a new result pass. If a change is genuinely needed, copy the
change back into a new version with a note and recompute the baseline.

The `moth_release.py` change is a packaging rule, not a scoring rule: without it
the validator rejects any budget above 25,000,000 bytes and any package not named
`moth-stt`, so it cannot validate a MOTH-50 release. Moth-20 packages still
validate with `--budget-bytes 20000000`.

`assets/mel_filters.bin` is Whisper's 80×201 Slaney filterbank (openai/whisper,
MIT; see `NOTICE`). `moth_features.load_mel_filters()` and `log_mel()` read it by
default and fail without it.

## Not vendored

- `stt/model.py`, `stt/train.py`, `stt/export.py` — specific to the 0.8.1
  architecture and recipe.
- `stt/decode.py` — depends on `stt/hotwords.py` and `stt/lm.py`; copy it with
  those dependencies if the new model reuses the Moth decoder.

When they are ported, note that the Moth pipeline hard-codes the Whisper-tiny
`EncoderConfig()` outside `stt/model.py` (see `docs/review-2026-11-17.md` §4).

## Re-sync

From a `moth-stt` checkout at the pinned commit:

```bash
SRC=../moth-stt/stt
DST=vendor/moth-stt
cp "$SRC/text.py" "$DST/moth_text.py"
cp "$SRC/manifest.py" "$DST/moth_manifest.py"
cp "$SRC/features.py" "$DST/moth_features.py"
cp "$SRC/eval.py" "$DST/moth_eval.py"
cp "$SRC/release.py" "$DST/moth_release.py"
cp "$SRC/assets/mel_filters.bin" "$DST/assets/mel_filters.bin"
# re-apply the import edits and the moth_release.py budget/name edits,
# then run the py_compile check and the tests
python -m py_compile "$DST"/*.py
```
