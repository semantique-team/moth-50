# Vendored Moth evaluation code

Source: `semantique-team/moth-stt` at commit
`f3ecd1b0f4f0c5c044fa637e5fb5dd8a0c798d97` (2026-11-15), Apache-2.0.

| File | Original | Change |
|---|---|---|
| `moth_text.py` | `stt/text.py` | none |
| `moth_manifest.py` | `stt/manifest.py` | `from stt.text` → `from moth_text` |
| `moth_features.py` | `stt/features.py` | none |
| `moth_eval.py` | `stt/eval.py` | `stt.text` imports → `moth_text` |
| `moth_release.py` | `stt/release.py` | none |

The import changes are mechanical; the scoring logic is unchanged. Do not edit
these files to make a new result pass. If a change is genuinely needed, copy the
change back into a new version with a note and recompute the baseline.

## Not vendored

- `stt/model.py`, `stt/train.py`, `stt/export.py` — specific to the 0.8.1
  architecture and recipe.
- `stt/decode.py` — depends on `stt/hotwords.py` and `stt/lm.py`; copy it with
  those dependencies if the new model reuses the Moth decoder.

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
# re-apply the import edits, then run the py_compile check
python -m py_compile "$DST"/*.py
```
