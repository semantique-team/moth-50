# E-025 (E-020) — LLM-written mixed IT–EN text for the LM: no effect, not adopted

> **Status: done 2026-11-23, not adopted.** 6,000 generated sentences (Qwen2.5-
> 1.5B-Instruct) → 3,521 kept after leakage/quality filtering → an E-034c LM
> with 1.9 % more sentences and a lossless 7,833,032 B pack. On R-S1-cal it is
> **12.8917 % / 75.6291 %** vs **12.8469 / 75.4967** for E-034c: WER **+0.0448
> pp** [−0.0900, +0.1807], ETA **+0.1325 pp** [0, +0.4178], both ties and 7/400
> clips changed. Gate 1 fails, so E-034c stays; S1-dev, dev-macro and T were not
> read (pre-registered stop).

## Generation and filter

- **Generator:** `Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; prompts and decoding in
  `experiments/E-025-prompts.json`; 6,000 sentences, 1,550 s on MPS (batch 8).
- **Seeds:** gold train terms + `../moth-stt/data/en_terms.txt`; never S1-dev,
  T or A.
- **Filter:** 3,522 kept, 2,478 dropped:
  `seed_missing` 1,385, `eval_4gram` **249**, `not_english` 223,
  `english_in_control_it` 155, `not_mixed` 152, `not_italianized` 62,
  `length` 113, exact duplicates 139. The 4-gram overlap check used the S1-dev,
  T and A scripts (5,558 distinct 4-grams). `filter-report.json`.
- **Combined corpus:** 185,103 base + 3,521 added (1 exact duplicate) =
  **188,624** rows, sha256 `de9fc474…`; `combine-report.json`.

## LM

| Artifact | E-034c | E-025 |
|---|---:|---:|
| MOTHLMB1 | 12,303,348 B | 12,375,336 B (+71,988) |
| bigrams / trigrams | 265,713 / 2,278,369 | 266,955 / 2,292,642 |
| packed MOTHLMB2 | 7,787,184 B | **7,833,032 B** (+45,848) |
| pack | lossless | **lossless** |
| package total | 49,203,776 B | 49,249,624 B (750,376 B room) |

The build is the E-034c recipe with one variable (the added text), term ×1,
unpruned, same tokenizer.

## Gate

| Gate | Result | Verdict |
|---|---|---|
| 1. R-S1-cal ETA win with WER guard (or WER win with ETA guard) | 12.8917 / 75.6291 vs 12.8469 / 75.4967; WER +0.0448 pp [−0.0900, +0.1807], ETA +0.1325 pp [0, +0.4178]; both **ties**, 7/400 clips differ | **FAIL: no win** |
| 2. S1-dev | not run (gate 1 failed) | — |
| 3. dev-macro | not run (gate 1 failed) | — |
| 4. Budget | 49,249,624 B, 750,376 B room, lossless | pass |
| 5. T-S3 | not run (gate 1 failed) | — |

The seven changed clips are single-word variants (`kickato`/`kiccato`,
`checktato`/`chektato`, `smtp` placement, `greencot`/`greencoute`); the aggregate
is noise. A 1.9 % corpus addition (3,521 sentences, ~90k tokens) is too small to
move the unpruned 2.3M-trigram LM at the frozen decoder, or the generated text
is not close enough to the product distribution.

## Verdict

**Not adopted. E-034c remains the product LM.** The generated text is
leakage-clean and fits the budget, but it does not buy a calibration win. A
larger generation (tens of thousands of filtered sentences, or a stronger
generator) is a new experiment with its own pre-registration; it is not
authorized here. No package, browser or T read changed.

## Artifacts

- `generated.jsonl`, `generated.report.json` — 6,000 raw sentences.
- `filtered.jsonl`, `filter-report.json` — 3,522 kept, 249 eval 4-gram drops.
- `combine-report.json` — 188,624-row corpus.
- `lm-build.json`, `lm-pack.json` — LM and lossless pack.
- `s1-cal.json/.hyp.jsonl` — 7/400 changed, tie.
- `lm_e025.bin`, `lm_e025_mothlmb2.bin` — gitignored binaries.

## Reproduce

```bash
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/e020_generate.py --model-dir "$ROOT/models/Qwen2.5-1.5B-Instruct" \
  --out runs/E-025/generated.jsonl --device mps --batch-size 8
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/e020_filter.py --generated runs/E-025/generated.jsonl \
  --out runs/E-025/filtered.jsonl --report runs/E-025/filter-report.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/e020_combine.py --base /tmp/lm-corpus-nos1dev.jsonl \
  --generated runs/E-025/filtered.jsonl --out /tmp/lm-corpus-e025.jsonl \
  --report runs/E-025/combine-report.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/build_lm_from_corpus.py --corpus /tmp/lm-corpus-e025.jsonl \
  --out runs/E-025/lm_e025 --term-weight 1 --report runs/E-025/lm-build.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/pack_lm.py --lm runs/E-025/lm_e025.bin \
  --out runs/E-025/lm_e025_mothlmb2.bin --bits 10 --report runs/E-025/lm-pack.json
```
