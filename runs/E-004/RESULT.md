# E-004 — where the freed bytes go: arm B wins

> Executed 2026-11-21 on the M5 Pro. Decode-only except the LM build: no
> acoustic training, no new data, no weights change. **Arm B (unpruned E-034 LM,
> lossless packed) beats A3 on dev-macro, S1-dev and T-600 and fits at 49.21 MB.
> It is adopted as the product package; the acoustic model remains E-005.**
> **Erratum 2026-11-21 (P2):** the S1-dev half of this evidence is withdrawn;
> 900 of 960 S1-dev cues are in the E-031 LM corpus. The corrected build
> (E-034c) still improves dev-macro and S1-dev against a clean A3-recipe LM, but
> the S1-dev gain is ~0.44 pp, not 2.43. See "Post-review checks" below.

## Arms

The E-012 package left 5,229,567 B of room. The pre-registered arms were:

| Arm | Acoustic | LM | Package | Status |
|---|---|---|---:|---|
| A (control) | base+6 E-005 | A3, packed | 44,770,433 B | measured |
| **B** | base+6 E-005 | **E-034 N2_1_1, unpruned, packed** | **49,210,224 B** | **adopted** |
| C | base+7 | A3, packed | ~47.88 MB est. | not run |
| D | base+4 long | large LM | ~46–49 MB est. | not run |

C and D were conditional: C needed an E-011-length run only if depth still paid,
and E-011 failed the S1-dev ETA veto; D needed base+4 to be competitive at a
long schedule, which was never shown. They are not authorized by this result.

## Arm B build

- Rebuilt the frozen E-031 LM corpus from its receipt (sha256 `c8d94192…`).
- Built E-034 `N2_1_1`: all bigrams, all trigrams, term weight ×1; 12,313,440 B
  MOTHLMB1.
- **The E-034 LM has 1,015 distinct bigram and 873 distinct trigram values**, so
  the 10-bit `MOTHLMB2` is **lossless**: 7,793,632 B, sha256 `86aa7cbf…`.
- Package: 40,039,997 (graph, byte-identical to E-005) + 7,793,632 (LM) +
  1,289,603 (16 kHz VAD) + 86,992 (misc) = **49,210,224 B**, **789,776 B** of
  room.

Decoder calibration on dev-macro (E-009 grid, 20 points) selected the frozen
**0.55 / 2.7**: the best point 0.45 / 2.0 reached 16.908 % versus 16.968 % for
the frozen point, but the 0.06 pp gain is not significant (`adopt: false`).

## Results

### Dev-macro (guard)

| LM | Macro | Δ vs A3 (95 % CI) |
|---|---:|---|
| A3 | 17.6212 % | — |
| **E-034 N2_1_1 packed** | **16.9863 %** | **−0.6349 pp [−0.7609, −0.5056]** |

The gain is smaller than the E-002 1 pp gate, but E-004's pre-registered guard is
dev-macro, not the 1 pp gate; the product sets decide.

### S1-dev (veto)

| LM | WER | ETA | Δ WER (95 % CI) | Δ ETA (95 % CI) | Pass |
|---|---:|---:|---|---|---|
| A3 | 14.0541 % | 75.5371 % | — | — | — |
| **E-034 N2_1_1 packed** | **11.6289 %** | **79.8340 %** | **−2.4251 pp** | **+4.2969 pp** | ✅ |

Both improvements are far outside the noise; no WER or ETA regression. The
per-utterance paired CIs exclude zero in the favourable direction.

**Withdrawn 2026-11-21 (P2):** 900 of the 960 S1-dev cues are exact sentences of
the E-031 corpus the LM was built on. The table is the as-measured number, not a
held-out one. The corrected comparison is below.

### T-600 (read once, final report)

T-600 mel shards were built for this readout (600 clips, 0.39 h).

| LM | WER | ETA | Δ vs A3 (paired) |
|---|---:|---:|---|
| A3 | 13.9908 % | 75.3595 % | — |
| **E-034 N2_1_1 packed** | **13.4748 %** | **76.5101 %** | **−0.5161 pp WER / +1.1505 pp ETA** |

For context, A3 on T-600 is already **−10.09 pp WER / +20.13 pp ETA** versus the
retained 0.8.1 hypotheses; arm B adds another half point of WER and one point of
ETA.

### Browser

The 49.21 MB package loads and verifies every hash in the existing runtime:

| Measure | Result |
|---|---|
| Load | **70 ms** |
| Transcript `T/S1/S1_0001.wav` | **"ho spostato il meeting a domani"** — exact |
| Single run | 251.6 ms, RTFx 15.86 |
| Benchmark median (5 runs) | 140.7 ms, **RTFx 28.36** |

## 2026-11-21 post-review checks (P0, P2)

Both come from `docs/review-2026-11-21.md` §7 and ran before any training.

### P0 — cross-machine parity: pass

E-004 decoded its S1-dev hypotheses on the second Mac (M5, 16 GB) and paired
them with E-005 references written on the M5 Pro. The A3 INT8 graph on this
machine has sha256 `14ba2034…e2d6`, the pin in `export-report-a3.json`.
Re-decoding S1-dev with that graph, the frozen A3 LM and the same decoder
reproduces the E-005 hypotheses **byte-for-byte**:

| File | sha256 |
|---|---|
| `runs/E-005/s1-dev-a3-int8.hyp.jsonl` | `d43af295…9acb29` |
| `runs/E-004/parity-a3-s1-dev.hyp.jsonl` | `d43af295…9acb29` |

WER/ETA 14.0541 / 75.5371, identical to the E-005 readout. The E-004 pairs are
therefore not confounded by decoding differences between the two Macs.
Artifacts: `parity-a3-s1-dev.json`, `parity-a3-s1-dev.hyp.jsonl`.

### P2 — S1-dev overlap with the E-031 LM corpus: leakage found

After Moth's `normalize`, on the frozen E-031 corpus
(`../moth-stt/runs/E-031/frozen/lm-corpus.jsonl`, 186,003 sentences):

| Measure | S1-dev scripts | T scripts |
|---|---:|---:|
| rows | 960 | 800 |
| exact normalized sentences in the corpus | **900** | 0 |
| unique word 4-grams present | **61.16 %** | 9.75 % |
| 4-gram occurrences present | **61.26 %** | 10.39 % |
| multi-word English terms with corpus count 1 | 69.8 % | 23.6 % |

The exact matches are the 900 short-form cues (0001–0900), across all eight
categories: acronym, brand, single_word and italianized 100 %, control_en 95 %,
control_it 93 %, switch 87.5 %, phrase 79 % (the 60 long cues, 0901–0960, are
30 of the 142 phrase rows). The 60 long cues are clean: 0 exact, 3.84 % 4-gram
share. **Root cause:** the S1-dev scripts were selected from
`../moth-cs-dataset/text/pool.jsonl` (README in `text-2026-11-dev`), and
`pool.jsonl` is a `train_script` of the E-031 corpus (3,137 rows). Dedupe at
script time checked T, A, R-S1 and R-S2, not the LM corpus. The pool was never
treated as an evaluation source, so this was invisible until now.

**Consequence.** The E-034 LM had seen 900 of the 960 S1-dev sentences, and the
unpruned LM is exactly the variant that keeps singleton n-grams A3 prunes. The
S1-dev gain was mostly memorisation:

| LM | Corpus | S1-dev WER / ETA | T-600 WER / ETA | Dev macro |
|---|---|---:|---:|---:|
| A3 | full | 14.0541 / 75.5371 | 13.9908 / 75.3595 | 17.6212 |
| E-034 B | full | 11.6289 / 79.8340 | 13.4748 / 76.5101 | 16.9863 |
| A3c (same recipe, clean) | minus 900 | 15.4127 / 72.8027 | 14.1915 / 75.0719 | 17.6253 |
| **E-034c B′** (same recipe, clean) | minus 900 | **14.9744 / 73.9258** | **13.7041 / 76.1266** | **16.9897** |

Paired, INT8, frozen decoder 0.55 / 2.7 (A3c = A3 recipe on the clean corpus):

- **B′ vs B, S1-dev:** +3.3455 pp WER [2.7920, 3.8572] and −5.9082 pp ETA
  [−7.1360, −4.6334]: removing the 900 sentences gives back almost the whole
  advertised gain.
- **B′ vs A3c, S1-dev:** **−0.4381 pp WER [−0.8570, −0.0450] and +1.1230 pp ETA
  [0.4100, 1.9150]** — the unpruned LM is still better than the pruned recipe
  when both are built without the dev sentences.
- **A3c vs A3, S1-dev:** +1.3590 pp WER [1.0000, 1.7460] and −2.7340 pp ETA
  [−3.5710, −1.9500]. A3's own S1-dev number was inflated too; every historical
  S1-dev absolute number is affected, though paired A3-vs-A3 comparisons are
  not.
- **T-600 (declared repeated read — B was read in this experiment, B′ again):**
  B′ vs A3 −0.2870 pp WER [−0.8210, 0.2290] / +0.7670 pp ETA [−0.3550, 1.9310];
  B′ vs A3c −0.4870 [−1.0190, 0.0580] / +1.0550 [−0.0960, 2.2770]. The gain is
  positive but no longer significant at 600 clips. B vs A3c is −0.7170
  [−1.3010, −0.2000] / +1.4380 [0.3740, 2.7080]: the full-corpus LM's slightly
  larger T gain is the 900 extra sentences showing, not a different recipe.
- **Dev macro:** B′ 16.9897 vs A3 17.6212, **−0.6315 pp [−0.7629, −0.5009]**;
  vs A3c 17.6253, **−0.6355 pp [−0.7657, −0.5072]**; B′ vs B +0.0034 pp
  [−0.0100, 0.0182]; A3c vs A3 +0.0040 pp [−0.0101, 0.0185]. The dev-macro gain
  is unaffected by the leak, as expected (the pool sentences are S1 scripts,
  not MLS/LS dev).

**Traces after cleaning:** S1-dev drops to 0 exact and 4.09 % 4-gram share
(below T's 8.30 %); multi-word terms at corpus count 1 fall from 69.8 % to
9.9 % (T: 21.8 %). The residual S1-dev advantage is term-inventory and domain
fit from the same script pipeline, not sentence memorisation.

**Corrected build.** E-034c is the same N2_1_1 recipe (all bigrams/trigrams,
term ×1) on the corpus minus the 900 matching sentences, 185,103 rows
(`lm-corpus-clean-report.json`; MOTHLMB1 12,303,348 B, packed MOTHLMB2
7,787,184 B, 6,448 B smaller than B). A3c is the A3 recipe (≥1 bigrams,
≥2 trigrams, term ×3) on the same clean corpus, 5,376,320 B, a diagnostic, not
a release candidate. The package on disk remains `release-e034/`; a decision to
ship E-034c is pending.

## Decision

- **Adopt arm B (as built).** The package on disk is E-005 acoustic + E-034
  N2_1_1 packed + 16 kHz VAD, version `0.9.5-e004-lm`, **49,210,224 B**. The
  acoustic checkpoint, the decoder 0.55 / 2.7 and the VAD are unchanged. Its
  S1-dev evidence is withdrawn by the 2026-11-21 erratum below.
- The A3 LM remains the historical E-002 gate reference and the package for
  reproducing earlier results. E-004 B is a different, pre-registered LM arm, so
  its readout is `gate_eligible: false` by design; it was judged on the E-004
  rule (dev-macro guard + S1-dev WER/ETA veto + T once), not the A3 1 pp gate.
- This answers E-004: the freed bytes go to **LM coverage**, not more depth. C
  (base+7) and D (base+4 long) are not run; E-011's product veto removed their
  precondition.
- A term-weighted ×3 version of the unpruned LM is not tested. The honest ×1
  gain is +1.12 pp ETA on clean S1-dev (vs A3c); ×3 remains a possible later arm,
  ranked on E-017, not a prerequisite.
- **Erratum 2026-11-21 (P2).** The S1-dev row of arm B is withdrawn: 900 of 960
  cues are exact sentences of the E-031 LM corpus. The corrected build (E-034c:
  same recipe on the corpus minus those sentences) is **16.9897 % dev-macro
  (−0.6355 pp vs A3c), 14.9744 % / 73.9258 % S1-dev (−0.4381 pp WER /
  +1.1230 pp ETA vs A3c, CIs exclude zero) and 13.7041 % / 76.1266 % T-600
  (−0.4870 / +1.0550 vs A3c, CIs include zero)**. The T-600 read is a declared
  repeated read. The honest product gain is ~0.4 pp WER / ~1.1 pp ETA on
  same-speaker mixed dictation and ~0.5 pp / ~1.1 pp on T-600, not the
  advertised 2.4 pp / 4.3 pp. **Decision pending:** ship E-034c in place of
  E-034 B, or keep B and quote T-600 only. Recommendation: ship E-034c; it is
  the same recipe on a corpus that respects the evaluation protocol.

## Artifacts

- `calib-e034.json` — 20-point decoder grid with the new LM.
- `lm-pack-e034.json`, `lm_e034_n2_mothlmb2.bin` (LM binary gitignored).
- `readout-e034-int8.json/.hyp.jsonl` — dev-macro vs E-005 A3.
- `s1-dev-e034-int8.json/.hyp.jsonl` — S1-dev veto.
- `t600-a3-int8.json/.hyp.jsonl`, `t600-e034-int8.json/.hyp.jsonl` — T-600 once.
- `export-report-e034.json`, `release-e034/` (package gitignored).
- `browser-run-e034.json` — load, exact transcript, RTFx.
- `parity-a3-s1-dev.json/.hyp.jsonl` — P0 cross-machine parity (byte-identical).
- `lm-corpus-overlap.json`, `lm-corpus-overlap-after-clean.json` — P2 overlap,
  before and after removing the 900 S1-dev sentences.
- `lm-corpus-clean-report.json` — derivation of the 185,103-row clean corpus.
- `lm-build-e034c.json`, `lm-pack-e034c.json`, `lm_e034c_n2.bin` (gitignored),
  `lm_e034c_n2_mothlmb2.bin` (gitignored) — E-034c.
- `lm-build-a3c.json`, `lm-pack-a3c.json`, `lm_a3c_n1_2_terms3.bin` (gitignored)
  — clean-corpus A3 recipe, diagnostic.
- `readout-e034c-int8.*`, `s1-dev-e034c-int8.*`, `t600-e034c-int8.*` — E-034c
  dev-macro, S1-dev and T-600 (repeated read).
- `readout-a3c-int8.*`, `s1-dev-a3c-int8.*`, `t600-a3c-int8.*` — A3c reference.
- `s1-dev-*-vs-*.json`, `t600-*-vs-*.json` — paired moth_eval comparisons.

## Reproduce

```bash
cd ../moth-stt
uv run --frozen --project . --extra training -- \
  python runs/E-031/reproduce_corpus.py --out runs/E-031/frozen/lm-corpus.jsonl
uv run --frozen --project . --extra training -- python runs/E-034/build_variants.py
cd ../moth-50
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/pack_lm.py --lm ../moth-stt/runs/E-034/build/N2_1_1.bin \
  --out runs/E-004/lm_e034_n2_mothlmb2.bin --bits 10 \
  --report runs/E-004/lm-pack-e034.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/calibrate_decoder.py \
  --checkpoint runs/E-005/base+6-16k/checkpoints/ckpt_best.pt \
  --config training/configs/base6-long.toml \
  --lm ../moth-stt/runs/E-034/build/N2_1_1.bin --allow-other-lm \
  --workers 8 --out runs/E-004/calib-e034.json
```

P0 and P2 (commands assume the outputs do not exist; both tools refuse to
overwrite):

```bash
# P0: re-decode S1-dev with the E-005 A3 graph on this machine, and compare
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed \
  --onnx runs/E-005/release-a3/model.int8.onnx \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --out runs/E-004/parity-a3-s1-dev.json --hyp-out runs/E-004/parity-a3-s1-dev.hyp.jsonl
shasum -a 256 runs/E-004/parity-a3-s1-dev.hyp.jsonl runs/E-005/s1-dev-a3-int8.hyp.jsonl

# P2: overlap, clean corpus, E-034c and the A3c reference
python tools/lm_corpus_overlap.py \
  --corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \
  --set s1-dev=../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \
  --set t=../moth-cs-dataset/text/T.jsonl --out runs/E-004/lm-corpus-overlap.json
python tools/clean_lm_corpus.py \
  --corpus ../moth-stt/runs/E-031/frozen/lm-corpus.jsonl \
  --scripts ../moth-cs-dataset/text-2026-11-dev/R-S1.jsonl \
  --out /tmp/lm-corpus-nos1dev.jsonl --report runs/E-004/lm-corpus-clean-report.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/build_lm_from_corpus.py --corpus /tmp/lm-corpus-nos1dev.jsonl \
  --out runs/E-004/lm_e034c_n2 --report runs/E-004/lm-build-e034c.json
python tools/pack_lm.py --lm runs/E-004/lm_e034c_n2.bin \
  --out runs/E-004/lm_e034c_n2_mothlmb2.bin --bits 10 \
  --report runs/E-004/lm-pack-e034c.json
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/build_lm_from_corpus.py --corpus /tmp/lm-corpus-nos1dev.jsonl \
  --out runs/E-004/lm_a3c_n1_2_terms3 --min-bigram 1 --min-trigram 2 \
  --term-weight 3 --report runs/E-004/lm-build-a3c.json

# readouts (same ONNX graph, product decoder; --lm is the MOTHLMB1 file)
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed --onnx runs/E-004/release-e034/model.int8.onnx \
  --lm runs/E-004/lm_e034c_n2.bin --allow-other-lm \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --out runs/E-004/s1-dev-e034c-int8.json --hyp-out runs/E-004/s1-dev-e034c-int8.hyp.jsonl
uv run --frozen --project ../moth-stt --extra training -- \
  python tools/frozen_readout.py --mode mixed --onnx runs/E-004/release-e034/model.int8.onnx \
  --lm runs/E-004/lm_a3c_n1_2_terms3.bin --allow-other-lm \
  --manifest ../moth-cs-dataset/manifests/R-S1-dev.jsonl --shards data/shards/s1-dev \
  --config training/configs/base6-long.toml \
  --out runs/E-004/s1-dev-a3c-int8.json --hyp-out runs/E-004/s1-dev-a3c-int8.hyp.jsonl
# dev-macro and T-600 follow the same pattern with --mode macro and the T manifest.
```
