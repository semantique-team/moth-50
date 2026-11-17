# Probe 2026-11-17 — real INT8 size, CPU latency, activation memory, packed LM

**Status: measured.** This is not an E-xxx experiment; it trains nothing. It checks
the arithmetic of `docs/architecture-options.md` §3–4 against the exact Moth 0.8.1
export recipe before E-002, and sizes the packed LM format of review proposal P5.

- **Code:** `tools/probe_export.py`, `tools/probe_activations.py`, run against the
  Moth checkout `moth-stt-dev` at `d5ae605` (`stt.model.MothSTT`, `stt.export`);
  `tools/probe_lm_pack.py` (§4), standalone.
- **Machine:** Windows 11 laptop, Intel Core Ultra 7 268V, 32 GB; Python 3.12.10,
  torch 2.14.1+cpu, onnx 1.23.2, onnxruntime 1.31.0.
- **Weights:** random. The INT8 graph size does not depend on weight values.
- **Raw data:** `export.json`, `export_lengths.json`, `activations.json`,
  `lm_pack.json`.

## 1. INT8 graph size (Moth recipe: `export_fp32` + `quantize_dynamic`, per-channel QInt8)

| Candidate | Params | INT8 bytes | B/param | Package with A3 LM + VAD + misc | Room to 50,000,000 B |
|---|---:|---:|---:|---:|---:|
| tiny+2 (control) | 11,575,169 | 11,876,257 | 1.0260 | 19,671,869 | — |
| base | 20,348,417 | 20,697,343 | 1.0172 | 28,492,955 | 21,507,045 |
| base+2 | 26,652,161 | 27,113,541 | 1.0173 | 34,909,153 | 15,090,847 |
| base+4 | 32,955,905 | 33,530,165 | 1.0174 | 41,325,777 | 8,674,223 |
| **base+6** | **39,259,649** | **39,946,582** | **1.0175** | **47,742,194** | **2,257,806** |
| base+7 | 42,411,521 | 43,154,777 | 1.0175 | 50,950,389 | **−950,389** |

Package = INT8 graph + `lm_bpe3.bin` 5,381,992 + `silero_vad.onnx` 2,327,524 +
vocab/mel filters/config/NOTICE 86,096 B (the Moth 0.8.1 files).

- The control reproduces the shipped 0.8.1 graph within 34,420 B (11,876,257 vs
  11,910,677 B; different torch/ORT versions).
- At width 512 the ratio is **1.0175 B/param**, not 1.03: base+6 is 39.95 MB, not
  40.4 MB, and the package is **47.74 MB with 2.26 MB of room**, not 48.2 / 1.8.
- **base+6 is the deepest model that fits with the A3 LM in its current
  `MOTHLMB1` format** and Silero; base+7 is 0.95 MB over. With a packed LM (§4),
  base+7 + A3 would be 48,926,329 B if A3 packs losslessly at 10 bits
  (unverified: only the 0.7.0 LM was inspected).
- LM room at 50,000,000 B: **7.64 MB** with base+6, **14.06 MB** with base+4.
- The CTC head and the conv stem are INT8 (`MatMulInteger`, `ConvInteger`), as in 0.8.1.

## 2. CPU latency (ONNX Runtime, batch 1)

First run (`export.json`, median of 7):

| Candidate | 10 s, 1 thread | 30 s, 1 thread | 30 s ratio vs tiny+2 |
|---|---:|---:|---:|
| tiny+2 | 0.062 s | 0.357 s | 1.00× |
| base | 0.095 s | 0.501 s | 1.40× |
| base+2 | 0.128 s | 0.657 s | 1.84× |
| base+4 | 0.165 s | 0.869 s | 2.43× |
| base+6 | 0.188 s | 0.983 s | 2.75× |
| base+7 | 0.231 s | 1.035 s | 2.90× |

Length-matched second run (`export_lengths.json`, 1 thread, median of 9):

| Candidate | 5 s | 10 s | 20 s | 30 s | Ratio vs tiny+2 (5 / 10 / 20 / 30 s) |
|---|---:|---:|---:|---:|---:|
| tiny+2 | 0.025 s | 0.061 s | 0.168 s | 0.302 s | 1 |
| base+4 | 0.058 s | 0.139 s | 0.376 s | 0.735 s | 2.38 / 2.29 / 2.24 / 2.43 |
| base+6 | 0.069 s | 0.168 s | 0.445 s | 0.861 s | 2.80 / 2.76 / 2.65 / 2.85 |

Between the two runs the base+6 ratio at the same length moves by up to 10 %
(10 s: 3.03 vs 2.76), so read it as **≈2.7–3.0×**. The 4-thread timings in
`export.json` are noisy on this laptop (base+6 slower than base+7) and are not
used.

Browser projection: the export has a dynamic time axis, so the ratio must match
the clip length. Moth E-022 measured the browser at 1 thread on the tiny+2
architecture: RTFx 24.3 / 22.9 / 20.2× at 5 / 10 / 20 s. Dividing by the
length-matched ratios gives **≈7–9× for base+6** (≈7–7.6× at 20 s) and ≈8.5–10×
for base+4. The only 4-thread figure is 0.8.1's E-031 acceptance run on one 3 s
fixture (46.7×), which suggests ≈15–17× for base+6. Features, VAD and the
decoder do not slow down with model size, so the end-to-end slowdown is somewhat
smaller than the model's; still, these are planning numbers, not browser
measurements. At 1 thread base+6 would miss Moth's ≥ 10× browser bar and sits at
the E-005 gate of ≥ 8×.

## 3. Activation memory kept for backward (bf16 autocast, 192 audio-s per batch)

| Candidate | 6 × 30 s | 12 × 16 s | 32 × 6 s | 6 × 30 s with per-block checkpointing (estimate) |
|---|---:|---:|---:|---:|
| tiny+2 | 3.69 GB | 2.49 GB | 1.45 GB | 0.73 GB |
| base | 4.92 GB | 3.31 GB | 1.93 GB | 0.96 GB |
| base+2 | 6.52 GB | 4.37 GB | 2.53 GB | 0.98 GB |
| base+4 | 8.13 GB | 5.44 GB | 3.13 GB | 1.00 GB |
| **base+6** | **9.73 GB** | **6.50 GB** | **3.74 GB** | **1.02 GB** |

- base+6 keeps **2.6×** the activations of tiny+2. About 80 % of the 30 s bucket is
  attention matrices: `stt.model._Block` materialises `B × heads × T × T` scores and
  keeps an fp32 softmax copy and its bf16 cast for backward.
- E-028 peaked at **7,495.7 MiB (7.86 GB)** of `torch.mps.driver_allocated_memory`
  (Moth `runs/E-028/train.log.jsonl`, step 7,950). Projecting that peak to base+6
  gives **≈12–14 GB** if only the activation term grows (plus 0.44 GB of extra
  weights, gradients and AdamW state) and **≈20 GB** if the whole peak scales with
  the activations. The first case leaves 2–4 GB on a 16 GB machine for macOS and
  the data pipeline; the second does not fit. Which one holds is what the E-002
  probe must measure.
- With per-block activation checkpointing the activation term drops to **≈1 GB**
  for base+6 at the cost of one extra forward pass (~4/3 of the step time).

The checkpointing column is an estimate (block inputs + the largest single block +
stem and head), not a measured peak. On Windows a single process crashed with a
native access violation after the base+6 6 × 30 s batch, so `activations.json` was
merged from one process per candidate. Confirm both numbers on the M5 Pro during
E-002 with `python tools/probe_activations.py --moth-stt ../moth-stt --device mps
--no-count --candidates base+6 --batches 6x30 [--checkpointing] --out ...`, which
runs a real forward/backward/AdamW step and reports MPS driver memory.

## 4. A packed LM format (review proposal P5)

`tools/probe_lm_pack.py` re-encodes a Moth `MOTHLMB1` LM as `MOTHLMB2`: 11-bit
tokens + a `b`-bit index into one log-prob codebook per order, trigram contexts
grouped by `prev2` with an 11-bit `prev1`, and minimal-width row offsets. Every
field is fixed-width, so a JS reader keeps O(1) random access; the backoff rule is
unchanged. The probe checks the round trip and that the written size equals the
layout formula, then scores the 600 unique T sentences with both files. Raw data:
`lm_pack.json`.

- **Input:** the only Moth LM on this machine, the 0.7.0 `lm_bpe3.bin`
  (5,116,464 B, sha256 `dcf1d59f…`; 196,132 bigrams, 804,969 trigrams, 137,976
  contexts). It stores only 967 distinct bigram and 853 distinct trigram values,
  so a 10-bit codebook is lossless.
- **Tokenizer:** the real `moth_bpe.model` is not on this machine. Sentences are
  tokenized by a SentencePiece-style BPE rebuilt from `vocab.json` (0 unknown
  tokens). It is a proxy for realistic contexts, not the shipped tokenizer.

| Codebook bits | Bytes (0.7.0 LM) | vs MOTHLMB1 | Entry error mean / max (nats) | Sentence Δ mean / max (nats) | Top-1 next token unchanged | Unpruned E-034 LM (12,313,440 B) → |
|---:|---:|---:|---:|---:|---:|---:|
| 10 (lossless) | 3,183,116 | 62.2 % | 0 / 0 | 0 / 0 | 100 % | 7,797,724 B |
| **8** | **2,926,700** | **57.2 %** | **0.0067 / 0.054** | **0.033 / 0.165** | **99.36 %** | **7,155,008 B** |
| 7 | 2,800,540 | 54.7 % | 0.0145 / 0.113 | 0.064 / 0.235 | 98.86 % | 6,835,696 B |
| 6 | 2,674,888 | 52.3 % | 0.0296 / 0.236 | 0.125 / 0.545 | 97.11 % | 6,516,896 B |

The mean T sentence log-prob is −45.6 nats; the decoder multiplies LM scores by
0.55. The last column is exact for the layout: entry counts from Moth's E-034
`build.json`, trigram contexts (265,001) derived from its `MOTHLMB1` byte size
(the file itself is not on this machine).

What it means for the 50 MB package with base+6 (39,946,582 B INT8 + VAD + misc):

| LM | Format | LM bytes | Package | Room to 50,000,000 B |
|---|---|---:|---:|---:|
| A3 (0.8.1) | MOTHLMB1 | 5,381,992 | 47,742,194 | 2,257,806 |
| A3 (0.8.1) | MOTHLMB2, 10-bit lossless | 3,357,932 | 45,718,134 | 4,281,866 |
| E-034 unpruned | MOTHLMB1 | 12,313,440 | 54,673,642 | ✗ |
| E-034 unpruned | MOTHLMB2, 10-bit lossless | 7,797,724 | 50,157,926 | ✗ (−157,926) |
| **E-034 unpruned** | **MOTHLMB2, 8-bit** | **7,155,008** | **49,515,210** | **484,790** |
| E-034 unpruned | MOTHLMB2, 7-bit | 6,835,696 | 49,195,898 | 804,102 |

- An 8-bit codebook fits **base+6 with the unpruned LM** under 50 MB with 0.48 MB
  to spare. That LM gained −0.644 pp on Moth's dev (E-034 N2_1_1), but it uses
  term weight ×1 instead of A3's ×3, and its T-600 and rare-term effects were
  never measured with the current decoder: E-034 read dev only and notes that
  E-023's gate deltas (T-600 −0.77) came from a different corpus and decoder
  state, so it does not claim a product gain. The LM-score change from packing is
  small (99.4 % of next-token argmaxes unchanged), but it is not a WER
  measurement: decode dev and T with both formats before adopting it.
- Lossless 10-bit packing would save 2.02 MB on the shipped A3 LM, provided its
  log-probs also take at most 1,024 distinct values per order (the 0.7.0 LM's do;
  11 bits are lossless for any range under 20 nats at Moth's 0.01 scale).
- A quantile-only k-means codebook left the sparse low-probability tail without
  centroids (max entry error 2.14 nats at 8 bits); the probe initializes half the
  centroids on a uniform grid, which bounds the max error at 0.054 nats.
- Not done here: the JS reader, decoder parity, and a WER readout. The unpruned
  LM may store more distinct values than the 0.7.0 LM, so its 8-bit error must be
  measured on the file itself.
