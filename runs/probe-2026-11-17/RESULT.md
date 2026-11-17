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
- **Raw data:** `export.json`, `activations.json`, `lm_pack.json`.

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
- **base+6 is the deepest model that fits** with the A3 LM and Silero; base+7 is
  0.95 MB over.
- LM room at 50,000,000 B: **7.64 MB** with base+6, **14.06 MB** with base+4.
- The CTC head and the conv stem are INT8 (`MatMulInteger`, `ConvInteger`), as in 0.8.1.

## 2. CPU latency (ONNX Runtime, batch 1, median of 7)

| Candidate | 10 s, 1 thread | 30 s, 1 thread | 30 s ratio vs tiny+2 |
|---|---:|---:|---:|
| tiny+2 | 0.062 s | 0.357 s | 1.00× |
| base | 0.095 s | 0.501 s | 1.40× |
| base+2 | 0.128 s | 0.657 s | 1.84× |
| base+4 | 0.165 s | 0.869 s | 2.43× |
| base+6 | 0.188 s | 0.983 s | 2.75× |
| base+7 | 0.231 s | 1.035 s | 2.90× |

The 4-thread timings in `export.json` are noisy on this laptop (base+6 slower than
base+7) and are not used. The only browser measurement of 0.8.1 is the E-031
acceptance run on one 3 s fixture: median RTFx 25.1× (1 thread) and 46.7×
(4 threads). Dividing by the 1-thread ratio gives **≈9× (1 thread) to ≈17×
(4 threads) for base+6**: a planning number, not a browser measurement.

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
  the activations. The 16 GB machine is at or over its limit either way.
- With per-block activation checkpointing the activation term drops to **≈1 GB**
  for base+6 at the cost of one extra forward pass (~4/3 of the step time).

The checkpointing column is an estimate (block inputs + the largest single block +
stem and head), not a measured peak. On Windows a single process crashed with a
native access violation after the base+6 6 × 30 s batch, so `activations.json` was
merged from one process per candidate. Confirm both numbers on the M5 Pro with
`python tools/probe_activations.py --device mps [--checkpointing]` during E-002.

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
0.55. The last column is exact for the layout, computed from the entry and context
counts in Moth's E-034 `build.json` (the file itself is not on this machine).

What it means for the 50 MB package with base+6 (39,946,582 B INT8 + VAD + misc):

| LM | Format | LM bytes | Package | Room to 50,000,000 B |
|---|---|---:|---:|---:|
| A3 (0.8.1) | MOTHLMB1 | 5,381,992 | 47,742,194 | 2,257,806 |
| A3 (0.8.1) | MOTHLMB2, 10-bit lossless | 3,357,932 | 45,718,134 | 4,281,866 |
| E-034 unpruned | MOTHLMB1 | 12,313,440 | 54,673,642 | ✗ |
| E-034 unpruned | MOTHLMB2, 10-bit lossless | 7,797,724 | 50,157,926 | ✗ (−157,926) |
| **E-034 unpruned** | **MOTHLMB2, 8-bit** | **7,155,008** | **49,515,210** | **484,790** |
| E-034 unpruned | MOTHLMB2, 7-bit | 6,835,696 | 49,195,898 | 804,102 |

- An 8-bit codebook fits **base+6 with the unpruned LM** (E-023/E-034: dev −0.65,
  T-600 −0.77 significant) under 50 MB with 0.48 MB to spare. The LM-score change
  is small (99.4 % of next-token argmaxes unchanged), but it is not a WER
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
