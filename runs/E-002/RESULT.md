# E-002 — RESULT: base+6 fits in 16 GB, and checkpointing is the safe default

> Executed 2026-11-17 on the M5 Pro. Two real 200-step MPS training probes
> (with and without per-block activation checkpointing), base+6, 6 × 30 s
> bucket, real `moth-cs` mels and the pinned tokenizer. No quality claim: the
> model is randomly initialized and the probe saves no checkpoint.

## Verdict

Both configurations pass the E-002 memory gate:

| Configuration | Peak MPS driver memory | Median step | Padded throughput | Non-finite steps |
|---|---:|---:|---:|---:|
| base+6, no checkpointing | **13.368 GB** | 3.461 s | 51.31 audio-s/s | 0 |
| base+6, per-block checkpointing | **3.680 GB** | 4.473 s | 40.08 audio-s/s | 0 |

- Without checkpointing, 13.37 GB fits the 16 GB machine (gate ≤ 14.5 GB) but
  leaves only ~3.8 GB for macOS and the data pipeline.
- With checkpointing, 3.68 GB is far below the 12.5 GB gate and stable for all
  200 steps; the cost is **1.29×** the step time (4.47 vs 3.46 s).
- **Decision: use per-block activation checkpointing from the first base run**
  (D15). No fallback to base+4 or a 128 s batch is needed.
- Memory is flat across the run (first step 13.324 GB, last 13.350 GB without
  checkpointing; 3.671 GB / 3.680 GB with it), so there is no growth from
  AdamW state or the allocator over 200 steps.

## Dev-macro reference for the E-003 gate

The retained E-037 dev hypotheses are the shipped 0.8.1 INT8 graph + A3 LM with
the frozen decoder. Re-scoring them with the vendored evaluator on the two dev
manifests reproduces E-037 exactly:

| Dev set | n | WER | 95 % CI |
|---|---:|---:|---|
| MLS-IT dev | 1,248 | **28.1040** | [27.3562, 28.8502] |
| LS-EN dev | 2,691 | **17.9299** | [17.4856, 18.4061] |
| **Macro** | 3,939 | **23.0170** | — |

The E-003 gate therefore uses **23.0170 % macro WER** as the measured 0.8.1
reference, not the stale 30.86 (E-014) or the acoustic-only 31.01 greedy
(E-028): those are different protocols. Details and hashes:
`runs/E-002/dev-reference.json`.

## Method

- Model: `MothSTT`, `EncoderConfig(width=512, n_layers=6, n_heads=8,
  ffn_dim=2048)`, `head_blocks=6`, 39,259,649 parameters, random init seed 0.
- Data: `../moth-cs-dataset/manifests/train.jsonl` (1,413 clips, mean 2.447 s,
  max 5.131 s), loaded through Moth's own `load_audio_16k` + `log_mel`; every
  batch is padded to the 6 × 30 s worst-case bucket, with true clip lengths
  passed to the model.
- Targets: the pinned tokenizer `data/tokenizer/moth_bpe.model`
  (sha256 `d6e1054e…`, 1,024 pieces), CTC ids.
- Step: bf16 autocast, fp32 CTC (`blank=0`, `zero_infinity=True`), AdamW
  (head 1e-4, encoder 1e-5, betas 0.9/0.98, weight decay 0.01), grad clip 1.0,
  `empty_cache` every 10 steps. No augmentation, no checkpoint saving.
- 200 optimizer steps after 10 warm-up steps; timing excludes the warm-up.
- Checkpointing wraps every encoder and head block with
  `torch.utils.checkpoint(..., use_reentrant=False)` and recomputes them in
  backward.

The one-step `tools/probe_activations.py` readings (13.322 GB / 3.671 GB) match
the 200-step maxima, so the earlier measurement was not a one-step artifact.

## Interpretation

The throughput numbers are deliberately conservative: every batch is padded to
30 s, while the real clips average 2.447 s and the trainer's batch planner caps
**audio seconds**, not per-clip length. Real variable-length batches should use
less memory and run faster per optimizer step. Do not use 51.3 / 40.1 audio-s/s
as the final training throughput; the point of this probe is the worst-case
memory and stability, which pass.

The checkpointed path is the better default even though it is 29 % slower per
step: it leaves >12 GB of headroom for the data pipeline, macOS and allocator
fragmentation, which is exactly where Moth lost runs (E-005, E-006, E-014,
E-016). The 192 audio-s batch cap stays.

## Remaining E-002 items

- Record and verify the new held-out **S1-dev session** (30–60 min, consent
  addendum): E-003 cannot be ranked without it.
- Re-fetch the public corpora at pinned revisions and rebuild the mel shards
  for the frozen E-003 mixture (`experiments/E-002-protocol.md` §2).
- The 0.8.1 dev-macro reference is re-scored (**23.0170 %**); E-003 starts once
  the two data prerequisites above exist.

## Artifacts

- `runs/E-002/train-probe.json` — no checkpointing.
- `runs/E-002/train-probe-checkpointing.json` — per-block checkpointing.
- `runs/E-002/dev-reference.json` — re-scored 0.8.1 dev-macro reference.
- `experiments/E-002-protocol.md` — frozen protocol.
- `data/tokenizer/moth_bpe.model` — pinned tokenizer
  (sha256 `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`).
- Platform: macOS 27.0.1 arm64, Python 3.12.12, torch 2.14.1.
