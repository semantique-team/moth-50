# Experiments — MOTH-50

New ledger, starting at **E-001**. Do not import Moth's E-xxx IDs; cite them as
external evidence.

## Template

```markdown
## E-0NN — <title> (H<k>)

- **Date**:
- **Hypothesis**:
- **Model configuration**: (encoder, head, vocab, variant; link to `runs/E-0NN/config.yaml`; git SHA)
- **Dataset**: (moth-cs revision + manifest hashes; external corpora + versions; hours per source)
- **Training parameters**: (optimizer, LR, schedule, batch seconds, epochs, augmentation, precision)
- **Hardware**:
- **Compute and cost**:
- **Metrics**: (WER/ETA, per-speaker range, CIs, link to `metrics.json`)
- **Result**:
- **Conclusion**: (confirmed/rejected; gate decision; next experiment)
```

## Discipline

- Pre-register the gate before running; do not move it after seeing the result.
- Report per-speaker numbers, not only the pooled mean; three speakers cannot
  estimate speaker variance.
- A tie is a tie: report the interval, not a winner.
- Keep negative results; they are the most useful part of the record.
- Every run has a `runs/<id>/` directory with config, metrics and logs.
  Checkpoints are private and gitignored.

## Planned

| ID | Question | Status |
|---|---|---|
| E-001 | Reproduce Moth 0.8.1 on the canonical dataset with the vendored evaluator | done: 24.0826 / 55.2253, matches E-038 |
| E-002 | Freeze the evaluation protocol and the data mixture (S1 new sessions + external test); MPS memory probe with and without activation checkpointing (`tools/probe_training.py`) | protocol frozen; 200-step probe passed (13.37 / 3.68 GB); 0.8.1 dev reference re-scored (23.0170 macro); data mixture and shards ready; S1-dev at ~660/960 (2026-11-19), 960/960 cues reviewed |
| E-003 | Architecture screen: Whisper-base vs base+2/+4/+6 on the same mixture | **done**: base+6 wins; greedy dev 48.12 / 36.90 / 34.73 / 33.46 %, two seeds on +4/+6 (33.55 % / 34.74 %); gate needs a longer schedule; S1-dev readout pending |
| E-004 | Package-budget trade-off; re-scoped after E-005: where the E-012 headroom goes (unpruned LM vs base+7 vs base+4 + large LM) | planned (`next-experiments.md`) |
| E-005 | Full fine-tune of the winner; INT8 export; browser run | **done**: 17.5816 / 17.6628 % frozen beam+LM vs 23.0170 % (paired −5.4354 / −5.3542 pp); 47.57 MB package; browser transcript exact, RTFx 22.7. **Erratum 2026-11-19:** decoded and packaged with the pre-A3 LM and Moth's NOTICE (D17); verdict stands, A3 readout/release and S1-dev no-regression pending (E-005 addendum) |
| E-006 | Data scale: VoxPopuli IT, gold repetition, word-level splicing, pseudo-labels, new S1 sessions, TTS | planned, re-scoped as fine-tune arms from E-011 |
| E-007 | Optional architecture: 25 Hz extra blocks, conv adapters, dropout/EMA, LID auxiliary | split into E-011 (EMA), E-013, E-014, E-015 |
| E-008 | External evaluation: CS-FLEURS-READ (test-only), FLEURS, per-speaker ranges | planned |
| E-009 | Decoder recalibration (LM weight × token bonus) on dev-macro + S1-dev | planned; `tools/calibrate_decoder.py` |
| E-010 | Encoder-LR screen: 1e-4 vs 3e-5 on base+4/base+6 | planned; `training/configs/base6-enc1e-4.toml` |
| E-011 | ~3 epochs + weight EMA, two seeds | planned; needs E-009, E-010 |
| E-012 | Package bytes: 16 kHz-only Silero graph, lossless packed A3 LM | planned; synthetic VAD preflight bit-identical (`runs/probe-2026-11-19/`) |
| E-013 | Extra blocks before `final_ln` | planned |
| E-014 | Conformer-style convolution in the extra blocks at equal bytes | planned |
| E-015 | 25 Hz extra blocks (review P7) | conditional on E-016 |
| E-016 | Browser: 1-thread RTFx, browser-vs-Python WER, vendored runtime | planned |

The rationale, parameter counts and gates for E-003–E-008 are in
`docs/architecture-options.md`; the corrected Moth evidence, the measurements
and the proposals that may change them are in `docs/review-2026-11-17.md`. The
post-E-005 plan (E-009–E-016 and the re-scoped E-004/E-006) and its gates are in
`experiments/next-experiments.md`. Do
not start E-003 before E-002 has frozen the evaluation protocol; T is a
comparability set, not a clean test. There is no 100 MB experiment: the tier is
rejected. No CC-BY-NC data in training.
