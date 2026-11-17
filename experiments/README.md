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
| E-002 | Freeze the evaluation protocol and the data mixture (S1 new sessions + external test); MPS memory probe with and without activation checkpointing (`tools/probe_training.py`) | protocol frozen; 200-step probe passed (13.37 / 3.68 GB); 0.8.1 dev reference re-scored (23.0170 macro); source lock pinned; S1-dev and audio download/mel rebuild pending |
| E-003 | Architecture screen: Whisper-base vs base+2/+4/+6 on the same mixture | planned |
| E-004 | Package-budget trade-off: base+6 + 5.38 MB LM vs base+4 + 12 MB LM | planned |
| E-005 | Full fine-tune of the winner; INT8 export; browser run | planned |
| E-006 | Data scale: 2–5 h new S1 recordings, permissive TTS, word/phrase-level splicing | planned |
| E-007 | Optional architecture: 25 Hz extra blocks, conv adapters, dropout/EMA, LID auxiliary | planned |
| E-008 | External evaluation: CS-FLEURS-READ (test-only), FLEURS, per-speaker ranges | planned |

The rationale, parameter counts and gates for E-003–E-008 are in
`docs/architecture-options.md`; the corrected Moth evidence, the measurements
and the proposals that may change them are in `docs/review-2026-11-17.md`. Do
not start E-003 before E-002 has frozen the evaluation protocol; T is a
comparability set, not a clean test. There is no 100 MB experiment: the tier is
rejected. No CC-BY-NC data in training.
