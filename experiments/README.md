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
| E-002 | Freeze the evaluation protocol and the data mixture (S1 new sessions + external test); MPS memory probe with and without activation checkpointing (`tools/probe_training.py`) | protocol frozen; 200-step probe passed (13.37 / 3.68 GB); 0.8.1 dev reference re-scored (23.0170 macro); data mixture and shards ready; S1-dev **960/960 (51.30 min, 2026-11-19)**, QC clean |
| E-003 | Architecture screen: Whisper-base vs base+2/+4/+6 on the same mixture | **done**: base+6 wins; greedy dev 48.12 / 36.90 / 34.73 / 33.46 %, two seeds on +4/+6 (33.55 % / 34.74 %); E-005 long run and gate complete |
| E-004 | Package-budget trade-off; re-scoped after E-005: where the E-012 headroom goes (unpruned LM vs base+7 vs base+4 + large LM) | **done 2026-11-21, arm B adopted, then superseded by E-034c (D25) and E-023 R1 (D27)**: E-034 N2_1_1 unpruned LM packed lossless, package 49,210,224 B; dev-macro 16.9863 % vs 17.6212 %, S1-dev 11.6289 / 79.8340 vs 14.0541 / 75.5371 (withdrawn by P2), T-600 13.4748 / 76.5101; browser RTFx 28.36. C/D not run (E-011 veto); `runs/E-004/RESULT.md` |
| E-005 | Full fine-tune of the winner; INT8 export; browser run | **done, gate PASS**: A3 readout **17.5565 / 17.6212 %** FP32/INT8 vs 23.0170 % (paired **−5.4605 / −5.3957 pp**); S1-dev **13.9956 / 14.0541 %** vs **22.2498 %**, ETA **75.34 / 75.54 %** vs **62.11 %**, no regression; A3 package **47,836,502 B** (2.16 MB headroom); browser exact, **RTFx 22.3**. Superseded by E-034c (D25) and E-023 R1 (D27); S1-dev term-error analysis in `tools/eta_breakdown.py` |
| E-006 | Data scale: VoxPopuli IT, gold repetition, word-level splicing, pseudo-labels, new S1 sessions, TTS | planned; fine-tune arms from the **R1 + E-034c** product baseline. Pseudo-labels → E-018/E-019; S1 sessions and TTS → E-019/E-021 |
| E-007 | Optional architecture: 25 Hz extra blocks, conv adapters, dropout/EMA, LID auxiliary | split into E-011 (EMA), E-013, E-014, E-015 |
| E-008 | External evaluation: CS-FLEURS-READ (test-only), FLEURS, per-speaker ranges; ceilings from the E-018 teacher and frontier APIs (evaluation-only) | planned |
| E-009 | Decoder recalibration (LM weight × token bonus) on dev-macro + S1-dev | **done 2026-11-19**: dev-macro best **0.45 / 2.0** (17.3427 %, −0.2786 pp) **rejected by the S1-dev veto** (+0.5844 pp WER / −1.1230 pp ETA); frozen **0.55 / 2.7** stays; `runs/E-009/RESULT.md` |
| E-010 | Encoder-LR screen: 1e-4 vs 3e-5 on base+4/base+6 | **done 2026-11-19**: 1e-4 wins both; base+4 33.4003 % (−1.33 pp), base+6 32.3912 % (−1.07 pp); `runs/E-010/RESULT.md` |
| E-011 | ~3 epochs + weight EMA, two seeds | **done 2026-11-21**: dev-macro 15.8584/15.7945 vs 17.5565 (−1.70/−1.76 pp), but S1-dev seed 0 ETA −2.20 pp (CI excludes 0); **not adopted**, E-005 stays; `runs/E-011/RESULT.md` |
| E-012 | Package bytes: 16 kHz-only Silero graph, lossless packed A3 LM | **done 2026-11-19, both gates pass**: 2,040 real clips bit-identical; A3 5,381,992 → 3,353,840 B; package 44,770,433 B; browser exact, RTFx 24.63; `runs/E-012/RESULT.md` |
| E-013 | Extra blocks before `final_ln` | planned |
| E-014 | Conformer-style convolution in the extra blocks at equal bytes | planned |
| E-015 | 25 Hz extra blocks (review P7) | not triggered: E-016 passed the 1-thread gate at 21.17× (20 s) |
| E-016 | Browser: 1-thread RTFx, browser-vs-Python WER, vendored runtime | **done 2026-11-23 (D28)**: 1-thread **10.93/19.74/21.17×** (pass); browser default +0.72 pp WER vs Python, `segment:false` +0.35 pp (tie); runtime vendored at `0e15442`; `runs/E-016/RESULT.md` |
| E-017 | Mixed calibration set separate from the S1-dev gate (new session or S1-dev split) | **done 2026-11-21, frozen**: 400/400 takes, level-matched, manifest + shards, E-002 §8; control E-005 + E-034c 13.9212 / 75.6291; `runs/E-017/RESULT.md` |
| E-018 | Teacher bake-off on the gold train set (Parakeet, Canary, Whisper-turbo, Voxtral; optional SOV/Nemotron) | proposed; no training |
| E-019 | Teacher labels: Granary Italian, VoxPopuli unlabelled, unscripted S1 sessions | proposed; needs E-018; fine-tune from R1 |
| E-020 | LLM-written mixed IT–EN text for the A3-style LM | proposed; needs E-017 (done) and a clean corpus |
| E-021 | Targeted acronym, brand and elision data (recordings + permissive TTS) | proposed; fine-tune from R1 |
| E-022 | Keyterm (contextual) biasing re-test as a product feature | proposed, conditional; Moth E-027 was negative |
| E-023 | R1 recipe ablation: E-011 with encoder LR 3e-5 (H1) | **done 2026-11-23, adopted (D27)**: 2b report-only (D26), T-600 guard passed 12.2993 / 78.1400 vs 13.7041 / 76.1266; product `0.9.7-e023-r1`; `runs/E-023/RESULT.md` |

The rationale, parameter counts and gates for E-003–E-008 are in
`docs/architecture-options.md`; the corrected Moth evidence, the measurements
and the proposals that may change them are in `docs/review-2026-11-17.md`. The
post-E-005 plan (E-009–E-022 and the re-scoped E-004/E-006) and its gates are in
`experiments/next-experiments.md`; the frontier-model review behind E-017–E-022
is `docs/frontier-stt-2026-11-19.md` (D24). Do
not start E-003 before E-002 has frozen the evaluation protocol; T is a
comparability set, not a clean test. There is no 100 MB experiment: the tier is
rejected. No CC-BY-NC data in training.
