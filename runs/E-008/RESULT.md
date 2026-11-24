# E-008 — CS-FLEURS-READ once: the final external number

**Result 2026-11-24.** The product `0.9.7-e023-r1` was decoded once on the
CS-FLEURS-READ Italian–English test: **WER 34.81 %** (95 % CI 32.89–36.81),
insertion rate 2.26 %, 176 utterances / 4,203 reference words / 0.51 h, one
speaker. Report-only: no model, decoder or LM change follows, and this read is
not used for selection.

## Set and protocol

- **Source:** `byan/cs-fleurs` (`data/raw/cs-fleurs/read/test`, pinned at
  revision `0cdbf166…` in `data/sources.lock.json`), language pair `ita-eng`.
  Licence **CC-BY-NC-4.0**, **test-only, never training**. 176 utterances,
  0.5085 h, one speaker (SC), fluency 0/1/2 = 86/40/50.
- **Manifest:** `data/manifests/cs-fleurs-read.jsonl`, built reproducibly by
  `tools/build_cs_fleurs_manifest.py` (report
  `runs/E-008/cs-fleurs-read-report.json`); `split: test`, `source:
  cs-fleurs`, no `en_spans` (no word alignment, so ETA is undefined). Mel
  shards: 176 clips, 1 file, 0.51 h, 0 errors
  (`data/shards/cs-fleurs-read/`).
- **Decoder:** product INT8 graph `runs/E-023/release-0.9.7/model.int8.onnx`
  (sha256 `85abd830…`), E-034c LM `runs/E-004/lm_e034c_n2.bin` (packed sha256
  `7a4332f1…`), frozen beam 12 / top-k 10 / LM weight 0.55 / token bonus 2.7,
  `tools/frozen_readout.py`. `gate_eligible: false` because the LM is E-034c,
  not A3; this is an external report, not an E-002 gate.

## Result

| Metric | Value |
|---|---:|
| WER | **34.8085 %** |
| 95 % CI (utterance bootstrap, 1,000) | 32.8929 – 36.8120 |
| Insertion rate | 2.2603 % |
| Reference words | 4,203 |
| Utterances | 176 |
| ETA | n/a (no aligned English spans) |
| Speaker range | n/a (one speaker) |

**Digit sensitivity (report-only).** 27 of the 176 transcripts carry digits
(`35 mm`, `400`, `40,000`, times). The canonical normalizer spells them with
`lang_hint = it`, so an English-read number is a scoring artifact rather than a
recognition error. All 176 rows are kept in the primary number; the split is
reported for transparency:

| Subset | n | WER | 95 % CI |
|---|---:|---:|---:|
| without digits | 149 | 33.9948 % | 31.8533 – 36.1435 |
| with digits | 27 | 38.7052 % | 34.5714 – 43.3432 |

The difference is about 4.7 pp inside the 27 rows and about 0.8 pp on the full
set; the external number is not an artifact of those rows.

## Reading

This is the final external number for the closed model line (D36). It is a
**read** mixed IT–EN test with one unseen speaker, FLEURS-style content, and no
in-domain adaptation; it is not comparable to T-600 or S1-dev, which are S1/S2/
S3 scripted sessions. It is higher than the S1 numbers because the speaker,
accent and domain are outside the training/calibration voices, and it is the
only non-Italian-accented mixed set in the protocol. No speaker range is
reported because the set has one speaker.

**Not run:** the report-only ceiling read (E-018 teacher and frontier APIs) in
the E-008 addition. It needs third-party models, consent for S2/S3 audio and
API cost; it never selects and is not part of the product number.

## Artifacts

- `runs/E-008/cs-fleurs-read-0.9.7.json` / `.hyp.jsonl` — readout and
  hypotheses.
- `runs/E-008/cs-fleurs-read-report.json` — manifest report.
- `data/manifests/cs-fleurs-read.jsonl`,
  `data/shards/cs-fleurs-read/` (gitignored),
  `tools/build_cs_fleurs_manifest.py`,
  `tests/test_build_cs_fleurs_manifest.py`.
