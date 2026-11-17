# Pinned tokenizer

`moth_bpe.model` is the exact 1,024-piece SentencePiece BPE tokenizer used by
Moth 0.8.1. The published `moth-stt` release ships only `vocab.json`; the LM is
indexed by these piece IDs, so the model must not be rebuilt or substituted.

- **File:** `moth_bpe.model`
- **Bytes:** 14,327
- **sha256:** `d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd`
- **Source:** `semantique-team/moth-stt@f3ecd1b0f4f0c5c044fa637e5fb5dd8a0c798d97`,
  `data/tokenizer/moth_bpe.model` (the file is gitignored in the Moth
  repository; this copy is the verified training artifact).
- **Pieces:** 1,024; CTC mapping `blank = 0`, `ctc_id = sp_id + 1`.
- **Checks:** `python -c "import sentencepiece as spm; ..."` loads it and reports
  `get_piece_size() == 1024`; see `tests/test_tokenizer.py`.
- **Provenance:** trained by Moth on normalized Italian–English transcripts plus
  the curated English-term list. The Moth repository is Apache-2.0; the Moth
  recorded data is CC BY 4.0 (`semantique-team/moth-cs`). Keep the attribution
  in `NOTICE`.

Do not train a new tokenizer for MOTH-50. A vocabulary change is a different
model and breaks LM/decoder comparability.
