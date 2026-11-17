import hashlib
from pathlib import Path

import sentencepiece as spm

TOKENIZER = Path(__file__).resolve().parents[1] / "data" / "tokenizer" / "moth_bpe.model"
TOKENIZER_SHA256 = "d6e1054e9273915db4e706898286f90d4b97e6b703386ed8b84f0a148caed2fd"


def test_tokenizer_is_the_pinned_moth_artifact():
    assert TOKENIZER.is_file()
    assert TOKENIZER.stat().st_size == 14_327
    assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == TOKENIZER_SHA256


def test_tokenizer_loads_with_1024_pieces():
    processor = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    assert processor.get_piece_size() == 1024
    pieces = processor.encode("oggi parliamo di machine learning", out_type=str)
    assert pieces
    assert processor.decode(processor.encode("meeting")) == "meeting"
