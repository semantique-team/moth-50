import struct

import moth_lm as lm2
import numpy as np
import pytest


def _synthetic_lm(seed: int = 0, vocab: int = lm2.VOCAB) -> dict:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 6, vocab)
    bigram_offsets = np.concatenate([[0], np.cumsum(counts)]).astype("<u4")
    bigram_tokens = np.concatenate(
        [np.sort(rng.choice(vocab, size=n, replace=False)) for n in counts]
    ).astype("<u2")
    keys = np.unique(rng.integers(0, vocab * vocab, 300)).astype("<u4")
    trigram_counts = rng.integers(1, 5, len(keys))
    trigram_offsets = np.concatenate([[0], np.cumsum(trigram_counts)]).astype("<u4")
    trigram_tokens = np.concatenate(
        [np.sort(rng.choice(vocab, size=n, replace=False)) for n in trigram_counts]
    ).astype("<u2")
    return {
        "backoff": 0.4,
        "log_scale": 0.01,
        "unigram": rng.normal(-8, 1, vocab).astype("<f4"),
        "bigram_offsets": bigram_offsets,
        "bigram_tokens": bigram_tokens,
        "bigram_logq": rng.integers(-1200, -90, len(bigram_tokens)).astype("<i2"),
        "trigram_keys": keys,
        "trigram_offsets": trigram_offsets,
        "trigram_tokens": trigram_tokens,
        "trigram_logq": rng.integers(-1100, 0, len(trigram_tokens)).astype("<i2"),
    }


def _mothlmb1_bytes(lm: dict) -> bytes:
    """Write the probe's MOTHLMB1 layout, for a real ``stt.lm.BinaryBPELM`` baseline."""
    buffer = bytearray(b"MOTHLMB1")
    buffer += struct.pack("<IIfIf", 1, lm2.VOCAB, lm["backoff"], lm2.VOCAB, lm["log_scale"])
    buffer += lm["unigram"].astype("<f4").tobytes()
    buffer += struct.pack("<I", lm2.VOCAB)
    buffer += lm["bigram_offsets"].astype("<u4").tobytes()
    buffer += lm["bigram_tokens"].astype("<u2").tobytes()
    buffer += lm["bigram_logq"].astype("<i2").tobytes()
    buffer += b"\0" * (-len(buffer) % 4)
    buffer += struct.pack("<I", len(lm["trigram_keys"]))
    buffer += lm["trigram_keys"].astype("<u4").tobytes()
    buffer += lm["trigram_offsets"].astype("<u4").tobytes()
    buffer += lm["trigram_tokens"].astype("<u2").tobytes()
    buffer += lm["trigram_logq"].astype("<i2").tobytes()
    return bytes(buffer)


def test_round_trip_is_exact_and_matches_the_size_formula():
    source = _synthetic_lm()
    packed = lm2.write_mothlmb2(source, bits=10)
    decoded = lm2.read_mothlmb2(packed)
    for key in (
        "unigram",
        "bigram_offsets",
        "bigram_tokens",
        "bigram_logq",
        "trigram_keys",
        "trigram_offsets",
        "trigram_tokens",
        "trigram_logq",
    ):
        assert np.array_equal(decoded[key].astype(np.int64), source[key].astype(np.int64)), key
    assert decoded["backoff"] == np.float32(source["backoff"])
    assert decoded["log_scale"] == np.float32(source["log_scale"])
    assert len(packed) == lm2.packed_size(
        len(source["bigram_tokens"]),
        len(source["trigram_tokens"]),
        len(source["trigram_keys"]),
        bits=10,
    )


def test_lossy_codebook_is_refused():
    source = _synthetic_lm()
    source["bigram_logq"] = np.arange(len(source["bigram_tokens"]), dtype="<i2")
    with pytest.raises(ValueError, match="do not fit in 2 bits"):
        lm2.write_mothlmb2(source, bits=2)


def test_packed_reader_validates_magic_and_trailing_bytes():
    packed = lm2.write_mothlmb2(_synthetic_lm(), bits=10)
    with pytest.raises(ValueError, match="not a MOTHLMB2"):
        lm2.read_mothlmb2(b"MOTHLMB1" + packed[8:])
    with pytest.raises(ValueError, match="trailing"):
        lm2.read_mothlmb2(packed + b"\0\0\0\0")


def test_scores_match_moth_binary_bpelmb():
    """Every stored entry and the fallback must be bit-identical to BinaryBPELM."""
    stt_lm = pytest.importorskip("stt.lm")
    source = _synthetic_lm()
    baseline = stt_lm.BinaryBPELM(_mothlmb1_bytes(source))
    candidate = lm2.MothLMB2(lm2.write_mothlmb2(source, bits=10))

    checked = 0
    for row, key in enumerate(source["trigram_keys"]):
        prev2, prev1 = divmod(int(key), lm2.VOCAB)
        start, end = source["trigram_offsets"][row : row + 2]
        for token in source["trigram_tokens"][start:end]:
            assert candidate.score(prev2, prev1, int(token)) == baseline.score(
                prev2, prev1, int(token)
            )
            checked += 1
    rng = np.random.default_rng(7)
    for _ in range(2_000):
        prev2, prev1, token = (int(value) for value in rng.integers(0, lm2.VOCAB, 3))
        assert candidate.score(prev2, prev1, token) == baseline.score(prev2, prev1, token)
        checked += 1
    assert checked > 2_500


def test_load_lm_dispatches_on_magic(tmp_path):
    source = _synthetic_lm()
    packed = lm2.write_mothlmb2(source, bits=10)
    path = tmp_path / "lm.bin"
    path.write_bytes(packed)
    assert isinstance(lm2.load_lm(path), lm2.MothLMB2)
    path.write_bytes(b"NOTALM!!" + packed[8:])
    with pytest.raises(ValueError, match="unsupported LM magic"):
        lm2.load_lm(path)
