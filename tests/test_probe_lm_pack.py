import numpy as np
import probe_lm_pack as pack


def _synthetic_lm(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    vocab = pack.VOCAB
    bigram_counts = rng.integers(0, 6, vocab + 0)
    bigram_offsets = np.concatenate([[0], np.cumsum(bigram_counts)]).astype("<u4")
    bigram_tokens = np.concatenate(
        [np.sort(rng.choice(vocab, size=n, replace=False)) for n in bigram_counts]
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
        "bigram_logq": rng.integers(-1400, -90, len(bigram_tokens)).astype("<i2"),
        "trigram_keys": keys,
        "trigram_offsets": trigram_offsets,
        "trigram_tokens": trigram_tokens,
        "trigram_logq": rng.integers(-1200, 0, len(trigram_tokens)).astype("<i2"),
    }


def test_bit_packing_round_trips():
    values = np.array([0, 1, 2047, 1024, 5, 2046], dtype=np.uint64)
    assert np.array_equal(pack.unpack_bits(pack.pack_bits(values, 11), 11, len(values)), values)


def test_packed_lm_round_trips_and_matches_the_size_formula():
    lm = _synthetic_lm()
    for bits in (10, 8, 6):
        data, values = pack.write_mothlmb2(lm, bits)
        decoded = pack.read_mothlmb2(data)
        for key in (
            "bigram_offsets",
            "bigram_tokens",
            "trigram_keys",
            "trigram_offsets",
            "trigram_tokens",
        ):
            assert np.array_equal(decoded[key].astype(np.int64), lm[key].astype(np.int64))
        assert np.array_equal(decoded["bigram_values"], values["bigram_values"])
        assert len(data) == pack.packed_size(
            len(lm["bigram_tokens"]), len(lm["trigram_tokens"]), len(lm["trigram_keys"]), bits
        )


def test_codebook_is_exact_when_the_values_fit():
    values = np.array([-500, -20, -20, -3, 0], dtype=np.int64)
    book, index = pack.codebook(values, 3)
    assert np.array_equal(book[index.astype(np.int64)], values)


def test_codebook_bounds_the_tail_error():
    rng = np.random.default_rng(1)
    dense = rng.integers(-300, 0, 100_000)
    tail = rng.integers(-1400, -1000, 50)
    values = np.concatenate([dense, tail])
    book, index = pack.codebook(values, 6)
    error = np.abs(book[index.astype(np.int64)] - values)
    assert error.max() < 40  # 0.4 nats at log_scale 0.01, despite only 64 levels
