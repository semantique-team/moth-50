"""MOTHLMB2: lossless, bit-packed BPE n-gram LM for MOTH-50.

``MOTHLMB1`` (Moth 0.8.1) stores every n-gram entry as a uint16 token + an int16
log-prob (4 B) and every trigram context as a uint32 key + a uint32 row offset
(8 B), although a token needs 11 bits and the A3 file has only 1,018 distinct
bigram and 859 distinct trigram log-probs.  ``MOTHLMB2`` keeps the exact int16
values in a per-order codebook and packs every entry as:

    token (11 bits) | codebook index (``bits`` bits)

The trigram keys are reconstructed from a uint32 ``prev2`` offset table and an
11-bit ``prev1`` array, and the trigram row offsets use the minimal bit width.
Every stored value and every score is **identical** to ``MOTHLMB1``: the codebook
is only accepted when it is lossless, and scores are computed as
``int(codebook[code]) * log_scale`` just like ``stt.lm.BinaryBPELM``.

Layout (little-endian), all bitstreams padded to a multiple of 4 bytes::

    magic "MOTHLMB2" (8)
    header <IIffIIIIIII>: version=2, vocab, backoff(f32), log_scale(f32), bits,
        token_bits, offset_bits, n_bigram, n_context, n_trigram, reserved (44)
    bigram codebook int16 * 2**bits
    trigram codebook int16 * 2**bits
    unigram float32 * vocab
    bigram row offsets uint32 * (vocab + 1)
    bigram entries bit-packed
    prev2 row offsets uint32 * (vocab + 1)
    prev1 ids bit-packed (token_bits)
    trigram row offsets bit-packed (offset_bits)
    trigram entries bit-packed

Reader and writer live here, outside ``vendor/moth-stt``, because the archived
Moth repository is read-only; ``load_lm`` still returns Moth's own
``BinaryBPELM`` for ``MOTHLMB1`` so the baseline semantics stay untouched.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = [
    "BOS",
    "EOS",
    "VOCAB",
    "MothLMB2",
    "load_lm",
    "packed_size",
    "read_mothlmb1",
    "read_mothlmb2",
    "write_mothlmb2",
]

VOCAB = 1026  # 1,024 BPE pieces + BOS + EOS
BOS, EOS = 1024, 1025
TOKEN_BITS = 11
MAGIC1 = b"MOTHLMB1"
MAGIC2 = b"MOTHLMB2"
HEADER = struct.Struct("<IIffIIIIIII")
HEADER_SIZE = 8 + HEADER.size


# --- MOTHLMB1 reader (the source format for packing) ---------------------------


def read_mothlmb1(data: bytes) -> dict:
    """Decode a ``MOTHLMB1`` buffer into arrays, without importing Moth."""
    if data[:8] != MAGIC1:
        raise ValueError("not a MOTHLMB1 file")
    version, vocab, backoff, unigram_length, log_scale = struct.unpack_from("<IIfIf", data, 8)
    if version != 1 or vocab != VOCAB or unigram_length != VOCAB:
        raise ValueError("unsupported MOTHLMB1 header")
    offset = 28

    def read(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        array = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += array.nbytes
        return array

    unigram = read("<f4", VOCAB)
    read("<u4", 1)
    bigram_offsets = read("<u4", VOCAB + 1)
    entries = int(bigram_offsets[-1])
    bigram_tokens = read("<u2", entries)
    bigram_logq = read("<i2", entries)
    offset = (offset + 3) // 4 * 4
    contexts = int(read("<u4", 1)[0])
    trigram_keys = read("<u4", contexts)
    trigram_offsets = read("<u4", contexts + 1)
    entries = int(trigram_offsets[-1])
    trigram_tokens = read("<u2", entries)
    trigram_logq = read("<i2", entries)
    if offset != len(data):
        raise ValueError("trailing bytes")
    return {
        "backoff": backoff,
        "log_scale": log_scale,
        "unigram": unigram,
        "bigram_offsets": bigram_offsets,
        "bigram_tokens": bigram_tokens,
        "bigram_logq": bigram_logq,
        "trigram_keys": trigram_keys,
        "trigram_offsets": trigram_offsets,
        "trigram_tokens": trigram_tokens,
        "trigram_logq": trigram_logq,
    }


# --- bit packing ---------------------------------------------------------------


def pack_bits(values: np.ndarray, width: int) -> bytes:
    """Little-endian bitstream of ``width``-bit unsigned values, padded to 4 B."""
    values = np.asarray(values, dtype=np.uint64)
    bits = ((values[:, None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    return packed + b"\0" * (-len(packed) % 4)


def unpack_bits(data: bytes, width: int, count: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[: width * count].reshape(count, width)
    return (bits.astype(np.uint64) << np.arange(width, dtype=np.uint64)).sum(axis=1)


def bitstream_bytes(count: int, width: int) -> int:
    size = (count * width + 7) // 8
    return size + (-size % 4)


def packed_size(
    bigrams: int, trigrams: int, contexts: int, bits: int, offset_bits: int | None = None
) -> int:
    """Exact ``MOTHLMB2`` size for the given counts (same layout as the writer)."""
    entry = TOKEN_BITS + bits
    if offset_bits is None:
        offset_bits = max(1, trigrams.bit_length())
    return (
        HEADER_SIZE
        + 2 * 2 * 2**bits  # bigram and trigram int16 codebooks
        + 4 * VOCAB  # unigram
        + 4 * (VOCAB + 1)  # bigram row offsets
        + bitstream_bytes(bigrams, entry)
        + 4 * (VOCAB + 1)  # prev2 row offsets
        + bitstream_bytes(contexts, TOKEN_BITS)
        + bitstream_bytes(contexts + 1, offset_bits)
        + bitstream_bytes(trigrams, entry)
    )


def exact_codebook(values: np.ndarray, bits: int, order: str) -> tuple[np.ndarray, np.ndarray]:
    """Lossless codebook: every distinct int16 value, padded to ``2**bits`` entries."""
    unique, inverse = np.unique(values.astype(np.int64), return_inverse=True)
    if len(unique) > 2**bits:
        raise ValueError(
            f"{order}: {len(unique)} distinct values do not fit in {bits} bits "
            f"({2**bits}); a lossless MOTHLMB2 needs more bits"
        )
    book = np.zeros(2**bits, dtype="<i2")
    book[: len(unique)] = unique.astype("<i2")
    return book, inverse.astype(np.uint64)


def write_mothlmb2(lm: dict, bits: int = 10) -> bytes:
    """Encode a decoded MOTHLMB1 dictionary as ``MOTHLMB2``; refuses lossy input."""
    if not 1 <= bits <= 15:
        raise ValueError("bits must be between 1 and 15")
    scale = float(lm["log_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("log_scale must be finite and positive")
    if not np.isfinite(lm["backoff"]) or lm["backoff"] <= 0:
        raise ValueError("backoff must be finite and positive")
    if not np.isfinite(lm["unigram"]).all():
        raise ValueError("unigram must be finite")

    bigram_book, bigram_codes = exact_codebook(lm["bigram_logq"], bits, "bigram")
    trigram_book, trigram_codes = exact_codebook(lm["trigram_logq"], bits, "trigram")

    keys = lm["trigram_keys"].astype(np.int64)
    if np.any(keys < 0) or np.any(keys >= VOCAB * VOCAB):
        raise ValueError("trigram key outside the vocabulary")
    if np.any(keys[1:] <= keys[:-1]):
        raise ValueError("trigram keys must be strictly sorted")
    prev2 = keys // VOCAB
    prev1 = keys % VOCAB
    prev2_offsets = np.searchsorted(prev2, np.arange(VOCAB + 1)).astype("<u4")
    n_bi = int(lm["bigram_offsets"][-1])
    n_tri = int(lm["trigram_offsets"][-1])
    if len(lm["bigram_tokens"]) != n_bi or len(lm["trigram_tokens"]) != n_tri:
        raise ValueError("row offsets do not match the entry counts")
    if len(prev1) != len(keys):
        raise ValueError("trigram keys and prev1 disagree")
    offset_bits = max(1, n_tri.bit_length())
    if n_tri >= 2**offset_bits:
        raise ValueError("offset_bits cannot represent n_trigram")
    entry = TOKEN_BITS + bits

    buffer = bytearray(MAGIC2)
    buffer += HEADER.pack(
        2,
        VOCAB,
        lm["backoff"],
        scale,
        bits,
        TOKEN_BITS,
        offset_bits,
        n_bi,
        len(keys),
        n_tri,
        0,
    )
    buffer += bigram_book.tobytes()
    buffer += trigram_book.tobytes()
    buffer += lm["unigram"].astype("<f4").tobytes()
    buffer += lm["bigram_offsets"].astype("<u4").tobytes()
    buffer += pack_bits(lm["bigram_tokens"].astype(np.uint64) | (bigram_codes << TOKEN_BITS), entry)
    buffer += prev2_offsets.tobytes()
    buffer += pack_bits(prev1.astype(np.uint64), TOKEN_BITS)
    buffer += pack_bits(lm["trigram_offsets"].astype(np.uint64), offset_bits)
    buffer += pack_bits(
        lm["trigram_tokens"].astype(np.uint64) | (trigram_codes << TOKEN_BITS), entry
    )
    return bytes(buffer)


def read_mothlmb2(data: bytes) -> dict:
    """Decode ``MOTHLMB2`` into the same arrays as ``read_mothlmb1`` plus codebooks."""
    if data[:8] != MAGIC2:
        raise ValueError("not a MOTHLMB2 file")
    if len(data) < HEADER_SIZE:
        raise ValueError("truncated MOTHLMB2 header")
    (
        version,
        vocab,
        backoff,
        log_scale,
        bits,
        token_bits,
        offset_bits,
        n_bi,
        n_ctx,
        n_tri,
        _,
    ) = HEADER.unpack_from(data, 8)
    if version != 2:
        raise ValueError(f"unsupported MOTHLMB2 version: {version}")
    if vocab != VOCAB:
        raise ValueError("MOTHLMB2 vocab mismatch")
    if token_bits != TOKEN_BITS:
        raise ValueError("MOTHLMB2 token width mismatch")
    if not 1 <= bits <= 15 or not 1 <= offset_bits <= 32:
        raise ValueError("invalid MOTHLMB2 codebook or offset width")
    if not np.isfinite(backoff) or backoff <= 0:
        raise ValueError("MOTHLMB2 backoff must be finite and positive")
    if not np.isfinite(log_scale) or log_scale <= 0:
        raise ValueError("MOTHLMB2 log scale must be finite and positive")
    if n_ctx > VOCAB * VOCAB or n_tri >= 2**offset_bits:
        raise ValueError("invalid MOTHLMB2 counts")
    offset = HEADER_SIZE

    def take(size: int) -> bytes:
        nonlocal offset
        if size > len(data) - offset:
            raise ValueError("truncated MOTHLMB2 buffer")
        chunk = data[offset : offset + size]
        offset += size
        return chunk

    bigram_book = np.frombuffer(take(2 * 2**bits), dtype="<i2")
    trigram_book = np.frombuffer(take(2 * 2**bits), dtype="<i2")
    unigram = np.frombuffer(take(4 * vocab), dtype="<f4")
    if not np.isfinite(unigram).all():
        raise ValueError("MOTHLMB2 unigram must be finite")
    bigram_offsets = np.frombuffer(take(4 * (vocab + 1)), dtype="<u4")
    entry = token_bits + bits
    bigram = unpack_bits(take(bitstream_bytes(n_bi, entry)), entry, n_bi)
    prev2_offsets = np.frombuffer(take(4 * (vocab + 1)), dtype="<u4")
    prev1 = unpack_bits(take(bitstream_bytes(n_ctx, token_bits)), token_bits, n_ctx)
    trigram_offsets = unpack_bits(
        take(bitstream_bytes(n_ctx + 1, offset_bits)), offset_bits, n_ctx + 1
    )
    trigram = unpack_bits(take(bitstream_bytes(n_tri, entry)), entry, n_tri)
    if offset != len(data):
        raise ValueError("trailing MOTHLMB2 bytes")

    def validate_rows(offsets: np.ndarray, count: int, name: str) -> None:
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]) or int(offsets[-1]) != count:
            raise ValueError(f"MOTHLMB2 {name} offsets must be monotonic and end at {count}")

    validate_rows(bigram_offsets, n_bi, "bigram")
    validate_rows(prev2_offsets, n_ctx, "prev2")
    validate_rows(trigram_offsets, n_tri, "trigram")

    mask = np.uint64((1 << token_bits) - 1)
    bigram_tokens = (bigram & mask).astype(np.uint16)
    trigram_tokens = (trigram & mask).astype(np.uint16)
    if np.any(bigram_tokens >= vocab) or np.any(trigram_tokens >= vocab):
        raise ValueError("MOTHLMB2 token outside vocabulary")
    if np.any(prev1 >= vocab):
        raise ValueError("MOTHLMB2 prev1 outside vocabulary")

    def validate_sorted(offsets: np.ndarray, tokens: np.ndarray, name: str) -> None:
        if tokens.size == 0:
            return
        ordered = tokens[1:] > tokens[:-1]
        boundaries = offsets[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(tokens))]
        ordered[boundaries - 1] = True
        if not ordered.all():
            raise ValueError(f"MOTHLMB2 {name} row tokens must be strictly sorted")

    validate_sorted(bigram_offsets, bigram_tokens, "bigram")
    validate_sorted(trigram_offsets, trigram_tokens, "trigram")

    prev2 = np.repeat(np.arange(vocab), np.diff(prev2_offsets))
    trigram_keys = (prev2 * vocab + prev1.astype(np.int64)).astype("<u4")
    if np.any(trigram_keys[1:] <= trigram_keys[:-1]):
        raise ValueError("MOTHLMB2 trigram keys must be strictly sorted")

    return {
        "backoff": backoff,
        "log_scale": log_scale,
        "unigram": unigram,
        "bigram_offsets": bigram_offsets,
        "bigram_tokens": bigram_tokens,
        "bigram_logq": bigram_book[(bigram >> np.uint64(token_bits)).astype(np.int64)],
        "bigram_codes": (bigram >> np.uint64(token_bits)).astype(np.int64),
        "trigram_keys": trigram_keys,
        "trigram_offsets": trigram_offsets.astype(np.uint32),
        "trigram_tokens": trigram_tokens,
        "trigram_logq": trigram_book[(trigram >> np.uint64(token_bits)).astype(np.int64)],
        "trigram_codes": (trigram >> np.uint64(token_bits)).astype(np.int64),
        "bigram_codebook": bigram_book,
        "trigram_codebook": trigram_book,
        "bits": bits,
    }


# --- runtime scorer -------------------------------------------------------------


class MothLMB2:
    """Read-only ``MOTHLMB2`` scorer with the exact ``BinaryBPELM.score`` semantics."""

    def __init__(self, data: bytes) -> None:
        decoded = read_mothlmb2(bytes(data))
        self.backoff = decoded["backoff"]
        self.log_scale = decoded["log_scale"]
        self.unigram = decoded["unigram"]
        self.bigram_offsets = decoded["bigram_offsets"].astype(np.int64)
        self.bigram_tokens = decoded["bigram_tokens"].astype(np.int64)
        self.bigram_logq = decoded["bigram_logq"].astype(np.int64)
        self.trigram_keys = decoded["trigram_keys"]
        self.trigram_offsets = decoded["trigram_offsets"].astype(np.int64)
        self.trigram_tokens = decoded["trigram_tokens"].astype(np.int64)
        self.trigram_logq = decoded["trigram_logq"].astype(np.int64)
        self.unigram_floor = (self.unigram.astype(np.float64) + 2.0 * np.log(self.backoff)).astype(
            np.float32
        )
        self.unigram_floor.flags.writeable = False

    @classmethod
    def load(cls, path: Path | str) -> MothLMB2:
        return cls(Path(path).read_bytes())

    def score(self, prev2: int, prev1: int, token: int) -> float:
        """Log probability using trigram, bigram, then unigram fallback."""
        prev2, prev1, token = int(prev2), int(prev1), int(token)
        if not 0 <= token < VOCAB:
            raise ValueError("token outside LM vocabulary")
        token_id = np.uint16(token)
        if 0 <= prev1 < VOCAB:
            if 0 <= prev2 < VOCAB:
                key = np.uint32(prev2 * VOCAB + prev1)
                row = int(np.searchsorted(self.trigram_keys, key))
                if row < len(self.trigram_keys) and self.trigram_keys[row] == key:
                    start, end = self.trigram_offsets[row : row + 2]
                    index = int(start) + int(
                        np.searchsorted(self.trigram_tokens[start:end], token_id)
                    )
                    if index < end and self.trigram_tokens[index] == token:
                        return int(self.trigram_logq[index]) * self.log_scale
            start, end = self.bigram_offsets[prev1 : prev1 + 2]
            index = int(start) + int(np.searchsorted(self.bigram_tokens[start:end], token_id))
            if index < end and self.bigram_tokens[index] == token:
                return int(self.bigram_logq[index]) * self.log_scale
        return float(self.unigram_floor[token])


def load_lm(path: Path | str):
    """Load either format by magic; ``MOTHLMB1`` uses Moth's own ``BinaryBPELM``."""
    data = Path(path).read_bytes()
    if data[:8] == MAGIC2:
        return MothLMB2(data)
    if data[:8] == MAGIC1:
        from stt.lm import BinaryBPELM

        return BinaryBPELM(data)
    raise ValueError(f"unsupported LM magic in {path}")
