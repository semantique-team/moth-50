#!/usr/bin/env python3
"""Prototype a packed LM binary (MOTHLMB2) and measure its size and score error.

Moth's ``MOTHLMB1`` stores every n-gram as a uint16 token + an int16 log-prob
(4 B) and every trigram context as a uint32 key + a uint32 row offset (8 B),
although a token id needs 11 bits (vocabulary 1,026). This probe re-encodes a
``MOTHLMB1`` file with fixed-width, bit-packed fields, which keeps O(1) random
access for a JS reader:

- n-gram entry: 11-bit token + ``b``-bit index into a per-order log-prob codebook
  (2**b float32 values; exact when the distinct stored values fit, otherwise a
  1-D k-means over the stored log-probs);
- trigram contexts grouped by ``prev2`` (uint32 offsets), each storing an 11-bit
  ``prev1`` and a row offset with the minimal bit width;
- unigram log-probs and bigram row offsets unchanged.

Backoff semantics are unchanged: trigram, else bigram, else unigram floor. The
probe verifies the round trip, reports the exact size, predicts the size of other
Moth LMs from their entry/context counts, and compares scores with the original
on normalized text: per-token and per-sentence log-prob deltas and next-token
top-1/top-10 agreement. Tokenization uses a SentencePiece-style BPE encoder
rebuilt from ``vocab.json`` (merge priority = piece id); it is a proxy for the
real ``moth_bpe.model``, good enough to produce realistic contexts.

Usage (num2words is needed for ``moth_text``):
    python tools/probe_lm_pack.py --lm ../moth-stt/lm_bpe3.bin \
        --vocab ../moth-stt/vocab.json \
        --text ../moth-stt/recordings/manifests/T.jsonl \
        --out runs/probe-2026-11-17/lm_pack.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "moth-stt"))

VOCAB = 1026  # 1,024 BPE pieces + BOS + EOS
BOS, EOS = 1024, 1025
TOKEN_BITS = 11

#: Entry/context counts of other Moth LMs. Entries are from Moth runs/E-034 and
#: E-035 build.json; E-034's 265,001 trigram contexts are derived from its MOTHLMB1
#: byte size (12,313,440 = 8,252 fixed + 4 B per entry + 8 B per context).
KNOWN_LMS = {
    "E-031 A3 (0.8.1 shipped)": {
        "bytes": 5_381_992,
        "bigrams": 265_868,
        "trigrams": 802_437,
        "contexts": 137_565,
    },
    "E-034 N1_2_1 (bigram>=2, all trigrams)": {
        "bytes": 12_034_004,
        "bigrams": 196_009,
        "trigrams": 2_280_427,
        "contexts": 265_001,
    },
    "E-034 N2_1_1 (unpruned)": {
        "bytes": 12_313_440,
        "bigrams": 265_868,
        "trigrams": 2_280_427,
        "contexts": 265_001,
    },
}


# --- MOTHLMB1 -----------------------------------------------------------------


def read_mothlmb1(data: bytes) -> dict:
    if data[:8] != b"MOTHLMB1":
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


# --- packing helpers ------------------------------------------------------------


def pack_bits(values: np.ndarray, width: int) -> bytes:
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


def codebook(values: np.ndarray, bits: int, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Per-order log-prob codebook; returns (centroids padded to 2**bits, index per value).

    If the distinct stored values fit in 2**bits the codebook is exact (lossless).
    Otherwise 1-D weighted k-means, initialized half on value quantiles and half on
    a uniform grid over the range: quantile-only initialization leaves the sparse
    low-probability tail without centroids (max error > 2 nats at 8 bits), the
    mixed one keeps the mean error and bounds the max error.
    """
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) <= 2**bits:
        centroids = unique.astype(np.float64)
    else:
        half = 2**bits // 2
        cumulative = np.cumsum(counts) / counts.sum()
        quantile = unique[np.searchsorted(cumulative, (np.arange(half) + 0.5) / half)]
        grid = np.linspace(unique.min(), unique.max(), 2**bits - half)
        centroids = np.unique(np.concatenate([quantile.astype(np.float64), grid]))
        for _ in range(iterations):
            bounds = (centroids[1:] + centroids[:-1]) / 2
            assign = np.searchsorted(bounds, unique)
            sums = np.bincount(assign, weights=unique * counts, minlength=len(centroids))
            weights = np.bincount(assign, weights=counts, minlength=len(centroids))
            keep = weights > 0
            updated = np.unique(sums[keep] / weights[keep])
            if len(updated) == len(centroids) and np.allclose(updated, centroids):
                break
            centroids = updated
    bounds = (centroids[1:] + centroids[:-1]) / 2
    assign = np.searchsorted(bounds, unique)
    padded = np.zeros(2**bits, dtype=np.float64)
    padded[: len(centroids)] = centroids
    return padded, assign[inverse].astype(np.uint64)


# --- MOTHLMB2 -------------------------------------------------------------------


def packed_size(bigrams: int, trigrams: int, contexts: int, bits: int) -> int:
    """Exact MOTHLMB2 size for the given counts (same layout as ``write_mothlmb2``)."""
    entry = TOKEN_BITS + bits
    offset_bits = max(1, trigrams.bit_length())
    return (
        8  # magic
        + 4 * 10  # header
        + 2 * 4 * 2**bits  # bigram and trigram codebooks
        + 4 * VOCAB  # unigram
        + 4 * (VOCAB + 1)  # bigram row offsets
        + bitstream_bytes(bigrams, entry)
        + 4 * (VOCAB + 1)  # contexts grouped by prev2
        + bitstream_bytes(contexts, TOKEN_BITS)
        + bitstream_bytes(contexts + 1, offset_bits)
        + bitstream_bytes(trigrams, entry)
    )


def write_mothlmb2(lm: dict, bits: int) -> tuple[bytes, dict]:
    scale = lm["log_scale"]
    bigram_book, bigram_codes = codebook(lm["bigram_logq"].astype(np.int64), bits)
    trigram_book, trigram_codes = codebook(lm["trigram_logq"].astype(np.int64), bits)
    keys = lm["trigram_keys"].astype(np.int64)
    prev2, prev1 = keys // VOCAB, keys % VOCAB
    prev2_offsets = np.searchsorted(prev2, np.arange(VOCAB + 1)).astype("<u4")
    n_tri = int(lm["trigram_offsets"][-1])
    offset_bits = max(1, n_tri.bit_length())
    entry = TOKEN_BITS + bits

    buffer = bytearray(b"MOTHLMB2")
    buffer += struct.pack(
        "<IIfIIIIIII",
        2,
        VOCAB,
        lm["backoff"],
        bits,
        TOKEN_BITS,
        offset_bits,
        len(lm["bigram_tokens"]),
        len(keys),
        n_tri,
        0,
    )
    buffer += (bigram_book * scale).astype("<f4").tobytes()
    buffer += (trigram_book * scale).astype("<f4").tobytes()
    buffer += lm["unigram"].astype("<f4").tobytes()
    buffer += lm["bigram_offsets"].astype("<u4").tobytes()
    buffer += pack_bits(lm["bigram_tokens"].astype(np.uint64) | (bigram_codes << TOKEN_BITS), entry)
    buffer += prev2_offsets.tobytes()
    buffer += pack_bits(prev1, TOKEN_BITS)
    buffer += pack_bits(lm["trigram_offsets"], offset_bits)
    buffer += pack_bits(
        lm["trigram_tokens"].astype(np.uint64) | (trigram_codes << TOKEN_BITS), entry
    )
    return bytes(buffer), {
        "bigram_values": (bigram_book * scale)[bigram_codes.astype(np.int64)].astype(np.float32),
        "trigram_values": (trigram_book * scale)[trigram_codes.astype(np.int64)].astype(np.float32),
    }


def read_mothlmb2(data: bytes) -> dict:
    if data[:8] != b"MOTHLMB2":
        raise ValueError("not a MOTHLMB2 file")
    header = struct.unpack_from("<IIfIIIIIII", data, 8)
    _, vocab, backoff, bits, token_bits, offset_bits, n_bi, n_ctx, n_tri, _ = header
    offset = 48

    def take(size: int) -> bytes:
        nonlocal offset
        chunk = data[offset : offset + size]
        offset += size
        return chunk

    bigram_book = np.frombuffer(take(4 * 2**bits), dtype="<f4")
    trigram_book = np.frombuffer(take(4 * 2**bits), dtype="<f4")
    unigram = np.frombuffer(take(4 * vocab), dtype="<f4")
    bigram_offsets = np.frombuffer(take(4 * (vocab + 1)), dtype="<u4")
    entry = token_bits + bits
    bigram = unpack_bits(take(bitstream_bytes(n_bi, entry)), entry, n_bi)
    prev2_offsets = np.frombuffer(take(4 * (vocab + 1)), dtype="<u4").astype(np.int64)
    prev1 = unpack_bits(take(bitstream_bytes(n_ctx, token_bits)), token_bits, n_ctx)
    trigram_offsets = unpack_bits(
        take(bitstream_bytes(n_ctx + 1, offset_bits)), offset_bits, n_ctx + 1
    )
    trigram = unpack_bits(take(bitstream_bytes(n_tri, entry)), entry, n_tri)
    if offset != len(data):
        raise ValueError("trailing bytes")
    mask = np.uint64((1 << token_bits) - 1)
    prev2 = np.repeat(np.arange(vocab), np.diff(prev2_offsets))
    return {
        "backoff": backoff,
        "unigram": unigram,
        "bigram_offsets": bigram_offsets,
        "bigram_tokens": (bigram & mask).astype(np.uint16),
        "bigram_values": bigram_book[(bigram >> np.uint64(token_bits)).astype(np.int64)],
        "trigram_keys": (prev2 * vocab + prev1.astype(np.int64)).astype(np.uint32),
        "trigram_offsets": trigram_offsets.astype(np.uint32),
        "trigram_tokens": (trigram & mask).astype(np.uint16),
        "trigram_values": trigram_book[(trigram >> np.uint64(token_bits)).astype(np.int64)],
    }


# --- scoring --------------------------------------------------------------------


class Scorer:
    """Full next-token log-prob vectors with Moth's backoff (trigram > bigram > floor)."""

    def __init__(self, lm: dict, bigram_values: np.ndarray, trigram_values: np.ndarray) -> None:
        self.floor = lm["unigram"].astype(np.float64) + 2.0 * math.log(lm["backoff"])
        self.bigram_offsets = lm["bigram_offsets"]
        self.bigram_tokens = lm["bigram_tokens"].astype(np.int64)
        self.bigram_values = np.asarray(bigram_values, dtype=np.float64)
        self.rows = {int(key): row for row, key in enumerate(lm["trigram_keys"])}
        self.trigram_offsets = lm["trigram_offsets"]
        self.trigram_tokens = lm["trigram_tokens"].astype(np.int64)
        self.trigram_values = np.asarray(trigram_values, dtype=np.float64)

    def vector(self, prev2: int, prev1: int) -> np.ndarray:
        scores = self.floor.copy()
        start, end = self.bigram_offsets[prev1], self.bigram_offsets[prev1 + 1]
        scores[self.bigram_tokens[start:end]] = self.bigram_values[start:end]
        row = self.rows.get(prev2 * VOCAB + prev1)
        if row is not None:
            start, end = self.trigram_offsets[row], self.trigram_offsets[row + 1]
            scores[self.trigram_tokens[start:end]] = self.trigram_values[start:end]
        return scores


def bpe_encoder(vocab_path: Path):
    """SentencePiece-style BPE from ``vocab.json`` (CTC id -> piece; LM id = CTC id - 1)."""
    by_ctc = json.loads(vocab_path.read_text(encoding="utf-8"))
    piece_id = {piece: int(ctc) - 1 for ctc, piece in by_ctc.items()}

    def encode_word(word: str) -> list[int]:
        symbols = list("\u2581" + word)
        while len(symbols) > 1:
            best, best_id = None, None
            for index in range(len(symbols) - 1):
                merged = piece_id.get(symbols[index] + symbols[index + 1])
                if merged is not None and (best_id is None or merged < best_id):
                    best, best_id = index, merged
            if best is None:
                break
            symbols[best : best + 2] = [symbols[best] + symbols[best + 1]]
        return [piece_id.get(symbol, 0) for symbol in symbols]

    def encode(text: str) -> list[int]:
        return [token for word in text.split() for token in encode_word(word)]

    return encode


def compare(reference: Scorer, candidate: Scorer, sentences: list[list[int]]) -> dict:
    token_deltas, sentence_deltas, sentence_scores = [], [], []
    top1 = top10 = positions = 0
    for tokens in sentences:
        words = [BOS, BOS, *tokens, EOS]
        total_ref = total_new = 0.0
        for position in range(2, len(words)):
            prev2, prev1, token = words[position - 2], words[position - 1], words[position]
            ref = reference.vector(prev2, prev1)
            new = candidate.vector(prev2, prev1)
            total_ref += ref[token]
            total_new += new[token]
            token_deltas.append(abs(new[token] - ref[token]))
            top1 += int(np.argmax(ref) == np.argmax(new))
            top10 += len(set(np.argsort(ref)[-10:]) & set(np.argsort(new)[-10:]))
            positions += 1
        sentence_deltas.append(abs(total_new - total_ref))
        sentence_scores.append(total_ref)
    token_deltas = np.array(token_deltas)
    sentence_deltas = np.array(sentence_deltas)
    return {
        "sentences": len(sentences),
        "tokens": positions,
        "token_abs_delta_mean": round(float(token_deltas.mean()), 5),
        "token_abs_delta_p99": round(float(np.percentile(token_deltas, 99)), 5),
        "token_abs_delta_max": round(float(token_deltas.max()), 5),
        "sentence_abs_delta_mean": round(float(sentence_deltas.mean()), 4),
        "sentence_abs_delta_p95": round(float(np.percentile(sentence_deltas, 95)), 4),
        "sentence_abs_delta_max": round(float(sentence_deltas.max()), 4),
        "sentence_logprob_mean": round(float(np.mean(sentence_scores)), 3),
        "top1_agreement": round(top1 / positions, 5),
        "top10_overlap": round(top10 / (10 * positions), 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lm", type=Path, required=True, help="MOTHLMB1 file")
    parser.add_argument("--vocab", type=Path, required=True, help="release vocab.json")
    parser.add_argument("--text", type=Path, nargs="+", required=True, help="jsonl with text")
    parser.add_argument("--bits", type=int, nargs="+", default=[10, 8, 7, 6])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from moth_text import normalize

    raw = args.lm.read_bytes()
    lm = read_mothlmb1(raw)
    scale = lm["log_scale"]
    counts = {
        "bigrams": len(lm["bigram_tokens"]),
        "trigrams": len(lm["trigram_tokens"]),
        "contexts": len(lm["trigram_keys"]),
    }
    reference = Scorer(lm, lm["bigram_logq"] * scale, lm["trigram_logq"] * scale)

    encode = bpe_encoder(args.vocab)
    texts: dict[str, None] = {}
    for path in args.text:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                texts[row.get("norm") or normalize(row["text"])] = None
    sentences = [encode(text) for text in texts]
    unk = sum(token == 0 for sentence in sentences for token in sentence)

    variants = []
    for bits in args.bits:
        packed, values = write_mothlmb2(lm, bits)
        decoded = read_mothlmb2(packed)
        for key in (
            "bigram_offsets",
            "bigram_tokens",
            "trigram_keys",
            "trigram_offsets",
            "trigram_tokens",
        ):
            if not np.array_equal(decoded[key].astype(np.int64), lm[key].astype(np.int64)):
                raise AssertionError(f"round trip failed for {key}")
        for key in ("bigram_values", "trigram_values"):
            if not np.array_equal(decoded[key], values[key]):
                raise AssertionError(f"round trip failed for {key}")
        if len(packed) != packed_size(**counts, bits=bits):
            raise AssertionError("size formula does not match the written file")
        errors = np.concatenate(
            [
                np.abs(values["bigram_values"] - lm["bigram_logq"] * scale),
                np.abs(values["trigram_values"] - lm["trigram_logq"] * scale),
            ]
        )
        candidate = Scorer(lm, values["bigram_values"], values["trigram_values"])
        variants.append(
            {
                "bits": bits,
                "bytes": len(packed),
                "ratio": round(len(packed) / len(raw), 4),
                "entry_abs_error_mean": round(float(errors.mean()), 5),
                "entry_abs_error_p99": round(float(np.percentile(errors, 99)), 5),
                "entry_abs_error_max": round(float(errors.max()), 5),
                "text": compare(reference, candidate, sentences),
                "predicted_bytes": {
                    name: {
                        "mothlmb1": known["bytes"],
                        "mothlmb2": packed_size(
                            known["bigrams"], known["trigrams"], known["contexts"], bits
                        ),
                    }
                    for name, known in KNOWN_LMS.items()
                },
            }
        )
        print(json.dumps(variants[-1]), flush=True)

    report = {
        "tool": "tools/probe_lm_pack.py",
        "lm": {
            "file": str(args.lm.name),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            **counts,
            "log_scale": scale,
            "mothlmb1_quantization_max_error": scale / 2,
            "distinct_bigram_values": len(np.unique(lm["bigram_logq"])),
            "distinct_trigram_values": len(np.unique(lm["trigram_logq"])),
        },
        "text": {
            "files": [path.name for path in args.text],
            "unique_sentences": len(sentences),
            "unk_tokens": unk,
            "tokenizer": "SentencePiece-style BPE rebuilt from vocab.json (proxy)",
        },
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
