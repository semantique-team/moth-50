/**
 * Browser reader for the compact BPE trigram LM (stt/lm.py save_browser).
 *
 * The layout is sparse CSR: unigram float32, bigram rows by previous id, trigram rows keyed by
 * `prev2 * V + prev1` with sorted uint32 keys. Log-probs are int16-quantized; missing entries
 * back off exactly like the Python score().
 * Empty trigram rows may be omitted without changing scores or the MOTHLMB1 layout.
 */

export const LM_BOS_ID = 1024;
export const LM_EOS_ID = 1025;
export const LM_VOCAB_SIZE = 1026;

function align4(value) {
  return Math.ceil(value / 4) * 4;
}

function findContext(keys, key) {
  let low = 0;
  let high = keys.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const value = keys[middle];
    if (value === key) return middle;
    if (value < key) low = middle + 1;
    else high = middle - 1;
  }
  return -1;
}

export class BPELM {
  constructor(buffer) {
    if (buffer.byteLength < 28) throw new Error("truncated LM header");
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
    if (magic !== "MOTHLMB1") throw new Error(`bad LM magic: ${JSON.stringify(magic)}`);
    const view = new DataView(buffer);
    let offset = 8;
    const read = (Type, length) => {
      const size = Type.BYTES_PER_ELEMENT * length;
      if (size > buffer.byteLength - offset) throw new Error("truncated LM buffer");
      const result = new Type(buffer, offset, length);
      offset += size;
      return result;
    };
    const validateOffsets = (offsets) => {
      if (offsets[0] !== 0) throw new Error("LM offsets must start at zero");
      for (let i = 1; i < offsets.length; i += 1) {
        if (offsets[i] < offsets[i - 1]) throw new Error("LM offsets must be monotonic");
      }
    };
    const validateTokens = (offsets, tokens) => {
      for (let row = 0; row < offsets.length - 1; row += 1) {
        let previous = -1;
        for (let i = offsets[row]; i < offsets[row + 1]; i += 1) {
          if (tokens[i] >= this.vocab || tokens[i] <= previous) {
            throw new Error("LM row tokens must be in range and strictly sorted");
          }
          previous = tokens[i];
        }
      }
    };
    const version = view.getUint32(offset, true);
    offset += 4;
    if (version !== 1) throw new Error(`unsupported LM version: ${version}`);
    this.vocab = view.getUint32(offset, true);
    offset += 4;
    this.backoff = view.getFloat32(offset, true);
    offset += 4;
    const unigramLength = view.getUint32(offset, true);
    offset += 4;
    this.logScale = view.getFloat32(offset, true);
    offset += 4;

    if (this.vocab !== LM_VOCAB_SIZE || unigramLength !== LM_VOCAB_SIZE) {
      throw new Error(`LM vocab mismatch: ${this.vocab}/${unigramLength}`);
    }
    if (!Number.isFinite(this.backoff) || this.backoff <= 0) {
      throw new Error("LM backoff must be finite and positive");
    }
    if (!Number.isFinite(this.logScale) || this.logScale <= 0) {
      throw new Error("LM log scale must be finite and positive");
    }
    this.unigram = read(Float32Array, unigramLength);
    if (!this.unigram.every(Number.isFinite)) throw new Error("LM unigram must be finite");

    const bigramContexts = read(Uint32Array, 1)[0];
    if (bigramContexts !== LM_VOCAB_SIZE) {
      throw new Error(`LM bigram contexts: ${bigramContexts}`);
    }
    this.bigramOffsets = read(Uint32Array, bigramContexts + 1);
    validateOffsets(this.bigramOffsets);
    const bigramEntries = this.bigramOffsets[bigramContexts];
    this.bigramTokens = read(Uint16Array, bigramEntries);
    this.bigramLogq = read(Int16Array, bigramEntries);
    validateTokens(this.bigramOffsets, this.bigramTokens);
    offset = align4(offset);

    const trigramContexts = read(Uint32Array, 1)[0];
    if (trigramContexts > this.vocab * this.vocab) throw new Error("too many LM trigram contexts");
    this.trigramKeys = read(Uint32Array, trigramContexts);
    for (let i = 0; i < trigramContexts; i += 1) {
      if (this.trigramKeys[i] >= this.vocab * this.vocab ||
          (i > 0 && this.trigramKeys[i] <= this.trigramKeys[i - 1])) {
        throw new Error("LM trigram keys must be in range and strictly sorted");
      }
    }
    this.trigramOffsets = read(Uint32Array, trigramContexts + 1);
    validateOffsets(this.trigramOffsets);
    const trigramEntries = this.trigramOffsets[trigramContexts];
    this.trigramTokens = read(Uint16Array, trigramEntries);
    this.trigramLogq = read(Int16Array, trigramEntries);
    validateTokens(this.trigramOffsets, this.trigramTokens);
    if (offset !== buffer.byteLength) throw new Error("trailing LM bytes");

    this.unigramFloor = new Float32Array(this.vocab);
    const floor = 2 * Math.log(this.backoff);
    for (let token = 0; token < this.vocab; token += 1) {
      this.unigramFloor[token] = this.unigram[token] + floor;
    }
  }

  score(prev2, prev1, token) {
    const context = prev2 * this.vocab + prev1;
    const index = findContext(this.trigramKeys, context);
    if (index >= 0) {
      const start = this.trigramOffsets[index];
      const end = this.trigramOffsets[index + 1];
      for (let entry = start; entry < end; entry += 1) {
        if (this.trigramTokens[entry] === token) {
          return this.trigramLogq[entry] * this.logScale;
        }
      }
    }
    const start = this.bigramOffsets[prev1];
    const end = this.bigramOffsets[prev1 + 1];
    for (let entry = start; entry < end; entry += 1) {
      if (this.bigramTokens[entry] === token) return this.bigramLogq[entry] * this.logScale;
    }
    return this.unigramFloor[token];
  }
}

export function loadLM(buffer) {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(8, buffer.byteLength)));
  if (magic === "MOTHLMB2") return new BPELM2(buffer);
  if (magic === "MOTHLMB1") return new BPELM(buffer);
  throw new Error(`bad LM magic: ${JSON.stringify(magic)}`);
}

function bitBytes(count, width) {
  return align4(Math.ceil((count * width) / 8));
}

function unpackBits(buffer, offset, count, width) {
  const byteLength = bitBytes(count, width);
  if (byteLength > buffer.byteLength - offset) throw new Error("truncated LM bitstream");
  const bytes = new Uint8Array(buffer, offset, byteLength);
  const values = new Uint32Array(count);
  let bit = 0;
  for (let index = 0; index < count; index += 1) {
    let value = 0;
    for (let position = 0; position < width; position += 1) {
      value |= ((bytes[bit >> 3] >> (bit & 7)) & 1) << position;
      bit += 1;
    }
    values[index] = value;
  }
  return values;
}

/**
 * Lossless packed LM (MOTHLMB2): 11-bit tokens plus a per-order int16 codebook.
 * Scores are computed exactly like the Python scorer: int(codebook[code]) * logScale.
 */
export class BPELM2 {
  constructor(buffer) {
    if (buffer.byteLength < 52) throw new Error("truncated LM header");
    const view = new DataView(buffer);
    let offset = 8;
    const u32 = () => {
      const value = view.getUint32(offset, true);
      offset += 4;
      return value;
    };
    const f32 = () => {
      const value = view.getFloat32(offset, true);
      offset += 4;
      return value;
    };
    const version = u32();
    if (version !== 2) throw new Error(`unsupported LM version: ${version}`);
    this.vocab = u32();
    this.backoff = f32();
    this.logScale = f32();
    const bits = u32();
    const tokenBits = u32();
    const offsetBits = u32();
    const bigramEntries = u32();
    const contexts = u32();
    const trigramEntries = u32();
    u32();
    this.bits = bits;
    if (this.vocab !== LM_VOCAB_SIZE) throw new Error(`LM vocab mismatch: ${this.vocab}`);
    if (!Number.isFinite(this.backoff) || this.backoff <= 0) {
      throw new Error("LM backoff must be finite and positive");
    }
    if (!Number.isFinite(this.logScale) || this.logScale <= 0) {
      throw new Error("LM log scale must be finite and positive");
    }
    if (!(bits >= 1 && bits <= 15) || tokenBits !== 11 || !(offsetBits >= 1 && offsetBits <= 32)) {
      throw new Error("unsupported MOTHLMB2 field widths");
    }
    if (contexts > this.vocab * this.vocab || trigramEntries >= 2 ** offsetBits) {
      throw new Error("invalid MOTHLMB2 counts");
    }
    const take = (Type, length) => {
      const size = Type.BYTES_PER_ELEMENT * length;
      if (size > buffer.byteLength - offset) throw new Error("truncated LM buffer");
      const result = new Type(buffer, offset, length);
      offset += size;
      return result;
    };
    const bigramBook = take(Int16Array, 2 ** bits);
    const trigramBook = take(Int16Array, 2 ** bits);
    this.unigram = take(Float32Array, this.vocab);
    if (!this.unigram.every(Number.isFinite)) throw new Error("LM unigram must be finite");
    this.bigramOffsets = take(Uint32Array, this.vocab + 1);
    const entryBits = tokenBits + bits;
    const bigram = unpackBits(buffer, offset, bigramEntries, entryBits);
    offset += bitBytes(bigramEntries, entryBits);
    this.prev2Offsets = take(Uint32Array, this.vocab + 1);
    this.prev1 = unpackBits(buffer, offset, contexts, tokenBits);
    offset += bitBytes(contexts, tokenBits);
    const trigramOffsets = unpackBits(buffer, offset, contexts + 1, offsetBits);
    offset += bitBytes(contexts + 1, offsetBits);
    const trigram = unpackBits(buffer, offset, trigramEntries, entryBits);
    offset += bitBytes(trigramEntries, entryBits);
    if (offset !== buffer.byteLength) throw new Error("trailing LM bytes");

    const tokenMask = (1 << tokenBits) - 1;
    const decode = (packed, book) => {
      const tokens = new Uint16Array(packed.length);
      const values = new Int16Array(packed.length);
      for (let i = 0; i < packed.length; i += 1) {
        tokens[i] = packed[i] & tokenMask;
        values[i] = book[packed[i] >>> tokenBits];
      }
      return { tokens, values };
    };
    const bi = decode(bigram, bigramBook);
    const tri = decode(trigram, trigramBook);
    this.bigramTokens = bi.tokens;
    this.bigramValues = bi.values;
    this.trigramTokens = tri.tokens;
    this.trigramValues = tri.values;

    const validateOffsets = (offsets, count) => {
      if (offsets[0] !== 0 || offsets[offsets.length - 1] !== count) {
        throw new Error("LM offsets must start at zero and end at the entry count");
      }
      for (let i = 1; i < offsets.length; i += 1) {
        if (offsets[i] < offsets[i - 1]) throw new Error("LM offsets must be monotonic");
      }
    };
    const validateRows = (offsets, tokens) => {
      for (let row = 0; row < offsets.length - 1; row += 1) {
        let previous = -1;
        for (let i = offsets[row]; i < offsets[row + 1]; i += 1) {
          if (tokens[i] >= this.vocab || tokens[i] <= previous) {
            throw new Error("LM row tokens must be in range and strictly sorted");
          }
          previous = tokens[i];
        }
      }
    };
    validateOffsets(this.bigramOffsets, bigramEntries);
    validateOffsets(this.prev2Offsets, contexts);
    validateOffsets(trigramOffsets, trigramEntries);
    validateRows(this.bigramOffsets, this.bigramTokens);
    validateRows(trigramOffsets, this.trigramTokens);
    this.trigramOffsets = trigramOffsets;

    this.unigramFloor = new Float32Array(this.vocab);
    const floor = 2 * Math.log(this.backoff);
    for (let token = 0; token < this.vocab; token += 1) {
      this.unigramFloor[token] = this.unigram[token] + floor;
    }
  }

  contextRow(prev2, prev1) {
    const start = this.prev2Offsets[prev2];
    const end = this.prev2Offsets[prev2 + 1];
    let low = start;
    let high = end - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      const value = this.prev1[middle];
      if (value === prev1) return middle;
      if (value < prev1) low = middle + 1;
      else high = middle - 1;
    }
    return -1;
  }

  rowScore(start, end, tokens, values, token) {
    let low = start;
    let high = end - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      if (tokens[middle] === token) return values[middle] * this.logScale;
      if (tokens[middle] < token) low = middle + 1;
      else high = middle - 1;
    }
    return null;
  }

  score(prev2, prev1, token) {
    if (prev2 >= 0 && prev1 >= 0 && prev2 < this.vocab && prev1 < this.vocab) {
      const row = this.contextRow(prev2, prev1);
      if (row >= 0) {
        const value = this.rowScore(
          this.trigramOffsets[row],
          this.trigramOffsets[row + 1],
          this.trigramTokens,
          this.trigramValues,
          token,
        );
        if (value !== null) return value;
      }
      const value = this.rowScore(
        this.bigramOffsets[prev1],
        this.bigramOffsets[prev1 + 1],
        this.bigramTokens,
        this.bigramValues,
        token,
      );
      if (value !== null) return value;
    }
    return this.unigramFloor[token];
  }
}
