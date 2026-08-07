#!/usr/bin/env python3
"""
xrsc_to_obj.py - Convert Midnight Club: Los Angeles (MCLA) .xrsc model-resource
files to Wavefront .obj, so the mesh can be opened in Blender / 3ds Max.

------------------------------------------------------------------------------
WHERE THIS COMES FROM
------------------------------------------------------------------------------
.xrsc is a RAGE "RSC5" resource container (used by MCLA on Xbox 360 / PS3).
There is no public spec for it - this script is a from-scratch Python port of
the binary layout implemented in the CodeX tool's C# source
(CodeX.Games.MCLA / Files/XrscFile.cs, RSC5/Rsc5Data.cs, RSC5/Rsc5Drawable.cs),
which is the only real documentation of this format. I read that source and
translated the relevant structures (pointer/array primitives, vertex format
decoding, MCLA's axis convention) into pure Python below.

Pipeline, end to end:
  1. Container header (big-endian): magic 'RSC5', type, flags, unknown, len.
     flags encodes the virtual+physical segment sizes.
  2. The body is Xbox360 LZX-compressed ("XCompress"/XMemDecompress). This is
     decompressed into a single buffer: [virtual segment][physical segment].
  3. That buffer is walked as a graph of pointer-linked structs, all
     big-endian, with two "address spaces" (virtual @0x50000000,
     physical @0x60000000) that map into offsets in the one buffer.
  4. Rsc5ModelResource -> Rsc5Drawable -> Lod -> Model[] -> Geometry[]
     -> VertexBuffer + IndexBuffer -> raw vertex bytes decoded per the
     FVF (flexible-vertex-format) description attached to the buffer.
  5. MCLA stores vectors in its own axis order and packs normals/UVs in
     game-specific ways; those per-element conversions are reproduced in
     decode_vertex_element() below (worked out directly from the byte-swap
     and axis-remap code in Rsc5DrawableGeometry.Read in the C# source).
  6. Geometry (positions/normals/UV0) + triangle indices are written out
     as .obj, one "o"/"g" per drawable model x geometry submesh.

------------------------------------------------------------------------------
IMPORTANT CAVEATS - PLEASE READ
------------------------------------------------------------------------------
* LZX DECOMPRESSION IS THE RISKIEST PART OF THIS SCRIPT. The C# tool calls
  into a native "XCompress" (Xbox 360 XMemDecompress) implementation that
  isn't included in the source you gave me - there's no reference
  implementation to check against, and I have no sample .xrsc file to test
  with. decompress_lzx() below is a from-memory, best-effort implementation
  of the standard LZX algorithm (verbatim/aligned/uncompressed blocks,
  main/length/aligned Huffman trees). It is the part most likely to need
  debugging against a real file.
      -> If the resulting mesh comes out as garbage/NaNs/huge vertex counts,
         LZX decoding is the first thing to suspect, not the geometry parser.
      -> If you (or the CodeX GUI tool, which has a working, tested
         XCompress binding) can dump the *already-decompressed* resource
         buffer, pass it with --raw and skip our LZX code entirely - see
         --raw below. This is the safest path.
* Only geometry is exported: vertex positions, normals, and the first UV
  channel (TexCoord0), plus triangle indices. No materials/textures - the
  ModelResource format (.xrsc) doesn't actually carry shader/texture data
  in CodeX's own model (that lives in separate .xtl/.xdr files), so there's
  nothing to export there anyway.
* No skinning/bones, no collision bounds, no LODs beyond the single Lod the
  format stores.
* Assumes triangle-list topology (three indices per triangle), which is what
  the "TrianglesCount = IndicesCount/3" relationship in the C# source implies
  for these meshes.

Usage:
    python3 xrsc_to_obj.py input.xrsc output.obj
    python3 xrsc_to_obj.py input.xrsc output.obj --raw          (input is
        already the decompressed [virtual][physical] buffer, e.g. dumped by
        another tool - skips the container header + LZX step entirely)
    python3 xrsc_to_obj.py input.xrsc output.obj --window-bits 17
        (tune the LZX window size guess if default doesn't decode cleanly)
    python3 xrsc_to_obj.py input.xrsc output.obj --dump-only
        (just print the parsed structure/counts, don't write an .obj -
        useful for sanity-checking decompression before trusting the mesh)
"""

import argparse
import math
import struct
import sys
from dataclasses import dataclass, field


# =============================================================================
# Container header + LZX decompression
# =============================================================================

RSC5_MAGIC = 0x05435352  # 88298322 decimal, matches CodeX's `rscVersion != 88298322` check


def get_virtual_size(flag: int) -> int:
    return (flag & 0x7FF) << (((flag >> 11) & 15) + 8)


def get_physical_size(flag: int) -> int:
    return ((flag >> 15) & 0x7FF) << (((flag >> 26) & 15) + 8)


# =============================================================================
# Automatic --virtual-size detection (for --raw buffers with no header)
# =============================================================================
# When a buffer is dumped without its RSC5 container header (--raw), we lose
# the `flag` field that encodes virtual_size/physical_size, and have to guess
# virtual_size some other way. Two independent signals are used together:
#
#  1. QUANTIZATION: both get_virtual_size() and get_physical_size() only ever
#     produce values of the form `mantissa << (shift + 8)`, mantissa in
#     [1, 0x7FF], shift in [0, 15] - i.e. a fairly sparse set of ~32K possible
#     sizes. If virtual_size + physical_size must equal the total buffer
#     length (true for a clean [virtual][physical] dump with no extra
#     padding), very few splits of that length will have BOTH halves land on
#     a quantized value - usually just one. This alone is often decisive.
#  2. SANITY OF DECODED DATA: virtual_size only matters for *physical*
#     pointers (raw vertex/index bytes) - see Rsc5Reader._offset(). A wrong
#     virtual_size still "successfully" decodes bytes (no crash) as long as
#     it doesn't walk off the end of the buffer, but the resulting vertex
#     positions come out as huge/NaN/Inf garbage (misaligned IEEE-754 bit
#     patterns). The right virtual_size decodes small, finite, physically
#     plausible position values. This is used to break ties between multiple
#     quantized candidates, or as a fallback search if no exact quantized
#     split exists (e.g. the dump has extra padding/trailing bytes).

def _quantized_sizes(max_size: int):
    """All values `mantissa << (shift + 8)` (mantissa 1..0x7FF, shift 0..15)
    that are <= max_size - the same encoding get_virtual_size()/
    get_physical_size() use, so any real virtual_size or physical_size must
    be a member of this set."""
    sizes = set()
    for mantissa in range(1, 0x800):
        base = mantissa << 8
        if base > max_size:
            break
        v = base
        shift = 0
        while v <= max_size and shift < 16:
            sizes.add(v)
            shift += 1
            v = mantissa << (shift + 8)
    return sizes


def _score_virtual_size(decompressed: bytes, vsize: int):
    """Try parsing with this virtual_size and score how plausible the result
    looks. Returns None if it doesn't even parse without an out-of-bounds/
    invalid-pointer error. Otherwise returns (nonfinite_count, max_abs) where
    lower is better on both - nonfinite_count first (NaN/Inf positions are an
    unambiguous sign of a wrong split), max_abs second (real meshes have
    positions in the tens/hundreds, not 1e30)."""
    try:
        models, _shaders = parse_xrsc(decompressed, vsize)
    except Exception:
        return None

    vals = []
    for geoms in models:
        for geo in geoms:
            for p in geo.positions:
                if p != (0.0, 0.0, 0.0):
                    vals.extend(p)

    if not vals:
        return None  # parsed, but nothing to judge - not useful as a candidate

    nonfinite = sum(1 for v in vals if not math.isfinite(v))
    finite_vals = [abs(v) for v in vals if math.isfinite(v)]
    max_abs = max(finite_vals) if finite_vals else float("inf")
    return (nonfinite, max_abs)


def _best_candidate(decompressed: bytes, candidates, verbose=False, label=""):
    """Score every candidate vsize in `candidates` (in the order given) and
    return (score_tuple, vsize) for the best one, stopping early on a clean
    (0 non-finite) hit. Returns None if nothing in `candidates` scored."""
    best = None
    for vsize in candidates:
        score = _score_virtual_size(decompressed, vsize)
        if score is not None and (best is None or score < best[0]):
            best = (score, vsize)
            if score[0] == 0:
                break
    if verbose and label:
        if best is None:
            print(f"guess_virtual_size: {label}: no usable candidate", file=sys.stderr)
        else:
            print(f"guess_virtual_size: {label}: best so far vsize={best[1]} "
                  f"score={best[0]}", file=sys.stderr)
    return best


def guess_virtual_size(decompressed: bytes, verbose=False):
    """Best-effort recovery of virtual_size for a --raw buffer with no
    container header. See module notes above for the two signals combined."""
    total = len(decompressed)
    quantized_all = sorted(v for v in _quantized_sizes(total) if 0 < v < total)
    quantized_set = set(quantized_all)

    if verbose:
        print(f"guess_virtual_size: buffer is {total} bytes, "
              f"{len(quantized_all)} quantized size(s) below it", file=sys.stderr)

    # Pass 1: exact virtual+physical split (strongest signal - both halves
    # land on a quantized value with nothing left over).
    exact_splits = [v for v in quantized_all if (total - v) in quantized_set]
    best = _best_candidate(decompressed, exact_splits, verbose, "exact quantized split")
    if best is not None and best[0][0] == 0:
        return best[1]

    # Pass 2: any single quantized virtual_size, without requiring the
    # remainder to also be quantized - covers dumps with trailing padding
    # or extra bytes after the physical segment.
    best2 = _best_candidate(decompressed, quantized_all, verbose, "quantized virtual_size alone")
    if best2 is not None and (best is None or best2[0] < best[0]):
        best = best2
    if best is not None and best[0][0] == 0:
        return best[1]

    # Pass 3: no quantized value worked cleanly - brute-force scan purely on
    # decoded-data sanity, byte by byte on a coarse grid, then refine.
    if verbose:
        print(f"guess_virtual_size: no clean quantized match "
              f"(best so far: {best}) - falling back to brute-force scan",
              file=sys.stderr)

    coarse_step = max(1, total // 4000)
    coarse_best = best
    for vsize in range(1, total, coarse_step):
        score = _score_virtual_size(decompressed, vsize)
        if score is not None and (coarse_best is None or score < coarse_best[0]):
            coarse_best = (score, vsize)

    if coarse_best is None:
        nearest = min(quantized_all, key=lambda v: abs(v - total // 2)) if quantized_all else None
        raise ValueError(
            "Could not determine --virtual-size automatically: no candidate "
            f"decoded any finite vertex position (buffer is {total} bytes"
            + (f", nearest quantized size is {nearest}" if nearest else "")
            + "). This usually means either the file doesn't hold mesh "
            "geometry to score against (e.g. an animation/skeleton-only "
            "resource), or the buffer has a header/padding this script "
            "doesn't strip. Pass --virtual-size explicitly - if you have a "
            "sibling file (same model, different LOD/variant) that already "
            "works, its virtual_size is a reasonable first guess."
        )

    # Refine around the coarse winner at single-byte granularity.
    center = coarse_best[1]
    lo = max(1, center - coarse_step)
    hi = min(total - 1, center + coarse_step)
    fine_best = coarse_best
    for vsize in range(lo, hi + 1):
        score = _score_virtual_size(decompressed, vsize)
        if score is not None and score < fine_best[0]:
            fine_best = (score, vsize)

    if verbose:
        print(f"guess_virtual_size: best match vsize={fine_best[1]} "
              f"score={fine_best[0]}", file=sys.stderr)

    return fine_best[1]


def read_rsc5_container(data: bytes, window_bits=None, verbose=False):
    """Parse the RSC5 header and return (decompressed_buffer, virtual_size, physical_size)."""
    if len(data) < 20:
        raise ValueError("File too small to be an RSC5 resource")

    magic, rsc_type, flag, unk, length = struct.unpack_from(">IiiII", data, 0)
    if magic != RSC5_MAGIC:
        print(f"warning: file doesn't start with the RSC5 magic "
              f"(got 0x{magic:08X}, expected 0x{RSC5_MAGIC:08X}); trying anyway...",
              file=sys.stderr)

    vsize = get_virtual_size(flag)
    psize = get_physical_size(flag)
    dlen = vsize + psize
    compressed = data[20:20 + length]

    if verbose:
        print(f"container: type={rsc_type} flag=0x{flag:08X} "
              f"virtual={vsize} physical={psize} compressed_len={length}",
              file=sys.stderr)

    decompressed = decompress_lzx(compressed, dlen, window_bits=window_bits)
    return decompressed, vsize, psize


# --- LZX decompression -------------------------------------------------------
# Best-effort pure-Python port of the classic LZX decompression algorithm
# (verbatim / aligned-offset / uncompressed blocks with canonical Huffman
# main/length/aligned trees), adapted for the single-shot Xbox360
# XMemDecompress-style stream this container uses (no Intel E8 translation).
# See the big caveat in the module docstring - validate carefully.

class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.bitbuf = 0
        self.bitcount = 0

    def _fill16(self):
        # LZX packs bits into 16-bit units stored **little-endian** in the
        # byte stream (low byte first), with bits then consumed MSB-first
        # from that 16-bit value - i.e. the high bit of the *second* byte
        # comes out of take() before anything in the first byte.
        if self.pos + 1 < len(self.data):
            word = (self.data[self.pos + 1] << 8) | self.data[self.pos]
        elif self.pos < len(self.data):
            word = self.data[self.pos]
        else:
            word = 0
        self.pos += 2
        self.bitbuf = (self.bitbuf << 16) | word
        self.bitcount += 16

    def peek(self, n: int) -> int:
        while self.bitcount < n:
            self._fill16()
        return (self.bitbuf >> (self.bitcount - n)) & ((1 << n) - 1)

    def take(self, n: int) -> int:
        v = self.peek(n)
        self.bitcount -= n
        self.bitbuf &= (1 << self.bitcount) - 1 if self.bitcount > 0 else 0
        return v

    def align16(self):
        # discard bits so the stream is on a 16-bit boundary (used before
        # "uncompressed" blocks)
        drop = self.bitcount % 16
        if drop:
            self.take(drop)


class HuffmanTable:
    """Canonical Huffman decode table built from an array of code lengths."""

    def __init__(self, lengths, max_len=16):
        self.lengths = lengths
        self.max_len = max_len
        self._build()

    def _build(self):
        n = len(self.lengths)
        # count of codes per length
        bl_count = [0] * (self.max_len + 1)
        for l in self.lengths:
            if l:
                bl_count[l] += 1
        code = 0
        next_code = [0] * (self.max_len + 1)
        for bits in range(1, self.max_len + 1):
            code = (code + bl_count[bits - 1]) << 1
            next_code[bits] = code
        # symbol -> (length, code)
        self.decode_map = {}
        for sym in range(n):
            l = self.lengths[sym]
            if l == 0:
                continue
            c = next_code[l]
            next_code[l] += 1
            self.decode_map[(l, c)] = sym
        # build a fast lookup: for each possible length, dict of code->sym
        self.by_len = {}
        for (l, c), sym in self.decode_map.items():
            self.by_len.setdefault(l, {})[c] = sym

    def decode(self, br: BitReader) -> int:
        code = 0
        for l in range(1, self.max_len + 1):
            code = (code << 1) | br.take(1)
            d = self.by_len.get(l)
            if d is not None and code in d:
                return d[code]
        raise ValueError("Invalid Huffman code encountered (LZX stream corrupt or "
                          "decoder parameters wrong - try --window-bits)")


PRETREE_ELEMENTS = 20
ALIGNED_ELEMENTS = 8
NUM_CHARS = 256
NUM_PRIMARY_LENGTHS = 7
NUM_SECONDARY_LENGTHS = 249
MAIN_TREE_ELEMENTS = NUM_CHARS + 50 * 8  # sized for the largest supported window

EXTRA_BITS = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
              10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
              17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
POSITION_BASE = [0] * 51
for _i in range(1, 51):
    POSITION_BASE[_i] = POSITION_BASE[_i - 1] + (1 << EXTRA_BITS[_i - 1])


def _read_pretree_lengths(br: BitReader, n: int):
    """Read n code lengths (0-16) using the 20-symbol pretree that precedes
    each Huffman-coded length table in an LZX block."""
    pre_lengths = [br.take(4) for _ in range(PRETREE_ELEMENTS)]
    pre_table = HuffmanTable(pre_lengths, max_len=16)

    lens = [0] * n
    i = 0
    while i < n:
        sym = pre_table.decode(br)
        if sym <= 16:
            lens[i] = sym
            i += 1
        elif sym == 17:  # zero run, small
            run = br.take(4) + 4
            for _ in range(run):
                if i < n:
                    lens[i] = 0
                    i += 1
        elif sym == 18:  # zero run, large
            run = br.take(5) + 20
            for _ in range(run):
                if i < n:
                    lens[i] = 0
                    i += 1
        elif sym == 19:  # repeat previous nonzero-delta value 4-5 times
            run = br.take(1) + 4
            sym2 = pre_table.decode(br)
            if sym2 > 16:
                sym2 = 0
            for _ in range(run):
                if i < n:
                    lens[i] = sym2
                    i += 1
    return lens


def _read_lengths_delta(br: BitReader, prev_lengths, n):
    """LZX length tables are coded as (17-mod-17 delta) from the previous
    block's lengths using the pretree scheme above."""
    raw = _read_pretree_lengths(br, n)
    out = [0] * n
    for i in range(n):
        out[i] = (prev_lengths[i] - raw[i]) % 17
    return out


def _decode_lzx_chunk(compressed: bytes, out_len: int, window: bytearray, win_pos: int,
                       window_bits: int) -> bytes:
    """Decode a single self-contained LZX bitstream (its own E8 flag, its own
    fresh Huffman tables and R0/R1/R2) that produces exactly `out_len` bytes.
    `window` is the shared sliding-window history buffer (mutated in place so
    later chunks can still back-reference data from earlier chunks)."""
    window_size = len(window)
    out = bytearray()
    br = BitReader(compressed)

    main_lengths = [0] * MAIN_TREE_ELEMENTS
    length_lengths = [0] * NUM_SECONDARY_LENGTHS
    r0 = r1 = r2 = 1

    # Every independent LZX bitstream opens with a 1-bit "E8 call
    # translation" flag, followed by a 32-bit translation size if set.
    if br.take(1):
        br.take(32)

    while len(out) < out_len:
        block_type = br.take(3)
        block_size = br.take(24)
        if block_size <= 0:
            break
        block_end = len(out) + block_size

        if block_type == 1 or block_type == 2:  # verbatim / aligned offset
            aligned_table = None
            if block_type == 2:
                aligned_lengths = [br.take(3) for _ in range(ALIGNED_ELEMENTS)]
                aligned_table = HuffmanTable(aligned_lengths, max_len=7)

            main_lengths[:NUM_CHARS] = _read_lengths_delta(br, main_lengths[:NUM_CHARS], NUM_CHARS)
            main_lengths[NUM_CHARS:] = _read_lengths_delta(
                br, main_lengths[NUM_CHARS:], MAIN_TREE_ELEMENTS - NUM_CHARS)
            main_table = HuffmanTable(main_lengths, max_len=16)

            length_lengths[:] = _read_lengths_delta(br, length_lengths, NUM_SECONDARY_LENGTHS)
            length_table = HuffmanTable(length_lengths, max_len=16)

            while len(out) < block_end:
                sym = main_table.decode(br)
                if sym < NUM_CHARS:
                    out.append(sym)
                    window[win_pos] = sym
                    win_pos = (win_pos + 1) % window_size
                    continue

                sym -= NUM_CHARS
                length_header = sym & 7
                position_slot = sym >> 3

                if length_header == 7:
                    length_sym = length_table.decode(br)
                    match_len = 7 + 2 + length_sym
                else:
                    match_len = length_header + 2

                if position_slot == 0:
                    match_offset = r0
                elif position_slot == 1:
                    match_offset = r1
                    r1, r0 = r0, r1
                elif position_slot == 2:
                    match_offset = r2
                    r2, r0 = r0, r2
                else:
                    extra = EXTRA_BITS[position_slot]
                    base = POSITION_BASE[position_slot]
                    if block_type == 2 and extra >= 3:
                        verbatim_bits = extra - 3
                        v = br.take(verbatim_bits) if verbatim_bits else 0
                        aligned = aligned_table.decode(br)
                        match_offset = base + (v << 3) + aligned - 2
                    else:
                        v = br.take(extra) if extra else 0
                        match_offset = base + v - 2
                    r2, r1, r0 = r1, r0, match_offset

                for _ in range(match_len):
                    b = window[(win_pos - match_offset) % window_size]
                    out.append(b)
                    window[win_pos] = b
                    win_pos = (win_pos + 1) % window_size

        elif block_type == 3:  # uncompressed
            br.align16()
            raw = bytearray()
            for _ in range(12):
                raw.append(br.take(8))
            r0, r1, r2 = struct.unpack_from("<III", bytes(raw))
            for _ in range(block_size):
                b = br.take(8)
                out.append(b)
                window[win_pos] = b
                win_pos = (win_pos + 1) % window_size
            br.bitbuf = 0
            br.bitcount = 0
        else:
            raise ValueError(f"Unknown LZX block type {block_type} - stream is likely "
                              f"misaligned (try --window-bits)")

    return bytes(out[:out_len])


def decompress_lzx(compressed: bytes, out_len: int, window_bits: int = None) -> bytes:
    """RAGE-console (GTA IV/V, RDR, MC:LA, MP3 - Xbox360) flavor of
    XMemCompress/LZX: NOT one continuous bitstream. It's a sequence of
    small chunk headers, each followed by an independently-coded LZX
    bitstream (fresh Huffman tables, fresh R0-R2, its own E8 flag) that
    decompresses to at most 0x8000 (32768) bytes. The sliding-window match
    history *is* shared/carried across chunks even though the entropy
    coding resets - see Compression.LZX.pas (ReadBlockSize) in the
    RAGE-Console-Texture-Editor project for the reference framing this is
    ported from.

    Chunk header:
      b0 = next byte
      if b0 == 0xFF: read b1,b2,b3,b4
          uncompressed_size = (b1<<8)|b2      (usually 0x8000 except the
                                                last, shorter chunk)
          compressed_size   = (b3<<8)|b4
      else: read b1
          uncompressed_size = 0x8000
          compressed_size   = (b0<<8)|b1
    """
    if out_len <= 0:
        return b""

    if window_bits is None:
        window_bits = 17  # fixed by the reference implementation (LZXinit(17))

    window = bytearray(1 << window_bits)
    win_pos = 0
    out = bytearray()
    pos = 0

    while len(out) < out_len:
        if pos >= len(compressed):
            raise ValueError("LZX stream ran out of chunk headers before "
                              "reaching the expected output size - wrong "
                              "--window-bits, or this isn't RAGE-flavored LZX")
        b0 = compressed[pos]; pos += 1
        if b0 == 0xFF:
            b1, b2, b3, b4 = compressed[pos:pos + 4]
            pos += 4
            uncompressed_size = (b1 << 8) | b2
            compressed_size = (b3 << 8) | b4
        else:
            b1 = compressed[pos]; pos += 1
            uncompressed_size = 0x8000
            compressed_size = (b0 << 8) | b1

        uncompressed_size = min(uncompressed_size, out_len - len(out))
        chunk = compressed[pos:pos + compressed_size]
        pos += compressed_size

        chunk_out = _decode_lzx_chunk(chunk, uncompressed_size, window, win_pos, window_bits)
        out += chunk_out
        win_pos = (win_pos + len(chunk_out)) % len(window)

    return bytes(out[:out_len])


# =============================================================================
# RSC5 pointer/struct reader
# =============================================================================

VIRTUAL_BASE = 0x50000000
PHYSICAL_BASE = 0x60000000


class Rsc5Reader:
    def __init__(self, data: bytes, virtual_size: int):
        self.data = data
        self.virtual_size = virtual_size
        self.pos = VIRTUAL_BASE

    def _offset(self, addr=None):
        p = self.pos if addr is None else addr
        if (p & VIRTUAL_BASE) == VIRTUAL_BASE:
            return p & 0x0FFFFFFF
        if (p & PHYSICAL_BASE) == PHYSICAL_BASE:
            return (p & 0x1FFFFFFF) + self.virtual_size
        raise ValueError(f"Invalid pointer 0x{p:08X} - file is likely corrupt "
                         f"or was decompressed incorrectly")

    def u8(self):
        v = self.data[self._offset()]
        self.pos += 1
        return v

    def u16(self):
        v = struct.unpack_from(">H", self.data, self._offset())[0]
        self.pos += 2
        return v

    def i16(self):
        v = struct.unpack_from(">h", self.data, self._offset())[0]
        self.pos += 2
        return v

    def u32(self):
        v = struct.unpack_from(">I", self.data, self._offset())[0]
        self.pos += 4
        return v

    def i32(self):
        v = struct.unpack_from(">i", self.data, self._offset())[0]
        self.pos += 4
        return v

    def u64(self):
        v = struct.unpack_from(">Q", self.data, self._offset())[0]
        self.pos += 8
        return v

    def f32(self):
        v = struct.unpack_from(">f", self.data, self._offset())[0]
        self.pos += 4
        return v

    def raw(self, n):
        off = self._offset()
        b = self.data[off:off + n]
        self.pos += n
        return b

    def at(self, addr, func):
        """Temporarily jump to `addr`, run func(), then restore position -
        mirrors CodeX's ReadBlock(position, ...) pattern for pointers."""
        if not addr or addr == 0xCDCDCDCD:
            return None
        save = self.pos
        self.pos = addr
        try:
            result = func()
        finally:
            self.pos = save
        return result

    def cstr(self):
        """Read a null-terminated string at the current position (mirrors
        Rsc5DataReader.ReadString - used for Rsc5Str targets)."""
        b = bytearray()
        while True:
            c = self.u8()
            if c == 0:
                break
            b.append(c)
        return b.decode("latin1", errors="replace")

    def str_ptr(self):
        """Read a Rsc5Str: a u32 pointer field, then the C-string it points
        to (or None if the pointer is null)."""
        p = self.u32()
        return self.at(p, self.cstr) if p else None


# =============================================================================
# Vertex format decoding
# =============================================================================

# Rsc5VertexComponentType values (byte sizes, from Rsc5VertexComponentTypes.GetSizeInBytes)
COMPONENT_SIZE = {
    0: 2,   # Nothing / Half
    1: 4,   # Half2
    2: 6,   # Float (stored as Half3-sized slot in this table, but treated as scalar float, see note)
    3: 8,   # Half4
    4: 4,   # FloatUnk
    5: 8,   # Float2
    6: 12,  # Float3
    7: 16,  # Float4
    8: 4,   # UByte4
    9: 4,   # Colour
    10: 4,  # Dec3N (packed normal)
    11: 2,  # Unk1
    12: 4,  # Unk2
    13: 2,  # Unk3
    14: 4,  # UShort2N
    15: 8,  # Unk5
}

SEM_POSITION = 0
SEM_BLENDWEIGHTS = 1
SEM_BLENDINDICES = 2
SEM_NORMAL = 3
SEM_COLOUR0 = 4
SEM_COLOUR1 = 5
SEM_TEXCOORD0 = 6  # TexCoord0..7 are semantics 6..13


def dec3n_to_vec4(u: int):
    ux = (u & 0x3FF) << 22
    uy = ((u >> 10) & 0x3FF) << 22
    uz = ((u >> 20) & 0x3FF) << 22
    uw = u
    # arithmetic shift right by 22 / 30 on signed 32-bit values
    ux = _to_signed32(ux) >> 22
    uy = _to_signed32(uy) >> 22
    uz = _to_signed32(uz) >> 22
    uw = _to_signed32(uw) >> 30
    scale = 0.001956947162
    return (ux * scale, uy * scale, uz * scale, float(uw))


def _to_signed32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def half_to_float(h: int) -> float:
    """IEEE754 binary16 -> python float."""
    return struct.unpack("<e", struct.pack("<H", h))[0]


def parse_vertex_declaration(r: Rsc5Reader):
    fvf = r.u32()
    fvf_size = r.u8()
    flags = r.u8()
    dynamic_order = r.u8()
    channel_count = r.u8()
    types = r.u64()

    elements = []  # list of (semantic, component_type, byte_offset)
    offset = 0
    for i in range(16):
        if (fvf >> i) & 1:
            comp_type = (types >> (i * 4)) & 0xF
            elements.append((i, comp_type, offset))
            offset += COMPONENT_SIZE.get(comp_type, 0)

    return {"fvf": fvf, "fvf_size": fvf_size, "elements": elements}


def decode_vertex_element(data: bytes, base_off: int, comp_type: int):
    """Decode one FVF element from raw big-endian vertex bytes, applying the
    same MCLA-specific axis-remap CodeX applies in Rsc5DrawableGeometry.Read.
    Returns a tuple of floats, meaning depends on comp_type/semantic."""
    if comp_type == 6:  # Float3 - CodeX only byte-swaps this case, no axis
        a, b, c = struct.unpack_from(">fff", data, base_off)  # permute (confirmed
        return (a, b, c)                                       # against Rsc5DrawableGeometry.Read)
    if comp_type == 7:  # Float4 - same, byte-swap only, no axis permute
        a, b, c, d = struct.unpack_from(">ffff", data, base_off)
        return (a, b, c, d)
    if comp_type == 10:  # Dec3N packed normal - this IS axis-permuted to
        (u,) = struct.unpack_from(">I", data, base_off)          # (Z,X,Y,W) in CodeX
        fx, fy, fz, fw = dec3n_to_vec4(u)
        return (fz, fx, fy, fw)
    if comp_type == 1:  # Half2 (UV). CodeX's Rsc5DrawableGeometry.Read does
        # `new Half2(h.Y, h.X)` here, but that line runs on a buffer that
        # was already whole-array byte-swapped 4 bytes at a time
        # (Rpf3Crypto.Swap(byte[])) *before* this per-element fixup loop.
        # For a Half2 (two uint16s packed into exactly one 4-byte word),
        # reversing all 4 bytes and re-reading as two little-endian
        # uint16s has the side effect of implicitly swapping which half
        # occupies which slot *in addition to* fixing endianness - so by
        # the time `new Half2(h.Y, h.X)` runs, h.X/h.Y are already
        # transposed, and this explicit swap cancels that back out,
        # netting NO logical U/V swap overall. This script reads the
        # untouched big-endian file bytes directly (equivalent to just the
        # first implicit step), so it must NOT apply a second swap here -
        # doing so (as an earlier version of this script did) double-swaps
        # and scrambles the UVs.
        h0, h1 = struct.unpack_from(">HH", data, base_off)
        return (half_to_float(h0), half_to_float(h1))
    if comp_type == 5:  # Float2 (UV). Each float is its own 4-byte word, so
        # the whole-array Swap() has no field-transposing side effect here
        # (unlike Half2 above) - it's a plain per-float endian fix. Read
        # directly in stored order, no swap.
        a, b = struct.unpack_from(">ff", data, base_off)
        return (a, b)
    if comp_type == 2:  # Float (scalar)
        (a,) = struct.unpack_from(">f", data, base_off)
        return (a,)
    if comp_type == 14:  # UShort2N
        a, b = struct.unpack_from(">HH", data, base_off)
        return (a / 65535.0, b / 65535.0)
    # UByte4 / Colour / others: not needed for OBJ export (skin indices,
    # vertex colour) - caller should skip these semantics.
    return None


# =============================================================================
# RSC5 model-resource structure walk
# =============================================================================

@dataclass
class Geometry:
    positions: list = field(default_factory=list)
    normals: list = field(default_factory=list)
    uvs: list = field(default_factory=list)
    indices: list = field(default_factory=list)
    shader_id: int = 0


def parse_vertex_buffer(r: Rsc5Reader):
    vft = r.u32()
    vcount = r.u16()
    locked = r.u8()
    flags = r.u8()
    locked_ptr = r.u32()
    stride = r.u32()
    layout_ptr = r.u32()
    lock_thread = r.u32()
    vdata_ptr = r.u32()
    d3dvb_ptr = r.u32()

    layout = r.at(layout_ptr, lambda: parse_vertex_declaration(r))
    if layout is None:
        return None

    size = vcount * layout["fvf_size"]
    vdata = r.at(locked_ptr, lambda: r.raw(size)) if locked_ptr else None
    if vdata is None:
        vdata = r.at(vdata_ptr, lambda: r.raw(size)) if vdata_ptr else None

    return {"count": vcount, "stride": stride or layout["fvf_size"], "layout": layout, "data": vdata}


def parse_index_buffer(r: Rsc5Reader):
    vft = r.u32()
    idx_count = r.u32()
    idx_ptr = r.u32()
    d3d_ptr = r.u32()
    indices = None
    if idx_ptr:
        indices = r.at(idx_ptr, lambda: [r.u16() for _ in range(idx_count)])
    return indices or []


def parse_geometry(r: Rsc5Reader) -> Geometry:
    vft = r.u32()
    _u4 = r.u32()
    _u8 = r.u32()
    vb_ptrs = [r.u32() for _ in range(4)]
    ib_ptrs = [r.u32() for _ in range(4)]
    indices_count = r.u32()
    tri_count = r.u32()
    vcount = r.u16()
    prim_type = r.u16()
    boneids_ptr = r.u32()
    vstride = r.u16()
    boneids_count = r.u16()
    vdataref_ptr = r.u32()
    offset_buffer = r.u32()
    index_offset = r.u32()
    _u3c = r.u32()

    vb = r.at(vb_ptrs[0], lambda: parse_vertex_buffer(r)) if vb_ptrs[0] else None
    idx = r.at(ib_ptrs[0], lambda: parse_index_buffer(r)) if ib_ptrs[0] else []

    geo = Geometry()
    if idx:
        geo.indices = idx

    if vb and vb["data"]:
        layout = vb["layout"]
        stride = vb["stride"]
        data = vb["data"]
        n = vb["count"] or vcount
        elements = layout["elements"]

        for vi in range(n):
            base = vi * stride
            pos = normal = uv = None
            for sem, comp_type, elem_off in elements:
                val = decode_vertex_element(data, base + elem_off, comp_type)
                if val is None:
                    continue
                if sem == SEM_POSITION:
                    pos = val[:3]
                elif sem == SEM_NORMAL:
                    normal = val[:3]
                elif sem == SEM_TEXCOORD0:
                    uv = val[:2]
            geo.positions.append(pos or (0.0, 0.0, 0.0))
            geo.normals.append(normal)
            geo.uvs.append(uv)

    return geo


def parse_model(r: Rsc5Reader):
    vft = r.u32()
    geo_pos = r.u32()
    geo_count = r.u16()
    geo_cap = r.u16()
    bounds_ptr = r.u32()
    shadermap_ptr = r.u32()
    matrix_count = r.u8()
    flags = r.u8()
    type_ = r.u8()
    matrix_index = r.u8()
    stride = r.u8()
    skin_flag = r.u8()
    geo_count2 = r.u16()

    shader_ids = []
    if shadermap_ptr and geo_count:
        shader_ids = r.at(shadermap_ptr, lambda: [r.u16() for _ in range(geo_count)]) or []

    geoms = []
    if geo_pos and geo_count:
        ptrs = r.at(geo_pos, lambda: [r.u32() for _ in range(geo_count)])
        for i, p in enumerate(ptrs):
            g = r.at(p, lambda: parse_geometry(r))
            if g is not None:
                g.shader_id = shader_ids[i] if i < len(shader_ids) else 0
                geoms.append(g)
    return geoms


def parse_lod(r: Rsc5Reader):
    """Rsc5DrawableLod: an inline PtrArr<Rsc5DrawableModel> starting right at
    the current position (no extra indirection layer)."""
    pos = r.u32()
    count = r.u16()
    cap = r.u16()

    models = []
    if pos and count:
        ptrs = r.at(pos, lambda: [r.u32() for _ in range(count)])
        for p in ptrs:
            m = r.at(p, lambda: parse_model(r))
            if m is not None:
                models.append(m)
    return models


def parse_shader(r: Rsc5Reader):
    """Rsc5Shader (CodeX RSC5/Rsc5Drawable.cs): reads the shader/effect name
    plus its parameter list. Only the texture-type (Type==0) parameters are
    resolved here (each one points to an embedded Rsc5Texture *stub* -
    usually just a name, no pixel data - the pixel data lives in the
    matching-named texture inside the separate texture-dictionary file)."""
    _vft = r.u32()
    _blockmap_addr = r.u32()
    _version = r.u8(); _drawbucket = r.u8(); _usagecount = r.u8(); _unk1 = r.u8()
    _unk2 = r.u16()
    _shader_index = r.u16()
    paramsdata_ptr = r.u32()
    _unk3 = r.u32()
    params_count = r.u16()
    _effect_size = r.u16()
    paramstypes_ptr = r.u32()
    _hash = r.u32()
    _paramsnames_ptr = r.u32()
    _unk4 = r.u32(); _unk5 = r.u32()
    shader_name = r.str_ptr()
    _unk6 = r.u32()
    _shaderfilename = r.str_ptr()
    _unk7 = r.u32()

    pc = params_count
    params_data = (r.at(paramsdata_ptr, lambda: [r.u32() for _ in range(pc)])
                   if paramsdata_ptr and pc else []) or []
    params_types = (r.at(paramstypes_ptr, lambda: [r.u8() for _ in range(pc)])
                    if paramstypes_ptr and pc else []) or []

    textures = []       # texture names referenced by this shader, in param order
    textures_full = []  # the full parsed texture dicts (name/width/height/format/data)
    for i in range(pc):
        ptr = params_data[i] if i < len(params_data) else 0
        ptype = params_types[i] if i < len(params_types) else 1
        if ptype == 0 and ptr:  # 0 == texture parameter
            tex = r.at(ptr, lambda: parse_texture(r))
            if tex and tex.get("name"):
                textures.append(tex["name"])
                textures_full.append(tex)

    return {"name": shader_name, "textures": textures, "textures_full": textures_full}


def parse_shader_group(r: Rsc5Reader):
    """Rsc5ShaderGroup: Rsc5BlockBaseMap header (VFT + BlockMap ptr) followed
    by a PtrArr<Rsc5Shader>."""
    _vft = r.u32()
    _blockmap_ptr = r.u32()
    shaders_ptr = r.u32()
    count = r.u16()
    _cap = r.u16()

    shaders = []
    if shaders_ptr and count:
        ptrs = r.at(shaders_ptr, lambda: [r.u32() for _ in range(count)]) or []
        for p in ptrs:
            s = r.at(p, lambda: parse_shader(r)) if p else None
            if s is not None:
                shaders.append(s)
    return shaders


def parse_drawable_base(r: Rsc5Reader):
    """The Drawable ptr in this build's Rsc5ModelResource leads to a
    DrawableBase-shaped struct (empirically verified byte-for-byte against
    real file data - sane bounding box, sane bucket masks, sane LOD array):
      VFT, BlockMap ptr, ShaderGroup ptr, SkeletonRef ptr,
      BoundingCenter/Min/Max (3x Vector4), then four Rsc5Ptr<Rsc5DrawableLod>
      slots (High/Med/Low/Vlow), 4 LOD-distance floats, 4 draw-bucket masks,
      bounding-sphere radius. We take the first non-null LOD slot."""
    _vft = r.u32()
    _blockmap_ptr = r.u32()
    shadergroup_ptr = r.u32()
    _skeleton_ptr = r.u32()
    for _ in range(3):  # BoundingCenter, BoundingBoxMin, BoundingBoxMax
        r.f32(); r.f32(); r.f32(); r.f32()
    lod_ptrs = [r.u32() for _ in range(4)]  # High, Med, Low, Vlow

    shaders = r.at(shadergroup_ptr, lambda: parse_shader_group(r)) if shadergroup_ptr else []

    models = []
    for p in lod_ptrs:
        if p:
            models = r.at(p, lambda: parse_lod(r))
            break
    return models, shaders


def parse_xrsc(decompressed: bytes, virtual_size: int):
    r = Rsc5Reader(decompressed, virtual_size)
    # Rsc5ModelResource header. The decompiled CodeX source shows only
    # VFT + BlockMap ptr before Drawable ptr, but real file data shows the
    # Drawable ptr sits 8 bytes further out (offset 16, not 8) - confirmed
    # by walking pointers forward from real vertex/geometry data back to
    # the root and checking the target is a plausible DrawableBase struct.
    _vft = r.u32()
    _blockmap_ptr = r.u32()
    _unknown1 = r.u32()
    _unknown2 = r.u32()
    drawable_ptr = r.u32()

    models, shaders = (r.at(drawable_ptr, lambda: parse_drawable_base(r))
                        if drawable_ptr else ([], []))
    return models, shaders  # list of models (each a list of Geometry), list of shaders


# =============================================================================
# Texture / material extraction
# =============================================================================
# Ported from CodeX.Games.MCLA.RSC5/Rsc5Texture.cs (Rsc5Texture/Rsc5TextureBase
# .Read) and CodeX.Games.MCLA.RPF3/Rpf3Crypto.cs (GetBaseAdressFromDirect3D,
# GetVirtualSize, UnswizzleXbox360Data, XGAddress2DTiledX/Y). Two files are
# involved:
#   - the *model* .xrsc's Drawable.ShaderGroup.Shaders[].Params[] hold
#     "texture stub" Rsc5Texture blocks - just a name, no pixel data
#     (D3DBaseTexture ptr is null for these).
#   - the *texture-dictionary* .xrsc (the "_h" file for this game) holds the
#     full Rsc5Texture blocks with real pixel data, keyed by the same names.
# We match shader texture-stub names to texture-dictionary entries by name.

RSC5_BASE_ADDRESSES = {0x50: 0x50000000, 0x51: 0x51000000, 0x52: 0x52000000}

TEXFMT_L8 = "L8"
TEXFMT_BC1 = "BC1"
TEXFMT_BC2 = "BC2"
TEXFMT_BC3 = "BC3"
TEXFMT_A8R8G8B8 = "A8R8G8B8"

_D3DFMT_TO_TEXFMT = {
    2: TEXFMT_L8,          # D3DFMT_L8
    82: TEXFMT_BC1,        # D3DFMT_DXT1
    83: TEXFMT_BC2,        # D3DFMT_DXT3
    84: TEXFMT_BC3,        # D3DFMT_DXT5
    134: TEXFMT_A8R8G8B8,  # D3DFMT_A8R8G8B8
}


def convert_to_engine_format(d3dfmt_byte: int) -> str:
    return _D3DFMT_TO_TEXFMT.get(d3dfmt_byte, TEXFMT_BC1)  # default matches CodeX


def get_virtual_texture_size(size: int) -> int:
    """Rpf3Crypto.GetVirtualSize: pads sub-128 sizes up to 128."""
    if (size % 128 != 0) and size < 128:
        return 128
    return size


def calc_texture_data_size(fmt: str, width: int, height: int) -> int:
    if fmt == TEXFMT_BC1:
        return width * height // 2
    if fmt in (TEXFMT_BC2, TEXFMT_BC3, TEXFMT_A8R8G8B8, TEXFMT_L8):
        return width * height
    raise NotImplementedError(f"unsupported texture format for size calc: {fmt}")


def get_base_address_from_d3d(d3d_value: int, virtual_size: int) -> int:
    """Rpf3Crypto.GetBaseAdressFromDirect3D: reconstructs the virtual/physical
    address the raw (swizzled) texel data lives at from the packed D3D
    common-header value read at D3DBaseTexture+0x20."""
    is_physical = (d3d_value & 0x60000000) == 0x60000000
    top_byte = (d3d_value >> 24) & 0xFF
    size_field = d3d_value & 0xFF
    mapped_base = RSC5_BASE_ADDRESSES.get(top_byte, 0)
    if mapped_base == 0:
        mapped_base = 0x50000000
    return (d3d_value & 0xFFFFFF) - size_field + (virtual_size if is_physical else 0) + mapped_base


def _xg_tiled_common(offset: int, width: int, texel_pitch: int):
    aligned_width = (width + 31) & ~31
    log_bpp = (texel_pitch >> 2) + ((texel_pitch >> 1) >> (texel_pitch >> 2))
    offset_b = offset << log_bpp
    offset_t = ((offset_b & ~4095) >> 3) + ((offset_b & 1792) >> 2) + (offset_b & 63)
    offset_m = offset_t >> (7 + log_bpp)
    return aligned_width, log_bpp, offset_b, offset_t, offset_m


def xg_address_2d_tiled_x(offset: int, width: int, texel_pitch: int) -> int:
    aligned_width, log_bpp, offset_b, offset_t, offset_m = _xg_tiled_common(offset, width, texel_pitch)
    macro_x = (offset_m % (aligned_width >> 5)) << 2
    # Verified against a real extracted texture (see below): this addition
    # is genuinely summed *before* the final mask - masking (offset_b >> 6)
    # down to one bit first (as a "cleaner-looking" rewrite would suggest)
    # actually breaks it and produces a checkerboarded texture. Leave as-is.
    tile = (((offset_t >> (5 + log_bpp)) & 2) + (offset_b >> 6)) & 3
    macro = (macro_x + tile) << 3
    micro = ((((offset_t >> 1) & ~15) + (offset_t & 15)) & ((texel_pitch << 3) - 1)) >> log_bpp
    return macro + micro


def xg_address_2d_tiled_y(offset: int, width: int, texel_pitch: int) -> int:
    aligned_width, log_bpp, offset_b, offset_t, offset_m = _xg_tiled_common(offset, width, texel_pitch)
    macro_y = (offset_m // (aligned_width >> 5)) << 2
    tile = ((offset_t >> (6 + log_bpp)) & 1) + ((offset_b & 2048) >> 10)
    macro = (macro_y + tile) << 3
    micro = (((offset_t & ((texel_pitch << 6) - 1) & ~31) + ((offset_t & 15) << 1)) >> (3 + log_bpp)) & ~1
    return macro + micro + ((offset_t & 16) >> 4)


def unswizzle_xbox360_data(data: bytes, width: int, height: int, fmt: str) -> bytes:
    """Port of Rpf3Crypto.UnswizzleXbox360Data: undoes the Xbox 360 texture
    tiling/swizzle and the 16-bit byte-swap the console GPU expects."""
    if fmt in (TEXFMT_L8, TEXFMT_A8R8G8B8):
        return data  # not swizzled in CodeX's implementation

    if fmt == TEXFMT_BC1:
        block_size_row, texel_pitch = 4, 8
    elif fmt in (TEXFMT_BC2, TEXFMT_BC3):
        block_size_row, texel_pitch = 4, 16
    else:
        raise NotImplementedError(f"unsupported format for unswizzle: {fmt}")

    data = bytearray(data)
    # Reverse every two bytes (16-bit byte swap).
    for i in range(0, len(data) - 1, 2):
        data[i], data[i + 1] = data[i + 1], data[i]

    virtual_width = get_virtual_texture_size(width)
    virtual_height = get_virtual_texture_size(height)
    vbw = virtual_width // block_size_row
    vbh = virtual_height // block_size_row
    out = bytearray(len(data))

    for j in range(vbh):
        for i in range(vbw):
            block_offset = j * vbw + i
            x = xg_address_2d_tiled_x(block_offset, vbw, texel_pitch)
            y = xg_address_2d_tiled_y(block_offset, vbw, texel_pitch)
            src_off = j * vbw * texel_pitch + i * texel_pitch
            dst_off = y * vbw * texel_pitch + x * texel_pitch
            if src_off + texel_pitch <= len(data) and dst_off + texel_pitch <= len(out):
                out[dst_off:dst_off + texel_pitch] = data[src_off:src_off + texel_pitch]

    if width < 128 or height < 128:
        abw = width // block_size_row
        abh = height // block_size_row
        trimmed = bytearray(abw * abh * texel_pitch)
        for j in range(abh):
            src_off = j * vbw * texel_pitch
            dst_off = j * abw * texel_pitch
            trimmed[dst_off:dst_off + abw * texel_pitch] = out[src_off:src_off + abw * texel_pitch]
        out = trimmed

    return bytes(out)


def parse_texture(r: Rsc5Reader):
    """Rsc5TextureBase.Read + Rsc5Texture.Read: name, dimensions, and (if the
    D3DBaseTexture pointer is non-null) the actual unswizzled pixel bytes."""
    _vft = r.u32()
    _u4 = r.u32(); _u8 = r.u32(); _uc = r.u32(); _u10 = r.u32(); _u14 = r.u32()
    raw_name = r.str_ptr()
    d3d_ptr = r.u32()

    name = None
    if raw_name:
        name = raw_name.replace(".dds", "").replace("pack:/", "")

    result = {"name": name, "width": 0, "height": 0, "format": None, "data": None}
    if not d3d_ptr:
        return result  # texture stub only (as embedded in shader params)

    width = r.u16()
    height = r.u16()
    result["width"] = width
    result["height"] = height

    _stride = r.u16()
    _texture_type = r.u8()
    _mip_levels = r.u8()
    for _ in range(6):  # ColorExpR/G/B, ColorOfsR/G/B
        r.f32()

    d3d_value = r.at(d3d_ptr + 0x20, lambda: r.u32())
    if d3d_value is None:
        return result

    fmt = convert_to_engine_format(d3d_value & 0xFF)
    vw = get_virtual_texture_size(width)
    vh = get_virtual_texture_size(height)
    try:
        size = calc_texture_data_size(fmt, vw, vh)
        addr = get_base_address_from_d3d(d3d_value, r.virtual_size)
        raw = r.at(addr, lambda: r.raw(size))
        if raw:
            result["format"] = fmt
            result["data"] = unswizzle_xbox360_data(raw, width, height, fmt)
    except Exception:
        pass  # leave data=None; caller skips textures it couldn't decode

    return result


def _score_virtual_size_textures(decompressed: bytes, vsize: int):
    """Same idea as _score_virtual_size, but for texture-dictionary buffers:
    scores on whether textures parse to sane (nonzero, non-huge) dimensions
    with successfully-decoded pixel data, instead of vertex positions."""
    try:
        textures = parse_texture_dictionary(decompressed, vsize)
    except Exception:
        return None
    if not textures:
        return None

    bad = 0
    good = 0
    for t in textures.values():
        w, h = t.get("width", 0), t.get("height", 0)
        if w <= 0 or h <= 0 or w > 8192 or h > 8192:
            bad += 1
            continue
        if t.get("data") is None:
            bad += 1
            continue
        good += 1
    if good == 0:
        return None
    return (bad, -good)


def guess_virtual_size_textures(decompressed: bytes, verbose=False):
    """--textures-raw counterpart to guess_virtual_size(): a texture
    dictionary has no vertex positions to score against, so this scores on
    successfully-decoded texture dimensions/pixel data instead."""
    total = len(decompressed)
    quantized_all = sorted(v for v in _quantized_sizes(total) if 0 < v < total)
    quantized_set = set(quantized_all)

    exact_splits = [v for v in quantized_all if (total - v) in quantized_set]
    candidates = exact_splits or quantized_all

    best = None
    for vsize in candidates:
        score = _score_virtual_size_textures(decompressed, vsize)
        if score is not None and (best is None or score < best[0]):
            best = (score, vsize)
            if score[0] == 0:
                break

    if best is None:
        raise ValueError(
            "Could not determine --textures-virtual-size automatically: no "
            f"candidate split of this {total}-byte buffer decoded any texture "
            "with sane dimensions. Pass --textures-virtual-size explicitly."
        )
    if verbose:
        print(f"guess_virtual_size_textures: best match vsize={best[1]} "
              f"score={best[0]}", file=sys.stderr)
    return best[1]


def parse_texture_dictionary(decompressed: bytes, virtual_size: int):
    """Rsc5TextureDictionary root (as loaded by CodeX's XtdFile): VFT +
    BlockMap ptr, ParentDictionary, UsageCount, Arr<JenkHash> Hashes,
    PtrArr<Rsc5Texture> Textures. Returns {name: texture_dict}."""
    r = Rsc5Reader(decompressed, virtual_size)
    _vft = r.u32()
    _blockmap_ptr = r.u32()
    _parent_dict = r.u32()
    _usage_count = r.u32()
    _hashes_ptr = r.u32(); _hashes_count = r.u16(); _hashes_cap = r.u16()
    tex_ptr = r.u32(); tex_count = r.u16(); _tex_cap = r.u16()

    textures = {}
    if tex_ptr and tex_count:
        ptrs = r.at(tex_ptr, lambda: [r.u32() for _ in range(tex_count)]) or []
        for p in ptrs:
            t = r.at(p, lambda: parse_texture(r)) if p else None
            if t and t.get("name"):
                textures[t["name"].lower()] = t
    return textures


def force_bc_opaque(data: bytes, fmt: str) -> bytes:
    """Rewrite a BC2/BC3 block-compressed buffer so every texel decodes to
    full alpha (255), without touching the color blocks.

    Diffuse maps in this data reuse their alpha channel as an in-engine
    specular/gloss mask rather than real transparency. Left as-is, that
    partial alpha (commonly ~0-180/255 here) gets picked up by importers
    (e.g. Blender's OBJ importer) as material transparency, making the
    model render mostly see-through/black. BC1 doesn't need this: without
    the DXT1 explicit-alpha bit set, it already decodes fully opaque.
    """
    if fmt not in (TEXFMT_BC2, TEXFMT_BC3):
        return data
    out = bytearray(data)
    block = 16  # both BC2 and BC3 are 16 bytes/block, alpha is the first 8
    if fmt == TEXFMT_BC2:
        # Explicit 4-bit alpha per texel (16 texels x 4 bits = 8 bytes).
        # 0xF per nibble = full alpha.
        for off in range(0, len(out) - block + 1, block):
            for i in range(8):
                out[off + i] = 0xFF
    else:  # TEXFMT_BC3
        # 2 reference alpha values + 16x3-bit indices. Setting both
        # reference values to 255 and every index to 0 makes every
        # interpolated/selected alpha come out as 255, regardless of the
        # (alpha0 > alpha1) vs (alpha0 <= alpha1) interpolation mode.
        for off in range(0, len(out) - block + 1, block):
            out[off] = 0xFF
            out[off + 1] = 0xFF
            for i in range(2, 8):
                out[off + i] = 0x00
    return bytes(out)


def write_dds(path: str, width: int, height: int, fmt: str, data: bytes):
    """Write a minimal (no-mipmap) DDS file. BC1/2/3 map straight to the
    corresponding FourCC; A8R8G8B8/L8 are written as uncompressed formats."""
    header = bytearray(128)
    struct.pack_into("<4s", header, 0, b"DDS ")
    struct.pack_into("<I", header, 4, 124)  # header size
    flags = 0x1 | 0x2 | 0x4 | 0x1000  # CAPS | HEIGHT | WIDTH | PIXELFORMAT
    flags |= 0x80000 if fmt in (TEXFMT_BC1, TEXFMT_BC2, TEXFMT_BC3) else 0x8  # LINEARSIZE | PITCH
    struct.pack_into("<I", header, 8, flags)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    pitch_or_size = len(data) if fmt in (TEXFMT_BC1, TEXFMT_BC2, TEXFMT_BC3) else width * (4 if fmt == TEXFMT_A8R8G8B8 else 1)
    struct.pack_into("<I", header, 20, pitch_or_size)
    struct.pack_into("<I", header, 28, 1)  # mip count

    struct.pack_into("<I", header, 76, 32)  # pixel format block size
    if fmt in (TEXFMT_BC1, TEXFMT_BC2, TEXFMT_BC3):
        fourcc = {TEXFMT_BC1: b"DXT1", TEXFMT_BC2: b"DXT3", TEXFMT_BC3: b"DXT5"}[fmt]
        struct.pack_into("<I", header, 80, 0x4)  # DDPF_FOURCC
        struct.pack_into("<4s", header, 84, fourcc)
    elif fmt == TEXFMT_A8R8G8B8:
        struct.pack_into("<I", header, 80, 0x41)  # DDPF_ALPHAPIXELS | DDPF_RGB
        struct.pack_into("<I", header, 88, 32)
        struct.pack_into("<I", header, 92, 0x00FF0000)
        struct.pack_into("<I", header, 96, 0x0000FF00)
        struct.pack_into("<I", header, 100, 0x000000FF)
        struct.pack_into("<I", header, 104, 0xFF000000)
    else:  # L8
        struct.pack_into("<I", header, 80, 0x20000)  # DDPF_LUMINANCE
        struct.pack_into("<I", header, 88, 8)
        struct.pack_into("<I", header, 92, 0xFF)

    struct.pack_into("<I", header, 108, 0x1000)  # DDSCAPS_TEXTURE

    with open(path, "wb") as f:
        f.write(header)
        f.write(data)


def build_texture_dict_from_shaders(shaders):
    """The paired '_h' file (despite the generic .xrsc extension CodeX gives
    ModelResource-typed entries) turns out to be structured exactly like the
    model file - a Drawable with its own ShaderGroup - except its shader
    texture parameters carry *real* embedded pixel data (non-null
    D3DBaseTexture) instead of the name-only stubs the main model's shaders
    have. So: parse it with parse_xrsc() same as the model, then flatten all
    of its shaders' textures into one {name: texture_dict} map."""
    textures = {}
    for shader in shaders:
        for tex in shader.get("textures_full", []):
            if tex.get("name") and tex.get("data") is not None:
                textures[tex["name"].lower()] = tex
    return textures


def export_materials(shaders, texture_dict, out_dir: str, name_prefix: str):
    """Export every texture parameter of every shader (not just the first),
    build a {shader_index: material_name} map plus .mtl file text.

    Slot 0 is treated as the base/diffuse map (map_Kd) - its alpha channel
    is forced fully opaque first (see force_bc_opaque) since it's a
    specular/gloss mask in this data, not real transparency, and importers
    otherwise pick it up as material transparency. Slot 1 is treated as a
    normal/bump map (map_Bump / bump). Any further slots (detail/secondary
    maps) are still exported to disk and referenced via a plain comment
    line, since standard OBJ/MTL has no dedicated slot for them - the file
    is delivered even though nothing auto-wires it.

    A texture already written under one shader is not re-encoded/rewritten
    for the next shader that reuses it - only the map_* line is repeated.

    Returns (shader_material_name, mtl_lines, exported_count) where
    exported_count is the number of distinct texture files written."""
    import os
    shader_material = {}
    mtl_lines = []
    seen_materials = set()
    written_dds = {}  # texture name (lower) -> dds filename, avoids re-writing dupes

    def safe(s):
        return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in s)

    def export_one(tex, force_opaque=False):
        """Write tex's pixel data to a .dds (if not already written) and
        return its filename, or None if there's no decoded data."""
        if not tex or not tex.get("name") or tex.get("data") is None:
            return None
        key = tex["name"].lower()
        if key in written_dds:
            return written_dds[key]
        data = tex["data"]
        if force_opaque:
            data = force_bc_opaque(data, tex["format"])
        dds_name = safe(f"{name_prefix}_{tex['name']}.dds")
        write_dds(os.path.join(out_dir, dds_name), tex["width"], tex["height"], tex["format"], data)
        written_dds[key] = dds_name
        return dds_name

    for si, shader in enumerate(shaders):
        tex_names = shader.get("textures") or []
        mat_name = safe(f"mat_{si}_{shader.get('name') or 'shader'}")
        shader_material[si] = mat_name
        if mat_name in seen_materials:
            continue
        seen_materials.add(mat_name)

        mtl_lines.append(f"newmtl {mat_name}")
        mtl_lines.append("Ka 1.000 1.000 1.000")
        mtl_lines.append("Kd 1.000 1.000 1.000")
        mtl_lines.append("Ks 0.000 0.000 0.000")
        mtl_lines.append("d 1.0")
        mtl_lines.append("illum 1")

        for i, tex_name in enumerate(tex_names):
            tex = texture_dict.get(tex_name.lower()) if tex_name else None
            if i == 0:
                dds_name = export_one(tex, force_opaque=True)
                if dds_name:
                    mtl_lines.append(f"map_Kd {dds_name}")
            elif i == 1:
                dds_name = export_one(tex)
                if dds_name:
                    mtl_lines.append(f"map_Bump {dds_name}")
                    mtl_lines.append(f"bump {dds_name}")
            else:
                dds_name = export_one(tex)
                if dds_name:
                    mtl_lines.append(f"# detail map (not auto-applied): {dds_name}")
        mtl_lines.append("")

    return shader_material, mtl_lines, len(written_dds)


# =============================================================================
# OBJ writer
# =============================================================================

def write_obj(models, out_path: str, name_prefix="mesh", mtl_filename=None, shader_material=None):
    vert_offset = 0
    uv_offset = 0
    norm_offset = 0
    total_tris = 0
    total_verts = 0
    shader_material = shader_material or {}

    with open(out_path, "w") as f:
        f.write(f"# generated by xrsc_to_obj.py\n")
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")
        for mi, geoms in enumerate(models):
            for gi, geo in enumerate(geoms):
                if not geo.positions:
                    continue
                f.write(f"o {name_prefix}_m{mi}_g{gi}\n")
                mat = shader_material.get(geo.shader_id)
                if mat:
                    f.write(f"usemtl {mat}\n")

                for (x, y, z) in geo.positions:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                has_uv = any(uv is not None for uv in geo.uvs)
                if has_uv:
                    for uv in geo.uvs:
                        u, v = uv if uv is not None else (0.0, 0.0)
                        f.write(f"vt {u:.6f} {1.0 - v:.6f}\n")
                has_n = any(n is not None for n in geo.normals)
                if has_n:
                    for n in geo.normals:
                        nx, ny, nz = n if n is not None else (0.0, 0.0, 1.0)
                        f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

                idx = geo.indices
                ntris = len(idx) // 3
                for t in range(ntris):
                    i0, i1, i2 = idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]

                    def vref(i):
                        vi = i + 1 + vert_offset
                        ti = i + 1 + uv_offset
                        ni = i + 1 + norm_offset
                        if has_uv and has_n:
                            return f"{vi}/{ti}/{ni}"
                        if has_uv:
                            return f"{vi}/{ti}"
                        if has_n:
                            return f"{vi}//{ni}"
                        return f"{vi}"

                    f.write(f"f {vref(i0)} {vref(i1)} {vref(i2)}\n")

                total_tris += ntris
                total_verts += len(geo.positions)
                vert_offset += len(geo.positions)
                if has_uv:
                    uv_offset += len(geo.positions)
                if has_n:
                    norm_offset += len(geo.positions)

    return total_verts, total_tris


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .xrsc file")
    ap.add_argument("output", nargs="?", help="output .obj file")
    ap.add_argument("--raw", action="store_true",
                     help="input is already the decompressed [virtual][physical] "
                          "buffer (no RSC5 header, no LZX) - requires --virtual-size")
    ap.add_argument("--virtual-size", type=int, default=None,
                     help="virtual segment size in bytes (only used with --raw; "
                          "if omitted, it's auto-detected from the buffer - see "
                          "guess_virtual_size())")
    ap.add_argument("--window-bits", type=int, default=None,
                     help="override the LZX window size guess (15-21) if decoding "
                          "looks wrong")
    ap.add_argument("--dump-only", action="store_true",
                     help="parse and print stats only, don't write an .obj "
                          "(sanity-check decompression first)")
    ap.add_argument("--textures",
                     help="the paired texture-dictionary .xrsc file (e.g. the "
                          "'_h.xrsc' file) to pull material textures from")
    ap.add_argument("--textures-raw", action="store_true",
                     help="the --textures file is already decompressed, like --raw "
                          "(auto-detected if omitted, same as the model file)")
    ap.add_argument("--textures-virtual-size", type=int, default=None,
                     help="virtual segment size for --textures, if it needs --raw "
                          "and auto-detection picks the wrong split")
    args = ap.parse_args()

    def load_raw_buffer(path, raw_flag, window_bits, label):
        """Decompress one .xrsc to its [virtual][physical] buffer only - no
        vsize guessing here, since the *right* guesser depends on which
        struct layout we end up interpreting the buffer as (Drawable vs.
        flat Rsc5TextureDictionary), which differs for --textures."""
        with open(path, "rb") as f:
            raw_data = f.read()

        has_magic = len(raw_data) >= 4 and struct.unpack_from(">I", raw_data, 0)[0] == RSC5_MAGIC
        use_raw = raw_flag or not has_magic

        if not use_raw:
            try:
                return read_rsc5_container(raw_data, window_bits=window_bits, verbose=True)[0:2][0]
            except Exception as e:
                print(f"warning: {label}: RSC5/LZX decode failed ({e}), "
                      f"falling back to treating the file as a raw buffer", file=sys.stderr)
        return raw_data

    def load_buffer(path, raw_flag, vsize_override, window_bits, label):
        """load_raw_buffer() + the Drawable-oriented vsize guess (scores on
        sane vertex positions) - used for the model file."""
        decompressed = load_raw_buffer(path, raw_flag, window_bits, label)
        if vsize_override is None:
            print(f"{label}: no virtual-size given, auto-detecting...", file=sys.stderr)
            vsize = guess_virtual_size(decompressed, verbose=True)
            print(f"{label}: auto-detected virtual-size {vsize}", file=sys.stderr)
        else:
            vsize = vsize_override
        return decompressed, vsize

    decompressed, vsize = load_buffer(args.input, args.raw, args.virtual_size,
                                       args.window_bits, "model")

    models, shaders = parse_xrsc(decompressed, vsize)

    total_geo = sum(len(g) for g in models)
    total_verts = sum(len(geo.positions) for g in models for geo in g)
    total_tris = sum(len(geo.indices) // 3 for g in models for geo in g)
    print(f"parsed: {len(models)} model(s), {total_geo} geometry submesh(es), "
          f"{total_verts} vertices, {total_tris} triangles, {len(shaders)} shader(s)",
          file=sys.stderr)

    texture_dict = {}
    if args.textures:
        tex_decompressed = load_raw_buffer(
            args.textures, args.textures_raw, args.window_bits, "textures")

        # Try the real Rsc5TextureDictionary layout first (flat Hashes +
        # PtrArr<Rsc5Texture> - the normal shape of an MCLA "_h" file, with
        # no shaders involved at all). This is the path that was previously
        # dead code: build_texture_dict_from_shaders() alone only recovers
        # textures that happen to be reachable as a shader's texture
        # *parameter*, silently dropping every dictionary entry that isn't -
        # which is why not all textures were coming out.
        try:
            dict_vsize = (args.textures_virtual_size if args.textures_virtual_size is not None
                          else guess_virtual_size_textures(tex_decompressed, verbose=True))
            texture_dict = parse_texture_dictionary(tex_decompressed, dict_vsize)
        except Exception as e:
            print(f"note: textures: flat-dictionary parse failed ({e})", file=sys.stderr)
            texture_dict = {}

        if texture_dict:
            n_with_data = sum(1 for t in texture_dict.values() if t.get("data") is not None)
            print(f"parsed textures: {len(texture_dict)} named textures from texture "
                  f"dictionary ({n_with_data} with decoded pixel data)", file=sys.stderr)
        else:
            # Fall back to the Drawable+ShaderGroup interpretation, in case
            # this particular file really is shaped that way instead.
            print("note: textures: no entries from flat-dictionary parse, "
                  "falling back to shader-based parse", file=sys.stderr)
            shader_vsize = (args.textures_virtual_size if args.textures_virtual_size is not None
                             else guess_virtual_size(tex_decompressed, verbose=True))
            _tex_models, tex_shaders = parse_xrsc(tex_decompressed, shader_vsize)
            texture_dict = build_texture_dict_from_shaders(tex_shaders)
            print(f"parsed textures: {len(texture_dict)} named textures with "
                  f"decoded pixel data (from {len(tex_shaders)} shader(s))", file=sys.stderr)

    if args.dump_only:
        return

    if not args.output:
        ap.error("output .obj path is required unless --dump-only is given")

    import os
    name_prefix = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    out_base = os.path.splitext(os.path.basename(args.output))[0]

    shader_material = {}
    mtl_filename = None
    if shaders and texture_dict:
        shader_material, mtl_lines, n_exported = export_materials(
            shaders, texture_dict, out_dir, out_base)
        mtl_filename = f"{out_base}.mtl"
        with open(os.path.join(out_dir, mtl_filename), "w") as f:
            f.write("\n".join(mtl_lines) + "\n")
        print(f"wrote {mtl_filename}: {len(shader_material)} material(s), "
              f"{n_exported} texture(s) exported as .dds", file=sys.stderr)
    elif shaders and not args.textures:
        print("note: shaders found but no --textures file given - "
              "writing geometry only, no materials", file=sys.stderr)

    v, t = write_obj(models, args.output, name_prefix=name_prefix,
                      mtl_filename=mtl_filename, shader_material=shader_material)
    print(f"wrote {args.output}: {v} vertices, {t} triangles", file=sys.stderr)


if __name__ == "__main__":
    main()
