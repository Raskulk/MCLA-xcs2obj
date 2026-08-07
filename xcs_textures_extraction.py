#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xcs_textures_extraction.py
======================
Extracts textures from ALL .xcs files (Midnight Club: Los Angeles,
City Sector, RSC5 / Xbox 360 format) in the specified folder and saves
them as .png, using names from Codex.Games.MCLA.strings.txt
(matched via the Jenkins one-at-a-time hash).

If a name for a texture's hash is not found in strings.txt -> the name
is left "as is" (0xHHHHHHHH.png).
If two different .xcs files produce a texture with the same final name ->
the file is overwritten (the last one processed wins).

Usage:
    python3 xcs_textures_extraction.py <xcs_folder> <Codex.Games.MCLA.strings.txt> <png_folder>

Example:
    python3 xcs_textures_extraction.py ./xcs_files ./Codex.Games.MCLA.strings.txt ./textures_png

Dependencies: Pillow (pip install --break-system-packages pillow)
"""

import sys
import os
import glob
import struct

try:
    from PIL import Image
except ImportError:
    print("Pillow module required: pip install --break-system-packages pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Jenkins one-at-a-time hash (used by the MCLA engine for resource names)
# ---------------------------------------------------------------------------

def genhash(text: str) -> int:
    h = 0
    for ch in text:
        h = (h + (ord(ch) & 0xFF)) & 0xFFFFFFFF
        h = (h + ((h << 10) & 0xFFFFFFFF)) & 0xFFFFFFFF
        h ^= (h >> 6)
    h = (h + ((h << 3) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h ^= (h >> 11)
    h = (h + ((h << 15) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def load_string_table(path):
    """hash -> name, from the list of strings in Codex.Games.MCLA.strings.txt"""
    lookup = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\r\n")
            if not s:
                continue
            lookup[genhash(s.lower())] = s
            lookup[genhash(s)] = s
    return lookup


# ---------------------------------------------------------------------------
# Unswizzle (Xbox 360 texture layout, BC1/BC2/BC3)
# ---------------------------------------------------------------------------

_TEXEL_INFO = {
    "BC1": (4, 8),
    "BC2": (4, 16),
    "BC3": (4, 16),
}


def _get_virtual_size(size):
    if (size % 128 != 0) and size < 128:
        return 128
    return size


def _xg_tiled_x(offset, width, texel_pitch):
    aligned_width = (width + 31) & ~31
    log_bpp = (texel_pitch >> 2) + ((texel_pitch >> 1) >> (texel_pitch >> 2))
    offset_b = offset << log_bpp
    offset_t = ((offset_b & ~4095) >> 3) + ((offset_b & 1792) >> 2) + (offset_b & 63)
    offset_m = offset_t >> (7 + log_bpp)
    macro_x = (offset_m % (aligned_width >> 5)) << 2
    tile = (((offset_t >> (5 + log_bpp)) & 2) + (offset_b >> 6)) & 3
    macro = (macro_x + tile) << 3
    micro = ((((offset_t >> 1) & ~15) + (offset_t & 15)) & ((texel_pitch << 3) - 1)) >> log_bpp
    return macro + micro


def _xg_tiled_y(offset, width, texel_pitch):
    aligned_width = (width + 31) & ~31
    log_bpp = (texel_pitch >> 2) + ((texel_pitch >> 1) >> (texel_pitch >> 2))
    offset_b = offset << log_bpp
    offset_t = ((offset_b & ~4095) >> 3) + ((offset_b & 1792) >> 2) + (offset_b & 63)
    offset_m = offset_t >> (7 + log_bpp)
    macro_y = (offset_m // (aligned_width >> 5)) << 2
    tile = ((offset_t >> (6 + log_bpp)) & 1) + ((offset_b & 2048) >> 10)
    macro = (macro_y + tile) << 3
    micro = (((offset_t & ((texel_pitch << 6) - 1) & ~31) + ((offset_t & 15) << 1)) >> (3 + log_bpp)) & ~1
    return macro + micro + ((offset_t & 16) >> 4)


def unswizzle_xbox360_data(data: bytes, width: int, height: int, fmt: str) -> bytes:
    if fmt in ("L8", "A8R8G8B8"):
        return data

    block_size_row, texel_pitch = _TEXEL_INFO[fmt]
    data = bytearray(data)

    for i in range(0, len(data) - 1, 2):
        data[i], data[i + 1] = data[i + 1], data[i]

    virtual_width = _get_virtual_size(width)
    virtual_height = _get_virtual_size(height)
    vbw = virtual_width // block_size_row
    vbh = virtual_height // block_size_row

    unswizzled = bytearray(len(data))
    for j in range(vbh):
        for i in range(vbw):
            block_offset = j * vbw + i
            x = _xg_tiled_x(block_offset, vbw, texel_pitch)
            y = _xg_tiled_y(block_offset, vbw, texel_pitch)
            src = j * vbw * texel_pitch + i * texel_pitch
            dst = y * vbw * texel_pitch + x * texel_pitch
            unswizzled[dst:dst + texel_pitch] = data[src:src + texel_pitch]

    if width < 128 or height < 128:
        abw = width // block_size_row
        abh = height // block_size_row
        trimmed = bytearray(abw * abh * texel_pitch)
        for j in range(abh):
            src = j * vbw * texel_pitch
            dst = j * abw * texel_pitch
            trimmed[dst:dst + abw * texel_pitch] = unswizzled[src:src + abw * texel_pitch]
        unswizzled = trimmed

    return bytes(unswizzled)


# ---------------------------------------------------------------------------
# Building the DDS header
# ---------------------------------------------------------------------------

DDS_MAGIC = b"DDS "
DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40
DDPF_LUMINANCE = 0x20000
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSD_PITCH = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000
DDSCAPS_COMPLEX = 0x8


def build_dds(width, height, fmt, mip_levels, pixel_data):
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    caps = DDSCAPS_TEXTURE
    if mip_levels > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_MIPMAP | DDSCAPS_COMPLEX

    if fmt == "BC1":
        fourcc, pf_flags = b"DXT1", DDPF_FOURCC
        pitch = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 8
        flags |= DDSD_LINEARSIZE
    elif fmt == "BC2":
        fourcc, pf_flags = b"DXT3", DDPF_FOURCC
        pitch = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 16
        flags |= DDSD_LINEARSIZE
    elif fmt == "BC3":
        fourcc, pf_flags = b"DXT5", DDPF_FOURCC
        pitch = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 16
        flags |= DDSD_LINEARSIZE
    elif fmt == "A8R8G8B8":
        fourcc, pf_flags = b"\x00\x00\x00\x00", DDPF_RGB | DDPF_ALPHAPIXELS
        pitch = width * 4
        flags |= DDSD_PITCH
    elif fmt == "L8":
        fourcc, pf_flags = b"\x00\x00\x00\x00", DDPF_LUMINANCE
        pitch = width
        flags |= DDSD_PITCH
    else:
        raise ValueError(f"unsupported format {fmt}")

    buf = bytearray()
    buf += DDS_MAGIC
    buf += struct.pack("<I", 124)
    buf += struct.pack("<I", flags)
    buf += struct.pack("<I", height)
    buf += struct.pack("<I", width)
    buf += struct.pack("<I", pitch)
    buf += struct.pack("<I", 0)
    buf += struct.pack("<I", max(1, mip_levels))
    buf += struct.pack("<11I", *([0] * 11))
    buf += struct.pack("<I", 32)
    buf += struct.pack("<I", pf_flags)
    buf += fourcc
    if fmt == "A8R8G8B8":
        buf += struct.pack("<5I", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    elif fmt == "L8":
        buf += struct.pack("<5I", 8, 0xFF, 0, 0, 0)
    else:
        buf += struct.pack("<5I", *([0] * 5))
    buf += struct.pack("<I", caps)
    buf += struct.pack("<4I", 0, 0, 0, 0)
    return bytes(buf) + pixel_data


# ---------------------------------------------------------------------------
# Parsing .xcs (Rsc5CitySector -> Rsc5TextureDictionary -> Rsc5Texture[])
# ---------------------------------------------------------------------------

VIRTUAL_BASE = 0x50000000
BASE_ADDRESSES = {0x50: 0x50000000, 0x51: 0x51000000, 0x52: 0x52000000}
FORMAT_MAP = {2: "L8", 82: "BC1", 83: "BC2", 84: "BC3", 134: "A8R8G8B8"}


class XcsReader:
    def __init__(self, data: bytes):
        self.data = data

    def u8(self, off): return self.data[off]
    def u16(self, off): return struct.unpack_from(">H", self.data, off)[0]
    def u32(self, off): return struct.unpack_from(">I", self.data, off)[0]

    def offset_of(self, ptr):
        if ptr == 0:
            return None
        if (ptr & VIRTUAL_BASE) != VIRTUAL_BASE:
            return None
        return ptr & 0x0FFFFFFF

    def cstr(self, off):
        end = self.data.index(b"\x00", off)
        return self.data[off:end].decode("utf-8", "replace")

    def base_address_from_d3d(self, d3d_value, virtual_size=0):
        phys = (d3d_value & 0x60000000) == 0x60000000
        address = (d3d_value >> 24) & 0xFF
        size = d3d_value & 0xFF
        mapped_base = BASE_ADDRESSES.get(address, 0) or VIRTUAL_BASE
        value = (d3d_value & 0xFFFFFF) - size + (virtual_size if phys else 0) + mapped_base
        return value & 0xFFFFFFFF


def parse_xcs_textures(data: bytes):
    """Returns a list of dicts: hash, name (from the file or None), width,
    height, format, mip_levels, dds_bytes (a ready-made .dds, or None)."""
    r = XcsReader(data)

    texdict_off = r.offset_of(r.u32(8))
    if texdict_off is None:
        return []

    hashes_pos = r.u32(texdict_off + 16)
    hashes_count = r.u16(texdict_off + 20)
    tex_pos = r.u32(texdict_off + 24)
    tex_count = r.u16(texdict_off + 28)

    hashes_off = r.offset_of(hashes_pos)
    hashes = [r.u32(hashes_off + i * 4) for i in range(hashes_count)] if hashes_off is not None else []

    tex_off = r.offset_of(tex_pos)
    tex_ptrs = [r.u32(tex_off + i * 4) for i in range(tex_count)] if tex_off is not None else []

    result = []
    for i, tptr in enumerate(tex_ptrs):
        toff = r.offset_of(tptr)
        if toff is None:
            continue

        name_ptr = r.u32(toff + 24)
        d3d_ptr = r.u32(toff + 28)
        width = r.u16(toff + 32)
        height = r.u16(toff + 34)
        name = r.cstr(r.offset_of(name_ptr)) if name_ptr else None
        tex_hash = hashes[i] if i < len(hashes) else 0

        entry = {
            "hash": tex_hash, "name": name,
            "width": width, "height": height,
            "dds": None,
        }
        result.append(entry)

        if d3d_ptr == 0:
            continue  # placeholder entry with no pixel data

        d3d_off = r.offset_of(d3d_ptr)
        mip_levels = max(1, r.u8(toff + 39))
        d3d_value = r.u32(d3d_off + 0x20)
        fmt = FORMAT_MAP.get(d3d_value & 0xFF, "BC1")

        vw, vh = _get_virtual_size(width), _get_virtual_size(height)
        data_off = r.offset_of(r.base_address_from_d3d(d3d_value))
        if data_off is None:
            continue

        size = (vw * vh) // 2 if fmt == "BC1" else vw * vh
        raw = data[data_off:data_off + size]
        pixels = unswizzle_xbox360_data(raw, width, height, fmt)
        entry["dds"] = build_dds(width, height, fmt, mip_levels, pixels)

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 xcs_textures_extraction.py <xcs_folder> <strings.txt> <png_folder>")
        sys.exit(1)

    in_dir, strings_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    print("Loading name table...")
    lookup = load_string_table(strings_path)
    print(f"  {len(lookup)} hashes in table")

    xcs_files = sorted(glob.glob(os.path.join(in_dir, "*.xcs")))
    if not xcs_files:
        print(f"No .xcs files found in folder {in_dir}")
        sys.exit(0)

    print(f"Found {len(xcs_files)} .xcs files")

    total_written = 0
    total_overwritten = 0
    seen_names = set()

    for xcs_path in xcs_files:
        fname = os.path.basename(xcs_path)
        try:
            data = open(xcs_path, "rb").read()
            textures = parse_xcs_textures(data)
        except Exception as e:
            print(f"  [!] {fname}: failed to parse ({e})")
            continue

        written_here = 0
        for tex in textures:
            if tex["dds"] is None:
                continue

            name = tex["name"] or lookup.get(tex["hash"])
            if name:
                base = os.path.splitext(os.path.basename(name))[0]
            else:
                base = f"0x{tex['hash']:08X}"  # no name - leave "as is"

            out_path = os.path.join(out_dir, base + ".png")
            if base in seen_names:
                total_overwritten += 1
            seen_names.add(base)

            try:
                import io
                img = Image.open(io.BytesIO(tex["dds"])).convert("RGBA")
                img.save(out_path)
                written_here += 1
                total_written += 1
            except Exception as e:
                print(f"  [!] {base}: PNG conversion error ({e})")

        print(f"  {fname}: {written_here} textures")

    print(f"\nDone. Total files saved: {total_written} "
          f"(of which overwritten due to name collisions: {total_overwritten})")
    print(f"Result in folder: {out_dir}")


if __name__ == "__main__":
    main()