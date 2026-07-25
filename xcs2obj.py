#!/usr/bin/env python3
"""
xcs2obj.py — converter for .xcs (Midnight Club: Los Angeles, City Sector) to Wavefront .obj

Manually ported from the CodeX C# sources (RSC5 / MCLA reader):
    CodeX.Games.MCLA/Files/XcsFile.cs
    CodeX.Games.MCLA/RPF3/Rpf3Crypto.cs / Rpf3File.cs
    CodeX.Games.MCLA/RSC5/Rsc5Data.cs
    CodeX.Games.MCLA/RSC5/Rsc5City.cs
    CodeX.Games.MCLA/RSC5/Rsc5Drawable.cs
    CodeX.Games.MCLA/RSC5/Rsc5Texture.cs

The .xcs format is an RSC5 resource (the same container used in GTA IV /
the Rockstar Xbox 360 generation): a large data blob with a "virtual" and
a "physical" segment, inside which objects reference each other via
32-bit pointers of the form 0x50xxxxxx (virtual) / 0x60xxxxxx (physical).

Besides geometry, the script also extracts textures (they are stored as
"raw" Xbox 360 D3D textures, most often DXT1/DXT3/DXT5, sometimes
A8R8G8B8/L8, and "swizzled" — a tiled rearrangement of blocks for the
X360 GPU) and saves them as .dds files next to the .obj, and also
generates an .mtl that assigns each material (by shader name) all the
textures bound to it, into the correct slots — the same way the engine
does it in Rsc5DrawableGeometry.SetShaderRef
(CodeX.Games.MCLA/RSC5/Rsc5Drawable.cs), by classifying shader samplers
by the hash of their name:
    - diffuse/base texture (diffusesampler, texturesampler,
      diffusesamplera/b/c, etc.)                -> map_Kd (+ map_d, if the
                                                    format has an alpha
                                                    channel, DXT3/DXT5)
    - normal/bump/height map (bumpsampler,
      normalsampler, normalsamplera/b/c, etc.)   -> map_bump / bump
    - decals, grime, puddles and extra diffuse
      layers of terrain shaders (decalsampler,
      grimesampler, puddlesampler,
      channelmapsampler, etc.)                   -> decal (+ a comment, if
                                                     there are several layers)
All encountered textures (including those not directly used in the
geometry, but present in the sector's shared texture dictionary) are
saved as .dds next to the .obj.

Two kinds of input data are supported:

1. A "raw" (--raw) file — an already-unpacked virtual+physical buffer, as
   received by XcsFile.Load(data) in CodeX. For it you need to explicitly
   specify the size of the virtual segment (--vsize) or a 32-bit packing
   flag (--flag), from which the size is computed the same way the engine
   does it.

2. A normal exported resource file (as stored by an RPF3 archive), which
   has a 20-byte header:
       uint32 magic       (0x05435352, "RSC5")
       int32  resourceType
       int32  flag         (encodes the sizes of the virtual/physical segments)
       uint32 unknown
       int32  compressedLen
       ...compressedLen bytes of LZX-compressed data (the same LZX as in
       Xbox 360 XCompress, xcompress32.dll -> LZXDecompress)
   This is what actually sits on disk / in an RPF3 archive. Decompression
   uses xcompress32.dll (Windows) via ctypes — just like in the original.

Usage:
    python xcs2obj.py sector.xcs -o sector.obj
    python xcs2obj.py sector.xcs -o sector.obj --lzx-dll C:/tools/xcompress32.dll
    python xcs2obj.py sector_raw.bin -o sector.obj --raw --flag 0x1A03C201
    python xcs2obj.py sector.xcs -o sector.obj --no-textures   # without DDS export
    python xcs2obj.py sector.xcs -o sector.obj --strings Codex.Games.MCLA.strings.txt
        # ^ names of stub textures (known only by hash) are restored
        #   from the engine's string table; if such a file sits next to the
        #   script/input .xcs, it is picked up automatically without this flag.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import struct
import sys
from dataclasses import dataclass, field

VIRTUAL_BASE = 0x50000000
PHYSICAL_BASE = 0x60000000
RSC5_MAGIC = 0x05435352


# --------------------------------------------------------------------------
# LZX decompression (Xbox 360 XCompress) — via xcompress32.dll (as in CodeX)
# --------------------------------------------------------------------------

def get_virtual_size(flag: int) -> int:
    # Rpf3ResourceFileEntry.GetVirtualSize()
    return (flag & 0x7FF) << (((flag >> 11) & 15) + 8)


def get_physical_size(flag: int) -> int:
    # Rpf3ResourceFileEntry.GetPhysicalSize()
    return ((flag >> 15) & 0x7FF) << (((flag >> 26) & 15) + 8)


def lzx_decompress(data: bytes, out_size: int, dll_path: str | None) -> bytes:
    """Calls LZXDecompress from xcompress32.dll (Xbox 360 XCompress), the
    same way Rpf3Crypto.DecompressLZX does in the original C# code.
    Requires Windows + xcompress32.dll present (it ships with many Xbox 360
    game-modding tools; it can also be extracted directly from the
    Xbox 360 XDK / System Update). Put the file next to the script or
    point to it with --lzx-dll."""
    if not dll_path:
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcompress32.dll"),
            "xcompress32.dll",
        ]
        dll_path = next((c for c in candidates if os.path.isfile(c)), candidates[0])

    if os.name != "nt":
        raise RuntimeError(
            "LZX decompression uses the native xcompress32.dll and is only "
            "available on Windows (same as the original CodeX tool). "
            "Either run the script on Windows with xcompress32.dll nearby, "
            "or unpack the .xcs yourself beforehand and run the converter "
            "with the --raw flag."
        )
    if not os.path.isfile(dll_path):
        raise RuntimeError(
            f"xcompress32.dll not found (looked in: {dll_path}). Download/copy "
            "it next to the script, or pass the path via --lzx-dll."
        )

    dll = ctypes.WinDLL(dll_path)
    func = dll.LZXDecompress
    func.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    func.restype = ctypes.c_int

    src_buf = ctypes.create_string_buffer(data, len(data))
    dst_buf = ctypes.create_string_buffer(out_size)
    dst_len = ctypes.c_int(out_size)

    ret = func(src_buf, len(data), dst_buf, ctypes.byref(dst_len))
    if ret != 0:
        raise RuntimeError(f"LZXDecompress returned error code {ret}")
    return dst_buf.raw[:out_size]


def load_xcs_bytes(path: str, raw: bool, flag_override: int | None,
                    vsize_override: int | None, psize_override: int | None,
                    lzx_dll: str | None) -> tuple[bytes, int]:
    """Returns (data, virtual_size) — the resulting buffer (virtual segment
    immediately followed by the physical segment) and the size of the
    virtual segment, needed for pointer translation."""
    with open(path, "rb") as f:
        raw_bytes = f.read()

    if raw:
        if vsize_override is not None:
            vsize = vsize_override
        elif flag_override is not None:
            vsize = get_virtual_size(flag_override)
        else:
            # Same as for external .xtd files (see load_texture_dictionary_file):
            # if neither --vsize nor --flag is given, fall back to using the
            # size of the file itself, instead of requiring the user to
            # specify it explicitly.
            vsize = len(raw_bytes)
            print(f"  [i] {path}: --vsize/--flag not given, using the file "
                  f"size ({vsize} bytes) as virtual_size.")
        return raw_bytes, vsize

    if len(raw_bytes) < 20:
        raise SystemExit("File is too small to be a valid RSC5 resource.")

    magic, rsc_type, flag, _unk, comp_len = struct.unpack_from(">IiIIi", raw_bytes, 0)
    if magic != RSC5_MAGIC:
        raise SystemExit(
            f"RSC5 signature not found (0x{RSC5_MAGIC:08X}), got "
            f"0x{magic:08X}. If this is already an unpacked buffer (virtual+physical "
            f"without a header), run with the --raw flag and specify --vsize/--flag."
        )

    comp_data = raw_bytes[20:20 + comp_len]
    vsize = get_virtual_size(flag) if vsize_override is None else vsize_override
    psize = get_physical_size(flag) if psize_override is None else psize_override
    out_size = vsize + psize

    decompressed = lzx_decompress(comp_data, out_size, lzx_dll)
    return decompressed, vsize


def load_texture_dictionary_file(path: str, vsize_override: int | None,
                                  flag_override: int | None, lzx_dll: str | None,
                                  strings_table: dict[int, str] | None = None) -> dict[int, Texture]:
    """Loads a separate texture-dictionary file (e.g. common.xtd) and
    returns {name hash -> Texture}, exactly the same as for the
    dictionary embedded in the .xcs (see parse_texture_dictionary). Such
    external dictionaries are used when sector meshes reference textures
    from a shared/global pool that is not part of the .xcs itself — this
    is exactly what happens for textures that resolve_stub_textures could
    not find in the sector's own dictionary.

    The file format is the same RSC5 container (see load_xcs_bytes): if
    the RSC5 signature is found, the file is unpacked via LZX as usual; if
    not, the file is assumed to already be an unpacked "raw" buffer
    (virtual+physical), and, just like for .xcs, it needs --vsize/--flag.
    If neither is given, the whole file size is used as a fallback:
    pointers in this format usually reference only the virtual segment
    (masked by bits, without adding vsize), so even an inaccurate vsize
    usually doesn't prevent extracting the textures — but it's better to
    give the exact value via --xtd-vsize/--xtd-flag if it's known."""
    with open(path, "rb") as f:
        head = f.read(4)
    magic = struct.unpack(">I", head)[0] if len(head) == 4 else 0

    if magic == RSC5_MAGIC:
        data, vsize = load_xcs_bytes(path, False, flag_override, vsize_override, None, lzx_dll)
    else:
        with open(path, "rb") as f:
            data = f.read()
        if vsize_override is not None:
            vsize = vsize_override
        elif flag_override is not None:
            vsize = get_virtual_size(flag_override)
        else:
            vsize = len(data)
            print(f"  [i] {path}: --xtd-vsize/--xtd-flag not given, using the "
                  f"file size ({vsize} bytes) as a fallback.")

    reader = Rsc5Reader(data, vsize)
    reader.seek(VIRTUAL_BASE)
    _textures, hash_map = parse_texture_dictionary(reader, strings_table)
    return hash_map


# --------------------------------------------------------------------------
# Base big-endian reader with virtual/physical pointer translation
# --------------------------------------------------------------------------

class Rsc5Reader:
    def __init__(self, data: bytes, virtual_size: int):
        self.data = data
        self.virtual_size = virtual_size
        self.pos = VIRTUAL_BASE  # as in Rsc5DataReader: starts at 0x50000000
        self.block_cache: dict[int, object] = {}

    def _offset(self, pos: int) -> int:
        if (pos & VIRTUAL_BASE) == VIRTUAL_BASE:
            return pos & 0x0FFFFFFF
        if (pos & PHYSICAL_BASE) == PHYSICAL_BASE:
            return (pos & 0x1FFFFFFF) + self.virtual_size
        raise ValueError(f"Invalid pointer 0x{pos:08X} — the file is corrupted or the format was misdetected.")

    def seek(self, pos: int):
        self.pos = pos

    def tell(self) -> int:
        return self.pos

    def _read(self, fmt: str, size: int):
        off = self._offset(self.pos)
        val = struct.unpack_from(fmt, self.data, off)[0]
        self.pos += size
        return val

    def u8(self) -> int:
        return self._read(">B", 1)

    def i8(self) -> int:
        return self._read(">b", 1)

    def u16(self) -> int:
        return self._read(">H", 2)

    def i16(self) -> int:
        return self._read(">h", 2)

    def u32(self) -> int:
        return self._read(">I", 4)

    def i32(self) -> int:
        return self._read(">i", 4)

    def u64(self) -> int:
        return self._read(">Q", 8)

    def f32(self) -> float:
        return self._read(">f", 4)

    def bytes(self, count: int) -> bytes:
        off = self._offset(self.pos)
        b = self.data[off:off + count]
        self.pos += count
        return b

    def cstr_at(self, pos: int) -> str:
        off = self._offset(pos)
        end = self.data.index(b"\x00", off)
        return self.data[off:end].decode("utf-8", "replace")

    # ---- high-level helpers, analogous to Rsc5DataReader ----

    def vec3_zxy(self) -> tuple[float, float, float]:
        x, y, z = self.f32(), self.f32(), self.f32()
        return (z, x, y)  # ToZXY: XYZ (MCLA) -> ZXY (regular coordinates)

    def vec4_zxyw(self) -> tuple[float, float, float, float]:
        x, y, z, w = self.f32(), self.f32(), self.f32(), self.f32()
        if w != w:  # NaN
            w = 0.0
        return (z, x, y, w)

    def read_ptr_pos(self) -> int:
        """Reads a 4-byte pointer (raw position), without dereferencing it."""
        return self.u32()

    def read_block(self, pos: int, parse_func, cache=True):
        """Analogous to Rsc5DataReader.ReadBlock<T>(position): follows the
        pointer, parses the block once (cached by position) and restores
        the cursor's current position afterwards."""
        if pos == 0 or pos == 0xCDCDCDCD:
            return None
        if cache and pos in self.block_cache:
            return self.block_cache[pos]
        saved = self.pos
        self.pos = pos
        try:
            result = parse_func(self)
        finally:
            self.pos = saved
        if cache:
            self.block_cache[pos] = result
        return result

    def read_str(self, pos: int) -> str:
        if pos == 0:
            return ""
        return self.cstr_at(pos)


# --------------------------------------------------------------------------
# Vertex format (rage::grcFvf) — Rsc5VertexDeclaration
# --------------------------------------------------------------------------

# Rsc5VertexComponentType -> (bytes_size, decoder_key)
COMPONENT_SIZE = {
    0: 2, 1: 4, 2: 6, 3: 8, 4: 4, 5: 8, 6: 12, 7: 16,
    8: 4, 9: 4, 10: 4, 11: 2, 12: 4, 13: 2, 14: 4, 15: 8,
}

SEM_POSITION = 0
SEM_NORMAL = 3
SEM_TEXCOORD0 = 6
SEM_TEXCOORD1 = 7  # extra UV channel (decals/grime/puddles use it for a local
                    # area of the texture, rather than one stretched over the whole surface)


def dec3n_to_xyz(u: int) -> tuple[float, float, float]:
    # Rpf3Crypto.Dec3NToVector4, signed 10-bit values (sign-extend)
    def sext10(v: int) -> int:
        v &= 0x3FF
        if v & 0x200:
            v -= 0x400
        return v

    ux = sext10(u & 0x3FF)
    uy = sext10((u >> 10) & 0x3FF)
    uz = sext10((u >> 20) & 0x3FF)
    scale = 0.001956947162
    return (ux * scale, uy * scale, uz * scale)


@dataclass
class VertexLayout:
    fvf: int
    types: int  # 16 x 4-bit nibbles (Rsc5VertexComponentType)
    stride: int

    def channel_present(self, sem: int) -> bool:
        return ((self.fvf >> sem) & 1) != 0

    def channel_type(self, sem: int) -> int:
        return (self.types >> (sem * 4)) & 0xF

    def channel_offset(self, sem: int) -> int:
        # Rsc5VertexDeclaration.GetComponentOffset
        offset = 0
        for k in range(sem):
            if (self.fvf >> k) & 1:
                offset += COMPONENT_SIZE.get(self.channel_type(k), 0)
        return offset


def parse_vertex_declaration(r: Rsc5Reader) -> VertexLayout:
    fvf = r.u32()
    fvf_size = r.u8()
    _flags = r.u8()
    _dyn_order = r.u8()
    _chan_count = r.u8()
    types = r.u64()
    return VertexLayout(fvf=fvf, types=types, stride=fvf_size)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@dataclass
class Geometry:
    name: str
    material: str = "default"
    texture: "Texture | None" = None          # material's primary (diffuse) texture
    normal_texture: "Texture | None" = None    # material's normal/bump texture, if any
    other_textures: list = field(default_factory=list)  # decals/grime/extra material layers
    draw_bucket: int = 0   # see ShaderInfo.draw_bucket
    positions: list = field(default_factory=list)
    normals: list = field(default_factory=list)
    uvs: list = field(default_factory=list)
    uvs1: list = field(default_factory=list)  # extra UV channel (TEXCOORD1) — the real,
                                               # local unwrap of decals/grime/puddles
    indices: list = field(default_factory=list)  # flat list (3 per triangle)


def parse_vertex_buffer(r: Rsc5Reader):
    """Rsc5VertexBuffer.Read -> (raw_vertex_bytes, VertexLayout, vertex_count)"""
    _vft = r.u32()
    vcount = r.u16()
    _locked = r.u8()
    _flags = r.u8()
    locked_data_ptr = r.read_ptr_pos()
    stride = r.u32()
    layout_ptr = r.read_ptr_pos()
    _lock_thread = r.u32()
    vertex_data_ptr = r.read_ptr_pos()
    _d3d_ptr = r.read_ptr_pos()

    layout = r.read_block(layout_ptr, parse_vertex_declaration)
    if layout is None:
        return None, None, 0

    nbytes = vcount * layout.stride
    data_ptr = locked_data_ptr if locked_data_ptr != 0 else vertex_data_ptr
    if data_ptr == 0 or nbytes == 0:
        return b"", layout, vcount

    saved = r.pos
    r.pos = data_ptr
    raw = r.bytes(nbytes)
    r.pos = saved
    return raw, layout, vcount


def parse_index_buffer(r: Rsc5Reader):
    _vft = r.u32()
    icount = r.u32()
    indices_ptr = r.read_ptr_pos()
    _d3d_ptr = r.read_ptr_pos()

    if indices_ptr == 0 or icount == 0:
        return []
    saved = r.pos
    r.pos = indices_ptr
    idx = [r.u16() for _ in range(icount)]
    r.pos = saved
    return idx


def decode_vertices(raw: bytes, layout: VertexLayout, vcount: int):
    positions, normals, uvs, uvs1 = [], [], [], []
    has_pos = layout.channel_present(SEM_POSITION)
    has_nrm = layout.channel_present(SEM_NORMAL)
    has_uv = layout.channel_present(SEM_TEXCOORD0)
    has_uv1 = layout.channel_present(SEM_TEXCOORD1)

    pos_off = layout.channel_offset(SEM_POSITION) if has_pos else 0
    pos_type = layout.channel_type(SEM_POSITION) if has_pos else -1
    nrm_off = layout.channel_offset(SEM_NORMAL) if has_nrm else 0
    nrm_type = layout.channel_type(SEM_NORMAL) if has_nrm else -1
    uv_off = layout.channel_offset(SEM_TEXCOORD0) if has_uv else 0
    uv_type = layout.channel_type(SEM_TEXCOORD0) if has_uv else -1
    uv1_off = layout.channel_offset(SEM_TEXCOORD1) if has_uv1 else 0
    uv1_type = layout.channel_type(SEM_TEXCOORD1) if has_uv1 else -1

    stride = layout.stride
    for i in range(vcount):
        base = i * stride
        if base + stride > len(raw):
            break

        if has_pos:
            o = base + pos_off
            if pos_type == 6:  # Float3
                x, y, z = struct.unpack_from(">fff", raw, o)
                positions.append((z, x, y))
            elif pos_type == 7:  # Float4
                x, y, z, w = struct.unpack_from(">ffff", raw, o)
                positions.append((z, x, y))
            else:
                positions.append((0.0, 0.0, 0.0))

        if has_nrm:
            o = base + nrm_off
            if nrm_type == 10:  # Dec3N
                (u,) = struct.unpack_from(">I", raw, o)
                nx, ny, nz = dec3n_to_xyz(u)
                normals.append((nz, nx, ny))
            elif nrm_type == 6:  # Float3
                x, y, z = struct.unpack_from(">fff", raw, o)
                normals.append((z, x, y))
            else:
                normals.append(None)

        if has_uv:
            o = base + uv_off
            if uv_type == 1:  # Half2
                u, v = struct.unpack_from(">ee", raw, o)
                uvs.append((u, v))
            elif uv_type == 5:  # Float2
                u, v = struct.unpack_from(">ff", raw, o)
                uvs.append((u, v))
            else:
                uvs.append(None)

        if has_uv1:
            o = base + uv1_off
            if uv1_type == 1:  # Half2
                u, v = struct.unpack_from(">ee", raw, o)
                uvs1.append((u, v))
            elif uv1_type == 5:  # Float2
                u, v = struct.unpack_from(">ff", raw, o)
                uvs1.append((u, v))
            else:
                uvs1.append(None)

    return positions, normals, uvs, uvs1


# --------------------------------------------------------------------------
# Textures (Rsc5Texture / Rsc5TextureDictionary) — Xbox 360 D3D + swizzling
# --------------------------------------------------------------------------

TEXFMT_L8 = 2
TEXFMT_DXT1 = 82
TEXFMT_DXT3 = 83
TEXFMT_DXT5 = 84
TEXFMT_A8R8G8B8 = 134

# Rsc5Shader.DrawBucket -> ShaderBucket (see Rsc5Drawable.cs): values that
# the engine renders with alpha blending ON THE DIFFUSE texture —
# 1=Alpha, 3=Cutout, 6=Water, 7=Glass (this includes, among others, the
# CityWindow* shaders, which use DXT1/A8R8G8B8, not just DXT3/DXT5).
# 2=Decal is intentionally NOT included here — the engine draws decals
# differently (a polygon on top of other geometry), and by default they
# already have their own handler.
ALPHA_DRAW_BUCKETS = {1, 3, 6, 7}

# see the detailed comment on unswizzle_xbox360(): in one verified .xcs
# dump the textures turned out to already be linear, and Xbox 360
# detiling broke them — so this used to be disabled by default. Per the
# user's explicit request it is now ENABLED by default (matches the
# original CodeX/Rpf3Crypto.UnswizzleXbox360Data logic). If textures come
# out noisy, disable it with the --no-xbox-tiled flag.
UNSWIZZLE_TEXTURES = True

_TEXFMT_NAMES = {
    TEXFMT_L8: "L8",
    TEXFMT_DXT1: "DXT1",
    TEXFMT_DXT3: "DXT3",
    TEXFMT_DXT5: "DXT5",
    TEXFMT_A8R8G8B8: "A8R8G8B8",
}


@dataclass
class Texture:
    name: str
    width: int
    height: int
    fmt: int
    data: bytes


def tex_virtual_dim(size: int) -> int:
    # Rpf3Crypto.GetVirtualSize(int size) — rounds a texture dimension
    if (size % 128 != 0) and size < 128:
        return 128
    return size


def calc_texture_data_size(fmt: int, width: int, height: int) -> int:
    # Rsc5Texture.CalcDataSize
    if fmt == TEXFMT_DXT1:
        return width * height // 2
    if fmt in (TEXFMT_DXT3, TEXFMT_DXT5, TEXFMT_A8R8G8B8, TEXFMT_L8):
        return width * height
    raise ValueError(f"Unsupported texture format: 0x{fmt:X}")


def xg_address_2d_tiled_x(offset: int, width: int, texel_pitch: int) -> int:
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


def xg_address_2d_tiled_y(offset: int, width: int, texel_pitch: int) -> int:
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


def trim_virtual_texture(data: bytes, width: int, height: int, fmt: int) -> bytes:
    """Trims the texture buffer from its "virtual" size (padded to at least
    128 on at least one side — see tex_virtual_dim/Rpf3Crypto.GetVirtualSize)
    down to the actual width x height.

    This is UNRELATED to Xbox 360 tiling/detiling (see UNSWIZZLE_TEXTURES) —
    it is a separate step that is ALWAYS required: textures smaller than
    128 pixels on at least one side are physically allocated and stored
    with padding up to 128, and the per-row size (pitch) of the data block
    that was read corresponds to the PADDED width, not the real one. If
    this isn't trimmed, the resulting .dds will have a header with the
    real width/height, but data whose blockPitch was computed for the
    larger (padded) width — i.e. each row of blocks "slides" relative to
    what's expected, producing the characteristic noise/moire pattern.
    Previously this trimming was hidden inside unswizzle_xbox360 and
    stopped running whenever that step was disabled (see
    UNSWIZZLE_TEXTURES=False) — because of this, every texture smaller
    than 128 on a side (roughly a third of all sector textures) came out
    corrupted."""
    if fmt in (TEXFMT_L8, TEXFMT_A8R8G8B8):
        return data  # these formats are read without block padding

    if fmt == TEXFMT_DXT1:
        block_size_row, texel_pitch = 4, 8
    elif fmt in (TEXFMT_DXT3, TEXFMT_DXT5):
        block_size_row, texel_pitch = 4, 16
    else:
        return data

    virtual_width = tex_virtual_dim(width)
    virtual_height = tex_virtual_dim(height)
    if virtual_width == width and virtual_height == height:
        return data  # no padding was applied — nothing to trim

    virtual_block_width = virtual_width // block_size_row
    actual_block_width = width // block_size_row
    actual_block_height = height // block_size_row

    trimmed = bytearray(actual_block_width * actual_block_height * texel_pitch)
    for j in range(actual_block_height):
        src_offset = j * virtual_block_width * texel_pitch
        dst_offset = j * actual_block_width * texel_pitch
        n = actual_block_width * texel_pitch
        if src_offset + n > len(data):
            break
        trimmed[dst_offset:dst_offset + n] = data[src_offset:src_offset + n]
    return bytes(trimmed)


def unswizzle_xbox360(data: bytes, width: int, height: int, fmt: int) -> bytes:
    # Rpf3Crypto.UnswizzleXbox360Data
    #
    # By default this function IS CALLED (see UNSWIZZLE_TEXTURES below) —
    # matches the original CodeX logic for Xbox 360 tiling/detiling (byte
    # swap every 2 bytes + rearranging DXT blocks per the
    # XGAddress2DTiled* formula). On one of the verified .xcs dumps the
    # textures turned out to already be linear (not tiled), and in that
    # case applying this function BREAKS the data (noise/"green static"
    # instead of a picture) — if that happens, disable this step with the
    # --no-xbox-tiled flag.
    if fmt in (TEXFMT_L8, TEXFMT_A8R8G8B8):
        return data

    if fmt == TEXFMT_DXT1:
        block_size_row, texel_pitch = 4, 8
    elif fmt in (TEXFMT_DXT3, TEXFMT_DXT5):
        block_size_row, texel_pitch = 4, 16
    else:
        raise ValueError(f"Swizzling not supported for format 0x{fmt:X}")

    data = bytearray(data)
    # Swap every two bytes
    for i in range(0, len(data) - 1, 2):
        data[i], data[i + 1] = data[i + 1], data[i]

    virtual_width = tex_virtual_dim(width)
    virtual_height = tex_virtual_dim(height)
    virtual_block_width = virtual_width // block_size_row
    virtual_block_height = virtual_height // block_size_row

    unswizzled = bytearray(len(data))
    for j in range(virtual_block_height):
        for i in range(virtual_block_width):
            block_offset = j * virtual_block_width + i
            x = xg_address_2d_tiled_x(block_offset, virtual_block_width, texel_pitch)
            y = xg_address_2d_tiled_y(block_offset, virtual_block_width, texel_pitch)

            src_offset = j * virtual_block_width * texel_pitch + i * texel_pitch
            dst_offset = y * virtual_block_width * texel_pitch + x * texel_pitch
            if src_offset + texel_pitch > len(data) or dst_offset + texel_pitch > len(unswizzled):
                continue
            unswizzled[dst_offset:dst_offset + texel_pitch] = data[src_offset:src_offset + texel_pitch]

    if width < 128 or height < 128:
        actual_block_width = width // block_size_row
        actual_block_height = height // block_size_row
        trimmed = bytearray(actual_block_width * actual_block_height * texel_pitch)
        for j in range(actual_block_height):
            src_offset = j * virtual_block_width * texel_pitch
            dst_offset = j * actual_block_width * texel_pitch
            n = actual_block_width * texel_pitch
            trimmed[dst_offset:dst_offset + n] = unswizzled[src_offset:src_offset + n]
        unswizzled = trimmed

    return bytes(unswizzled)


def get_base_address_from_d3d(d3d_value: int, virtual_size: int) -> int:
    # Rpf3Crypto.GetBaseAdressFromDirect3D
    base_addresses = {0x50: 0x50000000, 0x51: 0x51000000, 0x52: 0x52000000}
    d3d_value &= 0xFFFFFFFF
    is_phys = (d3d_value & PHYSICAL_BASE) == PHYSICAL_BASE
    top_byte = (d3d_value >> 24) & 0xFF
    fmt_byte = d3d_value & 0xFF

    mapped_base = base_addresses.get(top_byte, 0)
    if mapped_base == 0:
        mapped_base = VIRTUAL_BASE

    return (d3d_value & 0xFFFFFF) - fmt_byte + (virtual_size if is_phys else 0) + mapped_base


def parse_texture(r: Rsc5Reader) -> "Texture | None":
    """Rsc5TextureBase.Read + Rsc5Texture.Read"""
    r.u32()  # VFT
    r.u32()  # Unknown_4h
    r.u32()  # Unknown_8h
    r.u32()  # Unknown_Ch
    r.u32()  # Unknown_10h
    r.u32()  # Unknown_14h

    name_ptr = r.u32()  # Rsc5Str TextureName
    name = r.read_str(name_ptr) if name_ptr else ""
    if name:
        name = name.replace(".dds", "").replace("pack:/", "")

    d3dbase_ptr = r.read_ptr_pos()  # Rsc5Ptr<Rsc5BlockMap> D3DBaseTexture
    if not d3dbase_ptr:
        # A "stub" — a texture with no pixel data of its own, just a name
        # (see Rsc5TextureBase.Read in CodeX: when D3DBaseTexture.Item ==
        # null, Width/Height/Data are not read, but Name is still
        # assigned). For a City Sector this is a NORMAL situation: meshes/
        # shaders most often reference textures only by name, and the
        # actual pixel data lives in the sector's shared texture
        # dictionary (Rsc5TextureDictionary) and gets substituted later —
        # see resolve_stub_textures() (equivalent to the original
        # Rsc5CitySectorPiece.ApplyTextures()).
        # This used to just `return None`, which lost the name and made it
        # impossible to later look up the real texture in the dictionary —
        # because of this, materials never got a texture reference at all.
        if not name:
            return None  # nothing to save at all — no data, no name
        return Texture(name=name, width=0, height=0, fmt=0, data=b"")

    width = r.u16()
    height = r.u16()

    r.u16()  # Stride
    r.u8()   # TextureType
    r.u8()   # MipLevels
    r.f32(); r.f32(); r.f32()  # ColorExpR/G/B
    r.f32(); r.f32(); r.f32()  # ColorOfsR/G/B

    d3d_pos = d3dbase_ptr + 0x20
    saved = r.pos
    r.pos = d3d_pos
    d3d_value = r.u32()
    r.pos = saved

    fmt = d3d_value & 0xFF
    virtual_w = tex_virtual_dim(width)
    virtual_h = tex_virtual_dim(height)
    base_addr = get_base_address_from_d3d(d3d_value, r.virtual_size)

    try:
        size = calc_texture_data_size(fmt, virtual_w, virtual_h)
        r.pos = base_addr
        raw = r.bytes(size)
        if UNSWIZZLE_TEXTURES:
            raw = unswizzle_xbox360(raw, width, height, fmt)
        else:
            raw = trim_virtual_texture(raw, width, height, fmt)
    except (ValueError, struct.error, IndexError) as exc:
        print(f"  [!] Failed to read texture '{name}': {exc}", file=sys.stderr)
        return None

    if not name:
        name = f"tex_{d3dbase_ptr:08X}"
    return Texture(name=name, width=width, height=height, fmt=fmt, data=raw)


def parse_texture_dictionary(r: Rsc5Reader,
                              strings_table: dict[int, str] | None = None) -> tuple[list[Texture], dict[int, Texture]]:
    """Rsc5TextureDictionary.Read.

    Returns (textures, hash_map). hash_map is an exact analog of the C#
    field Rsc5TextureDictionary.Dict: {hash (Rsc5Arr<JenkHash> Hashes[i])
    -> texture (Textures[i])}. This used to not read the Hashes array at
    all ("not used"), even though the engine looks up a mesh's texture by
    exactly this hash in Rsc5CitySectorPiece.ApplyTextures — without it,
    there's no way to match the geometry's texture references (see
    parse_texture/resolve_stub_textures) with the real data from the
    dictionary, since the dictionary entries' own TextureName is very
    often empty (see Rsc5TextureDictionary.Read in CodeX: `t.Name ??=
    h.ToString()` — the name is restored FROM the hash, not the other way
    around).

    If strings_table is passed (see load_strings_table — usually
    Codex.Games.MCLA.strings.txt), dictionary entries whose TextureName is
    empty OR is a generated stub ("tex_...") are given a real name found
    by looking up Hashes[i] in that table — instead of a meaningless
    numeric/address-based name."""
    r.u32()          # VFT
    r.read_ptr_pos()  # BlockMap ptr (Rsc5BlockBaseMap)
    r.u32()          # ParentDictionary
    r.u32()          # UsageCount
    hashes_ptr = r.u32()
    hashes_count = r.u16()
    r.u16()  # capacity
    tex_ptr = r.u32()
    tex_count = r.u16()
    r.u16()  # capacity

    hashes: list[int] = []
    if hashes_ptr and hashes_count:
        saved = r.pos
        r.pos = hashes_ptr
        hashes = [r.u32() for _ in range(hashes_count)]
        r.pos = saved

    textures: list[Texture] = []
    hash_map: dict[int, Texture] = {}
    if tex_ptr and tex_count:
        saved = r.pos
        r.pos = tex_ptr
        ptrs = [r.u32() for _ in range(tex_count)]
        r.pos = saved
        for i, p in enumerate(ptrs):
            if not p:
                continue
            try:
                tex = r.read_block(p, parse_texture, cache=True)
            except (ValueError, struct.error, IndexError) as exc:
                print(f"  [!] Error reading a texture from the dictionary: {exc}", file=sys.stderr)
                tex = None
            if tex is not None:
                if i < len(hashes):
                    h = hashes[i]
                    if strings_table and (not tex.name or tex.name.startswith("tex_")):
                        resolved = strings_table.get(h)
                        if resolved:
                            tex.name = resolved[:-4] if resolved.lower().endswith(".dds") else resolved
                    if not tex.name:
                        # As in the original engine (`t.Name ??= h.ToString()`) —
                        # a fallback for when even strings.txt didn't help.
                        tex.name = f"hash_{h:08X}"
                textures.append(tex)
                if i < len(hashes):
                    hash_map[hashes[i]] = tex
    return textures, hash_map


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    return name or "texture"


def decal_material_name(base_mat_name: str, tex_name: str) -> str:
    """The .mtl material name for a separate decal layer (see write_obj/
    export_decals) — one material per extra texture (decalsampler/
    grimesampler/puddlesampler etc.); the name includes the base material
    name and the texture name so it doesn't collide with others."""
    return f"{base_mat_name}__decal_{_sanitize_filename(tex_name)}"


def build_dds_bytes(tex: Texture) -> bytes:
    """Builds the contents of a .dds file (128-byte header + data for one
    mip level) in memory, without touching disk. Factored out of
    write_dds() so the same code can be reused to decode a texture "on the
    fly" via Pillow (see defilter_channelmap_texture / --fix-channelmap) —
    without an intermediate file."""
    DDS_MAGIC = b"DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PITCH = 0x8
    DDSD_PIXELFORMAT = 0x1000
    DDSD_LINEARSIZE = 0x80000
    DDPF_ALPHAPIXELS = 0x1
    DDPF_FOURCC = 0x4
    DDPF_RGB = 0x40
    DDPF_LUMINANCE = 0x20000
    DDSCAPS_TEXTURE = 0x1000

    w, h = tex.width, tex.height

    if tex.fmt in (TEXFMT_DXT1, TEXFMT_DXT3, TEXFMT_DXT5):
        fourcc = {TEXFMT_DXT1: b"DXT1", TEXFMT_DXT3: b"DXT3", TEXFMT_DXT5: b"DXT5"}[tex.fmt]
        block_size = 8 if tex.fmt == TEXFMT_DXT1 else 16
        pitch = max(1, (w + 3) // 4) * block_size
        flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
        # DDPF_ALPHAPIXELS is added for block-compressed formats too: the
        # FourCC itself (DXT1/3/5) doesn't guarantee that a reading program
        # will look for an alpha channel — some importers (including
        # 3ds Max) check exactly this flag before even decoding the alpha
        # blocks. DXT1 carries an optional 1-bit alpha, DXT3/DXT5 always
        # carry a full/interpolated one, so the flag is correct in all
        # three cases.
        pf_flags = DDPF_FOURCC | DDPF_ALPHAPIXELS
        pf_fourcc = fourcc
        rgb_bitcount = 0
        r_mask = g_mask = b_mask = a_mask = 0
    elif tex.fmt == TEXFMT_A8R8G8B8:
        flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_PITCH
        pitch = w * 4
        pf_flags = DDPF_RGB | DDPF_ALPHAPIXELS
        pf_fourcc = b"\x00\x00\x00\x00"
        rgb_bitcount = 32
        r_mask, g_mask, b_mask, a_mask = 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000
    elif tex.fmt == TEXFMT_L8:
        flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_PITCH
        pitch = w
        pf_flags = DDPF_LUMINANCE
        pf_fourcc = b"\x00\x00\x00\x00"
        rgb_bitcount = 8
        r_mask, g_mask, b_mask, a_mask = 0xFF, 0, 0, 0
    else:
        raise ValueError(f"Unknown texture format for DDS: 0x{tex.fmt:X}")

    header = _build_dds_header(w, h, flags, pitch, pf_flags, pf_fourcc, rgb_bitcount,
                                r_mask, g_mask, b_mask, a_mask)

    return header + tex.data


def write_dds(tex: Texture, out_path: str) -> None:
    """Writes a minimal valid DDS header (128 bytes) + data for one mip
    level — enough for the file to open in most 3D editors (Blender,
    Maya, GIMP, Photoshop with a DDS plugin, etc.)."""
    with open(out_path, "wb") as f:
        f.write(build_dds_bytes(tex))


# --------------------------------------------------------------------------
# --fix-channelmap: remove the "green filter" from a material's primary
# (diffuse) texture — ported from fix_channelmap_texture.py ("preview" mode)
# --------------------------------------------------------------------------

def _import_channelmap_deps():
    """Lazy import of numpy/Pillow — only needed by those who enable
    --fix-channelmap; the rest of the script doesn't depend on them."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "--fix-channelmap requires the numpy and Pillow packages "
            "(pip install numpy pillow) — same as fix_channelmap_texture.py, "
            "whose logic ('preview') this flag replicates."
        ) from exc
    return np, Image


def defilter_channelmap_texture(tex: Texture, source_channel: str = "g") -> "Texture | None":
    """Removes the artificial green tint from a texture assigned as a
    material's PRIMARY (diffuse, map_Kd) texture.

    This is the same case as in fix_channelmap_texture.py: the R/G/B
    channels of a channel-map texture (channelmapsampler, CityGrass/
    CityTerrain/CityRoad shaders etc.) actually store NOT color, but
    independent grayscale masks of the blend weights for three diffuse
    layers (diffusesamplera/b/c), which is why, when viewed normally as
    RGB, the texture looks green (the G channel dominates). If such a
    texture ended up mistakenly assigned as the main diffuse texture of a
    material (instead of as a decal/extra layer), --fix-channelmap
    replaces it with the same "preview" transform: a plain grayscale image
    based on source_channel ("g" — G channel only, "max"/"avg" — across
    all three), instead of a "proper" blend of the three original diffuse
    textures (that would need the separate "blend" mode from
    fix_channelmap_texture.py — not implemented here, since at the point
    this function is called the three original diffusesamplera/b/c
    textures haven't yet been matched to each other at the material
    level).

    Decodes compressed (DXT1/3/5) or other texture data via Pillow (the
    same method fix_channelmap_texture.py uses to read .dds), and returns
    a NEW Texture in uncompressed A8R8G8B8 format with the same name —
    which should then be substituted for the original on the material.
    Returns None if the texture has no pixel data or it couldn't be
    decoded (in which case the original texture is left as-is)."""
    if not tex.data or not tex.width or not tex.height:
        return None

    np, Image = _import_channelmap_deps()

    import io
    try:
        img = Image.open(io.BytesIO(build_dds_bytes(tex))).convert("RGBA")
        arr = np.array(img)
    except Exception as exc:
        print(f"  [!] --fix-channelmap: failed to decode texture "
              f"'{tex.name}': {exc}", file=sys.stderr)
        return None

    r, g, b, a = (arr[..., i] for i in range(4))
    if source_channel == "g":
        gray = g
    elif source_channel == "max":
        gray = np.maximum(np.maximum(r, g), b)
    elif source_channel == "avg":
        gray = ((r.astype(np.uint16) + g.astype(np.uint16) + b.astype(np.uint16)) // 3).astype(np.uint8)
    else:
        raise ValueError(f"Unknown source_channel: {source_channel}")

    h, w = gray.shape[:2]
    # Pack as A8R8G8B8 using the same byte order write_dds uses for this
    # format (masks r=0x00FF0000 g=0x0000FF00 b=0x000000FF a=0xFF000000
    # -> byte order in the file B,G,R,A):
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[..., 0] = gray  # B
    bgra[..., 1] = gray  # G
    bgra[..., 2] = gray  # R
    bgra[..., 3] = a     # A

    return Texture(name=tex.name, width=w, height=h, fmt=TEXFMT_A8R8G8B8, data=bgra.tobytes())


def _build_dds_header(w, h, flags, pitch, pf_flags, pf_fourcc, rgb_bitcount,
                       r_mask, g_mask, b_mask, a_mask) -> bytes:
    parts = []
    parts.append(b"DDS ")
    parts.append(struct.pack("<I", 124))       # dwSize
    parts.append(struct.pack("<I", flags))
    parts.append(struct.pack("<I", h))
    parts.append(struct.pack("<I", w))
    parts.append(struct.pack("<I", pitch))
    parts.append(struct.pack("<I", 0))          # dwDepth
    parts.append(struct.pack("<I", 1))          # dwMipMapCount
    parts.append(struct.pack("<11I", *([0] * 11)))  # dwReserved1
    # DDS_PIXELFORMAT (32 bytes)
    parts.append(struct.pack("<I", 32))         # dwSize
    parts.append(struct.pack("<I", pf_flags))
    parts.append(pf_fourcc if len(pf_fourcc) == 4 else b"\x00\x00\x00\x00")
    parts.append(struct.pack("<I", rgb_bitcount))
    parts.append(struct.pack("<I", r_mask))
    parts.append(struct.pack("<I", g_mask))
    parts.append(struct.pack("<I", b_mask))
    parts.append(struct.pack("<I", a_mask))
    # caps
    parts.append(struct.pack("<I", 0x1000))     # dwCaps = DDSCAPS_TEXTURE
    parts.append(struct.pack("<I", 0))          # dwCaps2
    parts.append(struct.pack("<I", 0))          # dwCaps3
    parts.append(struct.pack("<I", 0))          # dwCaps4
    parts.append(struct.pack("<I", 0))          # dwReserved2
    data = b"".join(parts)
    assert len(data) == 128, f"DDS header size mismatch: {len(data)}"
    return data


# --------------------------------------------------------------------------
# Top-level RSC5 structures (CitySector -> Piece -> Model -> Geometry)
# --------------------------------------------------------------------------

# Sampler name hashes (32-bit hash of the lowercased shader parameter name,
# ParamsNames[i] in Rsc5Shader.Read) -> texture "role". The table was built
# from all the Setup*Shader() functions in
# CodeX.Games.MCLA/RSC5/Rsc5Drawable.cs (SetupDefaultShader,
# SetupDecalShader, SetupDecal2Shader, SetupDecalGrimeShader,
# SetupGrassTerrainShader, SetupTerrainShader, SetupWaterShader) — that's
# where these exact same samplers get sorted into Textures[0..N] slots
# depending on their name.
TEXROLE_DIFFUSE = "diffuse"
TEXROLE_NORMAL = "normal"
TEXROLE_OTHER = "other"

_SAMPLER_ROLE = {
    # diffuse / base samplers -> map_Kd
    0x3C870418: TEXROLE_DIFFUSE,  # neonsampler
    0xF1FE2B71: TEXROLE_DIFFUSE,  # diffusesampler
    0x50022388: TEXROLE_DIFFUSE,  # platebgsampler
    0x1CF5B657: TEXROLE_DIFFUSE,  # texturesamp
    0x605FCC60: TEXROLE_DIFFUSE,  # distancemapsampler
    0x2B5170FD: TEXROLE_DIFFUSE,  # texturesampler
    0x3E19076B: TEXROLE_DIFFUSE,  # detailmapsampler
    0xC9A79FED: TEXROLE_DIFFUSE,  # diffusesamplera
    0xF7357B08: TEXROLE_DIFFUSE,  # diffusesamplerb
    0xA4CFD63A: TEXROLE_DIFFUSE,  # diffusesamplerc
    # normal / bump / height -> map_bump
    0x46B7C64F: TEXROLE_NORMAL,   # bumpsampler
    0x65DF0BCE: TEXROLE_NORMAL,   # platebgbumpsampler
    0x8AC11CB0: TEXROLE_NORMAL,   # normalsampler
    0x332EBE1C: TEXROLE_NORMAL,   # heightspecularsampler
    0xBE97CA14: TEXROLE_NORMAL,   # normalsamplera
    0xCCDCE69E: TEXROLE_NORMAL,   # normalsamplerb
    0xA8419D68: TEXROLE_NORMAL,   # normalsamplerc
    0x00A67ACD: TEXROLE_NORMAL,   # (ripple) normal map
    # decals / grime / extra layers of terrain shaders -> decal / extra slots
    0xE3381C99: TEXROLE_OTHER,    # grimesampler
    0xA79AEEC0: TEXROLE_OTHER,    # decalsampler
    0xFE553678: TEXROLE_OTHER,    # puddlesampler
    0x8BFCEF8D: TEXROLE_OTHER,    # channelmapsampler
    0x63F7C0E8: TEXROLE_OTHER,    # wavefoamsampler
    0xC2B08918: TEXROLE_OTHER,    # foamsampler
}


@dataclass
class ShaderInfo:
    name: str
    diffuse: list = field(default_factory=list)       # list[Texture], role "diffuse"
    normal: list = field(default_factory=list)         # list[Texture], role "normal"/bump
    other: list = field(default_factory=list)          # list[Texture], decals/grime/extra layers
    unclassified: list = field(default_factory=list)   # list[Texture], role not recognized
    draw_bucket: int = 0   # Rsc5Shader.DrawBucket — see ShaderBucket in
                            # Rsc5Drawable.cs: 1=Alpha, 2=Decal, 3=Cutout,
                            # 6=Water, 7=Glass. Materials with these buckets
                            # are rendered by the engine with alpha blending
                            # ON THE DIFFUSE texture regardless of its format
                            # (DXT1 with a 1-bit alpha channel, A8R8G8B8,
                            # etc.) — this is used, for example, by the
                            # CityWindow* shaders.

    @property
    def primary_diffuse(self) -> "Texture | None":
        if self.diffuse:
            return self.diffuse[0]
        if self.unclassified:  # unknown sampler — better to show something than nothing
            return self.unclassified[0]
        return None

    @property
    def primary_normal(self) -> "Texture | None":
        return self.normal[0] if self.normal else None

    @property
    def primary_texture(self) -> "Texture | None":  # backward compatibility
        return self.primary_diffuse

    @property
    def all_textures(self) -> list:
        return self.diffuse + self.normal + self.other + self.unclassified


def parse_shader(r: Rsc5Reader) -> ShaderInfo:
    # Rsc5Shader.Read
    r.u32()               # VFT
    r.u32()                # BlockMapAdress
    r.u8()                 # Version
    draw_bucket = r.u8()   # DrawBucket — see ShaderInfo.draw_bucket above
    r.u8(); r.u8()          # UsageCount, Unknown1
    r.u16()                # Unknown2
    r.u16()                # ShaderIndex
    params_data_ptr = r.read_ptr_pos()   # ParamsData ptr
    r.u32()                 # Unknown3
    params_count = r.u16()
    r.u16()                 # EffectSize
    params_types_ptr = r.read_ptr_pos()  # ParamsTypes ptr
    r.u32()                  # Hash
    params_names_ptr = r.read_ptr_pos()   # ParamsNames ptr — a parallel array of
                                           # 32-bit sampler-name hashes (Rsc5Shader.Params[i].Hash
                                           # in the original), one per parameter, same as ParamsData/ParamsTypes
    r.u32(); r.u32()           # Unknown4, Unknown5
    name_ptr = r.u32()
    name = r.read_str(name_ptr) if name_ptr else ""
    name = name or "shader"

    info = ShaderInfo(name=name, draw_bucket=draw_bucket)
    if params_count and params_data_ptr and params_types_ptr:
        saved = r.pos
        r.pos = params_data_ptr
        data_ptrs = [r.u32() for _ in range(params_count)]
        r.pos = params_types_ptr
        types = [r.u8() for _ in range(params_count)]
        names = [0] * params_count
        if params_names_ptr:
            r.pos = params_names_ptr
            names = [r.u32() for _ in range(params_count)]
        r.pos = saved

        for i in range(params_count):
            if types[i] == 0 and data_ptrs[i]:  # 0 == texture parameter
                try:
                    tex = r.read_block(data_ptrs[i], parse_texture, cache=True)
                except (ValueError, struct.error, IndexError) as exc:
                    print(f"  [!] Error reading a shader texture '{name}': {exc}", file=sys.stderr)
                    tex = None
                if tex is None:
                    continue
                role = _SAMPLER_ROLE.get(names[i])
                if role == TEXROLE_DIFFUSE:
                    info.diffuse.append(tex)
                elif role == TEXROLE_NORMAL:
                    info.normal.append(tex)
                elif role == TEXROLE_OTHER:
                    info.other.append(tex)
                else:
                    info.unclassified.append(tex)

    return info


def parse_shader_group(r: Rsc5Reader) -> list[ShaderInfo]:
    r.u32()  # VFT
    r.read_ptr_pos()  # BlockMap ptr
    ptrs_pos = r.u32()
    count = r.u16()
    _cap = r.u16()
    infos: list[ShaderInfo] = []
    if ptrs_pos and count:
        saved = r.pos
        r.pos = ptrs_pos
        ptrs = [r.u32() for _ in range(count)]
        r.pos = saved
        for p in ptrs:
            info = r.read_block(p, parse_shader) if p else None
            infos.append(info or ShaderInfo(name="shader"))
    return infos


def parse_geometry(r: Rsc5Reader, idx: int, shader_infos: list[ShaderInfo], shader_id: int) -> Geometry:
    r.u32(); r.u32(); r.u32()  # VFT, Unknown_4h, Unknown_8h
    vb_ptr = r.read_ptr_pos()
    r.read_ptr_pos(); r.read_ptr_pos(); r.read_ptr_pos()  # VertexBuffer 2/3/4 (unused)
    ib_ptr = r.read_ptr_pos()
    r.read_ptr_pos(); r.read_ptr_pos(); r.read_ptr_pos()  # IndexBuffer 2/3/4

    r.u32()  # IndicesCount (we use the length from the index buffer itself)
    r.u32()  # TrianglesCount
    r.u16()  # VertexCount (we use it from the vertex buffer)
    r.u16()  # PrimitiveType
    r.read_ptr_pos()  # BoneIds ptr
    r.u16()  # VertexStride
    bone_ids_count = r.u16()
    r.read_ptr_pos()  # VertexDataRef ptr
    r.u32(); r.u32(); r.u32()  # OffsetBuffer, IndexOffset, Unknown_3Ch

    raw, layout, vcount = r.read_block(vb_ptr, parse_vertex_buffer, cache=False) or (b"", None, 0)
    indices = r.read_block(ib_ptr, parse_index_buffer, cache=False) or []

    shader_info = (shader_infos[shader_id]
                   if (shader_infos and 0 <= shader_id < len(shader_infos))
                   else None)
    mat_name = shader_info.name if shader_info else f"geom_{idx}"
    texture = shader_info.primary_diffuse if shader_info else None
    normal_texture = shader_info.primary_normal if shader_info else None
    other_textures = list(shader_info.other) if shader_info else []
    draw_bucket = shader_info.draw_bucket if shader_info else 0

    geo = Geometry(name=f"{mat_name}_{idx}", material=mat_name, texture=texture,
                    normal_texture=normal_texture, other_textures=other_textures,
                    draw_bucket=draw_bucket)
    if layout is not None and raw:
        positions, normals, uvs, uvs1 = decode_vertices(raw, layout, vcount)
        geo.positions = positions
        geo.normals = normals
        geo.uvs = uvs
        geo.uvs1 = uvs1
    geo.indices = indices
    return geo


def parse_model(r: Rsc5Reader, shader_infos: list[ShaderInfo]) -> list[Geometry]:
    r.u32()  # VFT
    geoms_pos = r.u32()
    geoms_count = r.u16()
    r.u16()  # capacity
    bounds_ptr = r.read_ptr_pos()
    shadermap_ptr = r.read_ptr_pos()
    r.u8(); r.u8(); r.u8(); r.u8(); r.u8(); r.u8()  # MatrixCount..SkinFlag
    r.u16()  # GeometriesCount (duplicate of geoms_count)

    shader_map = []
    if shadermap_ptr and geoms_count:
        saved = r.pos
        r.pos = shadermap_ptr
        shader_map = [r.u16() for _ in range(geoms_count)]
        r.pos = saved

    result = []
    if geoms_pos and geoms_count:
        saved = r.pos
        r.pos = geoms_pos
        geom_ptrs = [r.u32() for _ in range(geoms_count)]
        r.pos = saved

        for i, gp in enumerate(geom_ptrs):
            if gp == 0:
                continue
            sid = shader_map[i] if i < len(shader_map) else 0
            geo = r.read_block(gp, lambda rr, ii=i, ss=sid: parse_geometry(rr, ii, shader_infos, ss), cache=False)
            if geo is not None:
                result.append(geo)
    return result


def parse_drawable_lod(r: Rsc5Reader, shader_infos: list[ShaderInfo]) -> list[Geometry]:
    models_pos = r.u32()
    count = r.u16()
    r.u16()  # capacity

    all_geoms: list[Geometry] = []
    if models_pos and count:
        saved = r.pos
        r.pos = models_pos
        model_ptrs = [r.u32() for _ in range(count)]
        r.pos = saved

        for mp in model_ptrs:
            if mp == 0:
                continue
            geoms = r.read_block(mp, lambda rr: parse_model(rr, shader_infos), cache=False)
            if geoms:
                all_geoms.extend(geoms)
    return all_geoms


def parse_drawable_lod_map(r: Rsc5Reader, shader_infos: list[ShaderInfo]) -> list[Geometry]:
    r.u32()  # VFT
    r.read_ptr_pos()  # BlockMap ptr
    r.u32()  # ParentDictionary
    r.u32()  # RefCount
    r.u32()  # Hashes ptr
    r.u16(); r.u16()  # Hashes count/capacity
    # Rsc5DrawableLod is nested directly here (not via a pointer)
    return parse_drawable_lod(r, shader_infos)


def parse_city_sector_piece(r: Rsc5Reader) -> list[Geometry]:
    r.u32()  # VFT
    r.read_ptr_pos()  # BlockMap ptr
    shader_group_ptr = r.u32()
    lod_ptr = r.u32()

    shader_infos = r.read_block(shader_group_ptr, parse_shader_group) if shader_group_ptr else []
    geoms = r.read_block(lod_ptr, lambda rr: parse_drawable_lod_map(rr, shader_infos)) if lod_ptr else []
    return geoms or []


def jenkhash_genhash(text: str) -> int:
    """Jenkins one-at-a-time hash — ported from CodeX.Core.Utilities.JenkHash.GenHash
    (Hashing.cs). Used by the engine to match texture names against
    entries in Rsc5TextureDictionary (see Rsc5CitySectorPiece.ApplyTextures)."""
    h = 0
    for ch in text:
        h = (h + (ord(ch) & 0xFF)) & 0xFFFFFFFF
        h = (h + ((h << 10) & 0xFFFFFFFF)) & 0xFFFFFFFF
        h ^= (h >> 6)
    h = (h + ((h << 3) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h ^= (h >> 11)
    h = (h + ((h << 15) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def load_strings_table(path: str) -> dict[int, str]:
    """Loads a text file containing the engine's string table (e.g.
    Codex.Games.MCLA.strings.txt — a game-wide list of all known strings:
    texture/resource file names, shader sampler names, etc., one string
    per line) and returns {JenkHash(string) -> string}.

    Strings in the file are already stored in the same form the engine
    hashes them in (lowercase, with the extension for ".dds" entries —
    see GenHash in Rpf3Crypto/JenkHash), so the hash is computed from the
    string as-is, without extra normalization. Collisions (two different
    strings with the same hash) are practically nonexistent, but if one
    does occur, the first string read from the file wins.

    This table is used to recover the real names of textures for which
    the .xcs file itself only stores a hash (Rsc5Arr<JenkHash> Hashes in
    Rsc5TextureDictionary) and the TextureName field is empty — in the
    original C# engine the name is never recovered for these either
    (`t.Name ??= h.ToString()` — just a decimal number), but the real name
    can be found precisely via such an external string dictionary."""
    table: dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            h = jenkhash_genhash(s)
            if h not in table:
                table[h] = s
    return table


def normalize_tex_name(name: str) -> str:
    """Rpf3Crypto.NormalizeTexName: lowercase + trim ".dds" off the end."""
    if not name:
        return ""
    n = name.lower()
    if n.endswith(".dds"):
        n = n[:-4]
    return n


def resolve_stub_textures(geoms: list["Geometry"], dict_hash_map: dict[int, Texture]) -> int:
    """Equivalent of Rsc5CitySectorPiece.ApplyTextures(Rsc5TextureDictionary)
    from CodeX (Rsc5Drawable.cs): textures bound to geometry via shader
    parameters very often turn out, in a City Sector, to be mere "stubs" —
    entries with a name but no pixel data of their own (see parse_texture:
    D3DBaseTexture ptr == 0). The real data lives in the sector's shared
    texture dictionary and is matched by the hash of the normalized name
    (name.lower() + ".dds") via dict_hash_map — that same
    Rsc5Arr<JenkHash> Hashes of the dictionary (see
    parse_texture_dictionary) — and NOT by the name of the dictionary
    entry itself: dictionary entries' TextureName is very often empty, and
    in C# their Name is likewise recovered from this same hash (`t.Name
    ??= h.ToString()`), not the other way around. Without this step,
    materials would have not a single texture with actual data bound to
    them, even if the name was read correctly."""
    by_hash = dict_hash_map
    if not by_hash:
        return 0

    resolved_count = 0
    cache: dict[str, "Texture | None"] = {}

    def resolve(tex: "Texture | None") -> "Texture | None":
        nonlocal resolved_count
        if tex is None or tex.data or not tex.name:
            return tex
        if tex.name in cache:
            return cache[tex.name] or tex
        key = jenkhash_genhash(normalize_tex_name(tex.name) + ".dds")
        found = by_hash.get(key)
        cache[tex.name] = found
        if found is not None:
            resolved_count += 1
            return found
        return tex

    for geo in geoms:
        geo.texture = resolve(geo.texture)
        geo.normal_texture = resolve(geo.normal_texture)
        geo.other_textures = [resolve(t) for t in geo.other_textures]

    return resolved_count


def parse_city_sector(r: Rsc5Reader,
                       strings_table: dict[int, str] | None = None) -> tuple[list[Geometry], list[Texture], dict[int, Texture]]:
    r.u32()  # VFT
    r.read_ptr_pos()  # BlockMap ptr
    texdict_ptr = r.read_ptr_pos()  # TextureDictionary ptr
    r.u32()  # Unknown_Ch
    sector_piece_ptr = r.u32()

    if not sector_piece_ptr:
        raise SystemExit("No SectorPiece found in the file — this may not be a .xcs, or the file is corrupted.")

    dict_textures: list[Texture] = []
    dict_hash_map: dict[int, Texture] = {}
    if texdict_ptr:
        try:
            dict_textures, dict_hash_map = r.read_block(
                texdict_ptr, lambda rr: parse_texture_dictionary(rr, strings_table)) or ([], {})
        except (ValueError, struct.error, IndexError) as exc:
            print(f"  [!] Failed to read the texture dictionary: {exc}", file=sys.stderr)
            dict_textures, dict_hash_map = [], {}

    geoms = r.read_block(sector_piece_ptr, parse_city_sector_piece) or []
    return geoms, dict_textures, dict_hash_map


# --------------------------------------------------------------------------
# Writing OBJ + MTL + DDS
# --------------------------------------------------------------------------

def export_textures(all_textures: list[Texture], textures_dir: str) -> dict[str, str]:
    """Saves the unique textures into textures_dir and returns a dict
    {texture_name -> relative path to the .dds file}."""
    os.makedirs(textures_dir, exist_ok=True)
    name_to_relpath: dict[str, str] = {}
    seen: dict[str, Texture] = {}

    for tex in all_textures:
        if tex.name in seen:
            continue
        seen[tex.name] = tex

    for name, tex in seen.items():
        if not tex.data:
            continue
        safe_name = _sanitize_filename(name)
        filename = f"{safe_name}.dds"
        out_path = os.path.join(textures_dir, filename)
        try:
            write_dds(tex, out_path)
        except (ValueError, OSError, AssertionError) as exc:
            print(f"  [!] Failed to save texture '{name}': {exc}", file=sys.stderr)
            continue
        rel = os.path.join(os.path.basename(textures_dir.rstrip(os.sep)), filename)
        name_to_relpath[name] = rel.replace(os.sep, "/")

    return name_to_relpath


@dataclass
class MaterialInfo:
    diffuse: "Texture | None" = None
    normal: "Texture | None" = None
    other: list = field(default_factory=list)
    draw_bucket: int = 0   # see ShaderInfo.draw_bucket


def write_obj(geoms: list[Geometry], out_path: str, scale: float = 1.0,
              export_tex: bool = True, textures_dir: str | None = None,
              diffuse_only: bool = False, fix_channelmap: bool = False,
              fix_channelmap_channel: str = "g", fix_channelmap_sector: str = "cityroad",
              export_decals: bool = True, decal_offset: float = 0.02) -> int:
    v_i, vn_i, vt_i = 1, 1, 1

    # Materials are grouped NOT simply by shader name, but by the unique
    # combination of (shader + this geometry's actual set of textures). The
    # same shader (e.g. "CityBumpSpec") is used by dozens of different
    # meshes with DIFFERENT textures — if grouped only by shader name, they
    # would all collapse into a single .mtl material with the texture of
    # only the first geometry encountered, and the rest of the objects
    # would either end up without a texture or get someone else's. So a
    # separate material is built for each unique combination here, and
    # geo_mat_names stores, for each geometry (by index), the name of its
    # own personal .mtl material.
    #
    # diffuse_only: if the material in the .mtl will contain only
    # Diffuse/Opacity anyway (see below), normals/decals/other extra
    # textures shouldn't participate in the grouping either — otherwise
    # two meshes with the same diffuse texture but different bump maps
    # would get two different (though visually identical) materials.
    def tex_key(t: "Texture | None") -> str:
        return t.name if t is not None else ""

    combo_infos: dict[tuple, MaterialInfo] = {}
    combo_order: list[tuple] = []
    geo_combo: list[tuple] = []
    for g in geoms:
        if diffuse_only:
            key = (g.material, tex_key(g.texture))
        else:
            other_key = tuple(sorted(tex_key(t) for t in g.other_textures if t is not None))
            key = (g.material, tex_key(g.texture), tex_key(g.normal_texture), other_key)
        geo_combo.append(key)
        if key not in combo_infos:
            combo_order.append(key)
            mat = MaterialInfo()
            mat.diffuse = g.texture
            mat.draw_bucket = g.draw_bucket
            if not diffuse_only:
                mat.normal = g.normal_texture
                for tex in g.other_textures:
                    if tex is not None and all(o.name != tex.name for o in mat.other):
                        mat.other.append(tex)
            combo_infos[key] = mat

    # Assign the final .mtl material names: if a shader has only one unique
    # texture combination, keep its "plain" name as before; if there are
    # several combinations, add a numeric suffix to distinguish them
    combos_per_shader: dict[str, list[tuple]] = {}
    for key in combo_order:
        combos_per_shader.setdefault(key[0], []).append(key)

    combo_name: dict[tuple, str] = {}
    materials: dict[str, MaterialInfo] = {}
    for shader, keys in combos_per_shader.items():
        for idx, key in enumerate(keys, start=1):
            name = shader if len(keys) == 1 else f"{shader}_{idx}"
            combo_name[key] = name
            materials[name] = combo_infos[key]

    geo_mat_names = [combo_name[key] for key in geo_combo]

    # Which shader name (material) each final .mtl material name belongs
    # to — needed so --fix-channelmap-sector can filter by SHADER NAME (as
    # it's written inside the .xcs, see parse_shader), rather than by the
    # numeric "_2"/"_3" suffix added to distinguish texture combinations of
    # the same shader.
    shader_of_material: dict[str, str] = {}
    for shader, keys in combos_per_shader.items():
        for key in keys:
            shader_of_material[combo_name[key]] = shader

    channelmap_fixed_names: set[str] = set()
    if fix_channelmap:
        # Replaces the PRIMARY (diffuse) texture of EVERY matching material
        # with its "de-greened" version (see defilter_channelmap_texture).
        # "Matching" means this material's shader name (what's actually
        # written inside the .xcs, NOT the input filename) contains
        # fix_channelmap_sector case-insensitively — so textures of other
        # shaders/materials in the same sector are left untouched, even if
        # they also have a primary texture. An empty fix_channelmap_sector
        # disables the filter and applies --fix-channelmap to every
        # material in the file.
        fixed_cache: dict[str, "Texture | None"] = {}
        n_skipped_by_filter = 0
        for mat_name, mat in materials.items():
            if mat.diffuse is None:
                continue
            shader_name = shader_of_material.get(mat_name, mat_name)
            if fix_channelmap_sector and fix_channelmap_sector.lower() not in shader_name.lower():
                n_skipped_by_filter += 1
                continue
            orig_name = mat.diffuse.name
            if orig_name not in fixed_cache:
                fixed_cache[orig_name] = defilter_channelmap_texture(mat.diffuse, fix_channelmap_channel)
            fixed = fixed_cache[orig_name]
            if fixed is not None:
                mat.diffuse = fixed
                channelmap_fixed_names.add(fixed.name)
        n_fixed = sum(1 for v in fixed_cache.values() if v is not None)
        if n_fixed:
            print(f"     --fix-channelmap: primary textures processed: {n_fixed} "
                  f"(shader/material contains '{fix_channelmap_sector}')")
        elif fix_channelmap_sector and n_skipped_by_filter:
            print(f"  [i] --fix-channelmap: no material contains "
                  f"'{fix_channelmap_sector}' in its shader name — textures left untouched.")

    texture_paths: dict[str, str] = {}
    n_textures_exported = 0
    if export_tex:
        all_textures: list[Texture] = []
        seen_names: set[str] = set()
        for mat in materials.values():
            for tex in [mat.diffuse, mat.normal, *mat.other]:
                if tex is not None and tex.name not in seen_names:
                    seen_names.add(tex.name)
                    all_textures.append(tex)
        if textures_dir is None:
            textures_dir = os.path.splitext(out_path)[0] + "_textures"
        texture_paths = export_textures(all_textures, textures_dir)
        n_textures_exported = len(texture_paths)

    mtl_path = os.path.splitext(out_path)[0] + ".mtl"
    with open(mtl_path, "w", encoding="utf-8") as mf:
        for name, mat in materials.items():
            has_normal = (not diffuse_only) and mat.normal is not None and mat.normal.name in texture_paths
            mf.write(f"newmtl {name}\n")
            mf.write("Ka 1.0 1.0 1.0\n")
            mf.write("Kd 0.8 0.8 0.8\n")
            mf.write("Ks 0.0 0.0 0.0\n")
            mf.write("d 1.0\n")
            mf.write(f"illum {2 if has_normal else 1}\n")

            if mat.diffuse is not None and mat.diffuse.name in texture_paths:
                rel = texture_paths[mat.diffuse.name]
                mf.write(f"map_Kd {rel}\n")
                # map_d is needed not only for DXT3/DXT5 (full alpha channel
                # baked into the format itself) — DXT1 can also carry a
                # 1-bit "punch-through" alpha in its blocks, and A8R8G8B8
                # carries a full alpha channel, and both cases are already
                # CORRECTLY saved in the .dds (see
                # build_dds_bytes/unswizzle_xbox360). This used to not write
                # map_d for them, which caused materials the engine renders
                # with alpha blending (Rsc5Shader.DrawBucket -> ShaderBucket:
                # Alpha/Cutout/Water/Glass — e.g. CityWindow*) to lose their
                # transparency in the .obj/.mtl, even though the texture
                # itself was correct. L8 carries no alpha at all, so it's
                # excluded.
                # This used to only write map_d for DXT3/DXT5, or when the
                # shader's draw_bucket explicitly belonged to an alpha
                # bucket (Alpha/Cutout/Water/Glass) — because of this,
                # DXT1 and A8R8G8B8 textures with a normal opaque bucket (0,
                # the most common case in a City Sector) never got an
                # opacity-map reference in the .mtl at all, even when the
                # DXT1 blocks themselves carried a 1-bit alpha channel. The
                # format itself being capable of carrying an alpha channel
                # is enough to reference it; a fully opaque alpha (0xFF)
                # doesn't break anything on import, whereas the absence of
                # the reference deprives the importer (3ds Max etc.) of any
                # information about the alpha channel at all.
                can_have_alpha = mat.diffuse.fmt in (TEXFMT_DXT1, TEXFMT_DXT3, TEXFMT_DXT5, TEXFMT_A8R8G8B8)
                if can_have_alpha or mat.diffuse.name in channelmap_fixed_names:
                    mf.write(f"map_d {rel}\n")

            if diffuse_only:
                mf.write("\n")
                continue

            if has_normal:
                rel = texture_paths[mat.normal.name]
                # write both variants of the bump-map directive for
                # compatibility with different importers (Blender/3ds Max
                # use "bump", some other tools use "map_bump")
                mf.write(f"map_bump -bm 1.0 {rel}\n")
                mf.write(f"bump -bm 1.0 {rel}\n")

            # Extra layers (decals/grime/puddles/extra diffuse layers of
            # terrain shaders etc., e.g. decalsampler on CityRoad — actual
            # road markings). In the engine these aren't an alpha decal on
            # top of the geometry, but a second sampler that the pixel
            # shader blends with the diffuse texture via BlendMode — this
            # can't be reproduced directly in .obj/.mtl. The only previous
            # attempt to convey this was the Wavefront "decal" directive
            # for the first layer, but almost no importer supports it
            # (Blender etc. simply ignore it), so the markings effectively
            # never showed up anywhere, even though the .dds for them was
            # exported correctly.
            #
            # export_decals (enabled by default) instead creates a SEPARATE
            # material with normal map_Kd/map_d for each extra layer (these
            # directives are supported by every importer) — the geometry
            # for these materials is duplicated slightly above the original
            # surface (see decal_offset below, in the same place the
            # polygons themselves are written), so the layer is visible on
            # top of the base texture instead of getting lost in it due to
            # z-fighting. The one-layer-per-material limit (like with
            # "decal") is no longer needed — each extra layer gets its own
            # material.
            if not export_decals:
                other_rels = [texture_paths[t.name] for t in mat.other if t.name in texture_paths]
                if other_rels:
                    mf.write(f"decal {other_rels[0]}\n")
                    for rel in other_rels[1:]:
                        mf.write(f"# extra material layer (not directly supported by the .mtl format): {rel}\n")

            mf.write("\n")

            if export_decals:
                for tex in mat.other:
                    if tex.name not in texture_paths:
                        continue
                    rel = texture_paths[tex.name]
                    dname = decal_material_name(name, tex.name)
                    mf.write(f"newmtl {dname}\n")
                    mf.write("Ka 1.0 1.0 1.0\n")
                    mf.write("Kd 1.0 1.0 1.0\n")
                    mf.write("Ks 0.0 0.0 0.0\n")
                    mf.write("d 1.0\n")
                    mf.write("illum 1\n")
                    mf.write(f"map_Kd {rel}\n")
                    # for decals we always write the alpha channel
                    # (regardless of format) — otherwise the layer would
                    # completely cover the base texture with an opaque
                    # rectangle, instead of showing through only where the
                    # markings/grime/puddle actually are.
                    mf.write(f"map_d {rel}\n")
                    mf.write("\n")

    n_decal_layers = 0
    warned_no_normals_for_decal = False
    warned_no_uv1_for_decal = False

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")

        def write_group(group_name: str, mat_name: str, positions: list, uvs: list, normals: list,
                         indices: list) -> None:
            nonlocal v_i, vt_i, vn_i
            f.write(f"g {group_name}\n")
            f.write(f"usemtl {mat_name}\n")

            for (x, y, z) in positions:
                f.write(f"v {x*scale:.6f} {y*scale:.6f} {z*scale:.6f}\n")
            has_uv = any(u is not None for u in uvs) if uvs else False
            if has_uv:
                for uv in uvs:
                    if uv is None:
                        f.write("vt 0.0 0.0\n")
                    else:
                        f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
            has_nrm = any(n is not None for n in normals) if normals else False
            if has_nrm:
                for n in normals:
                    if n is None:
                        f.write("vn 0.0 0.0 0.0\n")
                    else:
                        f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            n_verts = len(positions)
            for t in range(0, len(indices) - 2, 3):
                a, b, c = indices[t], indices[t + 1], indices[t + 2]
                if a >= n_verts or b >= n_verts or c >= n_verts:
                    continue

                def ref(idx):
                    vi = v_i + idx
                    if has_uv and has_nrm:
                        return f"{vi}/{vt_i + idx}/{vn_i + idx}"
                    if has_uv:
                        return f"{vi}/{vt_i + idx}"
                    if has_nrm:
                        return f"{vi}//{vn_i + idx}"
                    return f"{vi}"

                f.write(f"f {ref(a)} {ref(b)} {ref(c)}\n")

            v_i += n_verts
            if has_uv:
                vt_i += n_verts
            if has_nrm:
                vn_i += n_verts

        for gi, geo in enumerate(geoms):
            if not geo.positions or not geo.indices:
                continue
            write_group(geo.name, geo_mat_names[gi], geo.positions, geo.uvs, geo.normals, geo.indices)

            # Decal layers (see export_decals above, and why this is NOT a
            # "decal" in the .mtl but full-fledged geometry instead): the
            # same topology (positions/UV/indices), shifted along the
            # normal by decal_offset so it doesn't "sink" into the base
            # surface due to z-fighting, with its own material
            # (map_Kd+map_d) for each extra layer.
            if export_decals and not diffuse_only and geo.other_textures:
                has_nrm = any(n is not None for n in geo.normals) if geo.normals else False
                if not has_nrm and not warned_no_normals_for_decal:
                    print("  [!] Some geometry has no normals — decal layers for it "
                          "are not offset and may suffer from z-fighting with the base surface.",
                          file=sys.stderr)
                    warned_no_normals_for_decal = True

                # In the original, decals only cover part of the surface —
                # this is set by a separate UV channel (TEXCOORD1), not the
                # diffuse texture's UV0 (which covers 0..1 across the whole
                # surface and would stretch the decal over all of it). If
                # TEXCOORD1 is present, use it for decal layers, otherwise
                # (as before) fall back to UV0.
                has_uv1 = any(u is not None for u in geo.uvs1) if geo.uvs1 else False
                decal_uvs = geo.uvs1 if has_uv1 else geo.uvs
                if not has_uv1 and not warned_no_uv1_for_decal:
                    print("  [!] Some geometry has no second UV channel (TEXCOORD1) — "
                          "decal layers use the diffuse texture's UV0 and may "
                          "stretch across the whole surface instead of a local area.",
                          file=sys.stderr)
                    warned_no_uv1_for_decal = True

                seen_decal_names: set[str] = set()
                layer = 0
                for tex in geo.other_textures:
                    if tex is None or tex.name in seen_decal_names or tex.name not in texture_paths:
                        continue
                    if "decal" not in tex.name.lower():
                        continue
                    seen_decal_names.add(tex.name)
                    layer += 1
                    # decal_offset is specified as a value in the FINAL .obj
                    # coordinates (after --scale) — but positions here have
                    # NOT been scaled yet (scaling is applied later, in
                    # write_group, when writing "v x*scale ..."). If
                    # decal_offset were simply added to the unscaled
                    # coordinates, the actual shift in the file would end up
                    # being decal_offset*scale, not decal_offset — with
                    # scale < 1 (a common case, e.g. converting to meters)
                    # the shift could become too small, and the decal layer
                    # would sink back into the base surface (z-fighting). We
                    # compensate by dividing by scale, so that after the
                    # multiplication in write_group we get exactly the
                    # requested decal_offset.
                    off = (decal_offset * layer) / scale if scale else decal_offset * layer
                    if has_nrm:
                        dpositions = [
                            (x + n[0] * off, y + n[1] * off, z + n[2] * off) if n is not None else (x, y, z)
                            for (x, y, z), n in zip(geo.positions, geo.normals)
                        ]
                    else:
                        dpositions = geo.positions
                    dname = decal_material_name(geo_mat_names[gi], tex.name)
                    write_group(f"{geo.name}_decal{layer}", dname, dpositions, decal_uvs, geo.normals, geo.indices)
                    n_decal_layers += 1

    if n_decal_layers:
        print(f"     decal layers added as separate geometry: {n_decal_layers}")

    return n_textures_exported


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def reorient_and_center(geoms: list["Geometry"], up_axis: str, center: bool) -> tuple[float, float, float]:
    """Brings the coordinates into the usual .obj convention and/or centers
    the model at the origin.

    up_axis:
      "y" (default) — rotate from the engine's native system (Z axis "up",
      as in most game engines/CX) into a system with the Y axis "up" (the
      standard convention for .obj and most 3D editors/viewers, including
      Blender). A 90° rotation around the X axis: (x, y, z) -> (x, z, -y).
      This is a PURE rotation (no reflection), so normals and the face
      vertex winding order (backface culling) are not broken — a mirror
      transform is not needed here.
      "z" — leave as-is (native Z-up axis), no rotation.

    center: subtract the bounding-box center from all vertices so the
      model ends up at the origin instead of at the sector's original
      "world" position (City sectors are usually cut out of a large map
      and have coordinates far from zero — because of this the model can
      appear "offset" far from the scene's center in an editor).

    Returns (ox, oy, oz) — the amount by which the model was shifted during
    centering (0,0,0 if center=False), so the sector's original "world"
    position can be reported to the user."""

    def rotate(p: tuple[float, float, float]) -> tuple[float, float, float]:
        if up_axis == "y":
            x, y, z = p
            return (x, z, -y)
        return p

    for g in geoms:
        g.positions = [rotate(p) for p in g.positions]
        g.normals = [rotate(n) if n is not None else None for n in g.normals]

    if not center:
        return (0.0, 0.0, 0.0)

    xs = [p[0] for g in geoms for p in g.positions]
    ys = [p[1] for g in geoms for p in g.positions]
    zs = [p[2] for g in geoms for p in g.positions]
    if not xs:
        return (0.0, 0.0, 0.0)

    ox = (min(xs) + max(xs)) / 2
    oy = (min(ys) + max(ys)) / 2
    oz = (min(zs) + max(zs)) / 2

    for g in geoms:
        g.positions = [(x - ox, y - oy, z - oz) for (x, y, z) in g.positions]

    return (ox, oy, oz)


def convert(xcs_path: str, out_path: str, raw: bool, flag: int | None,
            vsize: int | None, psize: int | None, lzx_dll: str | None,
            scale: float, export_tex: bool, textures_dir: str | None,
            export_dict_textures: bool, xtd_paths: list[str] | None = None,
            xtd_vsize: int | None = None, xtd_flag: int | None = None,
            up_axis: str = "y", center: bool = True, diffuse_only: bool = False,
            fix_channelmap: bool = False, fix_channelmap_channel: str = "g",
            fix_channelmap_sector: str = "cityroad",
            strings_path: str | None = None,
            export_decals: bool = True, decal_offset: float = 0.02):
    strings_table: dict[int, str] | None = None
    if strings_path:
        try:
            strings_table = load_strings_table(strings_path)
        except OSError as exc:
            print(f"  [!] Failed to load string table {strings_path}: {exc}", file=sys.stderr)
            strings_table = None
        else:
            print(f"  [i] {strings_path}: loaded {len(strings_table)} string(s)")

    data, vsize_final = load_xcs_bytes(xcs_path, raw, flag, vsize, psize, lzx_dll)
    reader = Rsc5Reader(data, vsize_final)
    reader.seek(VIRTUAL_BASE)
    geoms, dict_textures, dict_hash_map = parse_city_sector(reader, strings_table)

    if not geoms:
        raise SystemExit("No geometry found — the file was read, but no mesh could be extracted.")

    ox, oy, oz = reorient_and_center(geoms, up_axis, center)
    if center and (ox or oy or oz):
        print(f"     model centered (original center in world coordinates: "
              f"{ox:.2f}, {oy:.2f}, {oz:.2f})")

    # External texture dictionaries (e.g. common.xtd) — a shared/global
    # pool that sector meshes can reference if the needed texture isn't in
    # the .xcs's own dictionary. The sector's own dictionary takes
    # priority: if the same hash unexpectedly appears in both, the entry
    # from dict_hash_map (the sector's) is kept, and the external
    # dictionaries only add to it.
    combined_hash_map = dict_hash_map
    if xtd_paths:
        combined_hash_map = {}
        for xtd_path in xtd_paths:
            try:
                ext_map = load_texture_dictionary_file(xtd_path, xtd_vsize, xtd_flag, lzx_dll, strings_table)
            except (ValueError, struct.error, IndexError, SystemExit) as exc:
                print(f"  [!] Failed to read external texture dictionary {xtd_path}: {exc}", file=sys.stderr)
                continue
            print(f"  [i] {xtd_path}: loaded {len(ext_map)} texture(s)")
            combined_hash_map.update(ext_map)
        combined_hash_map.update(dict_hash_map)

    # Substitute real pixel data into stub textures (see
    # resolve_stub_textures) — without this step, materials in the .mtl
    # would be left without a single bound texture, even when all the
    # needed data is actually present in the sector's texture dictionary
    # (and/or in external .xtd files).
    n_resolved = resolve_stub_textures(geoms, combined_hash_map)
    if n_resolved:
        print(f"     textures matched against the dictionary (sector + external): {n_resolved}")

    # Additionally export textures from the dictionary
    # (Rsc5TextureDictionary), even if they aren't directly bound to any
    # geometry's material.
    extra_textures: list[Texture] = []
    if export_tex and export_dict_textures:
        used_names: set[str] = set()
        for g in geoms:
            if g.texture is not None:
                used_names.add(g.texture.name)
            if g.normal_texture is not None:
                used_names.add(g.normal_texture.name)
            used_names.update(t.name for t in g.other_textures)
        extra_textures = [t for t in dict_textures if t.name not in used_names]

    n_tex = write_obj(geoms, out_path, scale, export_tex=export_tex, textures_dir=textures_dir,
                       diffuse_only=diffuse_only, fix_channelmap=fix_channelmap,
                       fix_channelmap_channel=fix_channelmap_channel,
                       fix_channelmap_sector=fix_channelmap_sector,
                       export_decals=export_decals, decal_offset=decal_offset)

    if export_tex and extra_textures:
        tdir = textures_dir or (os.path.splitext(out_path)[0] + "_textures")
        extra_paths = export_textures(extra_textures, tdir)
        n_tex += len(extra_paths)

    total_tris = sum(len(g.indices) // 3 for g in geoms)
    total_verts = sum(len(g.positions) for g in geoms)
    print(f"OK: {len(geoms)} mesh(es), {total_verts} vertices, {total_tris} triangles -> {out_path}")
    if export_tex:
        print(f"     textures exported: {n_tex}")


def find_auto_xtd_files(input_path: str) -> list[str]:
    """Automatically finds external texture dictionaries (.xtd) if the user
    didn't specify --xtd manually. Looks in all the places they're usually
    placed next to the script or the file being converted:
      1) the folder containing xcs2obj.py itself;
      2) the folder containing the input .xcs;
      3) the current working directory.
    Matches by absolute path are not duplicated."""
    candidates_dirs = []
    try:
        candidates_dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates_dirs.append(os.path.dirname(os.path.abspath(input_path)))
    candidates_dirs.append(os.getcwd())

    found: list[str] = []
    seen: set[str] = set()
    for d in candidates_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".xtd"):
                full = os.path.abspath(os.path.join(d, name))
                if full not in seen:
                    seen.add(full)
                    found.append(full)
    return found


def find_auto_strings_file(input_path: str) -> str | None:
    """Automatically finds the engine's string table (*.strings.txt, e.g.
    Codex.Games.MCLA.strings.txt) if the user didn't explicitly specify
    --strings manually. Looks in the same places as for .xtd (see
    find_auto_xtd_files): next to the script, next to the input .xcs, and
    in the current working directory."""
    candidates_dirs = []
    try:
        candidates_dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates_dirs.append(os.path.dirname(os.path.abspath(input_path)))
    candidates_dirs.append(os.getcwd())

    for d in candidates_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".strings.txt"):
                return os.path.join(d, name)
    return None


def main():
    ap = argparse.ArgumentParser(description="Converter for .xcs (MCLA City Sector) -> .obj")
    ap.add_argument("input", help="path to the .xcs file")
    ap.add_argument("-o", "--output", help="path to the output .obj (defaults to next to the input)")
    ap.add_argument("--raw", action="store_true",
                     help="the input file is already unpacked (no RSC5 header/no LZX)")
    ap.add_argument("--flag", type=lambda x: int(x, 0), default=None,
                     help="32-bit RSC5 packing flag (for --raw, if --vsize is not given)")
    ap.add_argument("--vsize", type=lambda x: int(x, 0), default=None,
                     help="size of the virtual segment in bytes (for --raw); if neither this nor "
                          "--flag is given, the input file's own size is used by default")
    ap.add_argument("--psize", type=lambda x: int(x, 0), default=None,
                     help="size of the physical segment in bytes (optional)")
    ap.add_argument("--lzx-dll", default=None, help="path to xcompress32.dll")
    ap.add_argument("--scale", type=float, default=1.0, help="coordinate scale (default 1.0)")
    ap.add_argument("--no-textures", action="store_true",
                     help="don't extract textures or create .dds files")
    ap.add_argument("--textures-dir", default=None,
                     help="folder to save .dds files in (default '<obj_name>_textures')")
    ap.add_argument("--no-dict-textures", action="store_true",
                     help="don't export textures from the shared dictionary that aren't "
                          "directly bound to any material (by default all of them are exported)")
    ap.add_argument("--xtd", action="append", default=None,
                     help="path to an external texture dictionary (.xtd, e.g. common.xtd) — "
                          "can be given multiple times if there are several dictionaries; "
                          "used for textures not present in the .xcs itself")
    ap.add_argument("--xtd-vsize", type=lambda x: int(x, 0), default=None,
                     help="size of the virtual segment for --xtd, if it is already unpacked without "
                          "an RSC5 header (defaults to the file size)")
    ap.add_argument("--xtd-flag", type=lambda x: int(x, 0), default=None,
                     help="32-bit RSC5 packing flag for --xtd (alternative to --xtd-vsize)")
    ap.add_argument("--no-auto-xtd", action="store_true",
                     help="don't automatically look for .xtd files next to the script/input file "
                          "(by default, if --xtd isn't given explicitly, all *.xtd files from those "
                          "folders are picked up automatically)")
    ap.add_argument("--strings", default=None,
                     help="path to the engine's string table (e.g. Codex.Games.MCLA.strings.txt) — "
                          "used to give textures that only store a name hash (not the name itself) "
                          "inside the .xcs a real human-readable name instead of a stub like "
                          "'tex_XXXXXXXX'/'hash_XXXXXXXX'. By default, if not given explicitly, a "
                          "*.strings.txt file is searched for automatically next to the script/"
                          "input file (see --no-auto-strings).")
    ap.add_argument("--no-auto-strings", action="store_true",
                     help="don't automatically look for *.strings.txt next to the script/input file "
                          "(by default, if --strings isn't given explicitly, such a file is picked up "
                          "automatically if found)")
    ap.add_argument("--up", choices=("y", "z"), default="y",
                     help="which axis should be \"up\" in the output .obj: 'y' (default) — "
                          "rotate from the engine's native system (Z up) into the system usual for "
                          ".obj/Blender (Y up); 'z' — leave as in the engine, no rotation")
    ap.add_argument("--no-center", action="store_true",
                     help="don't center the model at the origin (by default the bounding-box "
                          "center is moved to (0,0,0) — City sectors are cut out of a large map "
                          "and by default have coordinates far from zero)")
    ap.add_argument("--no-xbox-tiled", action="store_true",
                     help="DISABLE texture de-tiling for Xbox 360 (Rpf3Crypto.UnswizzleXbox360Data). "
                          "By default de-tiling is ENABLED. In some .xcs dumps textures are already "
                          "stored linearly, and de-tiling corrupts them (noise/colored static instead "
                          "of a picture) — if you see that, try converting with this flag.")
    ap.add_argument("--diffuse-only", action="store_true",
                     help="write only Diffuse (map_Kd) and Opacity (map_d) into materials — bump/normal "
                          "maps (map_bump/bump) and extra layers (decal/grime/puddles etc.) are not "
                          "added. Also collapses materials that differ only in these maps: if several "
                          "meshes of the same shader use the same diffuse texture but different bump "
                          "maps, they will get ONE shared material instead of several.")
    ap.add_argument("--fix-channelmap", action="store_true",
                     help="if a material's MAIN (diffuse, map_Kd) texture is actually a channel-map "
                          "texture (blend weights in R/G/B, which is why it looks green), remove the "
                          "green tint and save it as a plain grayscale image instead. This is the same "
                          "transform as \"preview\" mode in fix_channelmap_texture.py, applied "
                          "automatically to the primary texture of matching materials. By default it "
                          "applies ONLY to materials whose shader name (as recorded INSIDE the .xcs "
                          "itself, see --fix-channelmap-sector) contains 'cityroad' — other "
                          "shaders/materials in the same sector are not affected. Requires "
                          "pip install numpy pillow.")
    ap.add_argument("--fix-channelmap-channel", choices=("g", "max", "avg"), default="g",
                     help="which channel to treat as brightness for --fix-channelmap: 'g' — the G "
                          "channel only (default, usually the most detailed), 'max' — the max across "
                          "R/G/B (doesn't lose any mask), 'avg' — the average across R/G/B")
    ap.add_argument("--fix-channelmap-sector", default="cityroad",
                     help="--fix-channelmap is only applied to materials whose SHADER NAME contains "
                          "this substring, case-insensitively (default 'cityroad') — the name is taken "
                          "from the .xcs file itself (see Rsc5Shader.Name), NOT from the input file's "
                          "name. So if a sector has several different shaders/materials (e.g. "
                          "CityRoad, CityGrass, CityBumpSpec), --fix-channelmap will only change the "
                          "primary texture of the ones whose shader name matches the filter; the rest "
                          "are left as-is. Pass an empty string ('') to disable this check and apply "
                          "--fix-channelmap to every material in the file.")
    ap.add_argument("--no-decals", action="store_true",
                     help="don't output decal layers (decalsampler/grimesampler/puddlesampler etc. — "
                          "e.g. CityRoad's road markings) as separate geometry with its own material "
                          "(map_Kd+map_d). By default this is ENABLED: for each such extra layer, a "
                          "duplicate of the original mesh's geometry is created, offset along the "
                          "normal by --decal-offset, with its own material — so the layer is visible "
                          "in any normal OBJ importer (Blender etc.), unlike the old .mtl \"decal\" "
                          "directive, which almost nothing supports. With this flag, the old behavior "
                          "is restored (a single \"decal\" directive in the .mtl per material, without "
                          "new geometry). Ignored together with --diffuse-only (extra layers aren't "
                          "read there in the first place).")
    ap.add_argument("--decal-offset", type=float, default=0.02,
                     help="how many units (in the final .obj coordinates, AFTER --scale) to offset "
                          "decal geometry along the surface normal from the base geometry, to avoid "
                          "z-fighting (default 0.02). If a piece of geometry has several extra layers "
                          "(e.g. decal+grime+puddle on CityRoad), each subsequent layer is offset by a "
                          "multiple of this value, so the layers don't coincide with each other either. "
                          "If the geometry has no normals, the offset is not applied (see the warning "
                          "in the output) — the layer will stay exactly on the original surface.")
    args = ap.parse_args()

    global UNSWIZZLE_TEXTURES
    UNSWIZZLE_TEXTURES = not args.no_xbox_tiled

    out_path = args.output or (os.path.splitext(args.input)[0] + ".obj")

    xtd_paths = args.xtd
    if not xtd_paths and not args.no_auto_xtd:
        xtd_paths = find_auto_xtd_files(args.input)
        for p in xtd_paths:
            print(f"  [i] automatically found texture dictionary: {p}")

    strings_path = args.strings
    if not strings_path and not args.no_auto_strings:
        strings_path = find_auto_strings_file(args.input)
        if strings_path:
            print(f"  [i] automatically found string table: {strings_path}")

    convert(args.input, out_path, args.raw, args.flag, args.vsize, args.psize,
            args.lzx_dll, args.scale, not args.no_textures, args.textures_dir,
            not args.no_dict_textures, xtd_paths, args.xtd_vsize, args.xtd_flag,
            args.up, not args.no_center, args.diffuse_only,
            args.fix_channelmap, args.fix_channelmap_channel, args.fix_channelmap_sector,
            strings_path, not args.no_decals, args.decal_offset)


if __name__ == "__main__":
    main()
