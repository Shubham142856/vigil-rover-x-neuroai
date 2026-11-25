"""Minimal MATLAB v5 (.mat) reader for dense numeric arrays. No scipy needed.
Supports: miDOUBLE, miSINGLE, miUINT8, miUINT16, miUINT32, miINT8, miINT16, miINT32, miINT64, miUINT64 arrays.
Only reads the first variable named 'patt' (or the first matrix found).
"""
import struct, numpy as np
from pathlib import Path

MI_INT8   = 1; MI_UINT8  = 2; MI_INT16  = 3; MI_UINT16 = 4
MI_INT32  = 5; MI_UINT32 = 6; MI_SINGLE = 7; MI_DOUBLE = 9
MI_INT64  = 12; MI_UINT64 = 13; MI_MATRIX = 14; MI_COMPRESSED = 15; MI_UTF8 = 16

DTYPE_MAP = {
    MI_INT8: np.int8, MI_UINT8: np.uint8, MI_INT16: np.int16, MI_UINT16: np.uint16,
    MI_INT32: np.int32, MI_UINT32: np.uint32, MI_SINGLE: np.float32, MI_DOUBLE: np.float64,
    MI_INT64: np.int64, MI_UINT64: np.uint64,
}
DTYPE_BYTES = {t: np.dtype(d).itemsize for t, d in DTYPE_MAP.items()}

def _read_tag(buf, pos):
    """Read a data element tag. Returns (data_type, num_bytes, pos_after_tag, is_small).
    Returns (0, 0, pos, False) if insufficient bytes remain (end of stream)."""
    if pos + 8 > len(buf):
        return 0, 0, pos, False   # end-of-stream sentinel
    dt, nb = struct.unpack_from('<II', buf, pos)
    if dt >> 16:          # small data element: high 2 bytes = num_bytes, low 2 bytes = data_type
        nb2 = dt >> 16; dt2 = dt & 0xFFFF
        return dt2, nb2, pos + 4, True
    return dt, nb, pos + 8, False

def _read_data(buf, pos, dt, nb, is_small):
    if is_small:
        raw = buf[pos:pos+nb]
        pos_next = pos + 4   # small elements always occupy 4 bytes total (already past tag)
    else:
        raw = buf[pos:pos+nb]
        pad = (8 - nb % 8) % 8
        pos_next = pos + nb + pad
    if dt in DTYPE_MAP:
        arr = np.frombuffer(raw, dtype=DTYPE_MAP[dt])
    else:
        arr = np.frombuffer(raw, dtype=np.uint8)
    return arr, pos_next

def _read_element(buf, pos):
    dt, nb, pos_data, is_small = _read_tag(buf, pos)
    data, pos_next = _read_data(buf, pos_data, dt, nb, is_small)
    return dt, data, pos_next

def load_patt(path):
    """Load the 'patt' variable from a MATLAB v5 file. Returns numpy array (H, W, T) float64."""
    buf = Path(path).read_bytes()
    # Header: 128 bytes. Version at [124:126], endian at [126:128]
    endian = buf[126:128]
    assert endian == b'IM', f"Only little-endian MAT files supported, got {endian}"

    pos = 128
    n = len(buf)
    while pos + 8 <= n:
        dt, nb, pos_data, is_small = _read_tag(buf, pos)
        if dt == 0:
            break   # end-of-stream sentinel
        if dt == MI_COMPRESSED:
            import zlib
            raw = buf[pos_data:pos_data+nb]
            inner = zlib.decompress(raw)
            # The inner decompressed buffer is a single miMATRIX element
            # Read its tag first (8 bytes)
            if len(inner) >= 8:
                idt, inb, ipos, iis_small = _read_tag(inner, 0)
                if idt == MI_MATRIX:
                    arr = _parse_matrix(inner, ipos)
                    if arr is not None:
                        return arr
            pad = (8 - nb % 8) % 8
            pos = pos_data + nb + pad
        elif dt == MI_MATRIX:
            arr = _parse_matrix(buf, pos_data)
            if arr is not None:
                return arr
            pad = (8 - nb % 8) % 8
            pos = pos_data + nb + pad
        else:
            # skip non-matrix top-level element
            pad = (8 - nb % 8) % 8 if not is_small else 0
            pos = pos_data + nb + pad
    raise ValueError("Variable 'patt' not found in mat file")

def _parse_matrix(buf, pos):
    """Parse a miMATRIX element starting at pos within buf. Returns ndarray or None."""
    n = len(buf)
    if pos >= n:
        return None

    # 1. Array flags (mxUINT32, 8 bytes data)
    dt, flags_arr, pos = _read_element(buf, pos)
    mclass = int(flags_arr[0]) & 0xFF   # class byte

    # 2. Dimensions array
    dt2, dims_arr, pos = _read_element(buf, pos)
    dims = tuple(int(d) for d in dims_arr)

    # 3. Array name (miINT8 / miUTF8)
    dt3, name_arr, pos = _read_element(buf, pos)
    name = bytes(name_arr).decode('latin-1', errors='replace').rstrip('\x00')

    # Bounds check: at minimum we need 8 bytes for the data element tag
    if pos >= len(buf):
        return None

    # Only handle numeric arrays (mxDOUBLE=6, mxSINGLE=7, mxUINT8=9, mxINT8=8, etc.)
    NUMERIC_CLASSES = {6: MI_DOUBLE, 7: MI_SINGLE, 8: MI_INT8, 9: MI_UINT8,
                       10: MI_INT16, 11: MI_UINT16, 12: MI_INT32, 13: MI_UINT32,
                       14: MI_INT64, 15: MI_UINT64}
    if mclass not in NUMERIC_CLASSES:
        return None   # cell, struct, char etc — skip

    # 4. Real part
    dt4, real_arr, pos = _read_element(buf, pos)

    # Reshape column-major (Fortran order) to dims
    total = 1
    for d in dims:
        total *= d
    if len(real_arr) != total:
        # try casting
        expected_dtype = DTYPE_MAP.get(dt4, np.float64)
        real_arr = real_arr.view(expected_dtype)
    arr = real_arr.astype(np.float64).reshape(dims, order='F')

    if name == 'patt':
        return arr
    return None   # wrong name — keep scanning


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\rover\references\EMD-LPLC2-GF-model\code\1_Response-Characteristics-Of-Single-LPLC2-Neurons-To-Different-Stimuli\Stimuli\Looming_50ms.mat"
    arr = load_patt(path)
    print(f"Loaded patt: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
