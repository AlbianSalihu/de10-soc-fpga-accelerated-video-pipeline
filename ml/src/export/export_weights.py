# -----------------------------------------------------------------------------
# export_weights.py
#
# Packs all quantized FPGA parameters into a single binary file (.bin)
# ready for the HPS to load into FPGA memory at boot.
#
# Prerequisites:
#   Run these two scripts first (in order):
#     1. python -m ml.src.export.find_scales      → produces act_scales_sy.json
#     2. python -m ml.src.export.quantize_weights → produces fpgaqparms.npz + fpgaqparms.json
#
# Requirements:
#   numpy >= 1.24
#
# Usage:
#   python -m ml.src.export.export_weights \
#       --npz  ml/checkpoints/fpgaqparms.npz  \
#       --meta ml/checkpoints/fpgaqparms.json \
#       --out  ml/checkpoints/fpgaqparms.bin
#
# Optional:
#   --sdram-threshold <bytes>   W_q size above which placement defaults to SDRAM
#                               (default: 131072 = 128 KB)
#
# Output:
#   A single .bin file with the following layout:
#     [16 B]    Header  : magic, version, num_sections, reserved
#     [N x 56B] Table   : one descriptor per (layer, param) section
#     [...]     Blob    : raw little-endian array bytes, 4-byte padded
#
#   Target addresses within SDRAM / BRAM are NOT written here.
#   The HPS C loader owns address assignment based on the Platform Designer map.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from ml.src.utils import latest_run_id

# W_q byte size above which placement defaults to SDRAM
SDRAM_THRESHOLD_DEFAULT = 128 * 1024


# -- Section definitions -------------------------------------------------------

class Placement(IntEnum):
    SDRAM = 0
    BRAM  = 1

class ParamType(IntEnum):
    Wq = 0
    Bq = 1
    m  = 2
    r  = 3

class DType(IntEnum):
    INT8  = 0
    INT32 = 1

@dataclass
class Section:
    layer_idx:  int
    param_type: ParamType
    placement:  Placement
    dtype:      DType
    arr:        np.ndarray  # actual data, offsets filled in later
    file_offset: int = 0   # filled by assign_file_offsets()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with fields: npz, meta, out, sdram_threshold.
    """
    p = argparse.ArgumentParser(
        description="Pack FPGA quantization parameters into a single .bin for HPS loading."
    )
    p.add_argument("--outputs-dir",  type=str, default="ml/outputs", help="Base dir for export outputs")
    p.add_argument("--run-id",       type=int, default=-1, help="Run ID to use (default: latest)")
    p.add_argument("--npz",  type=str, default="", help="Override .npz path (overrides --run-id)")
    p.add_argument("--meta", type=str, default="", help="Override .json path (overrides --run-id)")
    p.add_argument("--out",  type=str, default="", help="Override .bin output path (overrides --run-id)")
    p.add_argument("--fpga-config", type=str, default="ml/config/fpga.yaml",
                   help="Path to FPGA hardware config YAML (default: ml/config/fpga.yaml)")
    p.add_argument("--sdram-threshold", type=int, default=None,
                   help="W_q byte size above which placement defaults to SDRAM. "
                        "CLI wins over fpga.yaml, which wins over hard-coded default (128 KB).")
    return p.parse_args()


# -- Expected dtypes per param type -------------------------------------------
_EXPECTED_DTYPE: Dict[str, np.dtype] = {
    "W_q": np.dtype("int8"),
    "B_q": np.dtype("int32"),
    "m":   np.dtype("int32"),
    "r":   np.dtype("int32"),
}


def load_inputs(
    npz_path: Path,
    meta_path: Path,
) -> Tuple[List[Dict[str, Any]], np.lib.npyio.NpzFile]:
    """Load and validate fpgaqparms.json + fpgaqparms.npz.

    Returns:
        layers : ordered list of layer dicts from the JSON
        arrays : lazy NpzFile handle (index by key name)

    Raises:
        FileNotFoundError : if either file is missing
        KeyError          : if a JSON-referenced array key is absent from the npz
        TypeError         : if an array dtype does not match expectations
        ValueError        : if an array shape does not match the JSON metadata
    """
    # -- Load files ---------------------------------------------------------------
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Parameter archive not found: {npz_path}")

    meta   = json.loads(meta_path.read_text())
    arrays = np.load(npz_path, allow_pickle=False)
    layers: List[Dict[str, Any]] = meta["layers"]

    print(f"\n[load_inputs] {len(layers)} layers from {meta_path.name}")
    print(f"[load_inputs] {len(arrays.files)} arrays from {npz_path.name}")

    # -- Validate + report --------------------------------------------------------
    print(f"\n  {'Layer':<16} {'Type':<8} {'W_q shape':<22} {'B_q':^5} {'m':^5} {'r':^5}")
    print(f"  {'─'*62}")

    for linfo in layers:
        lname = linfo["name"]
        ltype = linfo["type"]
        keys  = linfo["export_keys"]

        # Validate each present param
        for param in ("W_q", "B_q", "m", "r"):
            key = keys.get(param)
            if key is None:
                continue  # absent params validated below

            # Key must exist in npz
            if key not in arrays:
                raise KeyError(
                    f"[{lname}] expected array '{key}' not found in npz. "
                    f"Re-run quantize_weights.py to regenerate."
                )

            arr = arrays[key]

            # Dtype must match
            if arr.dtype != _EXPECTED_DTYPE[param]:
                raise TypeError(
                    f"[{lname}] '{param}' dtype mismatch: "
                    f"expected {_EXPECTED_DTYPE[param]}, got {arr.dtype}"
                )

            # Shape must match JSON metadata
            if param == "W_q":
                expected_shape = tuple(linfo["weight_shape"])
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"[{lname}] W_q shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )
            elif param == "B_q":
                expected_shape = tuple(linfo["bias_shape"])
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"[{lname}] B_q shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )

        # Warn if m/r absent (expected only for logits layer)
        has_m = keys.get("m") is not None
        has_r = keys.get("r") is not None
        if not has_m or not has_r:
            no_requant_note = "← no requant (logits layer, argmax on HPS)"
        else:
            no_requant_note = ""

        wq_shape = str(tuple(linfo["weight_shape"]))
        print(
            f"  {lname:<16} {ltype:<8} {wq_shape:<22} "
            f"{'✓':^5} {'✓' if has_m else '✗':^5} {'✓' if has_r else '✗':^5}"
            f"  {no_requant_note}"
        )

    # -- Critical warnings --------------------------------------------------------
    logits_layers = [l["name"] for l in layers if not l["post_relu_quantized"]]
    for lname in logits_layers:
        print(f"\n  [WARNING] '{lname}' has no m/r — "
              f"raw int32 accumulator output, argmax must be computed on HPS side.")

    print()
    return layers, arrays


def build_section_list(
    layers:           List[Dict[str, Any]],
    arrays:           np.lib.npyio.NpzFile,
    sdram_threshold:  int,
) -> List[Section]:
    """Walk layers in forward order and produce a flat list of Sections.

    One Section is created per present param (W_q, B_q, m, r) per layer.
    Placement is decided here and tagged on each Section — no other function
    needs to know the placement rules.

    Placement policy:
        W_q : SDRAM if W_q.nbytes > sdram_threshold, else BRAM
        B_q : always BRAM (max ~4 KB)
        m   : always BRAM (max ~4 KB), absent for logits layer
        r   : always BRAM (max ~4 KB), absent for logits layer

    Args:
        layers:          ordered list of layer dicts from fpgaqparms.json
        arrays:          NpzFile handle from fpgaqparms.npz
        sdram_threshold: W_q byte size above which placement defaults to SDRAM

    Returns:
        Flat list of Section objects with placement tagged, file_offset not yet set.
    """
    sections: List[Section] = []

    for li, linfo in enumerate(layers):
        keys = linfo["export_keys"]

        # -- W_q ---------------------------------------------------------------
        W_q = arrays[keys["W_q"]]
        placement = Placement.SDRAM if W_q.nbytes > sdram_threshold else Placement.BRAM
        sections.append(Section(
            layer_idx  = li,
            param_type = ParamType.Wq,
            placement  = placement,
            dtype      = DType.INT8,
            arr        = W_q,
        ))

        # -- B_q (always BRAM) -------------------------------------------------
        if keys["B_q"] is not None:
            sections.append(Section(
                layer_idx  = li,
                param_type = ParamType.Bq,
                placement  = Placement.BRAM,
                dtype      = DType.INT32,
                arr        = arrays[keys["B_q"]],
            ))

        # -- m (always BRAM, absent for logits layer) --------------------------
        if keys["m"] is not None:
            sections.append(Section(
                layer_idx  = li,
                param_type = ParamType.m,
                placement  = Placement.BRAM,
                dtype      = DType.INT32,
                arr        = arrays[keys["m"]],
            ))

        # -- r (always BRAM, absent for logits layer) --------------------------
        if keys["r"] is not None:
            sections.append(Section(
                layer_idx  = li,
                param_type = ParamType.r,
                placement  = Placement.BRAM,
                dtype      = DType.INT32,
                arr        = arrays[keys["r"]],
            ))

    return sections


HEADER_SIZE  = 16
SECTION_SIZE = 56  # each section descriptor is 56 bytes (defined by write_bin)


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


MAGIC   = b"FPGA"
VERSION = 1

# Section descriptor layout (56 bytes = 14 × uint32, little-endian):
#   layer_idx, param_type, placement, dtype,
#   file_offset, data_size, num_elements,
#   ndim, shape[0], shape[1], shape[2], shape[3],
#   reserved[0], reserved[1]
SECTION_FMT = "<14I"
assert struct.calcsize(SECTION_FMT) == SECTION_SIZE


def _to_le_bytes(arr: np.ndarray) -> bytes:
    """Raw little-endian bytes regardless of host byte order."""
    if arr.dtype.itemsize == 1:
        return arr.tobytes()
    return arr.astype(arr.dtype.newbyteorder("<")).tobytes()


def write_bin(sections: List[Section], out_path: Path) -> None:
    """Serialize sections into a .bin file.

    File layout:
        [16 B]      Header: magic(4) version(4) num_sections(4) reserved(4)
        [N × 56 B]  Section table: one descriptor per Section
        [...]       Data blob: raw little-endian array bytes, 4-byte padded

    Each section descriptor (56 bytes = 14 × uint32):
        layer_idx, param_type, placement, dtype,
        file_offset, data_size, num_elements,
        ndim, shape[0..3], reserved[0..1]

    Args:
        sections: flat list of Sections with file_offset already assigned
        out_path: destination .bin path (parent dirs created if needed)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-build blob to compute CRC32 before writing the header
    blob_parts = []
    for sec in sections:
        raw = _to_le_bytes(sec.arr)
        pad = _align_up(len(raw), 4) - len(raw)
        blob_parts.append(raw + b"\x00" * pad)
    blob = b"".join(blob_parts)
    crc32 = zlib.crc32(blob) & 0xFFFFFFFF

    with out_path.open("wb") as f:

        # -- Header (16 bytes) -------------------------------------------------
        # magic(4) version(4) num_sections(4) crc32_of_blob(4)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(sections)))
        f.write(struct.pack("<I", crc32))

        # -- Section table -----------------------------------------------------
        for sec in sections:
            shape = list(sec.arr.shape) + [0, 0, 0, 0]
            f.write(struct.pack(
                SECTION_FMT,
                sec.layer_idx,
                int(sec.param_type),
                int(sec.placement),
                int(sec.dtype),
                sec.file_offset,
                sec.arr.nbytes,
                sec.arr.size,
                sec.arr.ndim,
                shape[0], shape[1], shape[2], shape[3],
                0, 0,  # reserved
            ))

        # -- Data blob ---------------------------------------------------------
        f.write(blob)


_PARAM_NAMES  = {ParamType.Wq: "W_q", ParamType.Bq: "B_q", ParamType.m: "m", ParamType.r: "r"}
_TARGET_NAMES = {Placement.SDRAM: "SDRAM", Placement.BRAM: "BRAM"}


def print_summary(sections: List[Section], layers: List[Dict[str, Any]], out_path: Path) -> None:
    """Print a human-readable report of what was packed into the .bin.

    Prints per-section table (layer, param, placement, shape, size) and
    memory usage totals: SDRAM overall and BRAM per layer.

    Args:
        sections: flat list of Sections after all steps are complete
        layers:   ordered list of layer dicts (for name lookup)
        out_path: path of the written .bin (for file size reporting)
    """
    print(f"\n[export_weights] Written: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    print(f"\n  {'Layer':<16} {'Param':<6} {'Placement':<8} {'Shape':<24} {'Size':>10}")
    print(f"  {'-'*68}")

    sdram_total = 0
    bram_totals: Dict[int, int] = {}

    for sec in sections:
        lname = layers[sec.layer_idx]["name"]
        shape = str(sec.arr.shape)
        size  = sec.arr.nbytes
        place = _TARGET_NAMES[sec.placement]
        param = _PARAM_NAMES[sec.param_type]

        print(f"  {lname:<16} {param:<6} {place:<8} {shape:<24} {size:>8,} B")

        if sec.placement == Placement.SDRAM:
            sdram_total += size
        else:
            bram_totals[sec.layer_idx] = bram_totals.get(sec.layer_idx, 0) + size

    print(f"\n  SDRAM total : {sdram_total:>10,} B  ({sdram_total / 1024**2:.2f} MB)")
    print(f"  BRAM per layer:")
    for li, total in bram_totals.items():
        lname = layers[li]["name"]
        print(f"    [{li}] {lname:<16} {total:>8,} B  ({total / 1024:.1f} KB)")
    print()


def assign_file_offsets(sections: List[Section]) -> None:
    """Stamp file_offset on each Section in-place.

    Cursor starts immediately after the header and section table.
    Each section's data is placed at the current cursor position,
    then the cursor advances by the array's byte size rounded up
    to the nearest 4 bytes.

    Args:
        sections: flat list of Sections (modified in-place)
    """
    cursor = HEADER_SIZE + len(sections) * SECTION_SIZE
    for sec in sections:
        sec.file_offset = cursor
        cursor = _align_up(cursor + sec.arr.nbytes, 4)


def main() -> int:
    """Entry point: load → build sections → assign offsets → write → summarise.

    Returns:
        0 on success. Raises on any validation or I/O error.
    """
    args = parse_args()

    # -- Resolve sdram_threshold: CLI > fpga.yaml > hard-coded default --------
    fpga_cfg_path = Path(args.fpga_config).expanduser().resolve()
    fpga_cfg: Dict[str, Any] = {}
    if fpga_cfg_path.exists():
        fpga_cfg = yaml.safe_load(fpga_cfg_path.read_text()) or {}
        print(f"[export_weights] fpga config  → {fpga_cfg_path}")

    sdram_threshold: int = (
        args.sdram_threshold
        if args.sdram_threshold is not None
        else fpga_cfg.get("placement", {}).get("sdram_threshold_bytes", SDRAM_THRESHOLD_DEFAULT)
    )

    # -- Resolve run-id based paths -------------------------------------------
    outputs_base = Path(args.outputs_dir).expanduser().resolve()
    run_id       = args.run_id if args.run_id >= 0 else latest_run_id(outputs_base)

    npz_path  = Path(args.npz).expanduser().resolve()  if args.npz  \
                else outputs_base / f"run{run_id}" / "fpgaqparms.npz"
    meta_path = Path(args.meta).expanduser().resolve() if args.meta \
                else outputs_base / f"run{run_id}" / "fpgaqparms.json"
    out_path  = Path(args.out).expanduser().resolve()  if args.out  \
                else outputs_base / f"run{run_id}" / "fpgaqparms.bin"

    action = "Overwriting" if out_path.exists() else "Creating"
    print(f"[export_weights] run{run_id}  {action} → {out_path}")
    print(f"[export_weights] sdram threshold : {sdram_threshold // 1024} KB")

    # 1. load_inputs()
    layers, arrays = load_inputs(npz_path=npz_path, meta_path=meta_path)

    # 2. build_section_list()
    sections = build_section_list(
        layers          = layers,
        arrays          = arrays,
        sdram_threshold = sdram_threshold,
    )

    # 3. assign_file_offsets()
    assign_file_offsets(sections)

    # 4. write_bin()
    write_bin(sections=sections, out_path=out_path)

    # 5. print_summary()
    print_summary(sections=sections, layers=layers, out_path=out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
