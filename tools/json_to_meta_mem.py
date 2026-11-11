"""json_to_meta_mem.py -- pack a kernel test_vector.json into the three
$readmemh memory files Mini_Dice's TB consumes:

    <stem>_meta.mem      (sequential pgraph_meta_t, one per pc entry)
    <stem>_cta_desc.mem  (single dice_cta_desc_t at @0)
    <stem>_runtime.json  (csr_values + axi.expected_writes -- non-readmemh sidecar)

The bit-packing matches the packed-struct layout in
Mini_Dice/rtl/includes/dice_pkg.sv and dice_config.vh. Widths default to the
values currently in those files; pass --dice-config / --dice-pkg to re-read
them from a local Mini_Dice checkout if the RTL parameters change.

Usage:
    python3 json_to_meta_mem.py --json kernels/srad_prepare/prepare_test_vector.json \
                                --out-dir build/srad_prepare \
                                --stem srad_prepare
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults sourced from Mini_Dice/rtl/includes/dice_config.vh as of 2026-05-20.
# These are auto-overridden if the file is found at one of DEFAULT_CONFIG_PATHS
# below, or via --dice-config. Verified that the defaults below reproduce
# Mini_Dice's reference _meta.mem packed width (102 bits) and _cta_desc.mem
# packed width (120 bits).
# ---------------------------------------------------------------------------

DEFAULT_RTL_DEFINES: dict[str, int] = {
    "DICE_ADDR_WIDTH": 16,
    "DICE_MAX_GRID_SIZE": 65536,
    "DICE_NUM_MAX_THREADS_PER_CORE": 16,
    "DICE_GPR_NUM": 8,
    "DICE_PR_NUM": 2,
    "DICE_CR_NUM": 8,
    "DICE_CGRA_MEM_PORTS": 4,
    "DICE_MAX_PGRAPHS": 32,
}

DEFAULT_CONFIG_PATHS = (
    Path(__file__).resolve().parents[1] / "collateral/dice_config.vh",
)

# tb_dice_core.sv:  WORD_SIZE = 256 bytes  -> 2048-bit metadata memory width
METADATA_MEM_DATA_WIDTH = 256 * 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _load_defines(path: Path | None) -> dict[str, int]:
    """Load RTL `define` widths. If --dice-config wasn't given, try the
    auto-detect locations; finally fall back to DEFAULT_RTL_DEFINES."""
    if path is None:
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return dict(DEFAULT_RTL_DEFINES)
    out = dict(DEFAULT_RTL_DEFINES)
    rgx = re.compile(r"^\s*`define\s+(\w+)\s+(.+?)\s*$")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        m = rgx.match(line)
        if m:
            name, value = m.groups()
            try:
                out[name] = int(value.strip(), 0)
            except ValueError:
                pass
    return out


def sv_clog2(n: int) -> int:
    return 0 if n <= 1 else math.ceil(math.log2(n))


def parse_int(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return int(v, 0)
    raise ValueError(f"Cannot parse {v!r} as int")


class BitPacker:
    """Packs fields MSB-first (first push becomes most-significant)."""
    def __init__(self) -> None:
        self.value = 0
        self.total_bits = 0

    def push(self, val: int, width: int) -> None:
        mask = (1 << width) - 1
        self.value = (self.value << width) | (val & mask)
        self.total_bits += width

    def to_hex(self, pad_width: int | None = None) -> str:
        width = pad_width or self.total_bits
        hex_chars = (width + 3) // 4
        return format(self.value, f"0{hex_chars}x")


# ---------------------------------------------------------------------------
# Width derivation (call once after loading RTL defines)
# ---------------------------------------------------------------------------

class Widths:
    def __init__(self, defines: dict[str, int]) -> None:
        self.addr_width        = defines["DICE_ADDR_WIDTH"]
        self.max_grid_size     = defines["DICE_MAX_GRID_SIZE"]
        self.max_threads       = defines["DICE_NUM_MAX_THREADS_PER_CORE"]
        self.gpr_num           = defines["DICE_GPR_NUM"]
        self.pr_num            = defines["DICE_PR_NUM"]
        self.cr_num            = defines["DICE_CR_NUM"]
        self.mem_ports         = defines["DICE_CGRA_MEM_PORTS"]
        self.max_pgraphs       = defines["DICE_MAX_PGRAPHS"]

        self.cta_id_width      = sv_clog2(self.max_grid_size)
        self.tid_width         = sv_clog2(self.max_threads)
        self.pr_index_width    = sv_clog2(self.pr_num)
        self.pgraph_off_width  = sv_clog2(self.max_pgraphs)
        self.bitstream_len     = 8
        self.reg_num           = self.gpr_num + self.pr_num + self.cr_num
        self.reg_index_width   = sv_clog2(self.reg_num)
        self.ld_dest_count     = self.mem_ports
        self.num_stores_width  = sv_clog2(self.mem_ports + 1)
        self.thread_count_w    = self.tid_width + 1

        self.branch_meta_width = (
            1 + 1 + self.pr_index_width + 1 + 1
            + self.pgraph_off_width + self.pgraph_off_width
        )
        self.pgraph_meta_width = (
            self.addr_width + 2 + 8
            + self.reg_num + self.reg_num
            + self.ld_dest_count * self.reg_index_width
            + self.num_stores_width
            + self.branch_meta_width
            + 1 + 1
        )
        self.grid_size_width      = 3 * (self.cta_id_width + 1)
        self.cta_id_width_total   = 3 * self.cta_id_width
        self.kernel_desc_width    = self.grid_size_width + self.thread_count_w + self.addr_width
        self.cta_desc_width       = self.kernel_desc_width + self.cta_id_width_total


# ---------------------------------------------------------------------------
# Packers (MSB-first to match SV packed struct order)
# ---------------------------------------------------------------------------

def pack_branch_meta(bm: dict, w: Widths) -> BitPacker:
    p = BitPacker()
    p.push(parse_int(bm["branch_ena"]),              1)
    p.push(parse_int(bm["branch_uni"]),              1)
    p.push(parse_int(bm["branch_pred_reg"]),         w.pr_index_width)
    p.push(parse_int(bm["branch_neg_pred"]),         1)
    p.push(parse_int(bm["is_return"]),               1)
    p.push(parse_int(bm["branch_jump_target_offset"]), w.pgraph_off_width)
    p.push(parse_int(bm["branch_reconv_offset"]),    w.pgraph_off_width)
    return p


def pack_pgraph_meta(meta: dict, w: Widths) -> BitPacker:
    p = BitPacker()
    p.push(parse_int(meta["bitstream_addr"]),   w.addr_width)
    p.push(parse_int(meta["unrolling_factor"]), 2)
    p.push(parse_int(meta["lat"]),              8)
    p.push(parse_int(meta["in_regs_bitmap"]),   w.reg_num)
    p.push(parse_int(meta["out_regs_bitmap"]),  w.reg_num)
    ld = meta["ld_dest_regs"]
    for i in range(w.ld_dest_count):
        v = parse_int(ld[i]) if i < len(ld) else 0
        p.push(v, w.reg_index_width)
    p.push(parse_int(meta["num_stores"]),       w.num_stores_width)
    bm = pack_branch_meta(meta["branch_meta"], w)
    p.push(bm.value, bm.total_bits)
    p.push(parse_int(meta["barrier"]),          1)
    p.push(parse_int(meta["parameter_load"]),   1)
    return p


def pack_grid_size(gs: dict, w: Widths) -> BitPacker:
    p = BitPacker()
    p.push(parse_int(gs["x"]), w.cta_id_width + 1)
    p.push(parse_int(gs["y"]), w.cta_id_width + 1)
    p.push(parse_int(gs["z"]), w.cta_id_width + 1)
    return p


def pack_cta_id(cid: dict, w: Widths) -> BitPacker:
    p = BitPacker()
    p.push(parse_int(cid["x"]), w.cta_id_width)
    p.push(parse_int(cid["y"]), w.cta_id_width)
    p.push(parse_int(cid["z"]), w.cta_id_width)
    return p


def pack_kernel_desc(kd: dict, w: Widths) -> BitPacker:
    p = BitPacker()
    gs = pack_grid_size(kd["grid_size"], w)
    p.push(gs.value, gs.total_bits)
    if "thread_count" in kd:
        tc = parse_int(kd["thread_count"])
    elif "cta_size" in kd:
        cs = kd["cta_size"]
        tc = parse_int(cs["x"]) * parse_int(cs["y"]) * parse_int(cs["z"])
    else:
        raise KeyError("kernel_desc must provide thread_count or cta_size")
    p.push(tc, w.thread_count_w)
    p.push(parse_int(kd["start_pc"]), w.addr_width)
    return p


def pack_cta_desc(desc: dict, w: Widths) -> BitPacker:
    p = BitPacker()
    kd = pack_kernel_desc(desc["kernel_desc"], w)
    p.push(kd.value, kd.total_bits)
    cid = pack_cta_id(desc["cta_id"], w)
    p.push(cid.value, cid.total_bits)
    return p


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def compute_mem_addr(pc: int, mem_data_width: int) -> int:
    word_bytes = mem_data_width // 8
    return pc // word_bytes


def write_meta_mem(pgraph_list: list, w: Widths, out_path: Path,
                   mem_data_width: int = METADATA_MEM_DATA_WIDTH) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated metadata memory file\n")
        f.write(f"// Memory data width: {mem_data_width} bits\n")
        f.write(f"// pgraph_meta_t packed width: {w.pgraph_meta_width} bits\n\n")
        for entry in pgraph_list:
            pc = parse_int(entry["pc"])
            addr = compute_mem_addr(pc, mem_data_width)
            packed = pack_pgraph_meta(entry["meta"], w)
            f.write(f"@{addr:08x} {packed.to_hex(pad_width=mem_data_width)}\n")
    print(f"  wrote {out_path} ({len(pgraph_list)} entries)")


def write_cta_desc_mem(cta_desc: dict, w: Widths, out_path: Path) -> None:
    packed = pack_cta_desc(cta_desc, w)
    pad = ((packed.total_bits + 3) // 4) * 4
    kd = cta_desc["kernel_desc"]
    cid = cta_desc["cta_id"]
    tc = parse_int(kd.get("thread_count", 0)) or 1
    with out_path.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated CTA descriptor ($readmemh format)\n")
        f.write(f"// dice_cta_desc_t packed width: {packed.total_bits} bits "
                f"(padded to {pad} bits)\n")
        f.write(f"// grid_size=({kd['grid_size']['x']},{kd['grid_size']['y']},"
                f"{kd['grid_size']['z']}), thread_count={tc}, "
                f"start_pc={kd['start_pc']}\n")
        f.write(f"// cta_id=({cid['x']},{cid['y']},{cid['z']})\n\n")
        f.write(f"@00000000 {packed.to_hex(pad_width=pad)}\n")
    print(f"  wrote {out_path} ({packed.total_bits} bits, padded to {pad})")


def write_runtime_sidecar(data: dict, out_path: Path) -> None:
    runtime = data.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a JSON object when present")
    csr = runtime.get("csr_values")
    if not isinstance(csr, dict):
        raise ValueError("runtime.csr_values required")
    csr_out = {}
    for i in range(8):
        k = f"csrX{i}"
        if k not in csr:
            raise ValueError(f"runtime.csr_values is missing {k}")
        csr_out[k] = parse_int(csr[k])
    axi = runtime.get("axi") or {}
    writes_in = (axi or {}).get("expected_writes") or []
    writes_out = []
    for e in writes_in:
        item = {"addr": parse_int(e["addr"]), "data": parse_int(e["data"])}
        if "strb"  in e: item["strb"]  = parse_int(e["strb"])
        if "count" in e: item["count"] = parse_int(e["count"])
        writes_out.append(item)
    payload = {
        "csr_values": csr_out,
        "axi": {"expected_writes": writes_out},
    }
    # Preserve per_cta_csr_overrides if the kernel uses them.
    if "per_cta_csr_overrides" in runtime:
        payload["per_cta_csr_overrides"] = runtime["per_cta_csr_overrides"]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json",    required=True, type=Path, help="kernel test_vector.json")
    ap.add_argument("--out-dir", required=True, type=Path, help="output directory")
    ap.add_argument("--stem",    type=str, default=None,
                    help="filename stem (default: input JSON stem with _test_vector stripped)")
    ap.add_argument("--dice-config", type=Path, default=None,
                    help="optional: path to Mini_Dice/rtl/includes/dice_config.vh "
                         "to re-derive widths from current RTL")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    defines = _load_defines(args.dice_config)
    w = Widths(defines)

    if not args.json.exists():
        raise SystemExit(f"JSON not found: {args.json}")

    data = json.loads(args.json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem or args.json.stem.replace("_test_vector", "")
    print(f"Packing {args.json.name}  ->  {args.out_dir}/{stem}_{{meta,cta_desc}}.mem  (mem_data_width={METADATA_MEM_DATA_WIDTH})")
    print(f"  pgraph_meta_t width: {w.pgraph_meta_width} bits")
    print(f"  dice_cta_desc_t width: {w.cta_desc_width} bits")

    if "pgraph_metadata" in data:
        write_meta_mem(data["pgraph_metadata"], w, args.out_dir / f"{stem}_meta.mem")
    if "dice_cta_desc" in data:
        write_cta_desc_mem(data["dice_cta_desc"], w, args.out_dir / f"{stem}_cta_desc.mem")
    write_runtime_sidecar(data, args.out_dir / f"{stem}_runtime.json")


if __name__ == "__main__":
    main()
