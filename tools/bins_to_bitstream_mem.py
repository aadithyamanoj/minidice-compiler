"""bins_to_bitstream_mem.py -- combine per-pgraph .bin files into a $readmemh
bitstream.mem file matching the format Mini_Dice's TB expects.

For each pgraph in the test_vector.json's pgraph_metadata[]:
  - look up the matching stage in stage_artifacts[] (same order)
  - read <bins_dir>/<stem>.bin where <stem> comes from stage_artifacts.fasm_path
    with .fasm replaced by .bin (or via --stage-name -> path map)
  - lay the bytes out as 32-bit little-endian words at byte address
    pgraph_metadata[i].meta.bitstream_addr, padded to `--bitstream-size` bits
  - emit `@<word_addr> <hex_data>` lines

Output format matches Mini_Dice/tb/test_vectors/*_bitstream.mem (see
gen_memfile.generate_bitstream_mem for the layout reference).

Usage:
    python3 bins_to_bitstream_mem.py \
        --json kernels/srad_prepare/prepare_test_vector.json \
        --bins-dir build/srad_prepare \
        --out build/srad_prepare/srad_prepare_bitstream.mem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BITSTREAM_MEM_DATA_WIDTH = 32   # axi4_full_crossbar.sv AxiDataWidth
DEFAULT_BITSTREAM_SIZE_BITS = 1074  # dora mini_dice default; override with --bitstream-size


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, type=Path,
                    help="Kernel test_vector.json (provides pgraph order + bitstream_addr)")
    ap.add_argument("--bins-dir", required=True, type=Path,
                    help="Directory containing per-stage .bin files")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output combined bitstream.mem path")
    ap.add_argument("--bitstream-size", type=int, default=DEFAULT_BITSTREAM_SIZE_BITS,
                    help=f"Bitstream payload size in bits (default {DEFAULT_BITSTREAM_SIZE_BITS})")
    return ap.parse_args()


def _stem_for_stage(stage: dict) -> str:
    """Pick the .bin stem for a stage_artifacts entry."""
    fasm_path = stage.get("fasm_path", "")
    if fasm_path:
        return Path(fasm_path).stem
    kernel = stage.get("kernel", "")
    if kernel:
        return kernel
    raise ValueError(f"stage entry missing fasm_path and kernel: {stage}")


def _bin_to_words(bin_path: Path, num_chunks: int) -> list[str]:
    """Read raw bytes and emit `num_chunks` 32-bit little-endian hex words.

    Last chunk is zero-padded if the bin is shorter than num_chunks * 4 bytes.
    """
    raw = bin_path.read_bytes()
    word_bytes = BITSTREAM_MEM_DATA_WIDTH // 8
    hex_chars  = BITSTREAM_MEM_DATA_WIDTH // 4
    words: list[str] = []
    for chunk_idx in range(num_chunks):
        byte_off = chunk_idx * word_bytes
        word = 0
        for off in range(word_bytes):
            if byte_off + off < len(raw):
                word |= raw[byte_off + off] << (8 * off)
        words.append(f"{word:0{hex_chars}x}")
    return words


def main() -> None:
    args = parse_args()
    data = json.loads(args.json.read_text())
    pgraph_meta = data.get("pgraph_metadata") or []
    stage_artifacts = data.get("stage_artifacts") or []
    if len(pgraph_meta) != len(stage_artifacts):
        raise SystemExit(
            f"pgraph_metadata has {len(pgraph_meta)} entries but stage_artifacts "
            f"has {len(stage_artifacts)} -- they must align 1:1.")

    word_bytes = BITSTREAM_MEM_DATA_WIDTH // 8
    num_chunks = (args.bitstream_size + BITSTREAM_MEM_DATA_WIDTH - 1) // BITSTREAM_MEM_DATA_WIDTH

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated bitstream memory file (CGRA-Solve mapping)\n")
        f.write(f"// Memory word width: {BITSTREAM_MEM_DATA_WIDTH} bits ({word_bytes} bytes)\n")
        f.write(f"// Bitstream payload size: {args.bitstream_size} bits\n")
        f.write(f"// Chunks per bitstream: {num_chunks}\n")
        f.write(f"// Source bins: {args.bins_dir}\n\n")

        for idx, (entry, stage) in enumerate(zip(pgraph_meta, stage_artifacts)):
            meta = entry["meta"]
            bs_addr = int(meta["bitstream_addr"])
            bs_len  = int(meta["bitstream_length"])
            base_word_addr = bs_addr // word_bytes
            stem = _stem_for_stage(stage)
            bin_path = args.bins_dir / f"{stem}.bin"
            if not bin_path.exists():
                raise SystemExit(f"missing bin for stage {idx} ({stem}): {bin_path}")
            words = _bin_to_words(bin_path, num_chunks)

            f.write(f"// pgraph[{idx}] ({stem}): bitstream_addr=0x{bs_addr:08x}, "
                    f"length={bs_len}, base_word_addr=0x{base_word_addr:08x}\n")
            for chunk_idx, hex_data in enumerate(words):
                f.write(f"@{base_word_addr + chunk_idx:08x} {hex_data}\n")
            f.write("\n")

    print(f"  wrote {args.out} ({len(pgraph_meta)} pgraphs * {num_chunks} chunks)")


if __name__ == "__main__":
    main()
