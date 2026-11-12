"""Compile a CUDA kernel into the Mini_Dice memory images.

    source env.sh
    python3 minidice_flow.py --cuda KERNEL.cu --kernel MANGLED --grid N \
        --block 16 --workload nn --out DIR

The CUDA is lowered by the Boxman frontend, retargeted onto Mini_Dice's INT16
fabric, placed and routed, and packed into the metadata, bitstream, and data
images the chip loads. A kernel the fabric cannot execute is reported by name
rather than miscompiled. The built-in `nn` workload supplies the quantized
inputs for Rodinia's euclid nearest-neighbor kernel and the stores those
inputs must produce.

Boxman and DORA are submodules. Place and route uses Boxman's DICE mapper
on an architecture built from the Mini_Dice routing graph in cgra-solve-local.
The packers in tools/ and collateral/dice_config.vh are copies. env.sh names
the interpreters and the LLVM plugin, which are build products and are not in
those repositories.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mdflow.lower import LowerError, build, lower, metadata  # noqa: E402
from mdflow.vegas_map import map_dots  # noqa: E402
from mdflow.workload import (  # noqa: E402
    assemble, attn_apply_inputs, attn_mix_inputs, attn_score_inputs,
    horner_inputs, sine_inputs,
    fir3_inputs, gemm_k2_inputs, hadamard_inputs, nn_inputs, saxpy_inputs,
)

ROOT = Path(__file__).resolve().parent
THIRD = ROOT / "third_party"
DORA_REPO = THIRD / "dora"
WORKSPACE = (DORA_REPO /
             "examples/devices/dice-isca/mini_dice/static-build/workspace.pkl")
PACK_META = ROOT / "tools/json_to_meta_mem.py"
PACK_BITS = ROOT / "tools/bins_to_bitstream_mem.py"
PACK_FASM = ROOT / "tools/fasm_to_bin.py"
DICE_CONFIG = ROOT / "collateral/dice_config.vh"

WORKLOADS = {
    "nn": nn_inputs,
    "saxpy": saxpy_inputs,
    "hadamard": hadamard_inputs,
    "fir3": fir3_inputs,
    "gemm_k2": gemm_k2_inputs,
    "attn_score": attn_score_inputs,
    "attn_mix": attn_mix_inputs,
    "attn_apply": attn_apply_inputs,
    "horner": horner_inputs,
    "sine": sine_inputs,
}


def stage(cmd: list, **kwargs) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def run_frontend(cuda: Path, kernel: str, grid: int, block: int,
                 out: Path, work: Path) -> None:
    """The frontend runs under VEGAS's interpreter, where its LLVM lives."""
    script = ROOT / "mdflow/frontend.py"
    inner = (
        f"source {ROOT}/env.sh && "
        f"\"$MINIDICE_COMPILER_PYTHON\" {script} --cuda {cuda} --kernel {kernel} "
        f"--grid {grid} --block {block} --out {out} --work {work}"
    )
    stage(["bash", "-lc", inner])


def run_mapper(dots: Path, out: Path) -> list:
    """Boxman's DICE mapper, given the Mini_Dice fabric rather than VEGAS's."""
    return map_dots(dots, out)


def run_bitgen(mapped: Path, bins: Path) -> None:
    python = os.environ["MINIDICE_DORA_PYTHON"]
    bins.mkdir(parents=True, exist_ok=True)
    for fasm in sorted(mapped.glob("*.fasm")):
        stage([python, str(PACK_FASM), "--fasm", str(fasm),
               "--out", str(bins / f"{fasm.stem}.bin"),
               "--dora-repo", str(DORA_REPO), "--workspace", str(WORKSPACE)])


def write_vector(program, mapping: list, result: dict, stem: str,
                 out: Path) -> None:
    """The testbench image: metadata, bitstream, CTA descriptor, data memory,
    and the stores the chip must produce."""
    out.mkdir(parents=True, exist_ok=True)
    lowered = program.lowered
    latencies = [entry["latency"] for entry in _ordered(mapping, program)]
    vector = {
        "dice_cta_desc": {
            "cta_id": {"x": 0, "y": 0, "z": 0},
            "kernel_desc": {
                "grid_size": {"x": lowered.grid, "y": 1, "z": 1},
                "kernel_id": 1,
                "start_pc": 0x1000,
                "thread_count": lowered.block,
            },
        },
        "pgraph_metadata": metadata(program, latencies),
        "runtime": _runtime(program, result),
        "stage_artifacts": [{"fasm_path": f"{g.name}.fasm", "kernel": g.name,
                             "notes": g.notes} for g in program.graphs],
    }
    vector_path = out / f"{stem}_test_vector.json"
    vector_path.write_text(json.dumps(vector, indent=2))
    stage([sys.executable, str(PACK_META), "--json", str(vector_path),
           "--out-dir", str(out), "--stem", stem,
           "--dice-config", str(DICE_CONFIG)])
    stage([sys.executable, str(PACK_BITS), "--json", str(vector_path),
           "--bins-dir", str(out / "bins"),
           "--out", str(out / f"{stem}_bitstream.mem")])

    data_path = out / f"{stem}_data.mem"
    with data_path.open("w") as handle:
        handle.write("// 16-bit data memory, one 'addr data' hex pair per line\n")
        for addr in sorted(result["image"]):
            handle.write(f"{addr:04x} {result['image'][addr]:04x}\n")
    print(f"  wrote {data_path}")


def _ordered(mapping: list, program) -> list:
    """Mapping results in p-graph order."""
    by_name = {Path(entry["dot"]).stem: entry for entry in mapping}
    return [by_name[graph.name] for graph in program.graphs]


def _runtime(program, result: dict) -> dict:
    per_cta = []
    for cta, values in enumerate(result["per_cta"]):
        per_cta.append({
            "cta_id": {"x": cta, "y": 0, "z": 0},
            "csr_values": {f"csrX{index}": value
                           for index, value in sorted(values.items())},
        })
    return {
        "csr_values": result["csr_values"],
        "per_cta_csr_overrides": per_cta,
        "axi": {"expected_writes": result["expected_writes"]},
    }


def compile_kernel(args: argparse.Namespace) -> None:
    if not args.dfg and not args.cuda:
        raise SystemExit("give --cuda and --kernel, or --dfg")
    missing = [name for name in
               ("MINIDICE_COMPILER_PYTHON", "MINIDICE_DORA_PYTHON")
               if name not in os.environ]
    if missing:
        raise SystemExit("source env.sh first (missing " + ", ".join(missing) + ")")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    dfg_path = out / "dfg.json"
    if args.dfg:
        dfg_path = args.dfg
    else:
        run_frontend(args.cuda, args.kernel, args.grid, args.block,
                     dfg_path, out / "frontend")
    dfg = json.loads(dfg_path.read_text())

    try:
        lowered = lower(dfg)
    except LowerError as error:
        raise SystemExit(f"cannot lower {dfg['meta']['kernel']}: {error}")

    print(f"kernel {lowered.kernel}: {lowered.grid} CTAs x {lowered.block} threads")
    for note in lowered.approximations:
        print(f"  approximation: {note}")

    program = build(lowered)
    print(f"{len(program.graphs)} p-graphs, "
          f"{len(set(program.gpr_of.values()))} registers")

    dots = out / "dots"
    dots.mkdir(exist_ok=True)
    for graph in program.graphs:
        (dots / f"{graph.name}.dot").write_text(graph.dot)

    mapping = run_mapper(dots, out / "mapped")
    if any(entry["status"] != "success" for entry in mapping):
        raise SystemExit("place and route failed for a p-graph")

    run_bitgen(out / "mapped", out / "bins")

    inputs = WORKLOADS[args.workload](lowered.grid * lowered.block)
    result = assemble(program, inputs)
    write_vector(program, mapping, result, args.stem, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--dfg", type=Path,
                        help="skip the CUDA frontend and lower this DFG")
    parser.add_argument("--kernel", default="")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--workload", default="nn", choices=sorted(WORKLOADS))
    parser.add_argument("--stem", default="kernel")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    compile_kernel(parse_args())
