"""Compile a CUDA kernel and run it on the Mini_Dice chip in simulation.

    source env.sh
    python3 minidice_flow.py --cuda KERNEL.cu --kernel MANGLED --grid N \
        --block 16 --workload nn --out DIR

The CUDA is lowered by the VEGAS frontend, retargeted onto Mini_Dice's INT16
fabric, placed and routed, and then run through the chip testbench. A kernel
the fabric cannot execute is reported by name rather than miscompiled. The
built-in `nn` workload supplies the quantized inputs for Rodinia's euclid
nearest-neighbor kernel and checks the chip's stores against them.

Collateral lives in third_party/ as git submodules. env.sh names the
interpreters and the LLVM plugin, which are build products and are not in
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
from mdflow.workload import assemble, nn_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parent
THIRD = ROOT / "third_party"
VEGAS = THIRD / "vegas"
CGRA_SOLVE = THIRD / "cgra-solve"
DORA_REPO = THIRD / "dora"
TESTKERNELS = THIRD / "testkernels"
MINI_DICE = THIRD / "mini_dice"
BSG_ROOT = THIRD / "basejump_stl"
WORKSPACE = (DORA_REPO /
             "examples/devices/dice-isca/mini_dice/static-build/workspace.pkl")
PACK_META = TESTKERNELS / "scripts/json_to_meta_mem.py"
PACK_BITS = TESTKERNELS / "scripts/bins_to_bitstream_mem.py"
PACK_FASM = TESTKERNELS / "scripts/fasm_to_bin.py"
DICE_CONFIG = MINI_DICE / "rtl/includes/dice_config.vh"

WORKLOADS = {"nn": nn_inputs}


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
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(CGRA_SOLVE / "src") + os.pathsep + env.get("PYTHONPATH", ""))
    env["MINIDICE_CGRA_SOLVE"] = str(CGRA_SOLVE)
    stage([os.environ["MINIDICE_MAPPER_PYTHON"], str(ROOT / "mdflow/map.py"),
           "--dots", str(dots), "--out", str(out)], env=env)
    return json.loads((out / "mapping.json").read_text())


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


def build_simv(build_dir: Path) -> Path:
    """Compile the chip testbench once and reuse it."""
    simv = build_dir / "simv"
    dpi = MINI_DICE / "tb/cgra_core/dice_core/dpi_dice_core_runtime.cpp"
    newest = max(_source_mtime(), dpi.stat().st_mtime)
    if simv.exists() and simv.stat().st_mtime >= newest:
        return simv

    build_dir.mkdir(parents=True, exist_ok=True)
    files = build_dir / "files.f"
    stage([sys.executable, str(MINI_DICE / "tb/vcs/gen_filelist.py"),
           "--bsg-root", str(BSG_ROOT), "--out", str(files)])
    vcs = (
        f"module load vcs/X-2025.06 && cd {build_dir} && "
        f"vcs -full64 -sverilog -timescale=1ns/1ps +lint=TFIPC-L "
        f"+define+NO_FSDB -assert svaext -ignore initializer_driver_checks "
        f"-f {files} {dpi} -CFLAGS '-std=c++17 -O2' -top tb_chip_top "
        f"-o {simv} -j16"
    )
    # VCS generates C that the conda compiler rejects, so this runs under the
    # system compiler with a clean environment.
    stage(["env", "-i", f"HOME={os.environ['HOME']}", f"USER={os.environ['USER']}",
           "PATH=/usr/bin:/bin", "bash", "-lc", vcs])
    return simv


def _source_mtime() -> float:
    newest = 0.0
    for path in (MINI_DICE / "rtl").rglob("*"):
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return newest


def run_sim(simv: Path, stem: str, out: Path) -> None:
    env = os.environ.copy()
    env["DICE_HOME"] = str(MINI_DICE)
    stage([str(simv), f"+TEST_VECTOR={stem}", f"+TEST_VECTOR_DIR={out}",
           "+SETTLE=40000"], cwd=out, env=env)


def compile_and_run(args: argparse.Namespace) -> None:
    if not args.dfg and not args.cuda:
        raise SystemExit("give --cuda and --kernel, or --dfg")
    missing = [name for name in
               ("MINIDICE_COMPILER_PYTHON", "MINIDICE_MAPPER_PYTHON",
                "MINIDICE_DORA_PYTHON", "MINIDICE_CGRA_SOLVE")
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
    print(f"{len(program.graphs)} p-graphs, {len(program.gpr_of)} registers")

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

    simv = build_simv(ROOT / "build")
    run_sim(simv, args.stem, out)


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
    compile_and_run(parse_args())
