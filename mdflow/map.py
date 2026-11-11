"""Place and route Mini_Dice p-graphs with CGRA-Solve and report each latency.

Runs under CGRA-Solve's virtualenv, which is the only interpreter with the
mapper installed. The latency it records is the switch-box flop count on the
longest input-to-output path, the same number the chip's writeback uses to
tag a result with the thread that produced it.

Usage:
    map.py --dots DIR --out DIR
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from cgrasolve.algorithms.mapper.heuristic_mapper import HeuristicMapper
from cgrasolve.architecture.architecture_loader import ArchitectureLoader
from cgrasolve.architecture.fasm_generator import FasmGenerator
from cgrasolve.graph.utils.pipeline_depth import compute_pipeline_depth
from cgrasolve.schedules.parsers.latency_dot_parser import LatencyDotParser
from cgrasolve.workload.parsers.dot_parser import DFGDotParser

CGRA_SOLVE = Path(os.environ["MINIDICE_CGRA_SOLVE"])
ARCH_DIR = CGRA_SOLVE / "benchmarks/architectures/mini_dice_static/II_1"
SRC_MAP = CGRA_SOLVE / "benchmarks/architectures/mini_dice_static/source_map.json"
LAT_DOT = CGRA_SOLVE / "benchmarks/schedules/mini_dice/mini_dice_supported_ops.dot"


def map_graph(dot: Path, out_dir: Path, bundle, latency, src_map, fasm_gen,
              node_ids: set) -> dict:
    dfg = DFGDotParser().parse(dot)
    dfg.preprocess_for_mapping()
    fixed = [(node.id, src_map[node.id]) for node in dfg.get_nodes()
             if node.id in src_map and src_map[node.id] in node_ids]
    started = time.time()
    result = HeuristicMapper(
        dfg=dfg, mrrg=bundle.mrrg, latency_spec=latency,
        max_time=120.0, max_iterations=40, router_max_iterations=300,
        min_ii=1, max_ii=1, fixed_placement=fixed, debug=False,
    ).map()
    elapsed = time.time() - started
    status = result["status"]
    if status != "success":
        return {"dot": dot.name, "status": status, "seconds": round(elapsed, 2)}

    fasm = out_dir / f"{dot.stem}.fasm"
    fasm_gen.generate(dfg=dfg, placement=result["placement"], routes=result["routes"],
                      output_path=str(fasm), ii=1,
                      route_metadata=result.get("route_metadata"))
    depth = compute_pipeline_depth(result["routes"], result["placement"])
    return {"dot": dot.name, "status": status, "seconds": round(elapsed, 2),
            "latency": depth, "fasm": str(fasm)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dots", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = ArchitectureLoader().load_from_directory(str(ARCH_DIR))
    node_ids = {node.id for node in bundle.mrrg.get_nodes()}
    latency = LatencyDotParser().parse(LAT_DOT)
    src_map = json.loads(SRC_MAP.read_text())
    src_map = {**src_map["sources"], **src_map["sinks"]}
    fasm_gen = FasmGenerator(compiler_arch_path=str(ARCH_DIR / "compiler_arch.json"))

    results = []
    for dot in sorted(args.dots.glob("*.dot")):
        outcome = map_graph(dot, args.out, bundle, latency, src_map, fasm_gen,
                            node_ids)
        results.append(outcome)
        if outcome["status"] == "success":
            print(f"  {dot.stem:<12} lat {outcome['latency']}  "
                  f"({outcome['seconds']}s)")
        else:
            print(f"  {dot.stem:<12} FAILED to map ({outcome['seconds']}s)")

    (args.out / "mapping.json").write_text(json.dumps(results, indent=2))
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
