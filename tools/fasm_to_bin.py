"""fasm_to_bin.py -- run dora's bitgen on one (workspace.pkl, fasm) pair.

This is the same path mini_dice/build_bitgen.py takes; we just expose it as a
self-contained CLI so the Makefile can call it per-pgraph without writing a
bitgen_inputs.json sidecar each time.

Usage:
    python3 fasm_to_bin.py --fasm KERNEL.fasm --workspace workspace.pkl \
                           --out OUTFILE.bin [--compile-report path.json]

Requires the dora repo on PYTHONPATH (or installed). The mini_dice callbacks
must be importable -- we add the dora mini_dice example dir to sys.path so
register_mini_dice_callback_factories() resolves.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_DORA_REPO = (Path(__file__).resolve().parents[1]
                     / "third_party/dora")
DEFAULT_WORKSPACE = (
    DEFAULT_DORA_REPO
    / "examples/devices/dice-isca/mini_dice/static-build/workspace.pkl"
)


def _add_dora_to_path(dora_repo: Path) -> None:
    """Ensure dora and the mini_dice callbacks dir are importable."""
    candidates = [
        dora_repo / "dora.py" / "src",
        dora_repo / "examples/devices/dice-isca/mini_dice",
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasm", required=True, type=Path, help="Input FASM file")
    ap.add_argument("--out",  required=True, type=Path, help="Output .bin path")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                    help=f"Dora workspace pickle (default: {DEFAULT_WORKSPACE})")
    ap.add_argument("--dora-repo", type=Path, default=DEFAULT_DORA_REPO,
                    help=f"Dora repo root (default: {DEFAULT_DORA_REPO})")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fasm.exists():
        raise SystemExit(f"FASM not found: {args.fasm}")
    if not args.workspace.exists():
        raise SystemExit(f"workspace.pkl not found: {args.workspace}")

    _add_dora_to_path(args.dora_repo)
    # Imports must happen after sys.path is configured.
    from dora.configfabric.persistence import load_workspace_pickle
    from dora.passes.flow import Flow
    from dora.passes.flow_steps import GenerateBitstreamFromFasmStep
    from callbacks import register_mini_dice_callback_factories

    register_mini_dice_callback_factories()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  bitgen: {args.fasm.name}  ->  {args.out}")
    ws = load_workspace_pickle(pickle_path=str(args.workspace))
    flow = Flow(steps=(
        GenerateBitstreamFromFasmStep(
            fasm_path=str(args.fasm),
            output_path=str(args.out),
            output_format="bin",
        ),
    ))
    result = flow.run(workspace=ws)
    artifact = dict(result.artifacts).get("bitstream_path")
    if not isinstance(artifact, str) or artifact == "":
        raise SystemExit("dora flow returned no bitstream_path artifact")
    if not Path(artifact).exists():
        raise SystemExit(f"dora reported writing {artifact} but it doesn't exist")
    size = os.path.getsize(artifact)
    print(f"           wrote {size} bytes")


if __name__ == "__main__":
    main()
