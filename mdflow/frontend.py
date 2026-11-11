"""Run the VEGAS CUDA frontend and write a step_dfg_v4 DFG.

Invoked with the VEGAS compiler interpreter, which is the one the Boxman
frontend plugin was built against:

    source VEGAS/env.sh
    python compiler/mdflow/frontend.py --cuda K.cu --kernel NAME \
        --grid N --block 16 --out dfg.json --work DIR
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["BOXMAN_ROOT"])
from src.lib.frontend.LLVMFrontend import generate_from_cuda  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", required=True, type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--grid", required=True, type=int)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    generate_from_cuda(args.cuda, args.out, args.work, args.kernel,
                       args.grid, args.block)


if __name__ == "__main__":
    main()
