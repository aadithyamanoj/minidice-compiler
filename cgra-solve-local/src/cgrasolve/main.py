"""Main entry point for FastMap CLI."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for FastMap command-line interface."""
    parser = argparse.ArgumentParser(
        description="FastMap: CGRA Mapping Framework"
    )

    parser.add_argument(
        "--dfg",
        type=str,
        required=True,
        help="Path to DFG file"
    )

    parser.add_argument(
        "--mrrg",
        type=str,
        required=True,
        help="Path to MRRG (architecture) file"
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="modulo",
        choices=["modulo", "list"],
        help="Scheduling algorithm to use"
    )

    parser.add_argument(
        "--placer",
        type=str,
        default="greedy",
        choices=["greedy", "sa", "ga", "ilp"],
        help="Placement algorithm to use"
    )

    parser.add_argument(
        "--router",
        type=str,
        default="dijkstra",
        choices=["dijkstra", "astar"],
        help="Routing algorithm to use"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for mapping result"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the mapping result"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # TODO: Implement actual mapping pipeline
    print(f"FastMap: Mapping {args.dfg} onto {args.mrrg}")
    print(f"Using scheduler={args.scheduler}, placer={args.placer}, router={args.router}")

    if args.validate:
        print("Validation enabled")

    # Example workflow:
    # 1. Load DFG and MRRG
    # 2. Run scheduler
    # 3. Run placer
    # 4. Run router
    # 5. Validate if requested
    # 6. Save results

    print("\nNote: Full implementation pending. See README for usage examples.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
