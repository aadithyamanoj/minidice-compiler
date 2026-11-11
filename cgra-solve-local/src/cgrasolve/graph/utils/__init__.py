"""Graph algorithm utilities for DFG and MRRG manipulation."""

from cgrasolve.graph.utils.traversal import dfs, bfs, topological_sort
from cgrasolve.graph.utils.pathfinding import dijkstra, astar, shortest_path
from cgrasolve.graph.utils.analysis import (
    find_critical_path,
    compute_asap_schedule,
    compute_alap_schedule,
    detect_cycles,
    find_strongly_connected_components
)

__all__ = [
    # Traversal
    "dfs",
    "bfs",
    "topological_sort",
    # Pathfinding
    "dijkstra",
    "astar",
    "shortest_path",
    # Analysis
    "find_critical_path",
    "compute_asap_schedule",
    "compute_alap_schedule",
    "detect_cycles",
    "find_strongly_connected_components",
]
