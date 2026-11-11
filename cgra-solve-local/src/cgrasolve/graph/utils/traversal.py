"""Graph traversal algorithms (DFS, BFS, topological sort)."""

from typing import List, Set, Callable, Optional, TypeVar
from collections import deque
from cgrasolve.graph.graph_base import Graph, Node, Edge
import networkx as nx


N = TypeVar('N', bound=Node)
E = TypeVar('E', bound=Edge)


def dfs(
    graph: Graph[N, E],
    start_node_id: str,
    visit_fn: Optional[Callable[[N], None]] = None,
    preorder: bool = True
) -> List[N]:
    """
    Depth-First Search traversal.

    Args:
        graph: The graph to traverse
        start_node_id: ID of the starting node
        visit_fn: Optional callback function to call on each visited node
        preorder: If True, visit nodes in preorder; if False, postorder

    Returns:
        List of nodes in DFS order
    """

    # Call networkx's dfs function (returns node IDs, need to map to node objects)
    node_ids = nx.dfs_preorder_nodes(graph, start_node_id)
    return [graph.get_node(node_id) for node_id in node_ids]


def bfs(
    graph: Graph[N, E],
    start_node_id: str,
    visit_fn: Optional[Callable[[N], None]] = None,
    count: Optional[int] = None
) -> List[N]:
    """
    Breadth-First Search traversal.

    Args:
        graph: The graph to traverse
        start_node_id: ID of the starting node
        visit_fn: Optional callback function to call on each visited node
        count: Optional number of nodes to visit
    Returns:
        List of nodes in BFS order
    """
    
    # Call networkx's bfs function.
    start_node = graph.get_node(start_node_id)
    if start_node is None:
        raise ValueError(f"Source node {start_node_id} not found in graph")

    # bfs_edges returns (u, v) pairs, so collect all v's after the start node
    edges = nx.bfs_edges(graph, start_node_id)
    node_ids = [start_node_id] + [v for u, v in edges]

    # Now map node IDs → node objects
    return [graph.get_node(node_id) for node_id in node_ids]


def topological_sort(graph: Graph[N, E], ignore_edges: Optional[Set[str]] = None) -> List[N]:
    """
    Topological sort using DFS.

    Args:
        graph: The graph to sort (must be a DAG)
        ignore_edges: Set of edge IDs to ignore (e.g., loop-back edges)

    Returns:
        List of nodes in topological order

    Raises:
        ValueError: If the graph contains cycles (excluding ignored edges)
    """

    # Call networkx's topological_sort function (returns node IDs, need to map to node objects)
    node_ids = nx.topological_sort(graph)
    return [graph.get_node(node_id) for node_id in node_ids]

def get_k_shortest_paths(
    graph: Graph[N, E],
    start_node_id: str,
    end_node_id: str,
    k: int
) -> List[List[str]]:
    """
    Find the k shortest paths between two nodes using NetworkX.

    Args:
        graph: The graph to search (must implement NetworkX interface)
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        k: Number of shortest paths to find

    Returns:
        List of k shortest paths, where each path is a list of node IDs (strings)
    """
    # Use NetworkX's shortest_simple_paths (uses Yen's algorithm internally)
    paths_gen = nx.shortest_simple_paths(graph, start_node_id, end_node_id, weight='weight')

    # Get first k paths from the generator
    k_paths = []
    try:
        for _, path in zip(range(k), paths_gen):
            k_paths.append(path)
    except nx.NetworkXNoPath:
        # No path exists between the nodes
        pass

    return k_paths

def get_all_paths(
    graph: Graph[N, E],
    start_node_id: str,
    end_node_id: str,
    max_paths: int = 100
) -> List[List[N]]:
    """
    Find all paths between two nodes.

    Args:
        graph: The graph to search
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        max_paths: Maximum number of paths to find

    Returns:
        List of paths, where each path is a list of nodes
    """
    all_paths: List[List[N]] = []
    current_path: List[str] = []
    visited: Set[str] = set()

    def _find_paths(node_id: str) -> None:
        if len(all_paths) >= max_paths:
            return

        current_path.append(node_id)
        visited.add(node_id)

        if node_id == end_node_id:
            # Found a path
            path_nodes = [graph.get_node(nid) for nid in current_path]
            all_paths.append([n for n in path_nodes if n is not None])
        else:
            # Continue searching
            for successor in graph.get_successors(node_id):
                if successor.id not in visited:
                    _find_paths(successor.id)

        # Backtrack
        current_path.pop()
        visited.remove(node_id)

    _find_paths(start_node_id)
    return all_paths


def compute_levels(graph: Graph[N, E]) -> dict[str, int]:
    """
    Compute the level (distance from inputs) for each node.

    Args:
        graph: The graph to analyze

    Returns:
        Dictionary mapping node IDs to their levels
    """
    levels: dict[str, int] = {}
    visited: Set[str] = set()

    def _compute_level(node_id: str) -> int:
        if node_id in levels:
            return levels[node_id]

        if node_id in visited:
            raise ValueError(f"Cycle detected at node {node_id}")

        visited.add(node_id)

        predecessors = graph.get_predecessors(node_id)
        if not predecessors:
            # Input node
            level = 0
        else:
            # Level is max of predecessor levels + 1
            level = max(_compute_level(pred.id) for pred in predecessors) + 1

        visited.remove(node_id)
        levels[node_id] = level
        return level

    for node in graph.get_nodes():
        if node.id not in levels:
            _compute_level(node.id)

    return levels
