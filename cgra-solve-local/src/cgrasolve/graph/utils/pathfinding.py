"""Pathfinding algorithms for routing in MRRGs."""

from typing import List, Dict, Set, Optional, Callable, Tuple
import heapq
from cgrasolve.graph.graph_base import Graph, Node, Edge


def dijkstra(
    graph: Graph,
    start_node_id: str,
    end_node_id: Optional[str] = None,
    weight_fn: Optional[Callable[[Edge], float]] = None
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Dijkstra's shortest path algorithm.

    Args:
        graph: The graph to search
        start_node_id: Starting node ID
        end_node_id: Optional ending node ID (if None, finds shortest paths to all nodes)
        weight_fn: Function to compute edge weight (default: uses edge latency or 1)

    Returns:
        Tuple of (distances dict, predecessors dict)
        - distances: node_id -> shortest distance from start
        - predecessors: node_id -> previous node in shortest path
    """
    if weight_fn is None:
        weight_fn = lambda edge: getattr(edge, 'latency', 1)

    distances: Dict[str, float] = {start_node_id: 0}
    predecessors: Dict[str, Optional[str]] = {start_node_id: None}
    visited: Set[str] = set()

    # Priority queue: (distance, node_id)
    pq: List[Tuple[float, str]] = [(0, start_node_id)]

    while pq:
        current_dist, current_id = heapq.heappop(pq)

        if current_id in visited:
            continue

        visited.add(current_id)

        # Early termination if we reached the target
        if end_node_id and current_id == end_node_id:
            break

        # Explore neighbors
        for edge in graph.get_outgoing_edges(current_id):
            neighbor_id = edge.destination.id
            if neighbor_id in visited:
                continue

            edge_weight = weight_fn(edge)
            new_dist = current_dist + edge_weight

            if neighbor_id not in distances or new_dist < distances[neighbor_id]:
                distances[neighbor_id] = new_dist
                predecessors[neighbor_id] = current_id
                heapq.heappush(pq, (new_dist, neighbor_id))

    return distances, predecessors


def reconstruct_path(
    predecessors: Dict[str, Optional[str]],
    start_node_id: str,
    end_node_id: str,
    graph: Graph
) -> Optional[List[Node]]:
    """
    Reconstruct a path from the predecessors dictionary.

    Args:
        predecessors: Dictionary from Dijkstra or A* algorithm
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        graph: The graph containing the nodes

    Returns:
        List of nodes forming the path, or None if no path exists
    """
    if end_node_id not in predecessors:
        return None

    path: List[str] = []
    current = end_node_id

    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()

    if path[0] != start_node_id:
        return None

    return [graph.get_node(nid) for nid in path if graph.get_node(nid)]


def astar(
    graph: Graph,
    start_node_id: str,
    end_node_id: str,
    heuristic_fn: Callable[[Node, Node], float],
    weight_fn: Optional[Callable[[Edge], float]] = None
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    A* pathfinding algorithm.

    Args:
        graph: The graph to search
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        heuristic_fn: Heuristic function h(node, goal) estimating distance to goal
        weight_fn: Function to compute edge weight (default: uses edge latency or 1)

    Returns:
        Tuple of (distances dict, predecessors dict)
    """
    if weight_fn is None:
        weight_fn = lambda edge: getattr(edge, 'latency', 1)

    start_node = graph.get_node(start_node_id)
    end_node = graph.get_node(end_node_id)

    if not start_node or not end_node:
        return {}, {}

    # g_score: actual distance from start
    g_score: Dict[str, float] = {start_node_id: 0}

    # f_score: g_score + heuristic
    f_score: Dict[str, float] = {start_node_id: heuristic_fn(start_node, end_node)}

    predecessors: Dict[str, Optional[str]] = {start_node_id: None}
    visited: Set[str] = set()

    # Priority queue: (f_score, node_id)
    pq: List[Tuple[float, str]] = [(f_score[start_node_id], start_node_id)]

    while pq:
        current_f, current_id = heapq.heappop(pq)

        if current_id in visited:
            continue

        visited.add(current_id)

        # Found the goal
        if current_id == end_node_id:
            break

        current_node = graph.get_node(current_id)
        if not current_node:
            continue

        # Explore neighbors
        for edge in graph.get_outgoing_edges(current_id):
            neighbor_id = edge.destination.id
            if neighbor_id in visited:
                continue

            neighbor_node = graph.get_node(neighbor_id)
            if not neighbor_node:
                continue

            tentative_g = g_score[current_id] + weight_fn(edge)

            if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                g_score[neighbor_id] = tentative_g
                f_score[neighbor_id] = tentative_g + heuristic_fn(neighbor_node, end_node)
                predecessors[neighbor_id] = current_id
                heapq.heappush(pq, (f_score[neighbor_id], neighbor_id))

    return g_score, predecessors


def shortest_path(
    graph: Graph,
    start_node_id: str,
    end_node_id: str,
    weight_fn: Optional[Callable[[Edge], float]] = None
) -> Optional[List[Node]]:
    """
    Find the shortest path between two nodes using Dijkstra's algorithm.

    Args:
        graph: The graph to search
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        weight_fn: Function to compute edge weight

    Returns:
        List of nodes forming the shortest path, or None if no path exists
    """
    _, predecessors = dijkstra(graph, start_node_id, end_node_id, weight_fn)
    return reconstruct_path(predecessors, start_node_id, end_node_id, graph)


def manhattan_distance(node1: Node, node2: Node) -> float:
    """
    Manhattan distance heuristic for A* in grid-based architectures.

    Args:
        node1: First node (must have 'coordinates' attribute)
        node2: Second node (must have 'coordinates' attribute)

    Returns:
        Manhattan distance between the two nodes
    """
    coords1 = getattr(node1, 'coordinates', None)
    coords2 = getattr(node2, 'coordinates', None)

    if coords1 is None or coords2 is None:
        return 0.0

    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


def euclidean_distance(node1: Node, node2: Node) -> float:
    """
    Euclidean distance heuristic for A*.

    Args:
        node1: First node (must have 'coordinates' attribute)
        node2: Second node (must have 'coordinates' attribute)

    Returns:
        Euclidean distance between the two nodes
    """
    coords1 = getattr(node1, 'coordinates', None)
    coords2 = getattr(node2, 'coordinates', None)

    if coords1 is None or coords2 is None:
        return 0.0

    return ((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)**0.5
