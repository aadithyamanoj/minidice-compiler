"""Graph matching and pattern detection algorithms."""

from typing import Dict, List, Set, Optional, Callable
from cgrasolve.graph.graph_base import Graph, Node


def subgraph_isomorphism(
    pattern: Graph,
    target: Graph,
    node_match_fn: Optional[Callable[[Node, Node], bool]] = None
) -> List[Dict[str, str]]:
    """
    Find subgraph isomorphisms between pattern and target graphs.

    Args:
        pattern: Pattern graph to match
        target: Target graph to search in
        node_match_fn: Function to check if two nodes match

    Returns:
        List of mappings (pattern node ID -> target node ID)
    """
    if node_match_fn is None:
        node_match_fn = lambda n1, n2: True

    pattern_nodes = pattern.get_nodes()
    target_nodes = target.get_nodes()

    if len(pattern_nodes) > len(target_nodes):
        return []

    mappings: List[Dict[str, str]] = []

    def _is_valid_mapping(mapping: Dict[str, str]) -> bool:
        """Check if current mapping preserves edges."""
        for edge in pattern.get_edges():
            src_id = mapping.get(edge.source.id)
            dst_id = mapping.get(edge.destination.id)

            if src_id is None or dst_id is None:
                continue

            # Check if corresponding edge exists in target
            has_edge = False
            for target_edge in target.get_outgoing_edges(src_id):
                if target_edge.destination.id == dst_id:
                    has_edge = True
                    break

            if not has_edge:
                return False

        return True

    def _backtrack(
        pattern_idx: int,
        mapping: Dict[str, str],
        used_targets: Set[str]
    ) -> None:
        if pattern_idx == len(pattern_nodes):
            if _is_valid_mapping(mapping):
                mappings.append(mapping.copy())
            return

        pattern_node = pattern_nodes[pattern_idx]

        for target_node in target_nodes:
            if target_node.id in used_targets:
                continue

            if not node_match_fn(pattern_node, target_node):
                continue

            mapping[pattern_node.id] = target_node.id
            used_targets.add(target_node.id)

            if _is_valid_mapping(mapping):
                _backtrack(pattern_idx + 1, mapping, used_targets)

            del mapping[pattern_node.id]
            used_targets.remove(target_node.id)

    _backtrack(0, {}, set())
    return mappings


def graph_edit_distance(
    graph1: Graph,
    graph2: Graph,
    node_cost: float = 1.0,
    edge_cost: float = 1.0
) -> float:
    """
    Compute approximate graph edit distance.

    This is a simplified version that counts node and edge differences.

    Args:
        graph1: First graph
        graph2: Second graph
        node_cost: Cost of node insertion/deletion
        edge_cost: Cost of edge insertion/deletion

    Returns:
        Approximate edit distance
    """
    node_diff = abs(graph1.num_nodes() - graph2.num_nodes())
    edge_diff = abs(graph1.num_edges() - graph2.num_edges())

    return node_diff * node_cost + edge_diff * edge_cost
