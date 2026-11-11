"""Dijkstra-based routing algorithm."""

from typing import Dict, List, Optional
from cgrasolve.graph.dfg import DFG, DFGEdge
from cgrasolve.graph.mrrg import MRRG
from cgrasolve.graph.utils.pathfinding import dijkstra, reconstruct_path
from cgrasolve.algorithms.mapper_base import Placement, Routing


class DijkstraRouter:
    """
    Dijkstra-based router for CGRA mapping.

    Routes DFG edges through MRRG using Dijkstra's shortest path algorithm.
    """

    def __init__(self, avoid_congestion: bool = True) -> None:
        """
        Initialize the Dijkstra router.

        Args:
            avoid_congestion: If True, consider routing congestion in cost
        """
        self.avoid_congestion = avoid_congestion
        self.congestion_weight = 10.0

    def route(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Placement
    ) -> Optional[Routing]:
        """
        Route all DFG edges through the MRRG.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            placement: Placement of DFG nodes onto MRRG

        Returns:
            Routing object, or None if routing fails
        """
        routing = Routing()

        # Track congestion: mrrg_node_id -> number of routes using it
        congestion: Dict[str, int] = {}

        # Sort edges by criticality or other metric
        edges = sorted(dfg.get_edges(), key=lambda e: e.id)

        for edge in edges:
            # Get source and destination MRRG nodes
            src_mrrg_id = placement.get_placement(edge.source.id)
            dst_mrrg_id = placement.get_placement(edge.destination.id)

            if src_mrrg_id is None or dst_mrrg_id is None:
                # Incomplete placement
                return None

            # Find a path from source to destination
            def edge_weight(mrrg_edge):
                base_weight = getattr(mrrg_edge, 'latency', 1)

                if self.avoid_congestion:
                    # Add congestion penalty
                    src_congestion = congestion.get(mrrg_edge.source.id, 0)
                    dst_congestion = congestion.get(mrrg_edge.destination.id, 0)
                    congestion_penalty = (src_congestion + dst_congestion) * self.congestion_weight
                    return base_weight + congestion_penalty

                return base_weight

            # Use Dijkstra to find shortest path
            _, predecessors = dijkstra(
                mrrg,
                src_mrrg_id,
                dst_mrrg_id,
                weight_fn=edge_weight
            )

            path_nodes = reconstruct_path(predecessors, src_mrrg_id, dst_mrrg_id, mrrg)

            if path_nodes is None:
                # No path found
                return None

            # Extract node IDs for the path
            path = [node.id for node in path_nodes]

            # Add route
            routing.route_edge(edge.id, path)

            # Update congestion
            for node_id in path:
                congestion[node_id] = congestion.get(node_id, 0) + 1

        return routing

    def route_with_timing(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Placement,
        schedule: Dict[str, int]
    ) -> Optional[Routing]:
        """
        Route with timing awareness (considering scheduled times).

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            placement: Node placement
            schedule: Node schedule

        Returns:
            Routing object, or None if routing fails
        """
        routing = Routing()

        # Track resource usage over time: (mrrg_node_id, time) -> used
        resource_usage: Dict[tuple, bool] = {}

        for edge in dfg.get_edges():
            src_mrrg_id = placement.get_placement(edge.source.id)
            dst_mrrg_id = placement.get_placement(edge.destination.id)

            if src_mrrg_id is None or dst_mrrg_id is None:
                return None

            src_time = schedule.get(edge.source.id, 0)
            dst_time = schedule.get(edge.destination.id, 0)

            # Route considering timing constraints
            # This is simplified - real implementation would consider pipelining
            path_nodes = self._find_timing_aware_path(
                mrrg,
                src_mrrg_id,
                dst_mrrg_id,
                src_time,
                dst_time,
                resource_usage
            )

            if path_nodes is None:
                return None

            path = [node.id for node in path_nodes]
            routing.route_edge(edge.id, path)

            # Mark resources as used
            for i, node_id in enumerate(path):
                time = src_time + i
                resource_usage[(node_id, time)] = True

        return routing

    def _find_timing_aware_path(
        self,
        mrrg: MRRG,
        src_id: str,
        dst_id: str,
        src_time: int,
        dst_time: int,
        resource_usage: Dict[tuple, bool]
    ) -> Optional[List]:
        """
        Find a path that respects timing constraints.

        Simplified version - real implementation would use time-expanded graph.
        """
        # For now, just use standard Dijkstra
        # TODO: Implement time-expanded graph routing
        _, predecessors = dijkstra(mrrg, src_id, dst_id)
        return reconstruct_path(predecessors, src_id, dst_id, mrrg)
