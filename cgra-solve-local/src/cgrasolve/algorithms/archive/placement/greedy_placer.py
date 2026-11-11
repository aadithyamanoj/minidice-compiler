"""Greedy placement algorithm."""

from typing import Dict, List, Optional, Set
from cgrasolve.graph.dfg import DFG, DFGNode
from cgrasolve.graph.mrrg import MRRG, MRRGNode
from cgrasolve.algorithms.mapper_base import Placement


class GreedyPlacer:
    """
    Greedy placement algorithm.

    Places DFG nodes onto compatible MRRG FUs using a greedy heuristic.
    """

    def __init__(self, prefer_spatial_locality: bool = True) -> None:
        """
        Initialize the greedy placer.

        Args:
            prefer_spatial_locality: If True, prefer placing connected nodes close together
        """
        self.prefer_spatial_locality = prefer_spatial_locality

    def place(
        self,
        dfg: DFG,
        mrrg: MRRG,
        schedule: Dict[str, int]
    ) -> Optional[Placement]:
        """
        Perform greedy placement of DFG onto MRRG.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            schedule: Node schedule (node ID -> time)

        Returns:
            Placement object, or None if placement fails
        """
        placement = Placement()

        # Sort nodes by schedule time (place earlier operations first)
        sorted_nodes = sorted(dfg.get_nodes(), key=lambda n: schedule.get(n.id, 0))

        # Track FU usage: (fu_id, time_mod_ii) -> dfg_node_id
        fu_usage: Dict[tuple, str] = {}

        # Assume II from schedule or set to 1 for non-pipelined
        ii = dfg.initiation_interval or self._infer_ii(schedule)

        for node in sorted_nodes:
            time = schedule.get(node.id, 0)
            time_mod = time % ii if ii > 0 else time

            # Find compatible FUs
            compatible_fus = mrrg.get_compatible_fus(node.operation)

            if not compatible_fus:
                # No compatible FU found
                return None

            # Find an available FU
            selected_fu = self._select_fu(
                node,
                compatible_fus,
                fu_usage,
                time_mod,
                dfg,
                mrrg,
                placement
            )

            if selected_fu is None:
                # Failed to find available FU
                return None

            # Place the node
            placement.map_node(node.id, selected_fu.id, time)
            fu_usage[(selected_fu.id, time_mod)] = node.id

        return placement

    def _select_fu(
        self,
        node: DFGNode,
        candidates: List[MRRGNode],
        fu_usage: Dict[tuple, str],
        time_mod: int,
        dfg: DFG,
        mrrg: MRRG,
        current_placement: Placement
    ) -> Optional[MRRGNode]:
        """
        Select the best FU from candidates.

        Args:
            node: DFG node to place
            candidates: List of compatible FUs
            fu_usage: Current FU usage
            time_mod: Time modulo II
            dfg: Data Flow Graph
            mrrg: MRRG
            current_placement: Current placement state

        Returns:
            Selected FU, or None if no FU available
        """
        # Filter out already-used FUs at this time
        available_fus = [
            fu for fu in candidates
            if (fu.id, time_mod) not in fu_usage
        ]

        if not available_fus:
            return None

        if not self.prefer_spatial_locality or not current_placement.node_mapping:
            # Just return first available
            return available_fus[0]

        # Prefer FUs close to already-placed predecessors/successors
        best_fu = available_fus[0]
        best_score = float('inf')

        for fu in available_fus:
            score = self._compute_locality_score(node, fu, dfg, mrrg, current_placement)
            if score < best_score:
                best_score = score
                best_fu = fu

        return best_fu

    def _compute_locality_score(
        self,
        node: DFGNode,
        fu: MRRGNode,
        dfg: DFG,
        mrrg: MRRG,
        placement: Placement
    ) -> float:
        """
        Compute locality score (lower is better).

        Sum of Manhattan distances to placed neighbors.

        Args:
            node: DFG node
            fu: Candidate FU
            dfg: Data Flow Graph
            mrrg: MRRG
            placement: Current placement

        Returns:
            Locality score
        """
        total_distance = 0.0
        num_neighbors = 0

        # Check predecessors
        for pred in dfg.get_predecessors(node.id):
            pred_fu_id = placement.get_placement(pred.id)
            if pred_fu_id:
                pred_fu = mrrg.get_node(pred_fu_id)
                if pred_fu and hasattr(pred_fu, 'coordinates') and fu.coordinates:
                    dist = abs(fu.coordinates[0] - pred_fu.coordinates[0]) + \
                           abs(fu.coordinates[1] - pred_fu.coordinates[1])
                    total_distance += dist
                    num_neighbors += 1

        # Check successors
        for succ in dfg.get_successors(node.id):
            succ_fu_id = placement.get_placement(succ.id)
            if succ_fu_id:
                succ_fu = mrrg.get_node(succ_fu_id)
                if succ_fu and hasattr(succ_fu, 'coordinates') and fu.coordinates:
                    dist = abs(fu.coordinates[0] - succ_fu.coordinates[0]) + \
                           abs(fu.coordinates[1] - succ_fu.coordinates[1])
                    total_distance += dist
                    num_neighbors += 1

        return total_distance / max(num_neighbors, 1)

    def _infer_ii(self, schedule: Dict[str, int]) -> int:
        """Infer II from schedule (simplified)."""
        if not schedule:
            return 1
        max_time = max(schedule.values())
        return max(1, max_time // 2)  # Heuristic
