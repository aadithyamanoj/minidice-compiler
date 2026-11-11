"""Modulo scheduling for pipelined CGRA mapping."""

from typing import Dict, Optional, Set
from cgrasolve.graph.dfg import DFG
from cgrasolve.graph.mrrg import MRRG
from cgrasolve.graph.utils.analysis import (
    compute_asap_schedule,
    compute_alap_schedule,
    find_critical_path
)


class ModuloScheduler:
    """
    Modulo scheduler for pipelined CGRA execution.

    This scheduler assigns operations to time slots modulo an initiation interval (II),
    enabling pipelined execution of loop iterations.
    """

    def __init__(self, min_ii: Optional[int] = None, max_ii: Optional[int] = None) -> None:
        """
        Initialize the modulo scheduler.

        Args:
            min_ii: Minimum initiation interval to try (if None, computed from DFG)
            max_ii: Maximum initiation interval to try (if None, set to 2x min_ii)
        """
        self.min_ii = min_ii
        self.max_ii = max_ii

    def compute_min_ii(self, dfg: DFG, mrrg: MRRG) -> int:
        """
        Compute the minimum theoretical initiation interval.

        The minimum II is determined by:
        1. Resource constraints (RecMII): based on resource availability
        2. Recurrence constraints (RecMII): based on loop-carried dependencies

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG (for resource constraints)

        Returns:
            Minimum theoretical II
        """
        # Compute resource-constrained minimum II
        rec_mii = self._compute_resource_mii(dfg, mrrg)

        # Compute recurrence-constrained minimum II
        recurrence_mii = self._compute_recurrence_mii(dfg)

        return max(rec_mii, recurrence_mii)

    def _compute_resource_mii(self, dfg: DFG, mrrg: MRRG) -> int:
        """
        Compute resource-constrained minimum II.

        RecMII = ceil(total operations of type / number of resources for that type)

        For time-expanded MRRGs, resources are replicated across II cycles,
        so we count unique base resources (not time-expanded copies).

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG

        Returns:
            Resource-constrained minimum II
        """
        from collections import Counter

        # Count operations by type
        op_counts: Counter = Counter()
        for node in dfg.get_nodes():
            op_counts[node.operation] += 1

        # Count available unique resources by operation type
        # For time-expanded MRRGs, divide by II to get base resource count
        resource_counts: Counter = Counter()

        # Track unique FU base names to avoid counting time-expanded copies
        unique_fus: Set[str] = set()
        for fu in mrrg.get_fu_nodes():
            base_name = fu.get_base_name()
            if base_name not in unique_fus:
                unique_fus.add(base_name)
                for op in fu.supported_operations:
                    resource_counts[op] += 1

        # Compute RecMII
        rec_mii = 1
        for op_type, count in op_counts.items():
            if op_type in resource_counts and resource_counts[op_type] > 0:
                required_ii = (count + resource_counts[op_type] - 1) // resource_counts[op_type]
                rec_mii = max(rec_mii, required_ii)

        return rec_mii

    def _compute_recurrence_mii(self, dfg: DFG) -> int:
        """
        Compute recurrence-constrained minimum II.

        RecMII = max(loop latency / loop distance) for all recurrences

        Args:
            dfg: Data Flow Graph

        Returns:
            Recurrence-constrained minimum II
        """
        # Get loop-back edges
        loop_edges = dfg.get_loop_back_edges()

        if not loop_edges:
            return 1

        # For each loop-back edge, compute the recurrence MII
        max_rec_mii = 1
        for loop_edge in loop_edges:
            # Compute the path length from destination to source
            # Use 'dist' attribute (iteration distance) for recurrence analysis
            # NOT latency (which represents hardware pipeline stages)
            loop_distance = loop_edge.get_attribute('loop_distance', 1)
            loop_dist = loop_edge.get_attribute('dist', 0)

            if loop_distance > 0:
                rec_mii = (loop_dist + loop_distance - 1) // loop_distance
                max_rec_mii = max(max_rec_mii, rec_mii)

        return max_rec_mii

    def schedule(self, dfg: DFG, mrrg: MRRG, target_ii: Optional[int] = None) -> Optional[Dict[str, int]]:
        """
        Perform modulo scheduling on the DFG.

        Args:
            dfg: Data Flow Graph to schedule
            mrrg: MRRG (for resource constraints)
            target_ii: Target initiation interval (if None, uses minimum)

        Returns:
            Dictionary mapping node IDs to scheduled times (modulo II), or None if failed
        """
        if target_ii is None:
            target_ii = self.compute_min_ii(dfg, mrrg)

        # Compute ASAP and ALAP schedules
        asap = compute_asap_schedule(dfg)
        alap = compute_alap_schedule(dfg)

        # Sort nodes by mobility (ALAP - ASAP)
        nodes = sorted(
            dfg.get_nodes(),
            key=lambda n: alap.get(n.id, 0) - asap.get(n.id, 0)
        )

        # Resource usage tracking: time_mod_ii -> resource -> allocated
        resource_usage: Dict[int, Dict[str, bool]] = {}

        schedule: Dict[str, int] = {}

        # Try to schedule each node
        for node in nodes:
            earliest = asap.get(node.id, 0)
            latest = alap.get(node.id, earliest)

            # Try time slots from earliest to latest
            scheduled = False
            for time in range(earliest, latest + 1):
                time_mod = time % target_ii

                # Check resource availability
                # This is simplified - real implementation would check FU availability
                if self._can_schedule_at(node, time_mod, resource_usage, mrrg):
                    schedule[node.id] = time
                    self._mark_resource_used(node, time_mod, resource_usage)
                    scheduled = True
                    break

            if not scheduled:
                # Failed to schedule this node
                return None

        return schedule

    def _can_schedule_at(
        self,
        node,
        time_mod: int,
        resource_usage: Dict[int, Dict[str, bool]],
        mrrg: MRRG
    ) -> bool:
        """
        Check if a node can be scheduled at a given time (modulo II).

        For time-expanded MRRGs, we query FUs at the specific cycle.
        """
        # Get FUs at this cycle that support the operation
        cycle = time_mod % mrrg.II
        fus_at_cycle = [
            fu for fu in mrrg.get_nodes_at_cycle(cycle)
            if hasattr(fu, 'can_execute') and fu.can_execute(node.operation)
        ]

        if not fus_at_cycle:
            return False

        # Check if any FU is available at this time
        if time_mod not in resource_usage:
            return True

        # Check if there's an available FU that's not already used
        for fu in fus_at_cycle:
            if fu.id not in resource_usage[time_mod]:
                return True

        return False

    def _mark_resource_used(
        self,
        node,
        time_mod: int,
        resource_usage: Dict[int, Dict[str, bool]]
    ) -> None:
        """
        Mark a resource as used at a given time (modulo II).

        For time-expanded MRRGs, we need to track which specific FU node is used.
        This is a simplified version - full implementation would track the actual FU binding.
        """
        if time_mod not in resource_usage:
            resource_usage[time_mod] = {}
        # Store the operation node that's using a resource at this time
        # In a complete implementation, we'd also track which specific FU is bound
        resource_usage[time_mod][node.id] = True
