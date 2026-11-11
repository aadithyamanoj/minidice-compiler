"""List scheduling algorithm."""

from typing import Dict, List, Optional, Callable
from cgrasolve.graph.dfg import DFG, DFGNode
from cgrasolve.graph.mrrg import MRRG
from cgrasolve.graph.utils.analysis import compute_asap_schedule


class ListScheduler:
    """
    List scheduling algorithm for DFG scheduling.

    Schedules operations based on priorities and resource availability.
    """

    def __init__(self, priority_fn: Optional[Callable[[DFGNode], float]] = None) -> None:
        """
        Initialize the list scheduler.

        Args:
            priority_fn: Function to compute node priority (higher = schedule earlier)
        """
        self.priority_fn = priority_fn or self._default_priority

    def _default_priority(self, node: DFGNode) -> float:
        """
        Default priority function based on ALAP time (critical path).

        Args:
            node: DFG node

        Returns:
            Priority value (higher = more critical)
        """
        return -(node.alap_time or 0)  # Negative so earlier ALAP has higher priority

    def schedule(
        self,
        dfg: DFG,
        mrrg: MRRG,
        asap: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Perform list scheduling.

        Args:
            dfg: Data Flow Graph to schedule
            mrrg: MRRG for resource constraints
            asap: ASAP schedule (if None, will be computed)

        Returns:
            Dictionary mapping node IDs to scheduled times
        """
        if asap is None:
            asap = compute_asap_schedule(dfg)

        schedule: Dict[str, int] = {}
        ready_list: List[DFGNode] = []
        scheduled: set = set()

        # Initialize ready list with nodes that have no predecessors
        for node in dfg.get_nodes():
            if not dfg.get_predecessors(node.id):
                ready_list.append(node)

        # Resource availability tracking
        resource_available_time: Dict[str, int] = {}

        current_time = 0

        while ready_list or len(scheduled) < dfg.num_nodes():
            # Sort ready list by priority
            ready_list.sort(key=self.priority_fn, reverse=True)

            # Try to schedule operations from ready list
            scheduled_this_cycle = []

            for node in ready_list:
                # Check if resources are available
                earliest_time = max(
                    asap.get(node.id, current_time),
                    self._get_earliest_resource_time(node, mrrg, resource_available_time)
                )

                if earliest_time <= current_time:
                    # Schedule this node
                    schedule[node.id] = current_time
                    scheduled.add(node.id)
                    scheduled_this_cycle.append(node)

                    # Update resource availability
                    self._update_resource_time(
                        node,
                        current_time,
                        mrrg,
                        resource_available_time
                    )

            # Remove scheduled nodes from ready list
            for node in scheduled_this_cycle:
                ready_list.remove(node)

                # Add successors to ready list if all predecessors are scheduled
                for successor in dfg.get_successors(node.id):
                    if successor.id not in scheduled:
                        predecessors = dfg.get_predecessors(successor.id)
                        if all(pred.id in scheduled for pred in predecessors):
                            if successor not in ready_list:
                                ready_list.append(successor)

            current_time += 1

        return schedule

    def _get_earliest_resource_time(
        self,
        node: DFGNode,
        mrrg: MRRG,
        resource_times: Dict[str, int]
    ) -> int:
        """Get the earliest time when resources are available for this node."""
        compatible_fus = mrrg.get_compatible_fus(node.operation)

        if not compatible_fus:
            return 0

        # Find the earliest available FU
        earliest = min(
            resource_times.get(fu.id, 0)
            for fu in compatible_fus
        )

        return earliest

    def _update_resource_time(
        self,
        node: DFGNode,
        schedule_time: int,
        mrrg: MRRG,
        resource_times: Dict[str, int]
    ) -> None:
        """Update resource availability after scheduling a node."""
        compatible_fus = mrrg.get_compatible_fus(node.operation)

        if not compatible_fus:
            return

        # Find the FU with earliest availability and use it
        best_fu = min(
            compatible_fus,
            key=lambda fu: resource_times.get(fu.id, 0)
        )

        # Mark resource as busy for one cycle
        resource_times[best_fu.id] = schedule_time + 1
