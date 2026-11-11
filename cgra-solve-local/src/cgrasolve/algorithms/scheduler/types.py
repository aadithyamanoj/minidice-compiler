"""Shared scheduler types and schedule-time helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from cgrasolve.graph.dfg import DFG, DFGNode


class SchedulingFailureReason(Enum):
    """Structured failure categories for scheduler attempts."""

    NONE = "none"
    INVALID_II = "invalid_ii"
    NEGATIVE_CYCLE = "negative_cycle"
    RESOURCE_CONFLICT = "resource_conflict"
    UPPER_BOUND_VIOLATION = "upper_bound_violation"
    SEARCH_EXHAUSTED = "search_exhausted"
    UNSUPPORTED = "unsupported"


@dataclass
class SchedulingAttempt:
    """One scheduler attempt for a specific II."""

    ii: int
    success: bool
    failure_reason: SchedulingFailureReason = SchedulingFailureReason.NONE
    message: str = ""
    node_times: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuloSchedule:
    """Accepted schedule for a DFG."""

    ii: int
    node_times: Dict[str, int]
    asap_times: Dict[str, int] = field(default_factory=dict)
    alap_times: Dict[str, int] = field(default_factory=dict)
    mobility: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def time_class(self, node_id: str) -> int:
        """Return the temporal class for the scheduled node."""
        return self.node_times[node_id] % self.ii


@dataclass
class SchedulingResult:
    """Structured scheduler result."""

    success: bool
    schedule: Optional[ModuloSchedule] = None
    attempts: List[SchedulingAttempt] = field(default_factory=list)
    failure_reason: SchedulingFailureReason = SchedulingFailureReason.NONE
    message: str = ""


def get_node_schedule_time(node: DFGNode) -> Optional[int]:
    """Return the canonical schedule time for a DFG node."""
    if node.scheduled_time is not None:
        return node.scheduled_time
    return node.asap_time


def require_node_schedule_time(node: DFGNode) -> int:
    """Return the canonical schedule time or raise if the node is unscheduled."""
    time = get_node_schedule_time(node)
    if time is None:
        raise ValueError(f"Node {node.id} is not scheduled")
    return time


def stamp_schedule_on_dfg(
    dfg: DFG,
    schedule: ModuloSchedule,
    *,
    update_asap: bool = False,
) -> None:
    """Stamp an accepted schedule onto the DFG in-place."""
    for node in dfg.get_nodes():
        if node.id not in schedule.node_times:
            continue
        node.scheduled_time = schedule.node_times[node.id]
        if update_asap:
            asap_time = schedule.asap_times.get(node.id, schedule.node_times[node.id])
            node.asap_time = asap_time
        if node.id in schedule.alap_times:
            node.alap_time = schedule.alap_times[node.id]
        if node.id in schedule.mobility:
            node.mobility = schedule.mobility[node.id]
