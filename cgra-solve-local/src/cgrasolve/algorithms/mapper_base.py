"""Base classes for CGRA mappers."""

from typing import Dict, List, Optional, Tuple, Any, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from cgrasolve.graph.dfg import DFG, DFGNode, DFGEdge
from cgrasolve.graph.mrrg import MRRG, MRRGNode


class MappingStatus(Enum):
    """Status of a mapping attempt."""
    SUCCESS = "success"
    FAILED_PLACEMENT = "failed_placement"
    FAILED_ROUTING = "failed_routing"
    FAILED_TIMING = "failed_timing"
    FAILED_RESOURCES = "failed_resources"
    TIMEOUT = "timeout"


@dataclass
class Placement:
    """Represents a placement of DFG nodes onto MRRG nodes."""

    # Mapping from DFG node ID to MRRG node ID (typically an FU)
    node_mapping: Dict[str, str] = field(default_factory=dict)

    # Mapping from DFG node ID to scheduled time step
    schedule: Dict[str, int] = field(default_factory=dict)

    # Additional placement metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def map_node(self, dfg_node_id: str, mrrg_node_id: str, time: int = 0) -> None:
        """Map a DFG node to an MRRG node at a specific time."""
        self.node_mapping[dfg_node_id] = mrrg_node_id
        self.schedule[dfg_node_id] = time

    def get_placement(self, dfg_node_id: str) -> Optional[str]:
        """Get the MRRG node where a DFG node is placed."""
        return self.node_mapping.get(dfg_node_id)

    def get_scheduled_time(self, dfg_node_id: str) -> Optional[int]:
        """Get the scheduled time for a DFG node."""
        return self.schedule.get(dfg_node_id)

    def is_complete(self, dfg: DFG) -> bool:
        """Check if all DFG nodes are placed."""
        return all(node.id in self.node_mapping for node in dfg.get_nodes())

    def get_resource_usage(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get resource usage per MRRG node.

        Returns:
            Dict mapping MRRG node ID to list of (DFG node ID, time) tuples
        """
        usage: Dict[str, List[Tuple[str, int]]] = {}
        for dfg_id, mrrg_id in self.node_mapping.items():
            time = self.schedule.get(dfg_id, 0)
            if mrrg_id not in usage:
                usage[mrrg_id] = []
            usage[mrrg_id].append((dfg_id, time))
        return usage


@dataclass
class Routing:
    """Represents routing of DFG edges through MRRG paths."""

    # Mapping from DFG edge ID to path in MRRG (list of MRRG node IDs)
    edge_routing: Dict[str, List[str]] = field(default_factory=dict)

    # Register allocation: edge ID -> list of (register node ID, time) pairs
    register_allocation: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)

    # Additional routing metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def route_edge(self, dfg_edge_id: str, mrrg_path: List[str]) -> None:
        """Route a DFG edge through a path in the MRRG."""
        self.edge_routing[dfg_edge_id] = mrrg_path

    def get_route(self, dfg_edge_id: str) -> Optional[List[str]]:
        """Get the routing path for a DFG edge."""
        return self.edge_routing.get(dfg_edge_id)

    def allocate_register(self, dfg_edge_id: str, register_id: str, time: int) -> None:
        """Allocate a register for an edge at a specific time."""
        if dfg_edge_id not in self.register_allocation:
            self.register_allocation[dfg_edge_id] = []
        self.register_allocation[dfg_edge_id].append((register_id, time))

    def is_complete(self, dfg: DFG) -> bool:
        """Check if all DFG edges are routed."""
        return all(edge.id in self.edge_routing for edge in dfg.get_edges())


@dataclass
class MappingResult:
    """Result of a mapping attempt."""

    status: MappingStatus
    placement: Optional[Placement] = None
    routing: Optional[Routing] = None

    # Performance metrics
    initiation_interval: Optional[int] = None
    latency: Optional[int] = None
    throughput: Optional[float] = None

    # Resource utilization
    pe_utilization: float = 0.0
    register_utilization: float = 0.0

    # Cost metrics
    total_cost: float = 0.0
    placement_cost: float = 0.0
    routing_cost: float = 0.0

    # Timing information
    mapping_time: float = 0.0  # Time taken to find this mapping (seconds)

    # Diagnostics
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if the mapping was successful."""
        return self.status == MappingStatus.SUCCESS

    def get_summary(self) -> str:
        """Get a human-readable summary of the mapping result."""
        if self.is_success():
            return (
                f"Mapping successful: II={self.initiation_interval}, "
                f"Latency={self.latency}, "
                f"PE util={self.pe_utilization:.2%}, "
                f"Cost={self.total_cost:.2f}"
            )
        else:
            return f"Mapping failed: {self.status.value} - {self.error_message}"

