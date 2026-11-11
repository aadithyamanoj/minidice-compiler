from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cgrasolve.graph.mrrg import MRRGNode, OperandTag
    from cgrasolve.graph.dfg import DFGNode

from cgrasolve.graph.dfg import DFGEdge
from cgrasolve.graph.hyperdfg import HyperVal


class TransportMode(Enum):
    """Logical transport mode for a routed value."""

    UNKNOWN = "unknown"
    FLAT = "flat"
    FRACTURED = "fractured"
    PACKED = "packed"
    TEMPORAL_FRACTURED = "temporal_fractured"


@dataclass(frozen=True)
class TransportProfile:
    """Legality and costing profile for one transport decision."""

    mode: TransportMode = TransportMode.UNKNOWN
    total_bitwidth: Optional[int] = None
    legal_min_bitwidth: Optional[int] = None
    preferred_bitwidth: Optional[int] = None
    exact_bitwidth: Optional[int] = None
    num_segments: int = 1
    segment_index: Optional[int] = None
    segment_widths: Tuple[int, ...] = ()
    segment_offsets: Tuple[int, ...] = ()
    lane_width: Optional[int] = None
    lane_index: Optional[int] = None

    @classmethod
    def unknown(cls) -> "TransportProfile":
        """Return a profile with no width constraints."""
        return cls()

    @classmethod
    def flat(cls, bitwidth: Optional[int]) -> "TransportProfile":
        """Create the profile for ordinary flat routing."""
        normalized = bitwidth if bitwidth and bitwidth > 0 else None
        return cls(
            mode=TransportMode.FLAT,
            total_bitwidth=normalized,
            legal_min_bitwidth=normalized,
            preferred_bitwidth=normalized,
            num_segments=1,
        )

    @classmethod
    def fractured_bundle(
        cls,
        total_bitwidth: Optional[int],
        chunk_widths: Tuple[int, ...],
        chunk_offsets: Tuple[int, ...],
    ) -> "TransportProfile":
        """Create the bundle-level profile for a fractured route."""
        normalized = total_bitwidth if total_bitwidth and total_bitwidth > 0 else None
        return cls(
            mode=TransportMode.FRACTURED,
            total_bitwidth=normalized,
            num_segments=len(chunk_widths),
            segment_widths=chunk_widths,
            segment_offsets=chunk_offsets,
        )

    @classmethod
    def fractured_chunk(
        cls,
        *,
        total_bitwidth: Optional[int],
        chunk_index: int,
        chunk_width: Optional[int],
        chunk_widths: Tuple[int, ...] = (),
        chunk_offsets: Tuple[int, ...] = (),
    ) -> "TransportProfile":
        """Create the profile for one fractured interior chunk route."""
        normalized_total = total_bitwidth if total_bitwidth and total_bitwidth > 0 else None
        normalized_chunk = chunk_width if chunk_width and chunk_width > 0 else None
        return cls(
            mode=TransportMode.FRACTURED,
            total_bitwidth=normalized_total,
            legal_min_bitwidth=normalized_chunk,
            preferred_bitwidth=normalized_chunk,
            exact_bitwidth=normalized_chunk,
            num_segments=len(chunk_widths) if chunk_widths else 1,
            segment_index=chunk_index,
            segment_widths=chunk_widths,
            segment_offsets=chunk_offsets,
        )

    @classmethod
    def packed(
        cls,
        *,
        total_bitwidth: Optional[int],
        lane_width: Optional[int],
        lane_index: Optional[int] = None,
    ) -> "TransportProfile":
        """Create the profile for packed routing on a wide carrier lane."""
        normalized_total = total_bitwidth if total_bitwidth and total_bitwidth > 0 else None
        normalized_lane = lane_width if lane_width and lane_width > 0 else None
        return cls(
            mode=TransportMode.PACKED,
            total_bitwidth=normalized_total,
            legal_min_bitwidth=normalized_lane,
            preferred_bitwidth=normalized_lane,
            lane_width=normalized_lane,
            lane_index=lane_index,
        )

    def rejects_node_width(self, node: "MRRGNode", *, is_sink: bool = False) -> bool:
        """Return True when ``node`` is illegal for this transport profile."""
        if self.mode == TransportMode.PACKED:
            if self.lane_width is None:
                return True
            if node.num_lanes > 1 and node.lane_width >= self.lane_width:
                return False
            return node.bitwidth < self.lane_width

        if self.legal_min_bitwidth is not None and node.bitwidth < self.legal_min_bitwidth:
            return True

        if (
            self.exact_bitwidth is not None
            and not is_sink
            and not node.get_attribute("is_instance", False)
            and node.bitwidth != self.exact_bitwidth
        ):
            return True

        return False

    def width_penalty(self, node: "MRRGNode", mismatch_cost: float) -> float:
        """Return the soft width mismatch penalty for using ``node``."""
        if (
            self.mode == TransportMode.PACKED
            and node.num_lanes > 1
            and self.lane_width is not None
            and node.lane_width >= self.lane_width
        ):
            return 0.0
        if self.exact_bitwidth is not None:
            return 0.0
        if self.preferred_bitwidth is None or node.bitwidth <= self.preferred_bitwidth:
            return 0.0
        width_ratio = node.bitwidth / self.preferred_bitwidth
        return (width_ratio - 1.0) * mismatch_cost


@dataclass
class RoutingStats:
    """Summary statistics for one routing run."""

    hypervals_considered: int = 0
    destinations_considered: int = 0
    negotiation_iterations: int = 0
    routed_nets: int = 0
    resources_used: int = 0
    total_occupancy: int = 0
    total_capacity: int = 0

    @property
    def utilization_percent(self) -> float:
        """Return routing-resource utilization percentage."""
        if self.total_capacity <= 0:
            return 0.0
        return self.total_occupancy / self.total_capacity * 100.0


@dataclass
class RoutingResult:
    """Structured public result for one PathFinder routing run."""

    success: bool
    timed_out: bool = False
    routing_solution: Dict[Tuple[HyperVal, int], 'RouteBundle'] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    failure_summary: Optional[str] = None
    message: str = ""
    stats: RoutingStats = field(default_factory=RoutingStats)


@dataclass
class RouteBundle:
    """Internal route representation, possibly fractured into sub-paths.

    For unfractured routes, sub_paths has a single element.
    For fractured routes, each sub-path corresponds to one chunk (e.g., 8b).
    """
    sub_paths: List[List['MRRGNode']]
    fractured: bool = False
    num_chunks: int = 1
    chunk_indices: List[int] = field(default_factory=list)
    chunk_widths: List[int] = field(default_factory=list)
    chunk_offsets: List[int] = field(default_factory=list)
    lane_assignments: Dict[str, int] = field(default_factory=dict)
    active_pack_nodes: Set[str] = field(default_factory=set)
    active_unpack_nodes: Set[str] = field(default_factory=set)
    transport_profile: TransportProfile = field(default_factory=TransportProfile.unknown)

    def __post_init__(self) -> None:
        """Require an explicit transport profile for every route bundle."""
        if self.transport_profile.mode == TransportMode.UNKNOWN:
            raise ValueError("RouteBundle requires an explicit transport_profile")

    @classmethod
    def from_flat_path(
        cls,
        path: List['MRRGNode'],
        transport_profile: TransportProfile,
    ) -> 'RouteBundle':
        """Wrap a legacy single-path route in a RouteBundle."""
        return cls(sub_paths=[path], transport_profile=transport_profile)

    @property
    def flat_path(self) -> List['MRRGNode']:
        """Get the single path for unfractured routes."""
        if self.fractured:
            raise ValueError("Cannot get flat_path from fractured RouteBundle")
        return self.sub_paths[0]

    @property
    def all_nodes(self) -> Set['MRRGNode']:
        """Get all unique nodes across all sub-paths."""
        return {n for path in self.sub_paths for n in path}

    def to_node_id_paths(self) -> List[List[str]]:
        """Return the bundle as lists of MRRG node IDs."""
        return [[node.id for node in path] for path in self.sub_paths]


def extract_predecessors(bundle: RouteBundle) -> Dict['MRRGNode', 'MRRGNode']:
    """Extract node -> predecessor mapping from all sub-paths in a bundle.

    For each node at position i > 0 in a sub-path, maps node -> sub_path[i-1].
    The first node in each sub-path (source output port) has no predecessor.

    If two sub-paths disagree on a node's predecessor (e.g., a fractured
    bundle's join node receiving from different chunk inputs), the node is
    excluded — it is a multi-input convergence point, not a mux conflict.
    """
    predecessors: Dict['MRRGNode', 'MRRGNode'] = {}
    conflicts: Set['MRRGNode'] = set()
    for sub_path in bundle.sub_paths:
        for i in range(1, len(sub_path)):
            node = sub_path[i]
            pred = sub_path[i - 1]
            if node in conflicts:
                continue
            if node in predecessors:
                if predecessors[node] != pred:
                    del predecessors[node]
                    conflicts.add(node)
            else:
                predecessors[node] = pred
    return predecessors


@dataclass
class RoutingNodePlacementState:
    """Node attributes for heuristic mapping."""

    # Number of distinct HyperVals using this routing node (NOT number of sinks)
    occupancy: int = 0

    # Base cost of the routing node
    base_cost: float = 1.0

    # Historical cost of the routing node
    historical_cost: float = 0.0

    # Values using this routing node (per-sink record for debugging)
    values: List[Tuple[HyperVal, int]] = field(default_factory=list)

    # Track number of sinks of each HyperVal using this node
    # Key: HyperVal, Value: count of sinks of that HyperVal using this node
    hyperval_usage: Dict[HyperVal, int] = field(default_factory=dict)

    # Full-width users on lane-capable nodes. When non-empty, the node is
    # effectively reserved as an ordinary wide resource.
    full_width_hyperval_usage: Dict[HyperVal, int] = field(default_factory=dict)

    # Packed users on lane-capable nodes: lane -> {HyperVal -> sink_count}
    packed_lane_usage: Dict[int, Dict[HyperVal, int]] = field(default_factory=dict)

    # True when this ROUTING_FUNCTION node is placed for compute (not available for routing)
    reserved_for_compute: bool = False


@dataclass
class SinkAndLatency:
    """
    Sink destination with timing information for priority routing.

    Used to prioritize which sinks to route first within a multi-sink value.
    Routes sinks with longer timing constraints first (critical paths).
    """

    # Timing constraint (cycles to sink) - used for ordering
    cycles_to_sink: int

    # Destination index in HyperVal.destination_ids (for tie-breaking)
    dest_idx: int

    # Destination DFG node
    dest_dfg_node: 'DFGNode'

    # Sink FU node in MRRG
    sink_fu: 'MRRGNode'

    # Required operand tag
    operand_tag: Optional['OperandTag'] = None

    def __lt__(self, other: 'SinkAndLatency') -> bool:
        """
        Comparison for priority queue (max-heap via reversal).

        Route sinks with longer timing constraints first.
        """
        if self.cycles_to_sink == other.cycles_to_sink:
            return self.dest_idx < other.dest_idx
        return self.cycles_to_sink > other.cycles_to_sink  # Reverse for max-heap


@dataclass
class HyperValNetInfo:
    """
    Information about a HyperVal for routing priority.

    Updated to work at HyperVal level (all destinations together)
    rather than per-destination.
    """

    # The HyperVal to route (all destinations)
    hyperval: HyperVal

    # Number of sinks not yet covered by any route
    uncovered_sinks: int = 0

    # Sum of overuse across nodes used by this HyperVal's routed sinks
    conflict_score: float = 0.0

    # Maximum timing constraint across all destinations
    max_cycles: int = 0

    # Fanout of the HyperVal (total number of destinations)
    fanout: int = 0

    def __lt__(self, other: 'HyperValNetInfo') -> bool:
        """
        Less than comparison for priority sorting.

        Priority order (descending):
            1. uncovered_sinks  – route unrouted nets first
            2. conflict_score   – then most-conflicted nets
            3. fanout           – then highest-fanout nets
            4. max_cycles       – then longest-path nets

        Returns:
            True if this net has higher priority than other
        """
        if self.uncovered_sinks != other.uncovered_sinks:
            return self.uncovered_sinks > other.uncovered_sinks
        if self.conflict_score != other.conflict_score:
            return self.conflict_score > other.conflict_score
        if self.fanout != other.fanout:
            return self.fanout > other.fanout
        return self.max_cycles > other.max_cycles


@dataclass
class VertexData:
    """
    Dijkstra state data for pathfinding.

    Stores the best path found to reach a specific (node, latency) state.
    """

    # Path to reach this state (list of MRRG nodes)
    fanin: List['MRRGNode']

    # Lowest known cost to reach this state
    lowest_known_cost: float

    # Accumulated latency at this state
    num_of_cycles: int

    # Whether operand tag has been satisfied
    tag_found: bool = False


@dataclass(order=True)
class VertexAndCost:
    """
    Priority queue entry for A* exploration.

    Ordered by f_cost = g_cost + heuristic (min-heap).
    """

    # f-cost: g_cost + heuristic estimate to sink (used for heap ordering)
    f_cost: float

    # Cycles/latency at this node (for state tracking)
    cycles: int = field(compare=False)

    # The MRRG node
    node: 'MRRGNode' = field(compare=False)

    # Whether the operand tag has been found on the path to this node
    tag_found: bool = field(default=False, compare=False)

    # g-cost: actual cost from source to this node
    g_cost: float = field(default=0.0, compare=False)

    def __hash__(self):
        """Allow use in sets."""
        return hash(id(self.node))
