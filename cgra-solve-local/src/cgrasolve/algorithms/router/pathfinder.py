"""PathFinder Algorithm for heuristic mapping."""

import heapq
import math
import time
from typing import Any, Dict, List, Set, Tuple, Optional

from cgrasolve.graph.dfg import DFG, DFGNode, OperationType
from cgrasolve.graph.mrrg import (
    MRRG,
    MRRGNode,
    NodeType,
    OperandTag,
    FractureEndpoint,
    PackingEndpoint,
)
from cgrasolve.graph.hyperdfg import HyperDFG, HyperVal
from cgrasolve.algorithms.scheduler.types import require_node_schedule_time
from cgrasolve.algorithms.router.types import (
    RoutingNodePlacementState,
    RouteBundle,
    TransportProfile,
    TransportMode,
    RoutingResult,
    RoutingStats,
    HyperValNetInfo,
    SinkAndLatency,
    VertexData,
    VertexAndCost
)
from cgrasolve.algorithms.router.candidate_generator import CandidateGenerator
from cgrasolve.algorithms.router.negotiated_router import NegotiatedRouter
from cgrasolve.algorithms.router.routing_state import RoutingState
from cgrasolve.algorithms.common.routing_validation import (
    resolve_sink_data_bitwidth,
    resolve_source_output_bitwidth_info,
)


class PathFinder:
    """PathFinder Algorithm for heuristic mapping."""

    # DFG and MRRG graphs
    _dfg: DFG
    _mrrg: MRRG
    _hyperdfg: HyperDFG

    # Mapping from a hyper edge and value index to a routing node
    _placement: Dict[DFGNode, MRRGNode]
    _routing_nodes: Dict[MRRGNode, RoutingNodePlacementState]

    # Penalty multipliers

    # Present congestion factor
    _p_factor: float
    _p_initial: float
    _p_growth_rate: float

    # Historical congestion factor
    _h_factor: float
    _h_initial: float
    _h_growth_rate: float

    # Routing solution
    _routing_solution: Dict[Tuple[HyperVal, int], RouteBundle]

    # Maximum routing iterations
    _max_iterations: int

    # Debug mode
    _debug: bool

    # Most recent sink-level routing failure summary (for debug/mapper reporting)
    _last_route_failure: Optional[Dict[str, Any]]

    # Most recent structured routing result
    _last_routing_result: Optional[RoutingResult]

    # Width mismatch cost: penalty per unit of width ratio above 1.0.
    # Penalizes routing narrow data on wide buses to prefer exact-width matches.
    WIDTH_MISMATCH_COST: float = 0.5

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Dict[DFGNode, MRRGNode],
        p_growth_rate: float = 1.5,
        h_growth_rate: float = 1.5,
        p_factor_initial: float = 1.0,
        h_factor_initial: float = 1.0,
        max_iterations: int = 70,
        debug: bool = False,
        deadline: Optional[float] = None,
        target_ii: Optional[int] = None,
        warm_start_historical_costs: Optional[Dict[str, float]] = None,
        warm_start_decay: float = 0.5,
    ) -> None:
        """
        Initialize PathFinder router.

        Args:
            dfg: Data flow graph to route
            mrrg: Modulo routing resource graph
            placement: Initial placement of DFG operations to MRRG function nodes
            p_growth_rate: Present congestion penalty growth rate per iteration
            h_growth_rate: Historical congestion penalty growth rate per iteration
            p_factor_initial: Initial present congestion penalty multiplier
            h_factor_initial: Initial historical congestion penalty multiplier
            max_iterations: Maximum number of negotiation iterations (default 70)
            debug: Enable debug output (default False)
            target_ii: Target initiation interval (defaults to mrrg.II)
            warm_start_historical_costs: Historical costs from a previous routing
                attempt, keyed by MRRG node ID. Used to bias routing away from
                previously congested resources.
            warm_start_decay: Decay factor (0.0-1.0) applied to warm-start
                historical costs. Previous placement was different, so not all
                congestion is relevant. Lower = forget more, higher = trust more.
        """
        self._dfg = dfg
        self._mrrg = mrrg
        self._target_ii = target_ii
        self._warm_start_historical_costs = warm_start_historical_costs
        self._warm_start_decay = warm_start_decay

        # Create HyperDFG representation
        self._hyperdfg = HyperDFG.from_dfg(dfg)

        # Initialize placement
        self._placement = {}
        for dfg_node, mrrg_node in placement.items():
            self._placement[dfg_node] = mrrg_node

        # Penalty factors and growth rates (not reset between calls)
        self._p_growth_rate = p_growth_rate
        self._h_growth_rate = h_growth_rate
        self._p_initial = p_factor_initial
        self._h_initial = h_factor_initial

        # Maximum negotiation iterations
        self._max_iterations = max_iterations

        # Debug mode
        self._debug = debug
        self._deadline = deadline

        # Initialize mutable per-call state
        self._reset_routing_state()

        # Precompute max Manhattan distance of any single MRRG edge (for A* heuristic).
        # Dividing the Manhattan distance by this value keeps the heuristic admissible
        # even when the MRRG contains long-wire or non-local routing edges.
        max_step = 0
        for edge in self._mrrg.get_edges():
            src_coords = edge.source.coordinates
            dst_coords = edge.destination.coordinates
            if src_coords and dst_coords:
                delta = abs(src_coords[0] - dst_coords[0]) + abs(src_coords[1] - dst_coords[1])
                if delta > max_step:
                    max_step = delta
        self._max_coord_manhattan_step = max_step

    def _deadline_exceeded(self) -> bool:
        """Return True when the routing deadline has expired."""
        return self._deadline is not None and time.time() >= self._deadline

    def _build_timeout_result(
        self,
        *,
        message: str,
        hypervals_considered: int,
        destinations_considered: int,
        negotiation_iterations: int,
    ) -> RoutingResult:
        """Build and publish a structured timeout result."""
        stats = self._collect_routing_stats(
            hypervals_considered=hypervals_considered,
            destinations_considered=destinations_considered,
            negotiation_iterations=negotiation_iterations,
        )
        # Snapshot congestion before rip-up zeroes occupancy
        self._pre_ripup_congestion_map = self.get_congestion_map()
        for hyperval in self._hyperdfg.get_edges():
            self._rip_up_hyperval(hyperval)
        self._p_factor = self._p_initial
        self._h_factor = self._h_initial
        result = RoutingResult(
            success=False,
            timed_out=True,
            failure_summary=self.get_last_route_failure_summary(),
            message=message,
            stats=stats,
        )
        self._last_routing_result = result
        return result

    def _reset_routing_state(self) -> None:
        """Reset all mutable per-call state so route_dfg() can be called again."""
        # Initialize routing nodes state for both ROUTING and ROUTING_FUNCTION
        self._routing_nodes = {}
        for mrrg_node in self._mrrg.get_nodes():
            if mrrg_node.node_type in (NodeType.ROUTING, NodeType.ROUTING_FUNCTION):
                self._routing_nodes[mrrg_node] = RoutingNodePlacementState()

        # Mark ROUTING_FUNCTION nodes placed for compute as reserved
        for dfg_node, mrrg_node in self._placement.items():
            if mrrg_node.node_type == NodeType.ROUTING_FUNCTION:
                if mrrg_node in self._routing_nodes:
                    self._routing_nodes[mrrg_node].reserved_for_compute = True

        # Routing solution: (HyperVal, dest_idx) -> path of MRRG nodes
        self._routing_solution = {}

        # Reset penalty factors
        self._p_factor = self._p_initial
        self._h_factor = self._h_initial
        self._last_route_failure = None
        self._last_routing_result = None
        self._pre_ripup_congestion_map: Dict[Tuple[int, int], float] = {}
        self._routing_state = RoutingState(self)
        self._candidate_generator = CandidateGenerator(self)
        self._negotiated_router = NegotiatedRouter(self)

    @staticmethod
    def _bump_debug_counter(
        counters: Optional[Dict[str, int]],
        key: str,
        amount: int = 1,
    ) -> None:
        """Increment a debug counter if diagnostics collection is enabled."""
        if counters is None:
            return
        counters[key] = counters.get(key, 0) + amount

    @staticmethod
    def _format_debug_counters(counters: Optional[Dict[str, int]]) -> str:
        """Format non-zero debug counters into a compact stable string."""
        if not counters:
            return "none"
        items = [(key, value) for key, value in sorted(counters.items()) if value]
        if not items:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in items)

    def get_last_route_failure_summary(self) -> Optional[str]:
        """Return a compact text summary for the most recent sink-level failure."""
        if not self._last_route_failure:
            return None

        info = self._last_route_failure
        return (
            f"{info['source_id']}->{info['dest_id']} [sink {info['sink_idx']}] "
            f"bw={info['token_bitwidth']} cycles={info['cycles_to_sink']} "
            f"fanouts={info['fanout_candidates']} flat_ok={info['flat_candidates']} "
            f"packed_ok={info.get('packed_candidates', 0)} "
            f"fractured_ok={info['fractured_candidates']} "
            f"flat_stats=({self._format_debug_counters(info.get('flat_stats'))}) "
            f"packed_stats=({self._format_debug_counters(info.get('packed_stats'))}) "
            f"fractured_stats=({self._format_debug_counters(info.get('fractured_stats'))})"
        )

    def get_last_route_failure(self) -> Optional[Dict[str, Any]]:
        """Return the most recent sink-level failure dictionary."""
        if self._last_route_failure is None:
            return None
        return dict(self._last_route_failure)

    def get_last_routing_result(self) -> Optional[RoutingResult]:
        """Return the most recent structured routing result."""
        return self._last_routing_result

    def get_routing_solution(self) -> Dict[Tuple[HyperVal, int], RouteBundle]:
        """Return a shallow copy of the current routing solution."""
        return dict(self._routing_solution)

    # ---- Congestion extraction (B2) and warm-start (B3) ----------------------

    def get_congestion_map(self) -> Dict[Tuple[int, int], float]:
        """Spatial congestion from current routing state.

        Returns (row, col) -> total overuse at that location.
        Available after route() returns (even on failure).

        On routing failure, route() rips up all routes (zeroing occupancy)
        before returning. In that case this method returns the congestion
        snapshot captured just before the rip-up.
        """
        congestion: Dict[Tuple[int, int], float] = {}
        for node, state in self._routing_nodes.items():
            cap = self._node_capacity(node)
            if state.occupancy > cap and node.coordinates:
                coord = node.coordinates
                congestion[coord] = congestion.get(coord, 0.0) + (state.occupancy - cap)
        # If live occupancy was wiped by rip-up, return the pre-rip-up snapshot.
        if not congestion and self._pre_ripup_congestion_map:
            return dict(self._pre_ripup_congestion_map)
        return congestion

    def get_historical_congestion_map(self) -> Dict[Tuple[int, int], float]:
        """Accumulated historical congestion by spatial location.

        Historical cost captures persistent bottlenecks across negotiation
        iterations within a single route() call.
        """
        congestion: Dict[Tuple[int, int], float] = {}
        for node, state in self._routing_nodes.items():
            if state.historical_cost > 0 and node.coordinates:
                coord = node.coordinates
                congestion[coord] = congestion.get(coord, 0.0) + state.historical_cost
        return congestion

    def get_historical_costs(self) -> Dict[str, float]:
        """Extract per-node historical costs for warm-starting next routing attempt.

        Returns MRRG node ID -> accumulated historical cost.
        """
        return {
            node.id: state.historical_cost
            for node, state in self._routing_nodes.items()
            if state.historical_cost > 0
        }

    def _emit(self, message: str) -> None:
        """Emit debug output only when router debug mode is enabled."""
        if self._debug:
            print(message)

    def _collect_routing_stats(
        self,
        hypervals_considered: int = 0,
        destinations_considered: int = 0,
        negotiation_iterations: int = 0,
    ) -> RoutingStats:
        """Collect aggregate routing statistics for the current solution."""
        resources_used = sum(
            1 for state in self._routing_nodes.values() if state.occupancy > 0
        )
        total_occupancy = sum(state.occupancy for state in self._routing_nodes.values())
        total_capacity = sum(self._node_capacity(node) for node in self._routing_nodes.keys())

        return RoutingStats(
            hypervals_considered=hypervals_considered,
            destinations_considered=destinations_considered,
            negotiation_iterations=negotiation_iterations,
            routed_nets=len(self._routing_solution),
            resources_used=resources_used,
            total_occupancy=total_occupancy,
            total_capacity=total_capacity,
        )

    def _emit_routing_stats(self, stats: RoutingStats) -> None:
        """Print routing statistics when debug output is enabled."""
        self._emit(f"\n[STATS] Number of routing resources used: {stats.resources_used}")
        self._emit(f"[STATS] Total hypervals routed: {stats.routed_nets}")
        self._emit(f"[STATS] Total occupancy: {stats.total_occupancy}")
        self._emit(f"[STATS] Total capacity: {stats.total_capacity}")
        if stats.total_capacity > 0:
            self._emit(f"[STATS] Utilization: {stats.utilization_percent:.2f}%")

    def _compute_cycles_source_to_sink(self, source_node: DFGNode, sink_node: DFGNode, edge_dist: int = 0) -> int:
        """
        Compute timing constraint from source to sink (static timing).

        Args:
            source_node: Source DFG operation
            sink_node: Sink DFG operation
            edge_dist: Loop iteration distance (for loop-carried dependencies)

        Returns:
            Required latency in cycles from source to sink
        """
        source_time = require_node_schedule_time(source_node)
        sink_time = require_node_schedule_time(sink_node)

        II = self._target_ii if self._target_ii is not None else self._mrrg.II

        # Unified formula: the sink fires at cycle (sink_time + edge_dist * II)
        # relative to the source at cycle source_time.
        if II is not None:
            cycles_to_sink = (sink_time + edge_dist * II) - source_time
        else:
            cycles_to_sink = sink_time - source_time

        # Non-loop-carried self-edges (same node, dist=0) need a full II wrap
        if II is not None and sink_node == source_node and edge_dist == 0:
            cycles_to_sink = II
        # Wrap negative results (backward edges within same iteration)
        elif II is not None and cycles_to_sink < 0:
            cycles_to_sink += II

        return cycles_to_sink

    @staticmethod
    def _node_capacity(node: MRRGNode) -> int:
        """Return effective routing capacity for a node."""
        if node.num_lanes > 1:
            return node.num_lanes
        return node.capacity

    @staticmethod
    def _route_uses_packed_lane(
        node: MRRGNode,
        transport_profile: TransportProfile,
    ) -> bool:
        """Return True when ``transport_profile`` consumes one carrier lane on ``node``."""
        return (
            transport_profile.mode == TransportMode.PACKED
            and transport_profile.lane_index is not None
            and node.num_lanes > 1
            and node.lane_width >= (transport_profile.lane_width or 0)
        )

    @staticmethod
    def _recompute_lane_node_occupancy(
        node: MRRGNode,
        state: RoutingNodePlacementState,
    ) -> None:
        """Recompute occupancy for lane-capable nodes from lane/full-width usage."""
        if node.num_lanes <= 1:
            return
        if state.full_width_hyperval_usage:
            state.occupancy = node.num_lanes
            return
        used_lanes = 0
        for lane_users in state.packed_lane_usage.values():
            if lane_users:
                used_lanes += 1
        state.occupancy = used_lanes

    @staticmethod
    def _lane_is_available(
        node: MRRGNode,
        state: RoutingNodePlacementState,
        hyperval: HyperVal,
        lane_index: int,
    ) -> bool:
        """Return True when ``lane_index`` is legal for ``hyperval`` on ``node``."""
        if node.num_lanes <= 1 or lane_index < 0 or lane_index >= node.num_lanes:
            return False
        if not node.allows_lane_index(lane_index):
            return False

        # A full-width occupant blocks all packed lanes unless it is the same
        # HyperVal already using the node.
        blocking_full_width = {
            hv for hv, count in state.full_width_hyperval_usage.items() if count > 0
        }
        if blocking_full_width and blocking_full_width != {hyperval}:
            return False

        lane_users = state.packed_lane_usage.get(lane_index, {})
        if not lane_users:
            return True
        return set(lane_users.keys()) == {hyperval}

    @staticmethod
    def _matches_speculative_lane_assignment(
        node: MRRGNode,
        lane_index: int,
        speculative_lane_assignments: Optional[Dict[str, int]] = None,
    ) -> bool:
        """Return True when ``node`` is either unclaimed or uses ``lane_index``."""
        if not speculative_lane_assignments or node.num_lanes <= 1:
            return True

        assigned_lane = speculative_lane_assignments.get(node.id)
        return assigned_lane is None or assigned_lane == lane_index

    @staticmethod
    def _node_supports_packed_transport(
        node: MRRGNode,
        transport_profile: TransportProfile,
    ) -> bool:
        """Return True when ``node`` can carry the packed transport profile."""
        lane_width = transport_profile.lane_width
        return (
            transport_profile.mode == TransportMode.PACKED
            and lane_width is not None
            and node.num_lanes > 1
            and node.lane_width >= lane_width
        )

    @staticmethod
    def _route_conflicts_with_speculative_transport(
        nodes: Set[MRRGNode],
        transport_profile: TransportProfile,
        speculative_lane_assignments: Optional[Dict[str, int]] = None,
        speculative_full_width_nodes: Optional[Set[str]] = None,
    ) -> bool:
        """Return True when ``nodes`` would mix incompatible speculative transport."""
        for node in nodes:
            if node.num_lanes <= 1:
                continue

            if transport_profile.mode == TransportMode.PACKED:
                if speculative_full_width_nodes and node.id in speculative_full_width_nodes:
                    return True
                if transport_profile.lane_index is None:
                    continue
                if not PathFinder._matches_speculative_lane_assignment(
                    node,
                    transport_profile.lane_index,
                    speculative_lane_assignments,
                ):
                    return True
                continue

            if speculative_lane_assignments and node.id in speculative_lane_assignments:
                return True

        return False

    def _compute_cost(self, node: MRRGNode) -> float:
        """
        PathFinder cost function using committed occupancy.

        Cost = base_cost + h_cost + oc_cost

        Where:
            - base_cost: Fixed wire length cost (always 1.0)
            - h_cost: Historical congestion penalty
            - oc_cost: Present overcapacity penalty

        Args:
            node: MRRG routing node

        Returns:
            Total cost for using this node
        """
        if node not in self._routing_nodes:
            return 1.0

        state = self._routing_nodes[node]

        occupancy = state.occupancy
        capacity = self._node_capacity(node)
        base_cost = state.base_cost
        historical_cost = state.historical_cost

        h_cost = self._h_factor * historical_cost
        overuse = max(0, occupancy - capacity)
        oc_cost = 1.0 + self._p_factor * overuse

        return base_cost + h_cost + oc_cost

    def _compute_prospective_cost(
        self,
        node: MRRGNode,
        hyperval: HyperVal,
        speculative_hv_nodes: Optional[Set[MRRGNode]] = None,
        transport_profile: Optional[TransportProfile] = None,
    ) -> float:
        """Cost as if this HyperVal were added to the node.

        Same-HyperVal reuse, whether already committed or already claimed
        speculatively by earlier sinks of the same HyperVal, does not increase
        prospective occupancy.
        """
        if node not in self._routing_nodes:
            return 1.0

        state = self._routing_nodes[node]
        profile = transport_profile or TransportProfile.unknown()

        # Start from current occupancy
        prospective_occ = state.occupancy

        if self._route_uses_packed_lane(node, profile):
            assert profile.lane_index is not None
            lane_users = state.packed_lane_usage.get(profile.lane_index, {})
            lane_already_used = bool(lane_users)
            uses_same_hv = set(lane_users.keys()) == {hyperval}
            if not lane_already_used:
                prospective_occ += 1
            elif uses_same_hv:
                prospective_occ = state.occupancy
            else:
                prospective_occ = self._node_capacity(node) + 1
        else:
            # Increment only if this HyperVal is not already using this node
            # and the node is not already claimed speculatively
            if hyperval not in state.hyperval_usage:
                if speculative_hv_nodes is None or node not in speculative_hv_nodes:
                    if node.num_lanes > 1:
                        prospective_occ = self._node_capacity(node)
                    else:
                        prospective_occ += 1

        base_cost = state.base_cost
        historical_cost = state.historical_cost
        h_cost = self._h_factor * historical_cost
        overuse = max(0, prospective_occ - self._node_capacity(node))
        oc_cost = 1.0 + self._p_factor * overuse

        return base_cost + h_cost + oc_cost

    def _score_candidate_route(
        self,
        fanout: MRRGNode,
        route: List[MRRGNode],
        transport_profile: TransportProfile,
        hyperval: HyperVal,
        speculative_hv_nodes: Optional[Set[MRRGNode]] = None,
    ) -> float:
        """Score a candidate route for fanout selection.

        Includes the root fanout itself, because Dijkstra seeds the start node
        with cost 0 and therefore does not account for root-fanout congestion.
        De-duplicates nodes so the root is not double-counted if it already
        appears in ``route``.
        """
        total_cost = 0.0
        seen: Set[MRRGNode] = set()
        for node in [fanout, *route]:
            if node in seen:
                continue
            seen.add(node)
            if node not in self._routing_nodes:
                continue
            total_cost += self._compute_prospective_cost(
                node,
                hyperval,
                speculative_hv_nodes,
                transport_profile,
            )
            total_cost += transport_profile.width_penalty(node, self.WIDTH_MISMATCH_COST)
        return total_cost

    def _score_candidate_bundle(
        self,
        bundle: RouteBundle,
        hyperval: HyperVal,
        speculative_hv_nodes: Optional[Set[MRRGNode]] = None,
    ) -> float:
        """Score a bundle candidate using unique-node occupancy cost only."""
        total_cost = 0.0
        for node in bundle.all_nodes:
            if node not in self._routing_nodes:
                continue
            total_cost += self._compute_prospective_cost(
                node,
                hyperval,
                speculative_hv_nodes,
                bundle.transport_profile,
            )
        return total_cost

    @staticmethod
    def _bundle_preference_rank(bundle: RouteBundle) -> int:
        """Return a deterministic transport tie-break preference rank."""
        if bundle.transport_profile.mode == TransportMode.PACKED:
            return 2
        if bundle.fractured:
            return 1
        return 0

    def _candidate_beats_best(
        self,
        *,
        candidate_bundle: RouteBundle,
        candidate_cost: float,
        best_bundle: Optional[RouteBundle],
        best_cost: float,
    ) -> bool:
        """Return True when a candidate should replace the current best bundle."""
        if best_bundle is None:
            return True

        epsilon = 1e-9
        if candidate_cost < best_cost - epsilon:
            return True
        if abs(candidate_cost - best_cost) > epsilon:
            return False

        return self._bundle_preference_rank(candidate_bundle) > self._bundle_preference_rank(
            best_bundle
        )

    def _track_bundle_nodes(
        self,
        claimed_nodes: Set[MRRGNode],
        bundle: RouteBundle,
    ) -> None:
        """Track the nodes claimed by earlier sinks of the same HyperVal."""
        claimed_nodes.update(bundle.all_nodes)

    def _commit_node(
        self,
        hyperval_net: Tuple[HyperVal, int],
        node: MRRGNode,
        transport_profile: TransportProfile,
    ) -> None:
        """
        Commit a routing decision - track node usage for a net.

        Updates occupancy (per distinct HyperVal) and tracks which hypervals use this node.
        Does NOT build the path in _routing_solution — the caller stores the RouteBundle.

        Args:
            hyperval_net: The (HyperVal, dest_idx) identifier for this net
            node: MRRG routing node to add to the route
        """
        if node not in self._routing_nodes:
            return

        hyperval, _ = hyperval_net

        # Update node state
        state = self._routing_nodes[node]

        # Increment per-HyperVal usage
        prev = state.hyperval_usage.get(hyperval, 0)
        state.hyperval_usage[hyperval] = prev + 1

        if self._route_uses_packed_lane(node, transport_profile):
            assert transport_profile.lane_index is not None
            lane_users = state.packed_lane_usage.setdefault(transport_profile.lane_index, {})
            lane_users[hyperval] = lane_users.get(hyperval, 0) + 1
            self._recompute_lane_node_occupancy(node, state)
        elif node.num_lanes > 1:
            state.full_width_hyperval_usage[hyperval] = (
                state.full_width_hyperval_usage.get(hyperval, 0) + 1
            )
            self._recompute_lane_node_occupancy(node, state)
        else:
            # Only bump occupancy when this HyperVal first uses the node
            if prev == 0:
                state.occupancy += 1

        state.values.append(hyperval_net)

    def _rip_up_hyperval_net(self, hyperval_net: Tuple[HyperVal, int]) -> None:
        """
        Remove routing for a net.

        This is called during negotiated congestion resolution to remove
        existing routes before re-routing.

        Args:
            hyperval_net: The (HyperVal, dest_idx) identifier for the net to rip up
        """
        if hyperval_net not in self._routing_solution:
            return

        bundle = self._routing_solution[hyperval_net]
        transport_profile = bundle.transport_profile
        hyperval, _ = hyperval_net

        for node in bundle.all_nodes:
            if node not in self._routing_nodes:
                continue

            state = self._routing_nodes[node]

            # Remove from mapped values list (per-sink record)
            try:
                state.values.remove(hyperval_net)
            except ValueError:
                raise ValueError(
                    f"Cannot find {hyperval_net} in {node.get_full_name()} values"
                ) from None

            # Update per-HyperVal usage and occupancy
            hv_count = state.hyperval_usage.get(hyperval, 0)
            if hv_count <= 0:
                raise ValueError(
                    f"HyperVal {hyperval} had non-positive usage on {node.get_full_name()}"
                )
            else:
                if hv_count == 1:
                    del state.hyperval_usage[hyperval]
                else:
                    state.hyperval_usage[hyperval] = hv_count - 1

            if self._route_uses_packed_lane(node, transport_profile):
                assert transport_profile.lane_index is not None
                lane_users = state.packed_lane_usage.get(transport_profile.lane_index)
                if not lane_users or hyperval not in lane_users:
                    raise ValueError(
                        f"Packed lane {transport_profile.lane_index} missing "
                        f"HyperVal {hyperval} on {node.get_full_name()}"
                    )
                if lane_users[hyperval] == 1:
                    del lane_users[hyperval]
                    if not lane_users:
                        del state.packed_lane_usage[transport_profile.lane_index]
                else:
                    lane_users[hyperval] -= 1
                self._recompute_lane_node_occupancy(node, state)
            elif node.num_lanes > 1:
                full_width_users = state.full_width_hyperval_usage
                if hyperval not in full_width_users:
                    raise ValueError(
                        f"Full-width packed-capable usage missing HyperVal {hyperval} "
                        f"on {node.get_full_name()}"
                    )
                if full_width_users[hyperval] == 1:
                    del full_width_users[hyperval]
                else:
                    full_width_users[hyperval] -= 1
                self._recompute_lane_node_occupancy(node, state)
            else:
                if hv_count == 1:
                    if state.occupancy <= 0:
                        raise ValueError(
                            f"Occupancy was already zero or less for {node.get_full_name()}: "
                            f"{state.occupancy}"
                        )
                    state.occupancy -= 1

        del self._routing_solution[hyperval_net]

    def _rip_up_hyperval(self, hyperval: HyperVal) -> None:
        """
        Remove routing for ALL destinations of a HyperVal.

        Args:
            hyperval: The HyperVal whose all destinations should be ripped up
        """
        for dest_idx in range(hyperval.cardinality):
            hyperval_net = (hyperval, dest_idx)
            self._rip_up_hyperval_net(hyperval_net)

    def _update_historical_costs(self) -> None:
        """Update historical congestion costs once per negotiation iteration.

        Node-based and order-independent: every overused node gets its
        historical cost incremented by its current overuse amount.
        """
        for node, state in self._routing_nodes.items():
            capacity = self._node_capacity(node)
            if state.occupancy > capacity:
                state.historical_cost += state.occupancy - capacity

    def _check_overuse(self, verbose: bool = False) -> bool:
        """
        Check if routing is legal (no capacity violations).

        Args:
            verbose: If True, print details of each overused node

        Returns:
            True if all nodes have occupancy <= capacity, False otherwise
        """
        is_legal = True
        num_overused = 0
        overused_nodes = []

        for node, state in self._routing_nodes.items():
            capacity = self._node_capacity(node)
            if state.occupancy > capacity:
                is_legal = False
                num_overused += 1
                overused_nodes.append((node, state, capacity))

        if not is_legal and (verbose or self._debug):
            print(f"  [OVERUSE] {num_overused} nodes overused")

            for node, state, capacity in overused_nodes:
                overuse_amount = state.occupancy - capacity
                print(f"    ! {node.get_full_name()}")
                print(f"      Occupancy: {state.occupancy}/{capacity} (overuse: +{overuse_amount})")
                if state.values:
                    print(f"      Used by {len(state.values)} net(s):")
                    for hyperval_net in state.values[:3]:
                        hyperval, dest_idx = hyperval_net
                        dest_id = hyperval.destination_ids[dest_idx]
                        src_hyper = self._hyperdfg.get_node(hyperval.source_id)
                        dst_hyper = self._hyperdfg.get_node(dest_id)
                        src_op = ""
                        dst_op = ""
                        if src_hyper:
                            src_dfg = self._dfg.get_node(src_hyper.original_dfg_node_id)
                            if src_dfg:
                                src_op = f" ({src_dfg.operation.value})"
                        if dst_hyper:
                            dst_dfg = self._dfg.get_node(dst_hyper.original_dfg_node_id)
                            if dst_dfg:
                                dst_op = f" ({dst_dfg.operation.value})"
                        print(f"        - {hyperval.source_id}{src_op} -> {dest_id}{dst_op} [sink {dest_idx}]")
                    if len(state.values) > 3:
                        print(f"        ... and {len(state.values) - 3} more net(s)")

        return is_legal

    def _compute_dfg_coverage(self, verbose: bool = False) -> bool:
        """
        Check if all hypervals are routed.

        Args:
            verbose: If True, print details of each uncovered hyperval

        Returns:
            True if every (HyperVal, dest_idx) has a routing, False otherwise
        """
        is_covered = True
        num_uncovered = 0
        uncovered_nets = []

        for hyperval in self._hyperdfg.get_edges():
            for dest_idx in range(hyperval.cardinality):
                hyperval_net = (hyperval, dest_idx)

                if hyperval_net not in self._routing_solution:
                    is_covered = False
                    num_uncovered += 1
                    uncovered_nets.append(hyperval_net)

        if not is_covered and (verbose or self._debug):
            print(f"  [UNCOVERED] {num_uncovered} hypervals not routed")

            for hyperval_net in uncovered_nets:
                hyperval, dest_idx = hyperval_net
                dest_id = hyperval.destination_ids[dest_idx]
                src_hyper = self._hyperdfg.get_node(hyperval.source_id)
                dst_hyper = self._hyperdfg.get_node(dest_id)
                src_info = f"{hyperval.source_id}"
                dst_info = f"{dest_id}"

                if src_hyper:
                    src_dfg = self._dfg.get_node(src_hyper.original_dfg_node_id)
                    if src_dfg:
                        src_op = src_dfg.operation.value
                        src_mrrg = self._placement.get(src_dfg)
                        src_place = f" @ {src_mrrg.get_full_name()}" if src_mrrg else ""
                        src_info = f"{hyperval.source_id} ({src_op}){src_place}"

                if dst_hyper:
                    dst_dfg = self._dfg.get_node(dst_hyper.original_dfg_node_id)
                    if dst_dfg:
                        dst_op = dst_dfg.operation.value
                        dst_mrrg = self._placement.get(dst_dfg)
                        dst_place = f" @ {dst_mrrg.get_full_name()}" if dst_mrrg else ""
                        dst_info = f"{dest_id} ({dst_op}){dst_place}"

                print(f"    x {src_info} -> {dst_info} [sink {dest_idx}]")

        return is_covered

    @staticmethod
    def _operand_tag_matches_request(
        requested: OperandTag, supported: Set[OperandTag]
    ) -> bool:
        """Check if a requested operand tag is satisfied by a node's supported set."""
        return MRRGNode.operand_tag_matches_request(requested, supported)

    def _compute_heuristic(
        self,
        node: MRRGNode,
        sink: MRRGNode,
        cycles: int,
        cycles_to_sink: int,
        transport_profile: Optional[TransportProfile] = None,
    ) -> float:
        """Admissible A* heuristic for MRRG pathfinding.

        Returns a lower bound on the cost from *node* to *sink*.  The bound is
        based on the Manhattan distance between grid coordinates, divided by the
        maximum coordinate step of any single MRRG edge.  This keeps the
        estimate safe even when the graph contains long-wire or non-local edges.

        For packed transport, the spatial heuristic is disabled because packed
        routing uses lane-aware cost functions that are not captured by the
        spatial bound.  Temporal pruning is always applied.
        """
        if cycles > cycles_to_sink:
            return float("inf")
        if node == sink:
            return 0.0
        # Disable spatial heuristic for packed/fractured routes to avoid
        # path-selection changes that conflict with lane assignment.
        if transport_profile is not None and transport_profile.mode not in (
            TransportMode.FLAT, TransportMode.UNKNOWN,
        ):
            return 0.0
        if not node.coordinates or not sink.coordinates:
            return 0.0
        if self._max_coord_manhattan_step <= 0:
            return 0.0
        dx = abs(node.coordinates[0] - sink.coordinates[0])
        dy = abs(node.coordinates[1] - sink.coordinates[1])
        manhattan = dx + dy
        min_steps = math.ceil(manhattan / self._max_coord_manhattan_step)
        return float(min_steps * 1.0)

    def _dijkstra_pathfinder(
        self,
        source: MRRGNode,
        sink: MRRGNode,
        tag_to_find: Optional[OperandTag],
        hyperval: HyperVal,
        routes: Set[MRRGNode],
        cycles_to_sink: int,
        latency_of_src: int,
        transport_profile: TransportProfile,
        dest_op: Optional[OperationType] = None,
        blocked_nodes: Optional[Set[MRRGNode]] = None,
        speculative_lane_assignments: Optional[Dict[str, int]] = None,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        debug_stats: Optional[Dict[str, int]] = None,
    ) -> List[MRRGNode]:
        """
        Dijkstra pathfinding with routing constraints.

        Args:
            source: Starting routing node
            sink: Destination FU node
            tag_to_find: Required operand tag (or None for UNTAGGED)
            hyperval: HyperVal being routed (for bitwidth checking)
            routes: Nodes already used by other sinks of this HyperVal
            cycles_to_sink: Required arrival time at sink
            latency_of_src: Starting latency
            transport_profile: Mode-aware legality/cost profile for this route
            dest_op: Destination operation type (for per-operation pin tag lookup)

        Returns:
            Path as list of MRRG nodes, or [] if not found
        """
        # Nodes already claimed by earlier sinks of this HyperVal.
        speculative_hv_nodes = set(routes)

        # State key: (node, cycles, tag_found) — tag_found separates tagged/untagged paths
        data: Dict[Tuple[MRRGNode, int, bool], VertexData] = {}
        to_visit: List[VertexAndCost] = []

        h_initial = self._compute_heuristic(source, sink, latency_of_src, cycles_to_sink, transport_profile)
        heapq.heappush(to_visit, VertexAndCost(
            f_cost=h_initial, cycles=latency_of_src, node=source,
            tag_found=False, g_cost=0.0,
        ))
        data[(source, latency_of_src, False)] = VertexData(
            fanin=[],
            lowest_known_cost=0.0,
            num_of_cycles=latency_of_src,
            tag_found=False
        )

        # Reject start node if illegal for the current transport profile.
        if transport_profile.mode == TransportMode.PACKED:
            if not self._node_supports_packed_transport(source, transport_profile):
                self._bump_debug_counter(debug_stats, "reject_source_not_packable")
                return []
            assert transport_profile.lane_index is not None
            if not self._lane_is_available(
                source,
                self._routing_nodes[source],
                hyperval,
                transport_profile.lane_index,
            ):
                self._bump_debug_counter(debug_stats, "reject_source_lane_busy")
                return []
            if not self._matches_speculative_lane_assignment(
                source,
                transport_profile.lane_index,
                speculative_lane_assignments,
            ):
                self._bump_debug_counter(
                    debug_stats,
                    "reject_source_speculative_lane_mismatch",
                )
                return []
        else:
            if (
                transport_profile.legal_min_bitwidth is not None
                and source.bitwidth < transport_profile.legal_min_bitwidth
            ):
                self._bump_debug_counter(debug_stats, "reject_source_too_narrow")
                return []
            if source.packing_inject_only:
                self._bump_debug_counter(debug_stats, "reject_source_inject_only")
                return []
            if transport_profile.rejects_node_width(source):
                self._bump_debug_counter(debug_stats, "reject_source_exact_width")
                return []

        found: Optional[MRRGNode] = None
        found_tag_state: bool = False

        while to_visit and not found:
            queue_top = heapq.heappop(to_visit)
            explore_curr = queue_top.node
            cycles_to_curr = queue_top.cycles
            tag_found_curr = queue_top.tag_found

            state_key = (explore_curr, cycles_to_curr, tag_found_curr)

            # Reopen-safe: skip stale queue entries whose g-cost is worse
            # than the best known g-cost for this state.
            best = data.get(state_key)
            if best is None:
                continue
            if queue_top.g_cost > best.lowest_known_cost + 1e-9:
                self._bump_debug_counter(debug_stats, "stale_pops")
                continue
            self._bump_debug_counter(debug_stats, "states_popped")

            if not found and sink == explore_curr:
                if cycles_to_sink == cycles_to_curr:
                    found = explore_curr
                    found_tag_state = tag_found_curr
                else:
                    self._bump_debug_counter(debug_stats, "reject_wrong_arrival_cycle")
                    if state_key in data:
                        del data[state_key]
                continue

            for fanout_edge in self._mrrg.get_outgoing_edges(explore_curr.id):
                fanout = self._mrrg.get_node(fanout_edge.destination.id)
                if not fanout:
                    continue

                if fanout == source:
                    self._bump_debug_counter(debug_stats, "skip_back_to_source")
                    continue

                if fanout.node_type == NodeType.FUNCTION and fanout != sink:
                    self._bump_debug_counter(debug_stats, "skip_other_function")
                    continue

                # Block routing through compute-reserved ROUTING_FUNCTION nodes
                if fanout.node_type == NodeType.ROUTING_FUNCTION:
                    if fanout in self._routing_nodes and self._routing_nodes[fanout].reserved_for_compute:
                        if fanout != sink:
                            self._bump_debug_counter(debug_stats, "skip_reserved_route_fu")
                            continue

                if blocked_nodes and fanout in blocked_nodes and fanout != sink:
                    self._bump_debug_counter(debug_stats, "skip_blocked_node")
                    continue

                # Enforce predecessor consistency for multi-fanout HyperVals.
                # If an earlier sink already claimed this node via a specific
                # predecessor, later sinks must enter from the same predecessor
                # to avoid crossbar mux select conflicts in FASM generation.
                if claimed_predecessors and fanout in claimed_predecessors:
                    if explore_curr != claimed_predecessors[fanout]:
                        self._bump_debug_counter(debug_stats, "skip_predecessor_mismatch")
                        continue

                # Reject width/lane-illegal paths for the selected transport mode.
                if transport_profile.mode == TransportMode.PACKED:
                    if not self._node_supports_packed_transport(fanout, transport_profile):
                        self._bump_debug_counter(debug_stats, "reject_not_packable")
                        continue
                    assert transport_profile.lane_index is not None
                    if not self._lane_is_available(
                        fanout,
                        self._routing_nodes[fanout],
                        hyperval,
                        transport_profile.lane_index,
                    ):
                        self._bump_debug_counter(debug_stats, "reject_lane_busy")
                        continue
                    if not self._matches_speculative_lane_assignment(
                        fanout,
                        transport_profile.lane_index,
                        speculative_lane_assignments,
                    ):
                        self._bump_debug_counter(
                            debug_stats,
                            "reject_speculative_lane_mismatch",
                        )
                        continue
                else:
                    if fanout.packing_inject_only:
                        self._bump_debug_counter(debug_stats, "reject_inject_only")
                        continue
                    if (
                        transport_profile.legal_min_bitwidth is not None
                        and fanout.bitwidth < transport_profile.legal_min_bitwidth
                    ):
                        self._bump_debug_counter(debug_stats, "reject_too_narrow")
                        continue
                    if transport_profile.rejects_node_width(fanout, is_sink=(fanout == sink)):
                        self._bump_debug_counter(debug_stats, "reject_exact_width")
                        continue

                data_curr = data[state_key]

                # Once tag found, only the sink is reachable
                if data_curr.tag_found and fanout != sink:
                    self._bump_debug_counter(debug_stats, "skip_after_tag_found")
                    continue

                # Prospective cost (accounts for speculative same-HyperVal usage)
                node_cost = self._compute_prospective_cost(
                    fanout,
                    hyperval,
                    speculative_hv_nodes,
                    transport_profile,
                )
                node_cost += transport_profile.width_penalty(
                    fanout,
                    self.WIDTH_MISMATCH_COST,
                )
                cost = queue_top.g_cost + node_cost
                cycles = data_curr.num_of_cycles

                if fanout in data_curr.fanin:
                    self._bump_debug_counter(debug_stats, "skip_cycle_in_path")
                    continue

                # Always accumulate latency from the current node
                cycles = cycles + explore_curr.latency

                if cycles > cycles_to_sink:
                    self._bump_debug_counter(debug_stats, "skip_cycles_exceeded")
                    continue

                # Determine tag_found state for this neighbor
                tag_found_new = data_curr.tag_found

                # Operation-aware operand tag check
                if tag_to_find is not None:
                    node_tags = (
                        fanout.get_supported_operand_tags_for_operation(dest_op)
                        if dest_op is not None
                        else fanout.supported_operand_tags
                    )
                    if node_tags:
                        if not self._operand_tag_matches_request(tag_to_find, node_tags):
                            self._bump_debug_counter(debug_stats, "reject_operand_tag")
                            continue

                        # Tag matches — mark as found if node is unused
                        if fanout in self._routing_nodes:
                            if len(self._routing_nodes[fanout].values) == 0 and \
                               fanout not in speculative_hv_nodes:
                                tag_found_new = True

                        if not tag_found_new:
                            self._bump_debug_counter(debug_stats, "reject_tag_node_busy")
                            continue
                    # If the pin advertises operand tags only for *other* ops,
                    # treat it as ordinary routing transit (not a sink for this
                    # route). Congestion accounting still tracks the use, so a
                    # later route that genuinely needs this pin as an operand
                    # sink will compete via PathFinder's negotiated cost.

                new_state_key = (fanout, cycles, tag_found_new)
                next_g = cost

                if new_state_key in data:
                    if data[new_state_key].lowest_known_cost <= next_g:
                        continue

                new_fanin = data_curr.fanin.copy()
                new_fanin.append(explore_curr)

                data[new_state_key] = VertexData(
                    fanin=new_fanin,
                    lowest_known_cost=next_g,
                    num_of_cycles=cycles,
                    tag_found=tag_found_new
                )

                h = self._compute_heuristic(fanout, sink, cycles, cycles_to_sink, transport_profile)
                heapq.heappush(to_visit, VertexAndCost(
                    f_cost=next_g + h, cycles=cycles, node=fanout,
                    tag_found=tag_found_new, g_cost=next_g,
                ))

        if not found:
            self._bump_debug_counter(debug_stats, "no_legal_path")
            return []

        final_data = data[(sink, cycles_to_sink, found_tag_state)]
        return final_data.fanin

    def _route_hyperval(self, hyperval: HyperVal) -> bool:
        """
        Route ALL destinations of a HyperVal together (multi-sink routing).

        Args:
            hyperval: The HyperVal to route (all destinations)

        Returns:
            True if routing succeeded for all destinations, False otherwise
        """
        self._last_route_failure = None

        if all(
            (hyperval, dest_idx) in self._routing_solution
            and self._routing_solution[(hyperval, dest_idx)]
            for dest_idx in range(hyperval.cardinality)
        ):
            return True
        plan = self._candidate_generator.build_hyperval_plan(hyperval)
        if plan is None:
            return False
        self._routing_state.commit_plan(plan)
        return True

    def _get_fracturable_output_candidates(
        self,
        src_fu: MRRGNode,
        src_operation: OperationType,
        required_bitwidth: Optional[int],
    ) -> List[FractureEndpoint]:
        """Return source output ports that expose a complete fracture layout."""
        candidates: List[FractureEndpoint] = []
        for port in self._mrrg.get_output_ports_for_operation(
            src_fu, src_operation, required_bitwidth=required_bitwidth
        ):
            for endpoint in self._mrrg.get_fracture_endpoints_for_output_port(port):
                if required_bitwidth is not None and endpoint.total_width != required_bitwidth:
                    continue
                candidates.append(endpoint)
        return candidates

    def _get_fracturable_input_candidates(
        self,
        sink_info: SinkAndLatency,
        required_bitwidth: Optional[int],
    ) -> List[FractureEndpoint]:
        """Return sink input pins that expose a matching fracture layout."""
        sink_fu = sink_info.sink_fu
        dest_op = sink_info.dest_dfg_node.operation

        candidates: List[FractureEndpoint] = []
        for port in self._mrrg.get_input_ports_for_operand(
            sink_fu, dest_op, sink_info.operand_tag
        ):
            for endpoint in self._mrrg.get_fracture_endpoints_for_input_port(port):
                if required_bitwidth is not None and endpoint.total_width != required_bitwidth:
                    continue
                candidates.append(endpoint)
        return candidates

    def _build_fractured_candidate(
        self,
        hyperval: HyperVal,
        source_endpoint: FractureEndpoint,
        sink_endpoint: FractureEndpoint,
        cycles_to_sink: int,
        speculative_hv_nodes: Set[MRRGNode],
        latency_of_src: int,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        debug_reasons: Optional[Dict[str, int]] = None,
    ) -> Optional[RouteBundle]:
        """Build a single fractured bundle candidate between explicit ports."""
        if (
            source_endpoint.chunk_indices,
            source_endpoint.chunk_widths,
            source_endpoint.chunk_offsets,
        ) != (
            sink_endpoint.chunk_indices,
            sink_endpoint.chunk_widths,
            sink_endpoint.chunk_offsets,
        ):
            self._bump_debug_counter(debug_reasons, "reject_layout_mismatch")
            return None

        split_chunks = list(source_endpoint.chunk_nodes)
        join_chunks = list(sink_endpoint.chunk_nodes)

        num_chunks = len(split_chunks)
        if len(join_chunks) != num_chunks:
            self._bump_debug_counter(debug_reasons, "reject_chunk_count_mismatch")
            return None

        source_boundaries: List[List[MRRGNode]] = []
        sink_boundaries: List[List[MRRGNode]] = []
        for chunk in split_chunks:
            path = self._mrrg.find_routing_path(source_endpoint.port, chunk)
            if not path:
                self._bump_debug_counter(debug_reasons, "reject_missing_source_boundary")
                return None
            source_boundaries.append(path)
        for chunk in join_chunks:
            path = self._mrrg.find_routing_path(chunk, sink_endpoint.port)
            if not path:
                self._bump_debug_counter(debug_reasons, "reject_missing_sink_boundary")
                return None
            sink_boundaries.append(path)

        blocked_chunk_nodes: Set[MRRGNode] = set()
        sub_paths: List[List[MRRGNode]] = []

        for idx in range(num_chunks):
            chunk_width = split_chunks[idx].bitwidth
            chunk_profile = TransportProfile.fractured_chunk(
                total_bitwidth=source_endpoint.total_width,
                chunk_index=idx,
                chunk_width=chunk_width,
                chunk_widths=source_endpoint.chunk_widths,
                chunk_offsets=source_endpoint.chunk_offsets,
            )
            prefix_latency = sum(node.latency for node in source_boundaries[idx][:-1])
            suffix_latency = sum(node.latency for node in sink_boundaries[idx])
            target_cycles = cycles_to_sink - suffix_latency
            start_cycles = latency_of_src + prefix_latency
            if target_cycles < start_cycles:
                self._bump_debug_counter(debug_reasons, "reject_negative_chunk_budget")
                return None

            dijkstra_path = self._dijkstra_pathfinder(
                source=split_chunks[idx],
                sink=join_chunks[idx],
                transport_profile=chunk_profile,
                cycles_to_sink=target_cycles,
                tag_to_find=None,
                dest_op=None,
                hyperval=hyperval,
                routes=speculative_hv_nodes,
                latency_of_src=start_cycles,
                blocked_nodes=blocked_chunk_nodes,
                claimed_predecessors=claimed_predecessors,
                debug_stats=debug_reasons,
            )
            if not dijkstra_path:
                self._bump_debug_counter(debug_reasons, "reject_chunk_path")
                return None

            full_path = (
                source_boundaries[idx][:-1]
                + dijkstra_path
                + sink_boundaries[idx]
            )
            sub_paths.append(full_path)

            blocked_chunk_nodes.update(dijkstra_path)
            blocked_chunk_nodes.add(join_chunks[idx])

        bundle = RouteBundle(
            sub_paths=sub_paths,
            fractured=True,
            num_chunks=num_chunks,
            chunk_indices=list(source_endpoint.chunk_indices),
            chunk_widths=list(source_endpoint.chunk_widths),
            chunk_offsets=list(source_endpoint.chunk_offsets),
            transport_profile=TransportProfile.fractured_bundle(
                total_bitwidth=source_endpoint.total_width,
                chunk_widths=source_endpoint.chunk_widths,
                chunk_offsets=source_endpoint.chunk_offsets,
            ),
        )

        for sub_path in bundle.sub_paths:
            total_latency = sum(node.latency for node in sub_path) + latency_of_src
            if total_latency != cycles_to_sink:
                self._bump_debug_counter(debug_reasons, "reject_chunk_latency_mismatch")
                return None

        return bundle

    def _build_fractured_candidates(
        self,
        hyperval: HyperVal,
        src_fu: MRRGNode,
        src_operation: OperationType,
        sink_info: SinkAndLatency,
        token_bitwidth: Optional[int],
        cycles_to_sink: int,
        speculative_hv_nodes: Set[MRRGNode],
        latency_of_src: int,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        debug_reasons: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[RouteBundle, float]]:
        """Enumerate legal fractured bundle candidates for a sink."""
        candidates: List[Tuple[RouteBundle, float]] = []

        output_candidates = self._get_fracturable_output_candidates(
            src_fu, src_operation, token_bitwidth
        )
        input_candidates = self._get_fracturable_input_candidates(sink_info, token_bitwidth)
        self._bump_debug_counter(debug_reasons, "output_candidates", len(output_candidates))
        self._bump_debug_counter(debug_reasons, "input_candidates", len(input_candidates))

        for source_endpoint in output_candidates:
            for sink_endpoint in input_candidates:
                self._bump_debug_counter(debug_reasons, "endpoint_pairs_considered")
                bundle = self._build_fractured_candidate(
                    hyperval=hyperval,
                    source_endpoint=source_endpoint,
                    sink_endpoint=sink_endpoint,
                    cycles_to_sink=cycles_to_sink,
                    speculative_hv_nodes=speculative_hv_nodes,
                    latency_of_src=latency_of_src,
                    claimed_predecessors=claimed_predecessors,
                    debug_reasons=debug_reasons,
                )
                if bundle is None:
                    continue
                cost = self._score_candidate_bundle(bundle, hyperval, speculative_hv_nodes)
                candidates.append((bundle, cost))
                self._bump_debug_counter(debug_reasons, "fractured_candidates")

        return candidates

    def _get_packable_output_candidates(
        self,
        src_fu: MRRGNode,
        src_operation: OperationType,
        required_bitwidth: Optional[int],
    ) -> List[PackingEndpoint]:
        """Return pack-capable source endpoints attached to an output port."""
        candidates: List[PackingEndpoint] = []
        for port in self._mrrg.get_output_ports_for_operation(src_fu, src_operation):
            for endpoint in self._mrrg.get_packing_endpoints_for_output_port(port):
                if required_bitwidth is not None and endpoint.lane_width < required_bitwidth:
                    continue
                candidates.append(endpoint)
        return candidates

    def _get_packable_input_candidates(
        self,
        sink_info: SinkAndLatency,
        required_bitwidth: Optional[int],
    ) -> List[PackingEndpoint]:
        """Return unpack-capable sink endpoints attached to an input port."""
        sink_fu = sink_info.sink_fu
        dest_op = sink_info.dest_dfg_node.operation

        candidates: List[PackingEndpoint] = []
        for port in self._mrrg.get_input_ports_for_operand(
            sink_fu,
            dest_op,
            sink_info.operand_tag,
        ):
            for endpoint in self._mrrg.get_packing_endpoints_for_input_port(port):
                if required_bitwidth is not None and endpoint.lane_width < required_bitwidth:
                    continue
                candidates.append(endpoint)
        return candidates

    def _collect_occupied_lane_nodes(
        self,
        *,
        hyperval: HyperVal,
    ) -> Set[MRRGNode]:
        """Return lane-capable nodes occupied by other HyperVals."""
        blocked: Set[MRRGNode] = set()
        for node, state in self._routing_nodes.items():
            if node.num_lanes <= 1:
                continue
            if state.occupancy <= 0:
                continue
            used_hypervals = set(state.hyperval_usage)
            if used_hypervals and used_hypervals == {hyperval}:
                continue
            blocked.add(node)
        return blocked

    def _find_piggyback_segments(
        self,
        source_node: MRRGNode,
        hyperval: HyperVal,
        lane_index: int,
    ) -> List[List[MRRGNode]]:
        """Return reusable packed prefixes starting at ``source_node``."""
        prefixes: List[List[MRRGNode]] = []
        seen: Set[Tuple[str, ...]] = set()
        packed_profile = TransportProfile.packed(
            total_bitwidth=None,
            lane_width=source_node.lane_width,
            lane_index=lane_index,
        )

        for route_key, bundle in self._routing_solution.items():
            routed_hyperval, _ = route_key
            if routed_hyperval == hyperval:
                continue
            if bundle.transport_profile.mode != TransportMode.PACKED:
                continue

            for sub_path in bundle.sub_paths:
                try:
                    start_idx = sub_path.index(source_node)
                except ValueError:
                    continue
                prefix: List[MRRGNode] = []
                for node in sub_path[start_idx:]:
                    if not self._route_uses_packed_lane(node, packed_profile):
                        break
                    state = self._routing_nodes.get(node)
                    if state is None or not self._lane_is_available(
                        node,
                        state,
                        hyperval,
                        lane_index,
                    ):
                        break
                    prefix.append(node)
                    prefix_key = tuple(prefix_node.id for prefix_node in prefix)
                    if prefix_key not in seen:
                        prefixes.append(list(prefix))
                        seen.add(prefix_key)

        return prefixes

    def _source_endpoint_has_existing_carrier(
        self,
        boundary_node: MRRGNode,
        speculative_lane_assignments: Optional[Dict[str, int]] = None,
    ) -> bool:
        """Return True when ``boundary_node`` already carries some packed traffic."""
        state = self._routing_nodes.get(boundary_node)
        if state is not None:
            for lane_users in state.packed_lane_usage.values():
                if lane_users:
                    return True
        return (
            speculative_lane_assignments is not None
            and boundary_node.id in speculative_lane_assignments
        )

    def _build_packed_candidate(
        self,
        hyperval: HyperVal,
        source_endpoint: PackingEndpoint,
        sink_endpoint: PackingEndpoint,
        token_bitwidth: int,
        cycles_to_sink: int,
        speculative_hv_nodes: Set[MRRGNode],
        latency_of_src: int,
        speculative_lane_assignments: Optional[Dict[str, int]] = None,
        speculative_full_width_nodes: Optional[Set[str]] = None,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        debug_reasons: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[RouteBundle, float]]:
        """Build packed route candidates for one explicit source/sink endpoint pair."""
        if source_endpoint.lane_width != sink_endpoint.lane_width:
            self._bump_debug_counter(debug_reasons, "reject_lane_width_mismatch")
            return []
        if source_endpoint.num_lanes != sink_endpoint.num_lanes:
            self._bump_debug_counter(debug_reasons, "reject_num_lanes_mismatch")
            return []
        if token_bitwidth > source_endpoint.lane_width:
            self._bump_debug_counter(debug_reasons, "reject_token_too_wide_for_lane")
            return []

        candidate_lane_indices = tuple(
            lane_idx
            for lane_idx in source_endpoint.allowed_lane_indices
            if lane_idx in sink_endpoint.allowed_lane_indices
        )
        if not candidate_lane_indices:
            self._bump_debug_counter(debug_reasons, "reject_no_common_lane")
            return []

        candidates: List[Tuple[RouteBundle, float]] = []
        source_boundary = list(source_endpoint.boundary_path)
        sink_boundary = list(sink_endpoint.boundary_path)
        blocked_lane_nodes = self._collect_occupied_lane_nodes(hyperval=hyperval)
        if speculative_full_width_nodes:
            for node_id in speculative_full_width_nodes:
                node = self._mrrg.get_node(node_id)
                if node is not None and node.num_lanes > 1:
                    blocked_lane_nodes.add(node)
        prefix_latency = sum(node.latency for node in source_boundary[:-1])
        suffix_latency = sum(node.latency for node in sink_boundary)
        start_cycles = latency_of_src + prefix_latency
        target_cycles = cycles_to_sink - suffix_latency
        if target_cycles < start_cycles:
            self._bump_debug_counter(debug_reasons, "reject_negative_packed_budget")
            return []

        for lane_index in candidate_lane_indices:
            packed_profile = TransportProfile.packed(
                total_bitwidth=token_bitwidth,
                lane_width=source_endpoint.lane_width,
                lane_index=lane_index,
            )

            piggyback_prefixes = self._find_piggyback_segments(
                source_endpoint.boundary_node,
                hyperval,
                lane_index,
            )
            if (
                source_endpoint.requires_existing_carrier
                and not piggyback_prefixes
                and self._source_endpoint_has_existing_carrier(
                    source_endpoint.boundary_node,
                    speculative_lane_assignments,
                )
            ):
                piggyback_prefixes = [[source_endpoint.boundary_node]]

            for prefix in piggyback_prefixes:
                branch_source = prefix[-1]
                blocked_prefix_nodes = blocked_lane_nodes - set(prefix)
                blocked_prefix_nodes.update(prefix[:-1])
                if branch_source == sink_endpoint.boundary_node:
                    carrier_path = prefix
                    full_path = source_boundary[:-1] + carrier_path + sink_boundary[1:]
                else:
                    branch_path = self._dijkstra_pathfinder(
                        source=branch_source,
                        sink=sink_endpoint.boundary_node,
                        tag_to_find=None,
                        hyperval=hyperval,
                        routes=speculative_hv_nodes,
                        cycles_to_sink=target_cycles,
                        latency_of_src=start_cycles + sum(
                            node.latency for node in prefix[:-1]
                        ),
                        transport_profile=packed_profile,
                        dest_op=None,
                        blocked_nodes=blocked_prefix_nodes,
                        speculative_lane_assignments=speculative_lane_assignments,
                        claimed_predecessors=claimed_predecessors,
                        debug_stats=debug_reasons,
                    )
                    if not branch_path:
                        self._bump_debug_counter(debug_reasons, "reject_packed_branch")
                        continue
                    carrier_path = prefix[:-1] + branch_path
                    full_path = source_boundary[:-1] + carrier_path + sink_boundary

                if self._route_conflicts_with_speculative_transport(
                    set(full_path),
                    packed_profile,
                    speculative_lane_assignments,
                    speculative_full_width_nodes,
                ):
                    self._bump_debug_counter(
                        debug_reasons,
                        "reject_speculative_transport_conflict",
                    )
                    continue
                total_latency = latency_of_src + sum(node.latency for node in full_path)
                if total_latency != cycles_to_sink:
                    self._bump_debug_counter(debug_reasons, "reject_packed_prefix_latency")
                    continue

                lane_assignments = {
                    node.id: lane_index
                    for node in full_path
                    if self._route_uses_packed_lane(node, packed_profile)
                }
                bundle = RouteBundle(
                    sub_paths=[full_path],
                    lane_assignments=lane_assignments,
                    active_pack_nodes={source_endpoint.boundary_node.id},
                    active_unpack_nodes={sink_endpoint.boundary_node.id},
                    transport_profile=packed_profile,
                )
                cost = self._score_candidate_bundle(bundle, hyperval, speculative_hv_nodes)
                candidates.append((bundle, cost))
                self._bump_debug_counter(debug_reasons, "packed_prefix_candidates")

            if (
                source_endpoint.requires_existing_carrier
                and not self._source_endpoint_has_existing_carrier(
                    source_endpoint.boundary_node,
                    speculative_lane_assignments,
                )
            ):
                self._bump_debug_counter(debug_reasons, "reject_missing_carrier")
                continue

            packed_path = self._dijkstra_pathfinder(
                source=source_endpoint.boundary_node,
                sink=sink_endpoint.boundary_node,
                tag_to_find=None,
                hyperval=hyperval,
                routes=speculative_hv_nodes,
                cycles_to_sink=target_cycles,
                latency_of_src=start_cycles,
                transport_profile=packed_profile,
                dest_op=None,
                blocked_nodes=blocked_lane_nodes,
                speculative_lane_assignments=speculative_lane_assignments,
                claimed_predecessors=claimed_predecessors,
                debug_stats=debug_reasons,
            )
            if not packed_path:
                self._bump_debug_counter(debug_reasons, "reject_packed_path")
                continue

            full_path = source_boundary[:-1] + packed_path + sink_boundary
            if self._route_conflicts_with_speculative_transport(
                set(full_path),
                packed_profile,
                speculative_lane_assignments,
                speculative_full_width_nodes,
            ):
                self._bump_debug_counter(
                    debug_reasons,
                    "reject_speculative_transport_conflict",
                )
                continue
            total_latency = latency_of_src + sum(node.latency for node in full_path)
            if total_latency != cycles_to_sink:
                self._bump_debug_counter(debug_reasons, "reject_packed_latency")
                continue

            lane_assignments = {
                node.id: lane_index
                for node in full_path
                if self._route_uses_packed_lane(node, packed_profile)
            }
            bundle = RouteBundle(
                sub_paths=[full_path],
                lane_assignments=lane_assignments,
                active_pack_nodes={source_endpoint.boundary_node.id},
                active_unpack_nodes={sink_endpoint.boundary_node.id},
                transport_profile=packed_profile,
            )
            cost = self._score_candidate_bundle(bundle, hyperval, speculative_hv_nodes)
            candidates.append((bundle, cost))
            self._bump_debug_counter(debug_reasons, "packed_candidates")

        return candidates

    def _build_packed_candidates(
        self,
        hyperval: HyperVal,
        src_fu: MRRGNode,
        src_operation: OperationType,
        sink_info: SinkAndLatency,
        token_bitwidth: Optional[int],
        cycles_to_sink: int,
        speculative_hv_nodes: Set[MRRGNode],
        latency_of_src: int,
        speculative_lane_assignments: Optional[Dict[str, int]] = None,
        speculative_full_width_nodes: Optional[Set[str]] = None,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        debug_reasons: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[RouteBundle, float]]:
        """Enumerate legal packed route candidates for a sink."""
        if token_bitwidth is None:
            self._bump_debug_counter(debug_reasons, "reject_unknown_token_bitwidth")
            return []

        output_candidates = self._get_packable_output_candidates(
            src_fu,
            src_operation,
            token_bitwidth,
        )
        input_candidates = self._get_packable_input_candidates(
            sink_info,
            token_bitwidth,
        )
        self._bump_debug_counter(debug_reasons, "pack_output_candidates", len(output_candidates))
        self._bump_debug_counter(debug_reasons, "pack_input_candidates", len(input_candidates))

        candidates: List[Tuple[RouteBundle, float]] = []
        for source_endpoint in output_candidates:
            for sink_endpoint in input_candidates:
                self._bump_debug_counter(debug_reasons, "pack_endpoint_pairs_considered")
                candidates.extend(
                    self._build_packed_candidate(
                        hyperval=hyperval,
                        source_endpoint=source_endpoint,
                        sink_endpoint=sink_endpoint,
                        token_bitwidth=token_bitwidth,
                        cycles_to_sink=cycles_to_sink,
                        speculative_hv_nodes=speculative_hv_nodes,
                        latency_of_src=latency_of_src,
                        speculative_lane_assignments=speculative_lane_assignments,
                        speculative_full_width_nodes=speculative_full_width_nodes,
                        claimed_predecessors=claimed_predecessors,
                        debug_reasons=debug_reasons,
                    )
                )
        return candidates

    def _map_operand_to_tag(self, operand_str: Optional[str]) -> Optional[OperandTag]:
        """Map operand string to OperandTag enum."""
        if operand_str is None:
            return None

        operand_map = {
            "": OperandTag.UNTAGGED,
            "LHS": OperandTag.BINARY_LHS,
            "RHS": OperandTag.BINARY_RHS,
            "any2input": OperandTag.BINARY_ANY,
            "any3input": OperandTag.TERNARY_ANY,
            "addr": OperandTag.ADDR,
            "data": OperandTag.DATA,
            "pred": OperandTag.PREDICATE,
            "branch_true": OperandTag.BR_TRUE,
            "branch_false": OperandTag.BR_FALSE,
            "binary_lhs": OperandTag.BINARY_LHS,
            "binary_rhs": OperandTag.BINARY_RHS,
            "ADDR": OperandTag.ADDR,
            "PRED": OperandTag.PREDICATE,
            "predicate": OperandTag.PREDICATE,
        }

        return operand_map.get(operand_str, None)

    def _compute_hyperval_priority(
        self,
        hyperval_info: HyperValNetInfo,
    ) -> None:
        """Recompute priority fields for a HyperValNetInfo in-place."""
        hyperval = hyperval_info.hyperval

        # Count uncovered sinks
        uncovered = 0
        for dest_idx in range(hyperval.cardinality):
            if (hyperval, dest_idx) not in self._routing_solution:
                uncovered += 1
        hyperval_info.uncovered_sinks = uncovered

        # Sum overuse across nodes used by this HyperVal's routed sinks
        conflict = 0.0
        seen_nodes: Set[MRRGNode] = set()
        for dest_idx in range(hyperval.cardinality):
            bundle = self._routing_solution.get((hyperval, dest_idx))
            if bundle is None:
                continue
            for node in bundle.all_nodes:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                if node in self._routing_nodes:
                    overuse = max(0, self._routing_nodes[node].occupancy - node.capacity)
                    conflict += overuse
        hyperval_info.conflict_score = conflict

    def route(self) -> RoutingResult:
        """Route the current DFG and return a structured result."""
        self._reset_routing_state()

        # Apply warm-start historical costs AFTER reset so they survive (B3).
        # Decay because previous placement was different — these are hints.
        if self._warm_start_historical_costs and self._warm_start_decay > 0:
            for mrrg_node, state in self._routing_nodes.items():
                warm_cost = self._warm_start_historical_costs.get(mrrg_node.id, 0.0)
                if warm_cost > 0:
                    state.historical_cost = warm_cost * self._warm_start_decay

        if self._deadline_exceeded():
            return self._build_timeout_result(
                message="Routing timed out before initial routing",
                hypervals_considered=0,
                destinations_considered=0,
                negotiation_iterations=0,
            )

        self._emit("\n" + "="*60)
        self._emit("PATHFINDER ROUTING")
        self._emit("="*60)

        self._emit("\n[PHASE 1] Initial routing of all HyperVals...")

        hypervals: List[HyperValNetInfo] = []

        for hyperval in self._hyperdfg.get_edges():
            src_hyper_node = self._hyperdfg.get_node(hyperval.source_id)
            if src_hyper_node:
                src_dfg_node = self._dfg.get_node(src_hyper_node.original_dfg_node_id)

                if src_dfg_node:
                    max_cycles = 0
                    for dest_idx in range(hyperval.cardinality):
                        dest_id = hyperval.destination_ids[dest_idx]
                        dest_hyper_node = self._hyperdfg.get_node(dest_id)

                        if dest_hyper_node:
                            dest_dfg_node = self._dfg.get_node(dest_hyper_node.original_dfg_node_id)

                            if dest_dfg_node:
                                edge_dist = hyperval.dists[dest_idx] if hyperval.dists[dest_idx] is not None else 0
                                cycles = self._compute_cycles_source_to_sink(src_dfg_node, dest_dfg_node, edge_dist)
                            max_cycles = max(max_cycles, cycles)

                    hypervals.append(HyperValNetInfo(
                        hyperval=hyperval,
                        max_cycles=max_cycles,
                        fanout=hyperval.cardinality
                    ))

        self._negotiated_router.route_initial_hypervals(hypervals)

        if self._debug:
            for hyperval in self._hyperdfg.get_edges():
                if any((hyperval, dest_idx) not in self._routing_solution for dest_idx in range(hyperval.cardinality)):
                    self._emit(f"[WARN] Initial routing failed for HyperVal {hyperval.source_id}")
                    summary = self.get_last_route_failure_summary()
                    if summary:
                        self._emit(f"       {summary}")

        # Compute priorities and sort
        for hv_info in hypervals:
            self._compute_hyperval_priority(hv_info)
        hypervals.sort()

        total_destinations = sum(hv_info.hyperval.cardinality for hv_info in hypervals)
        if self._deadline_exceeded():
            return self._build_timeout_result(
                message="Routing timed out after initial routing",
                hypervals_considered=len(hypervals),
                destinations_considered=total_destinations,
                negotiation_iterations=0,
            )
        self._emit(
            f"[INFO] Routed {len(hypervals)} HyperVals "
            f"({total_destinations} total destinations) in initial pass"
        )

        self._emit("\n[PHASE 2] Checking routing legality...")

        mapped = self._compute_dfg_coverage(verbose=self._debug) and self._check_overuse(
            verbose=self._debug
        )

        if mapped:
            valid, errors = self._run_shared_validation()
            if not valid:
                if self._debug:
                    self._emit("[VALIDATION FAILED]")
                    for err in errors:
                        self._emit(f"  - {err}")
                self._pre_ripup_congestion_map = self.get_congestion_map()
                for hyperval_info in hypervals:
                    self._rip_up_hyperval(hyperval_info.hyperval)
                self._p_factor = self._p_initial
                self._h_factor = self._h_initial
                mapped = False
                # Fall through to Phase 3 negotiation
            else:
                stats = self._collect_routing_stats(
                    hypervals_considered=len(hypervals),
                    destinations_considered=total_destinations,
                    negotiation_iterations=0,
                )
                self._emit("\n[SUCCESS] Initial routing is legal!")
                self._emit_routing_stats(stats)
                result = RoutingResult(
                    success=True,
                    routing_solution=self.get_routing_solution(),
                    stats=stats,
                    message="Initial routing is legal",
                )
                self._last_routing_result = result
                return result

        self._emit("\n[PHASE 3] Negotiated congestion resolution...")

        iteration = 0

        while not mapped and iteration < self._max_iterations:
            if self._deadline_exceeded():
                return self._build_timeout_result(
                    message="Routing timed out during negotiated routing",
                    hypervals_considered=len(hypervals),
                    destinations_considered=total_destinations,
                    negotiation_iterations=iteration,
                )
            iteration += 1
            self._emit(f"\n--- Iteration {iteration}/{self._max_iterations} ---")

            # Update historical costs once per iteration (node-based, order-independent)
            self._update_historical_costs()

            # Recompute priority for all HyperVals
            for hv_info in hypervals:
                self._compute_hyperval_priority(hv_info)
            hypervals.sort()

            self._negotiated_router.reroute_hypervals(hypervals)
            if self._debug:
                summary = self.get_last_route_failure_summary()
                if summary:
                    self._emit(f"  [REROUTE FAIL] {summary}")
            if self._deadline_exceeded():
                return self._build_timeout_result(
                    message="Routing timed out during negotiated reroute",
                    hypervals_considered=len(hypervals),
                    destinations_considered=total_destinations,
                    negotiation_iterations=iteration,
                )

            # Check coverage and overuse after the full pass
            opgraph_covered = self._compute_dfg_coverage(verbose=self._debug)
            mrrg_legal = self._check_overuse(verbose=self._debug)
            mapped = opgraph_covered and mrrg_legal

            if mapped:
                self._emit(f"\n[SUCCESS] Legal routing achieved in iteration {iteration}!")
                self._p_factor = self._p_initial
                self._h_factor = self._h_initial
                break

            # Increase congestion penalties for next iteration
            self._p_factor = self._p_factor * self._p_growth_rate
            self._h_factor = self._h_factor * self._h_growth_rate

        self._emit("\n" + "="*60)
        coverage = self._compute_dfg_coverage(verbose=self._debug)
        legality = self._check_overuse(verbose=self._debug)
        self._emit(f"ROUTING COMPLETE: covered={coverage}, legal={legality}")
        self._emit("="*60)

        if coverage and legality:
            # Validate with shared validator
            valid, errors = self._run_shared_validation()
            if not valid:
                if self._debug:
                    self._emit("[VALIDATION FAILED]")
                    for err in errors:
                        self._emit(f"  - {err}")
                stats = self._collect_routing_stats(
                    hypervals_considered=len(hypervals),
                    destinations_considered=total_destinations,
                    negotiation_iterations=iteration,
                )
                # Snapshot congestion before rip-up
                self._pre_ripup_congestion_map = self.get_congestion_map()
                # Rip up all routes and fail
                for hyperval_info in hypervals:
                    self._rip_up_hyperval(hyperval_info.hyperval)
                self._p_factor = self._p_initial
                self._h_factor = self._h_initial
                result = RoutingResult(
                    success=False,
                    validation_errors=errors,
                    failure_summary=self.get_last_route_failure_summary(),
                    message="Routing validation failed after negotiation",
                    stats=stats,
                )
                self._last_routing_result = result
                return result

            stats = self._collect_routing_stats(
                hypervals_considered=len(hypervals),
                destinations_considered=total_destinations,
                negotiation_iterations=iteration,
            )
            self._emit("\n[SUCCESS] Routing succeeded!")
            self._emit_routing_stats(stats)
            result = RoutingResult(
                success=True,
                routing_solution=self.get_routing_solution(),
                stats=stats,
                message="Routing succeeded",
            )
            self._last_routing_result = result
            return result

        stats = self._collect_routing_stats(
            hypervals_considered=len(hypervals),
            destinations_considered=total_destinations,
            negotiation_iterations=iteration,
        )
        self._emit(f"\n[PARTIAL] Routing incomplete after {iteration} iterations")
        self._emit("[INFO] Returning partial solution with warnings")
        self._emit_routing_stats(stats)

        # Snapshot current-overuse congestion BEFORE rip-up zeroes occupancy.
        self._pre_ripup_congestion_map = self.get_congestion_map()

        # Rip up all routes
        for hyperval_info in hypervals:
            self._rip_up_hyperval(hyperval_info.hyperval)

        self._p_factor = self._p_initial
        self._h_factor = self._h_initial

        result = RoutingResult(
            success=False,
            failure_summary=self.get_last_route_failure_summary(),
            message=f"Routing incomplete after {iteration} iterations",
            stats=stats,
        )
        self._last_routing_result = result
        return result

    def route_dfg(self) -> bool:
        """Backward-compatible bool-only routing entrypoint."""
        return self.route().success

    def _run_shared_validation(self) -> Tuple[bool, List[str]]:
        """Run the shared validation module against the current routing solution.

        Returns:
            (is_valid, errors) tuple
        """
        from cgrasolve.algorithms.common.routing_validation import validate_routing_solution
        return validate_routing_solution(
            dfg=self._dfg,
            mrrg=self._mrrg,
            placement=self._placement,
            routing_solution=self._routing_solution,
            debug=self._debug,
        )
