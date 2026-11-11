"""Anneal Placer for Heuristic Mapping."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import random
import math
import time
from collections import Counter, defaultdict, deque

from cgrasolve.graph.dfg import DFG, DFGNode, OperationType, DFGEdge
from cgrasolve.graph.hyperdfg import HyperDFG, HyperNode, HyperVal
from cgrasolve.schedules import LatencySpecification
from cgrasolve.graph.mrrg import MRRG, MRRGNode, MRRGEdge, NodeType, HWEntityType, OperandTag
from cgrasolve.algorithms.placer.types import MRRGNodePlacementState
from cgrasolve.algorithms.archive.models.cost_models import WirelengthLatencyCostModel
from cgrasolve.algorithms.scheduler.types import get_node_schedule_time


@dataclass
class _MoveDeltaResult:
    """Result of an incremental cost delta computation."""
    delta: float
    new_hv_costs: Dict[str, Tuple[float, float]]  # hv_id -> (wire, latency)
    new_precision: Dict[str, float]                 # dfg_node.id -> penalty
    new_congestion: Dict[str, float]                # dfg_node.id -> penalty
    new_mrc: Dict[str, float]                       # dfg_node.id -> mrc penalty
    new_peu: Dict[str, float]                       # dfg_node.id -> peu penalty


class AnnealPlacer:
    """Anneal Placer for Heuristic Mapping."""

    # Placer properties
    _dfg: DFG
    _latency_spec: LatencySpecification
    _mrrg: MRRG
    _ii: int
    _num_rows: int
    _num_cols: int

    # Mapping from MRRG node to its placement state
    _fu_node_placement_state: Dict[MRRGNode, MRRGNodePlacementState]

    # Mapping from DFG node to corresponding FUs
    _op_node_placement_map: Dict[DFGNode, MRRGNode]

    # Mapping from (OperationType, cycle, required_output_bw, required_input_bw)
    # to MRRG nodes that can execute the operation with compatible bitwidths.
    _temporal_op_type_to_fu_nodes: Dict[Tuple[OperationType, int, Optional[int], Optional[int]], List[MRRGNode]]

    # User reserved MRRG nodes
    _reserved_fu_nodes: Set[MRRGNode]

    # User reserved map from DFG node id to MRRG node id
    _reserved_dfg_node_id_to_fu_node_id: Dict[str, str]

    # Xbar channel reachability: how many independent xbar channels can reach each FU
    _fu_xbar_channel_counts: Dict[MRRGNode, int]

    # Per-node override candidate list for ops with >=2 fixed (iopad-bound) inputs
    _node_constrained_fus: Dict[str, List[MRRGNode]]

    # Random seed
    _random_seed: int

    # Swap factor for inner loop
    _swap_factor: int

    def __init__(self, num_rows: int, num_cols: int, dfg: DFG, latency_spec: LatencySpecification, mrrg: MRRG, fixed_placement: List[Tuple[str, str]], random_seed: int = 42, swap_factor: int = 10, target_ii: Optional[int] = None, precision_penalty_weight: float = 0.1, wire_weight: float = 0.5, latency_weight: float = 0.5, mrc_weight: float = 0.0, peu_weight: float = 0.0, normalize_costs: bool = True) -> None:
        """Initialize the Anneal Placer.

        Args:
            num_rows: Number of rows in the CGRA
            num_cols: Number of columns in the CGRA
            dfg: Data Flow Graph
            latency_spec: Latency specification
            mrrg: MRRG
            fixed_placement: List of (dfg_node_id, mrrg_node_id) tuples for fixed placements
            random_seed: Random seed for reproducibility
            swap_factor: Number of swap attempts = num_operations * swap_factor
            target_ii: Target initiation interval (defaults to mrrg.II)
            precision_penalty_weight: Weight for precision mismatch penalty
            wire_weight: Weight for wirelength cost component
            latency_weight: Weight for latency cost component
            mrc_weight: Weight for memory resource cost component
            peu_weight: Weight for PE utilization pressure component
            normalize_costs: Whether to normalize cost components
        """
        self._dfg = dfg
        self._latency_spec = latency_spec
        self._mrrg = mrrg
        self._ii = target_ii if target_ii is not None else mrrg.II
        self._num_rows = num_rows
        self._num_cols = num_cols

        # Shwet TODO: Do we want to generate a random seed here or make this user driven?
        self._random_seed = random_seed
        self._swap_factor = swap_factor
        self._rng = random.Random(self._random_seed)
        self._acceptance_rate = 0.0
        self._timed_out = False

        # Initialize state dictionaries
        self._fu_node_placement_state = {}
        self._op_node_placement_map = {}
        self._temporal_op_type_to_fu_nodes = {}
        self._reserved_fu_nodes = set()
        self._reserved_dfg_node_id_to_fu_node_id = {}

        # --- Incremental cost cache (B1) ---
        self._inc_hyperdfg: Optional[HyperDFG] = None
        self._inc_schedule: Dict[str, int] = {}
        self._per_hv_costs: Dict[str, Tuple[float, float]] = {}
        self._per_op_precision: Dict[str, float] = {}
        self._per_op_congestion: Dict[str, float] = {}
        self._total_cost_cached: float = 0.0
        self._cost_cache_valid: bool = False
        self._node_to_hv_ids: Dict[str, List[str]] = {}
        self._id_to_dfg_node: Dict[str, DFGNode] = {n.id: n for n in dfg.get_nodes()}

        # Routing congestion feedback (B2) — injected between outer iterations
        self._congestion_map: Dict[Tuple[int, int], float] = {}
        self._congestion_weight: float = 0.0

        # Tunable cost weights
        self._wire_weight = wire_weight
        self._latency_weight = latency_weight
        self._mrc_weight = mrc_weight
        self._peu_weight = peu_weight
        self._precision_penalty_weight = precision_penalty_weight
        self._normalize_costs = normalize_costs

        # MRC: Memory Resource Cost
        self._memory_fu_ids: Set[str] = {
            fu.id for fu in mrrg.get_fu_nodes()
            if fu.can_execute(OperationType.LOAD) or fu.can_execute(OperationType.STORE)
        }
        self._memory_dfg_ids: Set[str] = {
            n.id for n in dfg.get_nodes()
            if n.operation in (OperationType.LOAD, OperationType.STORE)
        }
        self._placed_memory_ops: int = 0
        self._memory_fus_occupied: int = 0
        self._per_op_mrc: Dict[str, float] = {}

        # Multi-cycle FU occupancy: base_name -> Counter of occupied cycle counts.
        # When an op is placed on a FU with latency > 1 at cycle C, cycles
        # C through C+latency-1 (mod II) are incremented.  A count > 1 at
        # any cycle indicates a multi-cycle conflict (e.g., back-to-back loads
        # on the same LSU port).
        self._multi_cycle_occ: Dict[str, Counter] = defaultdict(Counter)

        # PEU: PE Utilization Pressure
        self._coord_occupancy: defaultdict = defaultdict(int)
        self._per_op_peu: Dict[str, float] = {}

        # Normalization factors (fixed at first _init_incremental_cost call)
        self._norms_initialized: bool = False
        self._wire_norm: float = 1.0
        self._latency_norm: float = 1.0
        self._mrc_norm: float = 1.0
        self._peu_norm: float = 1.0

        # Compute the node classes and initialize FU node states
        classes = self.__compute_fu_node_classes()
        for fu_node_v in classes:
            for fu_node in fu_node_v:
                self._fu_node_placement_state[fu_node] = MRRGNodePlacementState(occupancy=0)

        # Initialize fixed placements (must run after _fu_node_placement_state is populated)
        for dfg_node_id, mrrg_node_id in fixed_placement:
            fu_node = mrrg.get_node(mrrg_node_id)
            if fu_node is None or fu_node not in self._fu_node_placement_state:
                raise ValueError(f"MRRG node {mrrg_node_id} not found in the MRRG")
            self._reserved_fu_nodes.add(fu_node)
            self._reserved_dfg_node_id_to_fu_node_id[dfg_node_id] = fu_node.id

        # Build Operation Type to FU nodes map per cycle, keyed by
        # (operation, cycle, req_out_bw, req_in_bw) for two-sided bitwidth filtering.
        for op in dfg.get_nodes():
            req_out, req_in = self._get_required_bws(op)
            for i in range(self._ii):
                key = (op.operation, i, req_out, req_in)
                if key in self._temporal_op_type_to_fu_nodes:
                    continue

                # Collect FU nodes that can execute op at cycle i with compatible bitwidths
                fu_nodes: List[MRRGNode] = []
                for mrrg_node in classes[i]:
                    if mrrg_node.can_execute(op.operation):
                        if not self._fu_bw_compatible(mrrg_node, op.operation, req_out, req_in):
                            continue
                        fu_nodes.append(mrrg_node)

                self._temporal_op_type_to_fu_nodes[key] = fu_nodes

        # Compute xbar channel min-latency reachability for both directions,
        # then build per-op filtered candidates for ops adjacent to fixed iopads.
        self._fu_xbar_channel_counts = self._compute_fu_xbar_channel_min_latency()
        self._fu_to_xbar_out_min_latency = self._compute_fu_to_xbar_out_min_latency()
        self._fu_to_fu_min_cycles = self._compute_fu_to_fu_min_cycles()
        self._node_constrained_fus: Dict[str, List[MRRGNode]] = {}
        for op in dfg.get_nodes():
            fixed_in_edges = [
                e for e in dfg.get_incoming_edges(op.id)
                if e.source.id in self._reserved_dfg_node_id_to_fu_node_id
            ]
            fixed_out_edges = [
                e for e in dfg.get_outgoing_edges(op.id)
                if e.destination.id in self._reserved_dfg_node_id_to_fu_node_id
            ]
            fixed_input_count = len(fixed_in_edges)
            fixed_output_count = len(fixed_out_edges)
            if fixed_input_count + fixed_output_count == 0:
                continue
            req_out, req_in = self._get_required_bws(op)
            op_time = get_node_schedule_time(op)
            if op_time is None:
                continue
            key = (op.operation, op_time % self._ii, req_out, req_in)
            all_candidates = self._temporal_op_type_to_fu_nodes.get(key, [])

            # Input-side budget: max budget across all fixed input sources
            in_budget = None
            if fixed_input_count > 0:
                src_times = [get_node_schedule_time(dfg.get_node(e.source.id)) or 0
                             for e in fixed_in_edges]
                in_budget = op_time - max(src_times)
                if in_budget <= 0:
                    in_budget = self._ii

            # Output-side budget: min cycles allowed from op to fixed sink
            out_budget = None
            if fixed_output_count > 0:
                snk_times = [get_node_schedule_time(dfg.get_node(e.destination.id)) or 0
                             for e in fixed_out_edges]
                out_budget = min(snk_times) - op_time
                if out_budget <= 0:
                    out_budget = self._ii

            filtered: List[MRRGNode] = []
            for fu in all_candidates:
                if fixed_input_count >= 2 and in_budget is not None:
                    in_lats = self._fu_xbar_channel_counts.get(fu, {})
                    n_reachable = sum(1 for lat in in_lats.values() if lat <= in_budget)
                    if n_reachable < fixed_input_count:
                        continue
                if fixed_output_count >= 1 and out_budget is not None:
                    out_lat = self._fu_to_xbar_out_min_latency.get(fu)
                    if out_lat is None or out_lat > out_budget:
                        continue
                filtered.append(fu)

            # Only record a constraint if we actually filtered something.
            if (fixed_input_count >= 2 and in_budget is not None) or \
               (fixed_output_count >= 1 and out_budget is not None):
                self._node_constrained_fus[op.id] = filtered

    def _compute_fu_to_xbar_out_min_latency(self) -> Dict[MRRGNode, int]:
        """For each FU, the min routing latency to ANY data-out xbar input port.

        Used to filter FU candidates for ops whose output edges target a fixed
        iopad sink. The exact xbar channel doesn't matter (any boundary path
        suffices), so we collapse to a single per-FU min latency.
        """
        # Build forward adjacency restricted to routing nodes (FU nodes are sources only).
        sinks = {
            n.id for n in self._mrrg.get_nodes()
            if "xbar" in n.id and "data_out" in n.id and ".data_i_" in n.id
        }
        if not sinks:
            return {}

        # Reverse Dijkstra from sinks to FUs.
        import heapq as _hq
        dist: Dict[str, int] = {sid: 0 for sid in sinks}
        heap: List[Tuple[int, str]] = [(0, sid) for sid in sinks]
        _hq.heapify(heap)

        # Pre-build reverse adjacency
        rev_adj: Dict[str, List[Tuple[str, int]]] = {}
        for edge in self._mrrg.get_edges():
            src_id = edge.source.id
            dst_id = edge.destination.id
            dst_node = edge.destination
            # Latency accumulated when traversing to dst (i.e., on the dst node)
            rev_adj.setdefault(dst_id, []).append((src_id, dst_node.latency))

        result: Dict[MRRGNode, int] = {}
        while heap:
            cyc, nid = _hq.heappop(heap)
            if cyc > dist.get(nid, cyc + 1):
                continue
            node = self._mrrg.get_node(nid)
            if node is None:
                continue
            if node.node_type == NodeType.FUNCTION:
                # Terminal — record min. Mux-class FUs (HW_MUX) are also
                # transit-capable, so continue traversing through them.
                if node not in result or cyc < result[node]:
                    result[node] = cyc
                if node.hw_entity_type != HWEntityType.HW_MUX:
                    continue
            for src_id, lat in rev_adj.get(nid, []):
                # When stepping backward across an edge, the dst node's latency
                # is what gets added (the register sits between src→dst).
                next_cyc = cyc + lat
                if next_cyc < dist.get(src_id, next_cyc + 1):
                    dist[src_id] = next_cyc
                    _hq.heappush(heap, (next_cyc, src_id))
        return result

    def _compute_fu_to_fu_min_cycles(self) -> Dict[MRRGNode, Dict[MRRGNode, int]]:
        """Min routing cycles from each FU's outputs to each other FU.

        Used by the placer's latency cost so it drives placements where
        actual MRRG reachability matches the schedule's c2s budget. Pure
        Manhattan distance is misleading on architectures (like mini_dice)
        where stride between FU coordinates is greater than 1.
        """
        import heapq as _hq
        result: Dict[MRRGNode, Dict[MRRGNode, int]] = {}
        for src_fu in self._fu_node_placement_state:
            dist: Dict[str, int] = {}
            heap: List[Tuple[int, str]] = []
            # Seed with src_fu's outgoing routing nodes
            for edge in self._mrrg.get_outgoing_edges(src_fu.id):
                dest = edge.destination
                cyc = dest.latency
                if dest.id not in dist or cyc < dist[dest.id]:
                    dist[dest.id] = cyc
                    _hq.heappush(heap, (cyc, dest.id))
            while heap:
                cyc, nid = _hq.heappop(heap)
                if cyc > dist.get(nid, cyc + 1):
                    continue
                node = self._mrrg.get_node(nid)
                if node is None:
                    continue
                if node.node_type == NodeType.FUNCTION and \
                        node.hw_entity_type != HWEntityType.HW_MUX:
                    # Don't traverse through compute FUs (terminal as sinks);
                    # mux-class FUs are transit-capable.
                    continue
                for edge in self._mrrg.get_outgoing_edges(nid):
                    dest = edge.destination
                    nc = cyc + dest.latency
                    if nc < dist.get(dest.id, nc + 1):
                        dist[dest.id] = nc
                        _hq.heappush(heap, (nc, dest.id))
            # For each other FU, find the min cycles to reach it via incoming routing nodes.
            per_fu: Dict[MRRGNode, int] = {}
            for tgt_fu in self._fu_node_placement_state:
                if tgt_fu == src_fu:
                    continue
                best = None
                for edge in self._mrrg.get_incoming_edges(tgt_fu.id):
                    src_route = edge.source
                    d = dist.get(src_route.id)
                    if d is None:
                        continue
                    if best is None or d < best:
                        best = d
                if best is not None:
                    per_fu[tgt_fu] = best
            result[src_fu] = per_fu
        return result

    def _compute_fu_xbar_channel_min_latency(self) -> Dict[MRRGNode, Dict[int, int]]:
        """For each FU, map xbar channel index → minimum routing latency (cycles) to reach it.

        Uses Dijkstra (min-latency) from each xbar data-in output node so that
        the caller can filter FU candidates by routing budget (cycles_to_sink).
        FU nodes are terminal — the path does not traverse through them.
        """
        xbar_out_nodes = [
            n for n in self._mrrg.get_nodes()
            if "xbar" in n.id and "data_in" in n.id and ".data_o_" in n.id
        ]
        # fu -> {channel_index: min_cycles}
        result: Dict[MRRGNode, Dict[int, int]] = {}

        for ch_idx, xbar_out in enumerate(xbar_out_nodes):
            # Dijkstra: min accumulated latency to each routing/FU node
            dist: Dict[str, int] = {xbar_out.id: 0}
            heap: List[Tuple[int, str]] = [(0, xbar_out.id)]
            import heapq as _hq
            while heap:
                cyc, nid = _hq.heappop(heap)
                if cyc > dist.get(nid, cyc + 1):
                    continue
                node = self._mrrg.get_node(nid)
                if node is None:
                    continue
                if node.node_type == NodeType.FUNCTION:
                    # Record but only stop for compute FUs; mux-class FUs are
                    # transit-capable, so keep exploring past them.
                    if node not in result:
                        result[node] = {}
                    if ch_idx not in result[node] or cyc < result[node][ch_idx]:
                        result[node][ch_idx] = cyc
                    if node.hw_entity_type != HWEntityType.HW_MUX:
                        continue
                next_cyc = cyc + node.latency
                for edge in self._mrrg.get_outgoing_edges(nid):
                    dest = edge.destination
                    if next_cyc < dist.get(dest.id, next_cyc + 1):
                        dist[dest.id] = next_cyc
                        _hq.heappush(heap, (next_cyc, dest.id))
        return result

    def _get_required_bws(self, op: DFGNode) -> Tuple[Optional[int], Optional[int]]:
        """Return (required_output_bw, required_input_bw) for a DFG node.

        required_output_bw: from node bitwidth attribute (width of produced data).
        required_input_bw: from max incoming edge bitwidth (width of consumed data).
        """
        # Output requirement: node bitwidth attribute
        node_bw = op.get_attribute("bitwidth")
        req_out = node_bw if isinstance(node_bw, int) and node_bw > 0 else None

        # Input requirement: max bitwidth across incoming edges
        incoming = self._dfg.get_incoming_edges(op.id)
        in_bws = [
            e.get_attribute("bitwidth") for e in incoming
            if isinstance(e.get_attribute("bitwidth"), int)
            and e.get_attribute("bitwidth") > 0
        ]
        req_in = max(in_bws) if in_bws else None

        return req_out, req_in

    @staticmethod
    def _fu_bw_compatible(
        fu: MRRGNode,
        op_type: OperationType,
        req_out: Optional[int],
        req_in: Optional[int],
    ) -> bool:
        """Check whether an FU satisfies both output and input bitwidth requirements.

        Each side is checked independently.  A check is skipped when either the
        FU has no metadata for that side or the requirement is None.
        """
        # Output side: FU must produce at least req_out bits
        if req_out is not None:
            fu_out = fu.operation_output_bitwidths.get(op_type)
            if fu_out is not None and fu_out < req_out:
                return False

        # Input side: FU must accept at least req_in bits
        if req_in is not None:
            fu_in = fu.operation_max_input_bitwidth.get(op_type)
            if fu_in is not None and fu_in < req_in:
                return False

        return True

    def anneal(self, initial_temperature: float, deadline: Optional[float] = None) -> None:
        """Anneal the placement."""
        temperature: float = initial_temperature
        self._timed_out = False

        while True:
            if deadline is not None and time.time() >= deadline:
                self._timed_out = True
                break

            # Inner loop: perform swaps at current temperature
            accept_rate: float = self.inner_loop(temperature, deadline=deadline)
            self._acceptance_rate = accept_rate
            if self._timed_out:
                break
            
            # Check termination
            current_cost: float = self.get_total_cost()
            if current_cost == 0:
                current_cost = 1

            threshold: float = 0.05 * (current_cost / len(self._dfg.get_nodes()))
            if temperature < threshold:
                break

            temperature = self.next_temperature(temperature, accept_rate)

    def _has_multi_cycle_conflict_at(self, fu_node: MRRGNode) -> bool:
        """Check if placing an op at *fu_node* would overlap an existing multi-cycle FU."""
        if fu_node.latency <= 1:
            return False
        base = fu_node.get_base_name()
        counts = self._multi_cycle_occ.get(base)
        if counts is None:
            return False
        for c in range(fu_node.latency):
            if counts[(fu_node.cycle + c) % self._ii] > 0:
                return True
        return False

    def update_op_placement(self, op: DFGNode, fu_node: MRRGNode) -> None:
        """Update the placement state and map for the given operation and FU node."""
        self._fu_node_placement_state[fu_node].occupancy += 1
        self._op_node_placement_map[op] = fu_node
        if fu_node.coordinates:
            self._coord_occupancy[fu_node.coordinates] += 1
        if op.operation in (OperationType.LOAD, OperationType.STORE):
            self._placed_memory_ops += 1
        if fu_node.id in self._memory_fu_ids:
            self._memory_fus_occupied += 1
        if fu_node.latency > 1:
            base = fu_node.get_base_name()
            for c in range(fu_node.latency):
                self._multi_cycle_occ[base][(fu_node.cycle + c) % self._ii] += 1

    def clear_op_placement(self, op: DFGNode) -> None:
        """Clear the placement state and map for the given operation."""
        fu_node = self._op_node_placement_map[op]
        self._fu_node_placement_state[fu_node].occupancy -= 1
        if fu_node.coordinates:
            self._coord_occupancy[fu_node.coordinates] -= 1
        if op.operation in (OperationType.LOAD, OperationType.STORE):
            self._placed_memory_ops -= 1
        if fu_node.id in self._memory_fu_ids:
            self._memory_fus_occupied -= 1
        if fu_node.latency > 1:
            base = fu_node.get_base_name()
            for c in range(fu_node.latency):
                self._multi_cycle_occ[base][(fu_node.cycle + c) % self._ii] -= 1
        self._op_node_placement_map.pop(op)

    def set_initial_placement(self) -> None:
        """Set the initial placement for all the DFG nodes."""
        for op in self._dfg.get_nodes():
            if op.id in self._reserved_dfg_node_id_to_fu_node_id:
                continue
            self.update_op_placement(op, self.get_random_unoccupied_fu(op))

    def get_random_fu(self, op: DFGNode) -> MRRGNode:
        """Get a random FU node that can execute the given operation, which maybe occupied or not."""

        op_time = get_node_schedule_time(op)
        if op_time is None:
            raise ValueError(f"Operation {op.id} is not scheduled")

        # Get candidate FU nodes that can execute the operation at the given cycle.
        # Use the xbar-channel-constrained list for ops with >=2 fixed iopad inputs.
        req_out, req_in = self._get_required_bws(op)
        key = (op.operation, op_time % self._ii, req_out, req_in)
        constrained = self._node_constrained_fus.get(op.id)
        candidate_fu_nodes: List[MRRGNode] = (
            constrained if constrained is not None
            else self._temporal_op_type_to_fu_nodes[key]
        )

        # Check if there are any candidate FU nodes
        if len(candidate_fu_nodes) < 1:
            raise ValueError(
                f"No FU nodes found that can execute the operation {op.operation} "
                f"at cycle {op_time % self._ii} with output_bw={req_out}, input_bw={req_in}"
            )

        # Grab a random FU node from the candidate FU nodes which is not reserved by the user.
        random_fu_node: MRRGNode

        # Set a max number of iterations to avoid infinite loop
        max_iterations: int = 100000
        iterations: int = 0
        while True:
            # Generate a random unsigned integer index in the range length of candidate_fu_nodes
            random_index: int = self._rng.randint(0, len(candidate_fu_nodes) - 1)
            random_fu_node = candidate_fu_nodes[random_index]

            # Check if the FU node is reserved by the user
            if random_fu_node not in self._reserved_fu_nodes:
                break

            # Increment the iterations
            iterations += 1

            # Check if the maximum number of iterations is reached
            if iterations >= max_iterations:
                raise ValueError(
                    "Maximum number of iterations reached while trying to find "
                    f"a random FU node that is not reserved by the user for "
                    f"operation {op.operation} at cycle {op_time % self._ii}"
                )

        return random_fu_node

    def get_random_unoccupied_fu(self, op: DFGNode) -> MRRGNode:
        """Get a random FU node that is not occupied."""

        op_time = get_node_schedule_time(op)
        if op_time is None:
            raise ValueError(f"Operation {op.id} is not scheduled")

        cycle: int = op_time % self._ii
        req_out, req_in = self._get_required_bws(op)
        key = (op.operation, cycle, req_out, req_in)
        constrained = self._node_constrained_fus.get(op.id)
        candidate_fu_nodes: List[MRRGNode] = (
            constrained if constrained is not None
            else self._temporal_op_type_to_fu_nodes[key]
        )

        # Get all the unoccupied FU nodes
        unoccupied_fu_nodes: List[MRRGNode] = []

        for fu_node in candidate_fu_nodes:
            if self._fu_node_placement_state[fu_node].occupancy >= fu_node.capacity:
                continue
            if self._has_multi_cycle_conflict_at(fu_node):
                continue
            unoccupied_fu_nodes.append(fu_node)

        if len(unoccupied_fu_nodes) < 1:
            raise ValueError(
                f"No unoccupied FU nodes found that can execute the operation "
                f"{op.operation} at cycle {op_time % self._ii}"
            )

        # Grab a random FU node from the unoccupied FU nodes
        random_fu_node: MRRGNode
        random_index: int = self._rng.randint(0, len(unoccupied_fu_nodes) - 1)
        random_fu_node = unoccupied_fu_nodes[random_index]

        return random_fu_node

    def clear_placement(self) -> None:
        """Clear the placement state for all the FU nodes."""
        for fu_node in self._fu_node_placement_state:
            self._fu_node_placement_state[fu_node].occupancy = 0
        self._op_node_placement_map.clear()
        self._coord_occupancy.clear()
        self._multi_cycle_occ.clear()
        self._placed_memory_ops = 0
        self._memory_fus_occupied = 0
        self._cost_cache_valid = False

    # Helper function to compute the FU node classes
    def __compute_fu_node_classes(self) -> List[List[MRRGNode]]:

        node_classes: List[List[MRRGNode]] = []

        # Compute the node classes
        for i in range(self._ii):
            node_classes.append([])

        for node in self._mrrg.get_fu_nodes():
            if node.cycle < self._ii:
                node_classes[node.cycle].append(node)

        return node_classes
        
    # ---- Incremental cost system (B1) ------------------------------------------

    def get_total_cost(self) -> float:
        """Return total placement cost. Builds cache on first call, O(1) thereafter."""
        if not self._cost_cache_valid:
            self._init_incremental_cost()
        return self._total_cost_cached

    def _get_total_cost_from_scratch(self) -> float:
        """Recompute total cost from scratch (used for debug validation only)."""
        hyperdfg = HyperDFG.from_dfg(self._dfg)
        total_wire = 0.0
        total_latency = 0.0
        for hv in hyperdfg.get_edges():
            w, l = self._compute_hyperval_cost(hv)
            total_wire += w
            total_latency += l
        total_precision = self._compute_precision_penalty_from_scratch()
        total_congestion = sum(self._congestion_at(mn) for mn in self._op_node_placement_map.values())
        total_mrc = sum(self._compute_single_mrc(dn, mn) for dn, mn in self._op_node_placement_map.items())
        total_peu = sum(self._compute_single_peu(dn, mn) for dn, mn in self._op_node_placement_map.items())
        return (
            self._wire_weight * total_wire / self._wire_norm
            + self._latency_weight * total_latency / self._latency_norm
            + total_precision + total_congestion
            + self._mrc_weight * total_mrc / self._mrc_norm
            + self._peu_weight * total_peu / self._peu_norm
        )

    def _init_incremental_cost(self) -> None:
        """Build cached cost structures for O(degree) incremental updates."""
        # Build HyperDFG once (DFG topology is static during placement)
        self._inc_hyperdfg = HyperDFG.from_dfg(self._dfg)

        # Build schedule from current DFG node times (rebuilt on cache invalidation; may change if schedule perturbation shifts an operation).
        self._inc_schedule = {
            n.id: get_node_schedule_time(n) or 0 for n in self._dfg.get_nodes()
        }

        # Build node-to-hyperval index (deduplicated)
        self._node_to_hv_ids = {}
        for hv in self._inc_hyperdfg.get_edges():
            touched_nodes = {hv.source_id} | set(hv.destination_ids)
            for nid in touched_nodes:
                self._node_to_hv_ids.setdefault(nid, []).append(hv.id)

        # Compute initial per-hyperval costs
        self._per_hv_costs = {}
        for hv in self._inc_hyperdfg.get_edges():
            self._per_hv_costs[hv.id] = self._compute_hyperval_cost(hv)

        # Compute initial per-op precision penalties
        self._per_op_precision = {}
        for dfg_node, mrrg_node in self._op_node_placement_map.items():
            self._per_op_precision[dfg_node.id] = self._compute_single_precision_penalty(
                dfg_node, mrrg_node
            )

        # Compute initial per-op congestion penalties
        self._per_op_congestion = {}
        for dfg_node, mrrg_node in self._op_node_placement_map.items():
            self._per_op_congestion[dfg_node.id] = self._congestion_at(mrrg_node)

        self._per_op_mrc = {}
        for dfg_node, mrrg_node in self._op_node_placement_map.items():
            self._per_op_mrc[dfg_node.id] = self._compute_single_mrc(dfg_node, mrrg_node)

        self._per_op_peu = {}
        for dfg_node, mrrg_node in self._op_node_placement_map.items():
            self._per_op_peu[dfg_node.id] = self._compute_single_peu(dfg_node, mrrg_node)

        # Sum it all up
        total_wire = sum(wl for wl, _ in self._per_hv_costs.values())
        total_latency = sum(lt for _, lt in self._per_hv_costs.values())
        total_precision = sum(self._per_op_precision.values())
        total_congestion = sum(self._per_op_congestion.values())
        total_mrc = sum(self._per_op_mrc.values())
        total_peu = sum(self._per_op_peu.values())

        if self._normalize_costs and not self._norms_initialized:
            self._wire_norm = max(total_wire, 1.0)
            self._latency_norm = max(total_latency, 1.0)
            self._mrc_norm = max(total_mrc, 1.0)
            self._peu_norm = max(total_peu, 1.0)
            self._norms_initialized = True

        self._total_cost_cached = (
            self._wire_weight * total_wire / self._wire_norm
            + self._latency_weight * total_latency / self._latency_norm
            + total_precision + total_congestion
            + self._mrc_weight * total_mrc / self._mrc_norm
            + self._peu_weight * total_peu / self._peu_norm
        )
        self._cost_cache_valid = True

    def _compute_hyperval_cost(self, hv: HyperVal) -> Tuple[float, float]:
        """Compute (wire_cost, latency_cost) for a single HyperVal."""
        src_dfg = self._id_to_dfg_node.get(hv.source_id)
        if src_dfg is None:
            return 0.0, 0.0
        src_mrrg = self._op_node_placement_map.get(src_dfg)
        if src_mrrg is None or not src_mrrg.coordinates:
            return 0.0, 0.0

        src_x, src_y = src_mrrg.coordinates
        x_min, x_max, y_min, y_max = src_x, src_x, src_y, src_y

        fanout = hv.cardinality
        q = (2.79 + 0.33 * (fanout - 3)) if fanout >= 3 else 1.0

        latency_cost = 0.0
        src_min_cycles_map = self._fu_to_fu_min_cycles.get(src_mrrg, {})
        for dest_id in hv.destination_ids:
            dest_dfg = self._id_to_dfg_node.get(dest_id)
            if dest_dfg is None:
                continue
            dest_mrrg = self._op_node_placement_map.get(dest_dfg)
            if dest_mrrg is None or not dest_mrrg.coordinates:
                continue

            dx, dy = dest_mrrg.coordinates
            x_min, x_max = min(x_min, dx), max(x_max, dx)
            y_min, y_max = min(y_min, dy), max(y_max, dy)

            src_t = self._inc_schedule.get(hv.source_id, 0)
            dst_t = self._inc_schedule.get(dest_id, 0)
            c2s = dst_t - src_t
            if c2s < 0:
                c2s += self._ii
            if c2s == 0:
                c2s = 1

            # Use real MRRG min-cycles when both endpoints are FUs in the
            # reachability map; fall back to Manhattan otherwise.
            min_cyc = src_min_cycles_map.get(dest_mrrg)
            if min_cyc is not None:
                # Heavy penalty when min_cyc > c2s (route cannot fit in schedule);
                # smaller penalty when min_cyc < c2s (slack ok but suboptimal).
                if min_cyc > c2s:
                    latency_cost += (min_cyc - c2s) * c2s * 10.0
                else:
                    latency_cost += (c2s - min_cyc) * c2s
            else:
                manhattan = abs(src_x - dx) + abs(src_y - dy)
                latency_cost += abs(manhattan - c2s) * c2s

        wire_cost = q * ((x_max - x_min) + (y_max - y_min))
        mobility = getattr(src_dfg, 'mobility', None) or 0
        mobility_factor = 1.0 / (1.0 + mobility)
        return wire_cost * mobility_factor, latency_cost * mobility_factor

    def _compute_single_precision_penalty(self, dfg_node: DFGNode, mrrg_node: MRRGNode) -> float:
        """Precision penalty for one operation."""
        if not mrrg_node.operation_output_bitwidths:
            return 0.0
        fu_output_bw = mrrg_node.operation_output_bitwidths.get(dfg_node.operation)
        if fu_output_bw is None:
            return 0.0

        expected_output_bw = None
        node_bw = dfg_node.get_attribute("bitwidth")
        if isinstance(node_bw, int) and node_bw > 0:
            expected_output_bw = node_bw

        outgoing = self._dfg.get_outgoing_edges(dfg_node.id)
        edge_widths = []
        for edge in outgoing:
            edge_bw = edge.get_attribute("bitwidth")
            if isinstance(edge_bw, int) and edge_bw > 0:
                edge_widths.append(edge_bw)

        if edge_widths:
            max_edge_bw = max(edge_widths)
            expected_output_bw = (
                min(expected_output_bw, max_edge_bw)
                if expected_output_bw is not None
                else max_edge_bw
            )

        if expected_output_bw is None or expected_output_bw <= 0:
            return 0.0

        if fu_output_bw != expected_output_bw:
            width_ratio = max(fu_output_bw, expected_output_bw) / min(
                fu_output_bw, expected_output_bw
            )
            return (width_ratio - 1.0) * self._precision_penalty_weight
        return 0.0

    def _compute_precision_penalty_from_scratch(self) -> float:
        """Full precision penalty (used by _get_total_cost_from_scratch only)."""
        penalty = 0.0
        for dfg_node, mrrg_node in self._op_node_placement_map.items():
            penalty += self._compute_single_precision_penalty(dfg_node, mrrg_node)
        return penalty

    def _congestion_at(self, mrrg_node: MRRGNode) -> float:
        """Congestion penalty for placing an op on this FU."""
        if self._congestion_weight == 0.0 or not self._congestion_map:
            return 0.0
        if mrrg_node.coordinates:
            return self._congestion_weight * self._congestion_map.get(
                mrrg_node.coordinates, 0.0
            )
        return 0.0

    def _compute_single_mrc(self, dfg_node, mrrg_node):
        """Memory Resource Cost: penalise non-memory ops placed on memory FUs."""
        if self._mrc_weight == 0.0:
            return 0.0
        if dfg_node.operation in (OperationType.LOAD, OperationType.STORE):
            return 0.0
        if mrrg_node.id not in self._memory_fu_ids:
            return 0.0
        total_mem_ops = len(self._memory_dfg_ids)
        unmapped = total_mem_ops - self._placed_memory_ops
        if unmapped <= 0:
            return 0.0
        total_mem_fus = len(self._memory_fu_ids)
        unused = max(total_mem_fus - self._memory_fus_occupied, 1)
        return unmapped / unused

    def _compute_single_peu(self, dfg_node, mrrg_node):
        """PE Utilization Pressure: penalise crowded coordinates."""
        if self._peu_weight == 0.0:
            return 0.0
        if not mrrg_node.coordinates:
            return 0.0
        count = self._coord_occupancy[mrrg_node.coordinates]
        return max(0, count - 1) ** 2

    def _compute_move_delta(
        self, moved_ops: Dict[DFGNode, Tuple[MRRGNode, MRRGNode]]
    ) -> _MoveDeltaResult:
        """Compute cost delta for moved operations. O(fanin+fanout) per op.

        Args:
            moved_ops: {dfg_node: (old_fu, new_fu)} — move already applied
        """
        delta = 0.0
        w_wn = self._wire_weight / self._wire_norm
        l_ln = self._latency_weight / self._latency_norm
        m_mn = self._mrc_weight / self._mrc_norm
        p_pn = self._peu_weight / self._peu_norm

        # Collect affected hypervals (deduplicated via set)
        affected_hv_ids: Set[str] = set()
        for dfg_node in moved_ops:
            affected_hv_ids.update(self._node_to_hv_ids.get(dfg_node.id, []))

        # Subtract old HV costs, compute new ones
        new_hv_costs: Dict[str, Tuple[float, float]] = {}
        for hv_id in sorted(affected_hv_ids):
            old_w, old_l = self._per_hv_costs[hv_id]
            delta -= w_wn * old_w + l_ln * old_l
            hv = self._inc_hyperdfg.get_edge(hv_id)
            new_w, new_l = self._compute_hyperval_cost(hv)
            delta += w_wn * new_w + l_ln * new_l
            new_hv_costs[hv_id] = (new_w, new_l)

        # Subtract old precision/congestion/mrc/peu, compute new
        new_precision: Dict[str, float] = {}
        new_congestion: Dict[str, float] = {}
        new_mrc: Dict[str, float] = {}
        new_peu: Dict[str, float] = {}
        for dfg_node, (old_fu, new_fu) in moved_ops.items():
            delta -= self._per_op_precision.get(dfg_node.id, 0.0)
            delta -= self._per_op_congestion.get(dfg_node.id, 0.0)
            delta -= m_mn * self._per_op_mrc.get(dfg_node.id, 0.0)
            delta -= p_pn * self._per_op_peu.get(dfg_node.id, 0.0)
            p = self._compute_single_precision_penalty(dfg_node, new_fu)
            c = self._congestion_at(new_fu)
            m = self._compute_single_mrc(dfg_node, new_fu)
            u = self._compute_single_peu(dfg_node, new_fu)
            delta += p + c + m_mn * m + p_pn * u
            new_precision[dfg_node.id] = p
            new_congestion[dfg_node.id] = c
            new_mrc[dfg_node.id] = m
            new_peu[dfg_node.id] = u

        return _MoveDeltaResult(delta, new_hv_costs, new_precision, new_congestion, new_mrc, new_peu)

    def _apply_move_delta(self, result: _MoveDeltaResult) -> None:
        """Commit cached costs from accepted move. O(1) dict updates."""
        self._per_hv_costs.update(result.new_hv_costs)
        self._per_op_precision.update(result.new_precision)
        self._per_op_congestion.update(result.new_congestion)
        self._per_op_mrc.update(result.new_mrc)
        self._per_op_peu.update(result.new_peu)
        self._total_cost_cached += result.delta

    def set_congestion_feedback(
        self, congestion_map: Dict[Tuple[int, int], float], weight: float
    ) -> None:
        """Inject routing congestion data into placement cost.

        Triggers a full cache rebuild to keep all per-HV and per-op
        caches consistent with the new congestion penalties.
        """
        self._congestion_map = congestion_map
        self._congestion_weight = weight
        if self._cost_cache_valid:
            self._init_incremental_cost()

    # Keep the old method name as an alias for backwards compatibility in tests
    def _compute_precision_penalty(self) -> float:
        """Compute penalty for precision mismatches (legacy, used by tests)."""
        return self._compute_precision_penalty_from_scratch()

    def inner_loop(self, temperature: float, deadline: Optional[float] = None) -> float:
        """
        Perform placement moves at the current temperature.

        Uses incremental O(degree) cost deltas instead of O(N) full recomputation.

        Args:
            temperature: Current annealing temperature

        Returns:
            float: Acceptance rate (accepted_moves / total_tries)
        """
        # Ensure incremental cost cache is built
        if not self._cost_cache_valid:
            self._init_incremental_cost()

        # Cache operation list for the loop
        operations = list(self._dfg.get_nodes())

        num_swaps = len(operations) * self._swap_factor
        total_accepted = 0
        total_tries = 0
        self._timed_out = False

        for i in range(num_swaps):
            if deadline is not None and time.time() >= deadline:
                self._timed_out = True
                break

            # Randomly select an operation
            op = self._rng.choice(operations)

            # Skip if this operation is reserved/fixed
            if op.id in self._reserved_dfg_node_id_to_fu_node_id:
                continue

            # Get a random compatible FU for this operation
            new_fu = self.get_random_fu(op)
            if new_fu is None:
                continue

            # Skip if it's the same FU (no change)
            current_fu = self._op_node_placement_map.get(op)
            if current_fu and new_fu.id == current_fu.id:
                continue

            # This is an actual attempt
            total_tries += 1

            # Perform the move/swap and track what changed
            moved_ops: Dict[DFGNode, Tuple[MRRGNode, MRRGNode]] = {}

            if self._fu_node_placement_state[new_fu].occupancy < new_fu.capacity:
                # FU is available, just move the operation
                moved_ops[op] = (current_fu, new_fu)
                self.clear_op_placement(op)
                self.update_op_placement(op, new_fu)
            else:
                # FU is occupied, need to swap
                displaced_op = None
                for dfg_op, fu_node in self._op_node_placement_map.items():
                    if fu_node.id == new_fu.id:
                        displaced_op = dfg_op
                        break

                if displaced_op is None:
                    continue

                # Reject the swap if the displaced op's new position (current_fu) is
                # not a legal candidate for it. This protects ops with constrained
                # candidate lists (e.g., must-be-near-boundary muls) from being
                # accidentally moved to invalid PEs by another op's swap.
                if current_fu is not None:
                    displaced_constrained = self._node_constrained_fus.get(displaced_op.id)
                    if displaced_constrained is not None:
                        if current_fu not in displaced_constrained:
                            continue

                moved_ops[op] = (current_fu, new_fu)
                moved_ops[displaced_op] = (new_fu, current_fu)

                self.clear_op_placement(op)
                self.clear_op_placement(displaced_op)
                self.update_op_placement(op, new_fu)
                self.update_op_placement(displaced_op, current_fu)

            # Reject moves/swaps that create multi-cycle FU conflicts
            # (e.g., back-to-back loads on the same LSU port).
            _mc_conflict = False
            for _dfg_op, (_, _new_fu) in moved_ops.items():
                if _new_fu.latency > 1:
                    _base = _new_fu.get_base_name()
                    _counts = self._multi_cycle_occ.get(_base)
                    if _counts and any(
                        _counts[(_new_fu.cycle + _c) % self._ii] > 1
                        for _c in range(_new_fu.latency)
                    ):
                        _mc_conflict = True
                        break
            if _mc_conflict:
                for _dfg_op, (old_fu, _) in moved_ops.items():
                    self.clear_op_placement(_dfg_op)
                    self.update_op_placement(_dfg_op, old_fu)
                continue

            # Compute delta cost incrementally — O(degree) not O(N)
            move_result = self._compute_move_delta(moved_ops)

            if self.accept_move(move_result.delta, temperature):
                total_accepted += 1
                self._apply_move_delta(move_result)
            else:
                # Revert placement
                for dfg_op, (old_fu, _new_fu) in moved_ops.items():
                    self.clear_op_placement(dfg_op)
                    self.update_op_placement(dfg_op, old_fu)

        # Return acceptance rate
        if total_tries == 0:
            return 0.0
        return total_accepted / total_tries

    def accept_move(self, delta_cost: float, temperature: float) -> bool:
        """Accept the move based on the delta cost and temperature."""
        if delta_cost < 0:
            return True
        probability: float = math.exp(-delta_cost / temperature)
        random_val: float = self._rng.random()  # Returns value in [0.0, 1.0)
        return probability > random_val          

    def next_temperature(self, temperature: float, accept_rate: float) -> float:
        """Get the next temperature based on the accept rate."""
        if accept_rate > 0.96:
            return temperature * 0.5
        elif accept_rate > 0.8:
            return temperature * 0.9
        elif accept_rate > 0.15:
            return temperature * 0.95
        else:
            return temperature * 0.8

    def pretty_print_placement(self) -> str:
        """
        Generate a pretty-printed visualization of the current placement.

        Shows the mapping of DFG operations to MRRG nodes with scheduled cycles.

        Returns:
            String representation of the placement
        """
        if not self._op_node_placement_map:
            return "No placement available."

        # Build the visualization string
        lines = []
        lines.append("=" * 90)
        lines.append("PLACEMENT VISUALIZATION")
        lines.append("=" * 90)
        lines.append(f"Total operations placed: {len(self._op_node_placement_map)}")
        lines.append("=" * 90)
        lines.append("")
        lines.append(f"{'DFG Operation':<30} {'Op Type':<15} {'Cycle':<8} {'MRRG Node':<35}")
        lines.append("-" * 90)

        # Sort by scheduled cycle first, then by DFG node id
        sorted_mappings = sorted(
            self._op_node_placement_map.items(),
            key=lambda x: (x[0].scheduled_time if x[0].scheduled_time is not None else 0, x[0].id)
        )

        for dfg_node, mrrg_node in sorted_mappings:
            dfg_id = dfg_node.id
            op_type = dfg_node.operation.name
            mrrg_id = mrrg_node.id

            # Get scheduled cycle
            cycle = dfg_node.scheduled_time if dfg_node.scheduled_time is not None else "N/A"
            cycle_str = str(cycle) if cycle != "N/A" else cycle

            lines.append(f"{dfg_id:<30} {op_type:<15} {cycle_str:<8} {mrrg_id:<35}")

        lines.append("=" * 90)

        return "\n".join(lines)


    # -- Placer Properties --------------------------------------------------------
    def op_node_placement_map(self) -> Dict[DFGNode, MRRGNode]:
        """Get the operation node placement map."""
        return self._op_node_placement_map

    def fu_node_placement_state(self) -> Dict[MRRGNode, MRRGNodePlacementState]:
        """Get the FU node placement state."""
        return self._fu_node_placement_state

    def temporal_op_type_to_fu_nodes(self) -> Dict[Tuple[OperationType, int, Optional[int], Optional[int]], List[MRRGNode]]:
        """Get the temporal operation type to FU nodes map."""
        return self._temporal_op_type_to_fu_nodes
