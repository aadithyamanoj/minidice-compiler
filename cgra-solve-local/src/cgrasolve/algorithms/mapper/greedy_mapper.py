"""HyCUBE-style greedy mapper (DAC'17 Algorithm 1).

Implements a single-pass greedy placement+routing algorithm for
apples-to-apples comparison with the iterative HeuristicMapper.
The mapper processes DFG nodes in topological order, greedily
selecting the minimum-cost FU for each node and immediately routing
its data dependencies via Dijkstra's shortest path.

Returns the *exact same* result dictionary contract as
:class:`HeuristicMapper` so that both mappers can be used
interchangeably in benchmarking scripts.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from cgrasolve.graph.dfg import DFG, DFGEdge, DFGNode, OperationType
from cgrasolve.graph.mrrg import MRRG, MRRGNode, NodeType, OperandTag
from cgrasolve.graph.hyperdfg import HyperDFG, HyperVal
from cgrasolve.graph.utils.pathfinding import dijkstra, reconstruct_path
from cgrasolve.schedules.latency_spec import LatencySpecification
from cgrasolve.algorithms.scheduler.asap_scheduler import ASAPScheduler
from cgrasolve.algorithms.scheduler.types import ModuloSchedule
from cgrasolve.algorithms.router.types import (
    RouteBundle,
    RoutingStats,
    TransportProfile,
)
from cgrasolve.algorithms.mapper.greedy_cost import GreedyCostEvaluator


# Sentinel for unreachable Dijkstra cost.
_INF: float = float("inf")


def _resolve_source_ports(
    mrrg: MRRG,
    source_fu: MRRGNode,
    source_op: OperationType,
    bitwidth: Optional[int],
) -> List[MRRGNode]:
    """Return legal output ports for *source_fu* running *source_op*."""
    try:
        ports = mrrg.get_output_ports_for_operation(source_fu, source_op, bitwidth)
        if ports:
            return ports
    except Exception:
        pass
    # Fallback: non-FU successors of the source FU.
    return [
        n for n in mrrg.get_successors(source_fu.id)
        if n.node_type != NodeType.FUNCTION
    ]


def _operand_to_tag(operand: Optional[str]) -> Optional[OperandTag]:
    """Map DFG edge operand string to MRRG OperandTag."""
    if operand is None:
        return None
    _MAP = {
        "LHS": OperandTag.BINARY_LHS,
        "RHS": OperandTag.BINARY_RHS,
        "DATA": OperandTag.DATA,
        "data": OperandTag.DATA,
        "addr": OperandTag.ADDR,
        "ADDR": OperandTag.ADDR,
        "pred": OperandTag.PREDICATE,
        "PRED": OperandTag.PREDICATE,
    }
    return _MAP.get(operand)


def _resolve_sink_ports(
    mrrg: MRRG,
    sink_fu: MRRGNode,
    sink_op: OperationType,
    operand: Optional[str],
) -> List[MRRGNode]:
    """Return legal input ports for *sink_fu* running *sink_op*."""
    tag = _operand_to_tag(operand)
    try:
        ports = mrrg.get_input_ports_for_operand(sink_fu, sink_op, tag)
        if ports:
            return ports
    except Exception:
        pass
    # Fallback: non-FU predecessors of the sink FU.
    return [
        n for n in mrrg.get_predecessors(sink_fu.id)
        if n.node_type != NodeType.FUNCTION
    ]


class GreedyMapper:
    """Single-pass greedy mapper inspired by HyCUBE (DAC'17).

    Unlike :class:`HeuristicMapper`, this mapper does **not** use
    simulated annealing or negotiated-congestion routing.  Placement
    and routing happen in a single topological-order sweep, guided by
    cost functions (SRC, UARC, MRC, Width).
    """

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        latency_spec: LatencySpecification,
        *,
        max_time: Optional[float] = None,
        max_ii_increment: int = 10,
        cost_weights: Optional[Dict[str, float]] = None,
        mrrg_loader: Optional[Callable[[int], Any]] = None,
        debug: bool = False,
    ) -> None:
        self._dfg = dfg
        self._mrrg = mrrg
        self._latency_spec = latency_spec
        self._max_time = max_time
        self._max_ii_increment = max_ii_increment
        self._mrrg_loader = mrrg_loader
        self._debug = debug

        weights = cost_weights or {}
        self._w_src = weights.get("src", 1.0)
        self._w_uarc = weights.get("uarc", 1.0)
        self._w_mrc = weights.get("mrc", 1.0)
        self._w_width = weights.get("width", 1.0)

        # State reset per map() call.
        self._mapping_start_time: Optional[float] = None
        self._mapping_deadline: Optional[float] = None
        self._iterations_run = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self) -> Dict[str, Any]:
        """Run the greedy mapping algorithm.

        Returns the same dictionary contract as ``HeuristicMapper.map()``.
        """
        start_time = time.time()
        self._mapping_start_time = start_time
        self._mapping_deadline = (
            start_time + self._max_time if self._max_time is not None else None
        )
        self._iterations_run = 0

        try:
            # Step 1: Compute MII.
            min_ii = self._compute_mii()

            # Step 2: Topological order (loop-back edges excluded).
            ordered_nodes = self._topological_sort_ignoring_loopbacks()

            # Step 3: Build HyperDFG once for result construction.
            hyperdfg = HyperDFG.from_dfg(self._dfg)
            edge_to_hv = self._build_edge_to_hyperval_map(hyperdfg)

            max_ii = min(min_ii + self._max_ii_increment, self._mrrg.II)
            last_error = "No II attempted"

            for ii in range(min_ii, max_ii + 1):
                if self._deadline_exceeded():
                    break

                self._iterations_run += 1
                mrrg = self._load_mrrg_for_ii(ii)

                if self._debug:
                    print(f"\n=== Greedy mapping attempt (II={ii}) ===")

                result = self._try_mapping(
                    ii, mrrg, ordered_nodes, hyperdfg, edge_to_hv,
                )
                if result is not None:
                    result["runtime"] = time.time() - start_time
                    return result
                last_error = f"Greedy mapping failed at II={ii}"

            runtime = time.time() - start_time
            if self._deadline_exceeded():
                last_error = f"Timeout after {runtime:.2f}s"
            return self._build_failure_result(
                runtime=runtime,
                error_message=last_error,
            )

        except Exception as exc:
            runtime = time.time() - start_time
            return self._build_failure_result(
                runtime=runtime,
                error_message=f"Mapping error: {exc}",
                validation_errors=[f"Mapping error: {exc}"],
            )

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _try_mapping(
        self,
        ii: int,
        mrrg: MRRG,
        ordered_nodes: List[DFGNode],
        hyperdfg: HyperDFG,
        edge_to_hv: Dict[Tuple[str, str], Tuple[HyperVal, int]],
    ) -> Optional[Dict[str, Any]]:
        """Attempt a full greedy mapping at the given II.

        Returns a success result dict, or ``None`` on failure.
        """
        self._seed_asap_schedule(ii)

        cost_eval = GreedyCostEvaluator(
            mrrg, self._dfg,
            w_src=self._w_src, w_uarc=self._w_uarc,
            w_mrc=self._w_mrc, w_width=self._w_width,
        )

        # Mapping state for this attempt.
        placement: Dict[DFGNode, MRRGNode] = {}
        placement_ids: Dict[str, str] = {}           # dfg_id -> fu_id
        schedule_times: Dict[str, int] = {}           # dfg_id -> time
        used_resources: Dict[str, str] = {}           # mrrg_id -> value_source_id
        placed_dfg_ids: Set[str] = set()
        placed_fu_ids: Set[str] = set()
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle] = {}
        # Track which link sets have been committed per source value
        # so SRC can reward multicast reuse.
        committed_link_sets: Dict[str, List[Set[str]]] = {}
        total_cost = 0.0

        for dfg_node in ordered_nodes:
            if self._deadline_exceeded():
                return None

            result = self._place_and_route_node(
                dfg_node,
                ii,
                mrrg,
                cost_eval,
                placement,
                placement_ids,
                schedule_times,
                used_resources,
                placed_dfg_ids,
                placed_fu_ids,
                routing_solution,
                committed_link_sets,
                edge_to_hv,
            )
            if result is None:
                if self._debug:
                    print(f"  Failed to place node {dfg_node.id}")
                return None
            total_cost += result

        # Post-pass: route loop-back edges.
        if not self._route_loopback_edges(
            ii, mrrg, placement, schedule_times, used_resources,
            routing_solution, edge_to_hv, committed_link_sets,
        ):
            if self._debug:
                print("  Failed to route loop-back edges")
            return None

        # Stamp schedule on DFG for validation.
        schedule = self._build_schedule(ii, schedule_times)
        self._stamp_schedule(schedule)

        # Validate.
        is_valid, validation_errors = self._validate(
            mrrg, placement, routing_solution, ii, schedule_times,
        )

        if not is_valid:
            if self._debug:
                n_errs = len(validation_errors)
                print(f"  Validation failed ({n_errs} errors) -- trying next II")
            return None

        routing_stats = self._compute_routing_stats(
            mrrg, routing_solution, used_resources,
        )

        return {
            "status": "success",
            "placement": placement_ids,
            "routes": self._routes_to_dict(routing_solution),
            "route_metadata": self._route_metadata_to_dict(routing_solution),
            "runtime": 0.0,  # filled in by caller
            "iterations": self._iterations_run,
            "final_cost": total_cost,
            "ii": ii,
            "schedule": self._schedule_to_dict(schedule),
            "attempt_history": [],
            "routing_stats": asdict(routing_stats),
            "validation": {"is_valid": True, "errors": []},
        }

    def _place_and_route_node(
        self,
        dfg_node: DFGNode,
        ii: int,
        mrrg: MRRG,
        cost_eval: GreedyCostEvaluator,
        placement: Dict[DFGNode, MRRGNode],
        placement_ids: Dict[str, str],
        schedule_times: Dict[str, int],
        used_resources: Dict[str, str],
        placed_dfg_ids: Set[str],
        placed_fu_ids: Set[str],
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
        committed_link_sets: Dict[str, List[Set[str]]],
        edge_to_hv: Dict[Tuple[str, str], Tuple[HyperVal, int]],
    ) -> Optional[float]:
        """Place one DFG node and route its incoming edges.

        Returns the cost of the best candidate, or ``None`` on failure.
        """
        # Collect non-loop-back incoming edges (parents already placed).
        incoming_edges = [
            e for e in self._dfg.get_incoming_edges(dfg_node.id)
            if not e.is_loop_back
        ]

        # Determine the earliest feasible schedule time from parents.
        earliest_time = 0
        for edge in incoming_edges:
            parent = edge.source
            parent_time = schedule_times.get(parent.id, 0)
            op_lat = self._safe_op_latency(parent.operation)
            net_lat = self._safe_network_latency_lower(
                parent.operation, dfg_node.operation,
            )
            earliest_time = max(earliest_time, parent_time + op_lat + net_lat)

        scheduled_hint = (
            dfg_node.scheduled_time
            if dfg_node.scheduled_time is not None
            else dfg_node.asap_time
        )
        if scheduled_hint is not None:
            earliest_time = max(earliest_time, scheduled_hint)

        # Find compatible FUs.  Try the preferred temporal class first,
        # then later classes if the preferred one is full.
        compatible_fus = mrrg.get_compatible_fus(dfg_node.operation)

        candidates: List[MRRGNode] = []
        chosen_time = earliest_time
        for offset in range(ii):
            t = earliest_time + offset
            tc = t % ii
            cands = [
                fu for fu in compatible_fus
                if fu.cycle == tc and fu.id not in placed_fu_ids
            ]
            if cands:
                candidates = cands
                chosen_time = t
                break

        if not candidates:
            return None

        # Update earliest_time to the chosen slot.
        earliest_time = chosen_time

        best_cost = _INF
        best_fu: Optional[MRRGNode] = None
        best_routes: Dict[Tuple[HyperVal, int], RouteBundle] = {}
        best_link_sets: List[Set[str]] = []
        best_edge_bitwidths: List[Optional[int]] = []

        for candidate_fu in candidates:
            # Try to route all incoming edges to this candidate.
            route_result = self._route_incoming_edges(
                dfg_node, candidate_fu, incoming_edges, ii, mrrg,
                placement, used_resources, edge_to_hv,
            )
            if route_result is None:
                continue  # At least one parent unreachable.

            routes_for_candidate, parent_link_sets, edge_bitwidths = route_result

            # Collect sibling link sets (routes from same parents
            # already committed for other destinations).
            sibling_link_sets: List[Set[str]] = []
            for edge in incoming_edges:
                sibling_link_sets.extend(
                    committed_link_sets.get(edge.source.id, [])
                )

            cost = cost_eval.evaluate(
                candidate_fu,
                dfg_node,
                parent_link_sets,
                sibling_link_sets,
                edge_bitwidths,
                set(used_resources.keys()),
                placed_dfg_ids,
                placed_fu_ids,
            )

            if cost < best_cost:
                best_cost = cost
                best_fu = candidate_fu
                best_routes = routes_for_candidate
                best_link_sets = parent_link_sets
                best_edge_bitwidths = edge_bitwidths

        if best_fu is None:
            return None

        # Commit placement.
        placement[dfg_node] = best_fu
        placement_ids[dfg_node.id] = best_fu.id
        placed_dfg_ids.add(dfg_node.id)
        placed_fu_ids.add(best_fu.id)
        schedule_times[dfg_node.id] = earliest_time
        used_resources[best_fu.id] = dfg_node.id

        # Commit routes.
        for key, bundle in best_routes.items():
            routing_solution[key] = bundle
            for node in bundle.all_nodes:
                if node.id != best_fu.id:
                    used_resources.setdefault(node.id, key[0].source_id)

        # Record link sets for SRC multicast reuse.
        for edge, link_set in zip(incoming_edges, best_link_sets):
            committed_link_sets.setdefault(edge.source.id, []).append(link_set)

        if self._debug:
            print(
                f"  Placed {dfg_node.id} ({dfg_node.operation.value}) -> "
                f"{best_fu.id} @ t={earliest_time}  cost={best_cost:.3f}"
            )

        return best_cost

    def _route_incoming_edges(
        self,
        dfg_node: DFGNode,
        candidate_fu: MRRGNode,
        incoming_edges: List[DFGEdge],
        ii: int,
        mrrg: MRRG,
        placement: Dict[DFGNode, MRRGNode],
        used_resources: Dict[str, str],
        edge_to_hv: Dict[Tuple[str, str], Tuple[HyperVal, int]],
    ) -> Optional[Tuple[
        Dict[Tuple[HyperVal, int], RouteBundle],
        List[Set[str]],
        List[Optional[int]],
    ]]:
        """Route all incoming edges to ``candidate_fu``.

        Returns ``(routes, link_sets, edge_bitwidths)`` or ``None``
        if any edge is unroutable.
        """
        routes: Dict[Tuple[HyperVal, int], RouteBundle] = {}
        parent_link_sets: List[Set[str]] = []
        edge_bitwidths: List[Optional[int]] = []
        speculative_used: Set[str] = set()

        for edge in incoming_edges:
            parent_fu = placement[edge.source]
            bw = edge.get_attribute("bitwidth")
            operand = edge.get_attribute("operand")

            path = self._find_route(
                parent_fu, candidate_fu,
                source_op=edge.source.operation,
                sink_op=dfg_node.operation,
                operand=operand,
                source_value_id=edge.source.id,
                bitwidth=bw,
                mrrg=mrrg,
                used_resources=used_resources,
                speculative_used=speculative_used,
            )
            if path is None:
                return None

            link_set = {n.id for n in path}
            parent_link_sets.append(link_set)
            edge_bitwidths.append(bw)
            speculative_used.update(link_set)

            # Build RouteBundle.
            profile = TransportProfile.flat(bw)
            bundle = RouteBundle.from_flat_path(path, profile)

            hv_key = edge_to_hv.get((edge.source.id, edge.destination.id))
            if hv_key is not None:
                routes[hv_key] = bundle

        return routes, parent_link_sets, edge_bitwidths

    def _find_route(
        self,
        source_fu: MRRGNode,
        sink_fu: MRRGNode,
        *,
        source_op: OperationType,
        sink_op: OperationType,
        operand: Optional[str],
        source_value_id: str,
        bitwidth: Optional[int],
        mrrg: MRRG,
        used_resources: Dict[str, str],
        speculative_used: Optional[Set[str]] = None,
    ) -> Optional[List[MRRGNode]]:
        """Find a Dijkstra shortest path between legal ports.

        Uses the MRRG port-resolution APIs to select legal output ports
        of *source_fu* and legal input ports of *sink_fu* (honouring
        operand tags), then routes through the interconnect via Dijkstra.
        """

        _spec = speculative_used or set()

        def weight_fn(e):
            dest_node = e.destination
            # Never route through an FU placed for compute.
            if dest_node.node_type == NodeType.FUNCTION:
                return _INF
            if dest_node.id in _spec:
                return _INF
            owner = used_resources.get(dest_node.id)
            if owner is not None and owner != source_value_id:
                return _INF  # blocked by a different value
            if owner == source_value_id:
                return 0.0  # multicast reuse
            # UARC-aware weight (per HyCUBE paper): base cost plus
            # number of used adjacent resources.  This steers Dijkstra
            # away from congested regions during pathfinding.
            uarc = 0
            for nb in mrrg.get_successors(dest_node.id):
                if nb.id in used_resources:
                    uarc += 1
            for nb in mrrg.get_predecessors(dest_node.id):
                if nb.id in used_resources:
                    uarc += 1
            return 1.0 + uarc

        # --- Resolve legal source output ports ---
        src_ports = _resolve_source_ports(mrrg, source_fu, source_op, bitwidth)

        # --- Resolve legal sink input ports (with operand tag) ---
        sink_ports = _resolve_sink_ports(mrrg, sink_fu, sink_op, operand)
        sink_ports = [p for p in sink_ports if p.id not in _spec]

        # If source directly connects to sink (no routing needed).
        if sink_fu in mrrg.get_successors(source_fu.id):
            return []

        if not src_ports or not sink_ports:
            return None

        best_path: Optional[List[MRRGNode]] = None
        best_cost = _INF

        for src_port in src_ports:
            distances, predecessors = dijkstra(
                mrrg, src_port.id, None, weight_fn=weight_fn,
            )
            for sink_port in sink_ports:
                sink_dist = distances.get(sink_port.id, _INF)
                if sink_dist >= _INF:
                    continue
                path = reconstruct_path(
                    predecessors, src_port.id, sink_port.id, mrrg,
                )
                if path is None:
                    continue
                # Filter out any FU nodes from the path.
                path = [n for n in path if n.node_type != NodeType.FUNCTION]
                cost = sink_dist
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

        return best_path

    # ------------------------------------------------------------------
    # Loop-back edge routing
    # ------------------------------------------------------------------

    def _route_loopback_edges(
        self,
        ii: int,
        mrrg: MRRG,
        placement: Dict[DFGNode, MRRGNode],
        schedule_times: Dict[str, int],
        used_resources: Dict[str, str],
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
        edge_to_hv: Dict[Tuple[str, str], Tuple[HyperVal, int]],
        committed_link_sets: Dict[str, List[Set[str]]],
    ) -> bool:
        """Route loop-back edges between already-placed FUs."""
        loopback_edges = self._dfg.get_loop_back_edges()
        for edge in loopback_edges:
            source_fu = placement.get(edge.source)
            sink_fu = placement.get(edge.destination)
            if source_fu is None or sink_fu is None:
                return False

            path = self._find_route(
                source_fu, sink_fu,
                source_op=edge.source.operation,
                sink_op=edge.destination.operation,
                operand=edge.get_attribute("operand"),
                source_value_id=edge.source.id,
                bitwidth=edge.get_attribute("bitwidth"),
                mrrg=mrrg,
                used_resources=used_resources,
            )
            if path is None:
                return False

            bw = edge.get_attribute("bitwidth")
            profile = TransportProfile.flat(bw)
            bundle = RouteBundle.from_flat_path(path, profile)

            hv_key = edge_to_hv.get((edge.source.id, edge.destination.id))
            if hv_key is not None:
                routing_solution[hv_key] = bundle

            # Commit routing resources.
            for node in bundle.all_nodes:
                used_resources.setdefault(node.id, edge.source.id)

            committed_link_sets.setdefault(edge.source.id, []).append(
                {n.id for n in path}
            )

        return True

    # ------------------------------------------------------------------
    # Topological ordering
    # ------------------------------------------------------------------

    def _topological_sort_ignoring_loopbacks(self) -> List[DFGNode]:
        """Return DFG nodes in topological order, ignoring loop-back edges."""
        g = nx.DiGraph()
        for node in self._dfg.get_nodes():
            g.add_node(node.id)
        for edge in self._dfg.get_edges():
            if not edge.is_loop_back:
                g.add_edge(edge.source.id, edge.destination.id)
        return [self._dfg.get_node(nid) for nid in nx.topological_sort(g)]

    # ------------------------------------------------------------------
    # MII computation
    # ------------------------------------------------------------------

    def _compute_mii(self) -> int:
        """Return the minimum initiation interval for the DFG + MRRG."""
        try:
            from cgrasolve.algorithms.scheduler.ii_analysis import IIAnalyzer
            analyzer = IIAnalyzer(self._dfg, self._mrrg, self._latency_spec)
            res_mii, _ = analyzer.compute_res_mii()
            rec_mii, _ = analyzer.compute_rec_mii()
            return max(res_mii, rec_mii, 1)
        except Exception:
            # Fallback: use MRRG's II or 1.
            return max(getattr(self._mrrg, "II", 1), 1)

    # ------------------------------------------------------------------
    # MRRG loading
    # ------------------------------------------------------------------

    def _load_mrrg_for_ii(self, ii: int) -> MRRG:
        """Resolve the MRRG to use for one II attempt.

        Priority:
        1. If an ``mrrg_loader`` is provided, use it (advanced override).
        2. Otherwise, slice the base MRRG to *ii* cycles via
           ``MRRG.slice_to_ii``.
        """
        if self._mrrg_loader is not None:
            loaded = self._mrrg_loader(ii)
            if isinstance(loaded, MRRG):
                return loaded
            if hasattr(loaded, "mrrg"):
                return loaded.mrrg
            raise TypeError(
                f"Unsupported mrrg_loader return type: {type(loaded).__name__}"
            )
        if ii <= self._mrrg.II:
            return self._mrrg.slice_to_ii(ii)
        raise ValueError(
            f"Cannot search II={ii}; base MRRG II is {self._mrrg.II} "
            f"and no mrrg_loader was provided"
        )

    # ------------------------------------------------------------------
    # HyperVal lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edge_to_hyperval_map(
        hyperdfg: HyperDFG,
    ) -> Dict[Tuple[str, str], Tuple[HyperVal, int]]:
        """Map ``(source_id, dest_id)`` to ``(HyperVal, dest_idx)``."""
        mapping: Dict[Tuple[str, str], Tuple[HyperVal, int]] = {}
        for hv in hyperdfg.get_vals():
            for idx, dest_id in enumerate(hv.destination_ids):
                mapping[(hv.source_id, dest_id)] = (hv, idx)
        return mapping

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _build_schedule(
        self, ii: int, node_times: Dict[str, int],
    ) -> ModuloSchedule:
        return ModuloSchedule(
            ii=ii,
            node_times=dict(node_times),
            asap_times=dict(node_times),
            alap_times={},
            mobility={nid: 0 for nid in node_times},
            diagnostics={},
        )

    def _stamp_schedule(self, schedule: ModuloSchedule) -> None:
        from cgrasolve.algorithms.scheduler.types import stamp_schedule_on_dfg
        stamp_schedule_on_dfg(self._dfg, schedule, update_asap=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        mrrg: MRRG,
        placement: Dict[DFGNode, MRRGNode],
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
        ii: int,
        schedule_times: Dict[str, int],
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        # 1. All DFG nodes placed.
        all_ids = {n.id for n in self._dfg.get_nodes()}
        placed_ids = {n.id for n in placement}
        missing = all_ids - placed_ids
        if missing:
            errors.append(f"Missing placements for nodes: {missing}")

        # 2. Operation support.
        for dfg_node, mrrg_node in placement.items():
            if not mrrg_node.can_execute(dfg_node.operation):
                errors.append(
                    f"Operation {dfg_node.operation.value} not supported on FU {mrrg_node.id}"
                )

        # 3. Resource conflicts.
        fu_usage: Dict[Tuple[str, int], str] = {}
        for dfg_node, mrrg_node in placement.items():
            key = (mrrg_node.get_base_name(), mrrg_node.cycle)
            if key in fu_usage:
                errors.append(
                    f"Resource conflict: {dfg_node.id} and {fu_usage[key]} "
                    f"both use {key[0]} at cycle {key[1]}"
                )
            else:
                fu_usage[key] = dfg_node.id

        # 4. Shared routing validation.
        try:
            from cgrasolve.algorithms.common.routing_validation import validate_routing_solution
            _, shared_errors = validate_routing_solution(
                dfg=self._dfg,
                mrrg=mrrg,
                placement=placement,
                routing_solution=routing_solution,
                debug=self._debug,
            )
            errors.extend(shared_errors)
        except Exception as exc:
            errors.append(f"Validation error: {exc}")

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    @staticmethod
    def _routes_to_dict(
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
    ) -> Dict[Tuple[str, int], List[List[str]]]:
        return {
            (hv.source_id, idx): bundle.to_node_id_paths()
            for (hv, idx), bundle in routing_solution.items()
        }

    @staticmethod
    def _route_metadata_to_dict(
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        metadata: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for (hv, idx), bundle in routing_solution.items():
            profile = bundle.transport_profile
            metadata[(hv.source_id, idx)] = {
                "paths": bundle.to_node_id_paths(),
                "transport_mode": profile.mode.value,
                "fractured": bundle.fractured,
                "num_chunks": bundle.num_chunks,
                "chunk_indices": list(bundle.chunk_indices),
                "chunk_widths": list(bundle.chunk_widths),
                "chunk_offsets": list(bundle.chunk_offsets),
                "lane_assignments": dict(bundle.lane_assignments),
                "active_pack_nodes": sorted(bundle.active_pack_nodes),
                "active_unpack_nodes": sorted(bundle.active_unpack_nodes),
                "total_bitwidth": profile.total_bitwidth,
                "lane_width": profile.lane_width,
                "lane_index": profile.lane_index,
            }
        return metadata

    @staticmethod
    def _schedule_to_dict(schedule: ModuloSchedule) -> Dict[str, Any]:
        return {
            "ii": schedule.ii,
            "node_times": dict(schedule.node_times),
            "asap_times": dict(schedule.asap_times),
            "alap_times": dict(schedule.alap_times),
            "mobility": dict(schedule.mobility),
            "diagnostics": dict(schedule.diagnostics),
        }

    @staticmethod
    def _compute_routing_stats(
        mrrg: MRRG,
        routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
        used_resources: Dict[str, str],
    ) -> RoutingStats:
        all_routing = [n for n in mrrg.get_nodes() if n.node_type != NodeType.FUNCTION]
        unique_hvs = {hv for hv, _ in routing_solution}
        return RoutingStats(
            hypervals_considered=len(unique_hvs),
            destinations_considered=len(routing_solution),
            negotiation_iterations=0,
            routed_nets=len(routing_solution),
            resources_used=len(used_resources),
            total_occupancy=len(used_resources),
            total_capacity=len(all_routing) if all_routing else 1,
        )

    def _build_failure_result(
        self,
        *,
        runtime: float,
        error_message: str,
        placement: Optional[Dict[str, str]] = None,
        routes: Optional[Dict[Tuple[str, int], List[List[str]]]] = None,
        route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        validation_errors: Optional[List[str]] = None,
        ii: Optional[int] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "failed",
            "placement": placement,
            "routes": routes,
            "route_metadata": route_metadata,
            "runtime": runtime,
            "iterations": self._iterations_run,
            "final_cost": 0.0,
            "ii": ii,
            "schedule": schedule,
            "attempt_history": [],
            "error_message": error_message,
            "routing_error_message": None,
            "routing_failure_summary": None,
            "routing_stats": None,
            "validation": {
                "is_valid": False,
                "errors": list(validation_errors or []),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deadline_exceeded(self) -> bool:
        return (
            self._mapping_deadline is not None
            and time.time() >= self._mapping_deadline
        )

    def _seed_asap_schedule(self, ii: int) -> None:
        """Seed node ASAP times so greedy placement honors latency slack."""
        self._dfg.initiation_interval = ii
        self._dfg.clear_scheduling_info()
        try:
            ASAPScheduler(
                self._dfg,
                self._latency_spec,
                debug=self._debug,
            ).schedule()
        except Exception:
            self._dfg.clear_scheduling_info()

    def _safe_op_latency(self, op: OperationType) -> int:
        """Return op latency, defaulting to 0 for unconfigured ops."""
        try:
            return self._latency_spec.get_op_latency(op)
        except (ValueError, KeyError):
            return 0

    def _safe_network_latency_lower(
        self,
        src_op: OperationType,
        sink_op: OperationType,
    ) -> int:
        """Return network lower bound, defaulting to 0 for unconfigured edges."""
        try:
            return self._latency_spec.get_network_latency_lower(src_op, sink_op)
        except (ValueError, KeyError):
            return 0
