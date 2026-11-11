"""Active modulo scheduler for heuristic mapping."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple

from cgrasolve.algorithms.scheduler.asap_scheduler import ASAPScheduler
from cgrasolve.algorithms.scheduler.ii_analysis import IIAnalyzer
from cgrasolve.algorithms.scheduler.types import (
    ModuloSchedule,
    SchedulingAttempt,
    SchedulingFailureReason,
    SchedulingResult,
)
from cgrasolve.graph.dfg import DFG, DFGEdge, DFGNode
from cgrasolve.graph.mrrg import MRRG
from cgrasolve.schedules import LatencySpecification


class ModuloScheduler:
    """Heuristic modulo scheduler with II diagnostics and resource repair."""

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        latency_spec: LatencySpecification,
        *,
        fixed_schedule: Optional[Dict[object, int]] = None,
        max_resource_repairs: int = 4096,
        upper_bound_repair_limit: int = 128,
        debug: bool = False,
    ) -> None:
        self._dfg = dfg
        self._mrrg = mrrg
        self._latency_spec = latency_spec
        self._debug = debug
        self._max_resource_repairs = max_resource_repairs
        self._upper_bound_repair_limit = upper_bound_repair_limit
        self._fixed_schedule = self._normalize_fixed_schedule(fixed_schedule or {})
        self._ii_analyzer = IIAnalyzer(dfg=dfg, mrrg=mrrg, latency_spec=latency_spec)

    def _emit(self, message: str) -> None:
        if self._debug:
            print(message)

    def _normalize_fixed_schedule(self, fixed_schedule: Dict[object, int]) -> Dict[str, int]:
        """Normalize fixed schedule keys to DFG node IDs."""
        normalized: Dict[str, int] = {}
        for key, time in fixed_schedule.items():
            if isinstance(key, DFGNode):
                normalized[key.id] = time
            else:
                normalized[str(key)] = time
        return normalized

    @staticmethod
    def _edge_iteration_distance(edge: DFGEdge) -> int:
        dist = edge.get_attribute("dist")
        if dist is not None:
            try:
                return int(dist)
            except (TypeError, ValueError):
                return 0
        return 1 if edge.is_loop_back else 0

    def _network_latency_lower(self, edge: DFGEdge) -> int:
        try:
            return self._latency_spec.get_network_latency_lower(
                edge.source.operation,
                edge.destination.operation,
            )
        except Exception:
            return 0

    def _network_latency_upper(self, edge: DFGEdge) -> int:
        try:
            return self._latency_spec.get_network_latency_upper(
                edge.source.operation,
                edge.destination.operation,
            )
        except Exception:
            return 1_000_000

    def _dependency_lower_bound(self, edge: DFGEdge) -> int:
        try:
            op_latency = self._latency_spec.get_op_latency(edge.source.operation)
        except Exception:
            op_latency = 0
        return op_latency + self._network_latency_lower(edge)

    def _dependency_upper_bound(self, edge: DFGEdge) -> int:
        return self._network_latency_upper(edge)

    def _topological_order_without_recurrences(self) -> List[DFGNode]:
        """Return a stable topological order over zero-distance edges."""
        indegree: Dict[str, int] = {node.id: 0 for node in self._dfg.get_nodes()}
        for edge in self._dfg.get_edges():
            if self._edge_iteration_distance(edge) > 0:
                continue
            indegree[edge.destination.id] += 1

        queue = deque(
            sorted(
                (self._dfg.get_node(node_id) for node_id, degree in indegree.items() if degree == 0),
                key=lambda node: node.id if node is not None else "",
            )
        )
        order: List[DFGNode] = []
        while queue:
            node = queue.popleft()
            if node is None:
                continue
            order.append(node)
            succs = sorted(
                self._dfg.get_outgoing_edges(node.id),
                key=lambda edge: (edge.destination.id, edge.id),
            )
            for edge in succs:
                if self._edge_iteration_distance(edge) > 0:
                    continue
                indegree[edge.destination.id] -= 1
                if indegree[edge.destination.id] == 0:
                    succ_node = self._dfg.get_node(edge.destination.id)
                    if succ_node is not None:
                        queue.append(succ_node)

        if len(order) != self._dfg.num_nodes():
            scheduled_ids = {node.id for node in order}
            remaining = sorted(
                (
                    node
                    for node in self._dfg.get_nodes()
                    if node.id not in scheduled_ids
                ),
                key=lambda node: node.id,
            )
            order.extend(remaining)
        return order

    def _compute_asap_seed(self) -> Dict[str, int]:
        """Run the upgraded ASAP scheduler to seed modulo scheduling."""
        fixed_schedule = {
            self._dfg.get_node(node_id): time
            for node_id, time in self._fixed_schedule.items()
            if self._dfg.get_node(node_id) is not None
        }
        scheduler = ASAPScheduler(
            dfg=self._dfg,
            latency_spec=self._latency_spec,
            fixed_schedule=fixed_schedule,
            debug=self._debug,
        )
        scheduler.schedule()
        return {
            node.id: scheduler._asap_schedule[node]
            for node in self._dfg.get_nodes()
        }

    def _compute_alap_seed(self, asap_times: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Compute latency-aware ALAP and mobility over the acyclic seed graph."""
        topo_order = self._topological_order_without_recurrences()
        max_latency = max(asap_times.values()) if asap_times else 0
        alap_times: Dict[str, int] = {}

        for node in reversed(topo_order):
            succ_edges = [
                edge
                for edge in self._dfg.get_outgoing_edges(node.id)
                if self._edge_iteration_distance(edge) == 0
            ]
            if node.id in self._fixed_schedule:
                latest = self._fixed_schedule[node.id]
            elif not succ_edges:
                latest = max_latency
            else:
                latest = min(
                    alap_times.get(edge.destination.id, max_latency)
                    - self._dependency_lower_bound(edge)
                    for edge in succ_edges
                )
            alap_times[node.id] = max(asap_times[node.id], latest)

        mobility = {
            node.id: max(0, alap_times[node.id] - asap_times[node.id])
            for node in topo_order
        }
        return alap_times, mobility

    def _initialize_times(self, asap_times: Dict[str, int]) -> Dict[str, int]:
        """Build the starting time assignment for modulo scheduling."""
        times = dict(asap_times)
        for node_id, time in self._fixed_schedule.items():
            times[node_id] = time
        return times

    def _relax_lower_bounds(
        self,
        times: Dict[str, int],
        ii: int,
    ) -> Tuple[bool, str]:
        """Propagate dependence lower bounds and detect infeasible recurrences."""
        edges = sorted(self._dfg.get_edges(), key=lambda edge: edge.id)
        node_count = self._dfg.num_nodes()
        for _ in range(max(0, node_count - 1)):
            changed = False
            for edge in edges:
                src_id = edge.source.id
                dst_id = edge.destination.id
                required = (
                    times[src_id]
                    + self._dependency_lower_bound(edge)
                    - self._edge_iteration_distance(edge) * ii
                )
                if required <= times[dst_id]:
                    continue
                if dst_id in self._fixed_schedule and required > self._fixed_schedule[dst_id]:
                    return (
                        False,
                        f"Fixed node {dst_id} violates lower bound from {src_id} at II={ii}",
                    )
                times[dst_id] = required
                changed = True
            if not changed:
                return True, ""

        for edge in edges:
            src_id = edge.source.id
            dst_id = edge.destination.id
            required = (
                times[src_id]
                + self._dependency_lower_bound(edge)
                - self._edge_iteration_distance(edge) * ii
            )
            if required <= times[dst_id]:
                continue
            if dst_id in self._fixed_schedule and required > self._fixed_schedule[dst_id]:
                return (
                    False,
                    f"Fixed node {dst_id} violates lower bound from {src_id} at II={ii}",
                )
            return False, f"Detected infeasible II={ii} via lower-bound relaxation cycle"

        return True, ""

    def _upper_bound_violations(
        self,
        times: Dict[str, int],
        ii: int,
    ) -> List[Tuple[DFGEdge, int, int]]:
        """Return all current upper-bound violations."""
        violations: List[Tuple[DFGEdge, int, int]] = []
        for edge in sorted(self._dfg.get_edges(), key=lambda item: item.id):
            actual_gap = (
                times[edge.destination.id]
                - times[edge.source.id]
                + self._edge_iteration_distance(edge) * ii
            )
            upper_bound = self._dependency_upper_bound(edge)
            if actual_gap > upper_bound:
                violations.append((edge, actual_gap, upper_bound))
        return violations

    def _repair_upper_bounds(
        self,
        times: Dict[str, int],
        ii: int,
    ) -> Tuple[bool, str]:
        """Repair upper-bound violations by delaying source operations."""
        for _ in range(self._upper_bound_repair_limit):
            violations = self._upper_bound_violations(times, ii)
            if not violations:
                return True, ""

            edge, actual_gap, upper_bound = violations[0]
            src_id = edge.source.id
            if src_id in self._fixed_schedule:
                return (
                    False,
                    f"Upper-bound violation on {src_id}->{edge.destination.id} cannot be repaired: source fixed",
                )

            delta = actual_gap - upper_bound
            times[src_id] += delta
            ok, message = self._relax_lower_bounds(times, ii)
            if not ok:
                return False, message

        return False, "Upper-bound repair limit exhausted"

    def _resource_overuse(
        self,
        times: Dict[str, int],
        ii: int,
        resource_classes: Dict[str, str],
        capacities: Dict[int, Dict[str, int]],
    ) -> Tuple[Optional[Tuple[str, int]], List[str], int]:
        """Return the most overfull (resource_class, cycle) bucket."""
        buckets: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        for node in self._dfg.get_nodes():
            resource_class = resource_classes[node.id]
            try:
                op_lat = self._latency_spec.get_op_latency(node.operation)
            except ValueError:
                op_lat = 0
            occupy_cycles = max(op_lat, 1)
            for c in range(occupy_cycles):
                buckets[(resource_class, (times[node.id] + c) % ii)].append(node.id)

        worst_key: Optional[Tuple[str, int]] = None
        worst_nodes: List[str] = []
        worst_excess = 0
        for key, node_ids in buckets.items():
            resource_class, cycle = key
            capacity = capacities.get(cycle, {}).get(resource_class, 0)
            excess = len(node_ids) - capacity
            if excess > worst_excess:
                worst_excess = excess
                worst_key = key
                worst_nodes = sorted(node_ids)
        return worst_key, worst_nodes, worst_excess

    def _repair_resource_conflicts(
        self,
        times: Dict[str, int],
        ii: int,
        resource_classes: Dict[str, str],
        capacities: Dict[int, Dict[str, int]],
        mobility: Dict[str, int],
    ) -> Tuple[bool, str]:
        """Repair resource conflicts by shifting selected ops forward."""
        for _ in range(self._max_resource_repairs):
            bucket_key, node_ids, excess = self._resource_overuse(
                times=times,
                ii=ii,
                resource_classes=resource_classes,
                capacities=capacities,
            )
            if bucket_key is None or excess <= 0:
                return True, ""

            movable = [node_id for node_id in node_ids if node_id not in self._fixed_schedule]
            if not movable:
                resource_class, cycle = bucket_key
                return (
                    False,
                    f"Resource class {resource_class} overfull at cycle {cycle} with only fixed operations",
                )

            movable.sort(
                key=lambda node_id: (
                    mobility.get(node_id, 0),
                    times[node_id],
                    node_id,
                )
            )
            victim_id = movable[-1]
            times[victim_id] += 1

            ok, message = self._relax_lower_bounds(times, ii)
            if not ok:
                return False, message
            ok, message = self._repair_upper_bounds(times, ii)
            if not ok:
                return False, message

        return False, "Resource repair limit exhausted"

    def _build_schedule(
        self,
        ii: int,
        analysis: Dict[str, object],
        asap_times: Dict[str, int],
        alap_times: Dict[str, int],
        mobility: Dict[str, int],
        times: Dict[str, int],
    ) -> ModuloSchedule:
        """Build the accepted schedule object."""
        diagnostics = dict(analysis)
        diagnostics["fixed_schedule"] = dict(self._fixed_schedule)
        diagnostics["final_schedule"] = dict(times)
        return ModuloSchedule(
            ii=ii,
            node_times=dict(times),
            asap_times=dict(asap_times),
            alap_times=dict(alap_times),
            mobility=dict(mobility),
            diagnostics=diagnostics,
        )

    def schedule(self, ii: Optional[int] = None) -> SchedulingResult:
        """Attempt modulo scheduling for the current MRRG II."""
        requested_ii = self._mrrg.II if ii is None else ii
        analysis = self._ii_analyzer.analyze_ii(requested_ii)

        if requested_ii > self._mrrg.II:
            attempt = SchedulingAttempt(
                ii=requested_ii,
                success=False,
                failure_reason=SchedulingFailureReason.INVALID_II,
                message=(
                    f"Requested II ({requested_ii}) does not match MRRG II ({self._mrrg.II})"
                ),
                diagnostics=analysis,
            )
            return SchedulingResult(
                success=False,
                attempts=[attempt],
                failure_reason=SchedulingFailureReason.INVALID_II,
                message=attempt.message,
            )

        unsupported_nodes = list(analysis.get("unsupported_nodes", []))
        if unsupported_nodes:
            message = (
                "No compatible FU resource exists for operations: "
                + ", ".join(sorted(unsupported_nodes))
            )
            attempt = SchedulingAttempt(
                ii=requested_ii,
                success=False,
                failure_reason=SchedulingFailureReason.UNSUPPORTED,
                message=message,
                diagnostics=analysis,
            )
            return SchedulingResult(
                success=False,
                attempts=[attempt],
                failure_reason=SchedulingFailureReason.UNSUPPORTED,
                message=message,
            )

        min_ii = int(analysis["min_ii"])
        if requested_ii < min_ii:
            reason = (
                SchedulingFailureReason.RESOURCE_CONFLICT
                if int(analysis["res_mii"]) >= int(analysis["rec_mii"])
                else SchedulingFailureReason.NEGATIVE_CYCLE
            )
            attempt = SchedulingAttempt(
                ii=requested_ii,
                success=False,
                failure_reason=reason,
                message=(
                    f"II={requested_ii} is below minimum feasible II={min_ii} "
                    f"(ResMII={analysis['res_mii']}, RecMII={analysis['rec_mii']})"
                ),
                diagnostics=analysis,
            )
            return SchedulingResult(
                success=False,
                attempts=[attempt],
                failure_reason=reason,
                message=attempt.message,
            )

        try:
            try:
                asap_times = self._compute_asap_seed()
                alap_times, mobility = self._compute_alap_seed(asap_times)
            except Exception:
                if len(self._fixed_schedule) != self._dfg.num_nodes():
                    raise
                asap_times = dict(self._fixed_schedule)
                alap_times = dict(self._fixed_schedule)
                mobility = {node_id: 0 for node_id in self._fixed_schedule}
            times = self._initialize_times(asap_times)
            ok, message = self._relax_lower_bounds(times, requested_ii)
            if not ok:
                attempt = SchedulingAttempt(
                    ii=requested_ii,
                    success=False,
                    failure_reason=SchedulingFailureReason.NEGATIVE_CYCLE,
                    message=message,
                    node_times=dict(times),
                    diagnostics=analysis,
                )
                return SchedulingResult(
                    success=False,
                    attempts=[attempt],
                    failure_reason=SchedulingFailureReason.NEGATIVE_CYCLE,
                    message=message,
                )

            ok, message = self._repair_upper_bounds(times, requested_ii)
            if not ok:
                attempt = SchedulingAttempt(
                    ii=requested_ii,
                    success=False,
                    failure_reason=SchedulingFailureReason.UPPER_BOUND_VIOLATION,
                    message=message,
                    node_times=dict(times),
                    diagnostics=analysis,
                )
                return SchedulingResult(
                    success=False,
                    attempts=[attempt],
                    failure_reason=SchedulingFailureReason.UPPER_BOUND_VIOLATION,
                    message=message,
                )

            resource_classes = analysis["operation_resource_classes"]
            capacities = analysis["cycle_resource_capacities"]
            ok, message = self._repair_resource_conflicts(
                times=times,
                ii=requested_ii,
                resource_classes=resource_classes,
                capacities=capacities,
                mobility=mobility,
            )
            if not ok:
                attempt = SchedulingAttempt(
                    ii=requested_ii,
                    success=False,
                    failure_reason=SchedulingFailureReason.RESOURCE_CONFLICT,
                    message=message,
                    node_times=dict(times),
                    diagnostics=analysis,
                )
                return SchedulingResult(
                    success=False,
                    attempts=[attempt],
                    failure_reason=SchedulingFailureReason.RESOURCE_CONFLICT,
                    message=message,
                )

            schedule = self._build_schedule(
                ii=requested_ii,
                analysis=analysis,
                asap_times=asap_times,
                alap_times=alap_times,
                mobility=mobility,
                times=times,
            )
            attempt = SchedulingAttempt(
                ii=requested_ii,
                success=True,
                node_times=dict(times),
                diagnostics=schedule.diagnostics,
            )
            return SchedulingResult(
                success=True,
                schedule=schedule,
                attempts=[attempt],
            )
        except Exception as exc:
            attempt = SchedulingAttempt(
                ii=requested_ii,
                success=False,
                failure_reason=SchedulingFailureReason.SEARCH_EXHAUSTED,
                message=str(exc),
                diagnostics=analysis,
            )
            return SchedulingResult(
                success=False,
                attempts=[attempt],
                failure_reason=SchedulingFailureReason.SEARCH_EXHAUSTED,
                message=str(exc),
            )
