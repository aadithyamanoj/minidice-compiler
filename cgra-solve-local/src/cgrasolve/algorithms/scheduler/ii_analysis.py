"""Initiation interval analysis helpers for modulo scheduling."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Set, Tuple

from cgrasolve.graph.dfg import DFG, DFGEdge, DFGNode
from cgrasolve.graph.mrrg import MRRG, MRRGNode, NodeType

# Type alias for a bitwidth-qualified resource key:
# (resource_class, operation, required_output_bw, required_input_bw)
_ResourceKey = Tuple[str, "OperationType", Optional[int], Optional[int]]
from cgrasolve.graph.utils.analysis import find_strongly_connected_components
from cgrasolve.schedules import LatencySpecification


@dataclass(frozen=True)
class RecurrenceConstraint:
    """Derived recurrence-pressure constraint from one loop-carried edge."""

    source_id: str
    destination_id: str
    iteration_distance: int
    latency: int
    recurrence_mii: int
    scc_nodes: Tuple[str, ...]


class IIAnalyzer:
    """Compute ResMII/RecMII and per-II diagnostics."""

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        latency_spec: LatencySpecification,
    ) -> None:
        self._dfg = dfg
        self._mrrg = mrrg
        self._latency_spec = latency_spec
        self._unique_fus = self._collect_unique_fus()
        self._resource_classes = self.resolve_operation_resource_classes()
        self._node_resource_keys = self._resolve_node_resource_keys()

    def _collect_unique_fus(self) -> Dict[str, MRRGNode]:
        """Return one representative FU per physical/base FU."""
        unique: Dict[str, MRRGNode] = {}
        for fu in self._mrrg.get_fu_nodes():
            unique.setdefault(fu.get_base_name(), fu)
        return unique

    @staticmethod
    def _edge_iteration_distance(edge: DFGEdge) -> int:
        """Return the loop iteration distance carried by ``edge``."""
        dist = edge.get_attribute("dist")
        if dist is not None:
            try:
                return int(dist)
            except (TypeError, ValueError):
                return 0
        return 1 if edge.is_loop_back else 0

    def _dependency_latency(self, edge: DFGEdge) -> int:
        """Return the lower-bound latency contribution for one edge."""
        try:
            op_latency = self._latency_spec.get_op_latency(edge.source.operation)
        except Exception:
            op_latency = 0

        try:
            net_latency = self._latency_spec.get_network_latency_lower(
                edge.source.operation,
                edge.destination.operation,
            )
        except Exception:
            net_latency = 0

        return op_latency + net_latency

    def _compatible_resource_classes(self, node: DFGNode) -> List[str]:
        """Return resource classes on which ``node`` can execute."""
        resource_classes: Set[str] = set()
        for fu in self._unique_fus.values():
            if not fu.can_execute(node.operation):
                continue
            resource_classes.add(fu.get_resource_class_for_operation(node.operation))

        if not resource_classes:
            return [node.operation.value]
        return sorted(resource_classes)

    def resolve_operation_resource_classes(self) -> Dict[str, str]:
        """Resolve a canonical scheduler resource class for each DFG node."""
        resolved: Dict[str, str] = {}
        for node in self._dfg.get_nodes():
            resolved[node.id] = self._compatible_resource_classes(node)[0]
        return resolved

    def _resolve_node_resource_keys(self) -> Dict[str, _ResourceKey]:
        """Resolve a bitwidth-qualified resource key for each DFG node.

        Returns a mapping from DFG node ID to
        ``(resource_class, operation, required_output_bw, required_input_bw)``.
        """
        keys: Dict[str, _ResourceKey] = {}
        for node in self._dfg.get_nodes():
            rc = self._resource_classes[node.id]

            # Output requirement: node bitwidth attribute
            node_bw = node.get_attribute("bitwidth")
            req_out = node_bw if isinstance(node_bw, int) and node_bw > 0 else None

            # Input requirement: max incoming edge bitwidth
            incoming = self._dfg.get_incoming_edges(node.id)
            in_bws = [
                e.get_attribute("bitwidth")
                for e in incoming
                if isinstance(e.get_attribute("bitwidth"), int)
                and e.get_attribute("bitwidth") > 0
            ]
            req_in = max(in_bws) if in_bws else None

            keys[node.id] = (rc, node.operation, req_out, req_in)
        return keys

    @staticmethod
    def _fu_bw_compatible(
        fu: MRRGNode,
        op_type: "OperationType",
        req_out: Optional[int],
        req_in: Optional[int],
    ) -> bool:
        """Check whether an FU meets both output and input bitwidth requirements."""
        if req_out is not None:
            fu_out = fu.operation_output_bitwidths.get(op_type)
            if fu_out is not None and fu_out < req_out:
                return False
        if req_in is not None:
            fu_in = fu.operation_max_input_bitwidth.get(op_type)
            if fu_in is not None and fu_in < req_in:
                return False
        return True

    def compute_resource_class_supply(self) -> Dict[str, int]:
        """Return physical FU supply by resource class (without bitwidth filtering)."""
        supply: Counter[str] = Counter()
        for fu in self._unique_fus.values():
            seen_for_fu: Set[str] = set()
            for op in fu.supported_operations:
                resource_class = fu.get_resource_class_for_operation(op)
                if resource_class in seen_for_fu:
                    continue
                seen_for_fu.add(resource_class)
                supply[resource_class] += 1
        return dict(supply)

    def compute_bitwidth_aware_supply(self) -> Dict[_ResourceKey, int]:
        """Return FU supply per bitwidth-qualified resource key.

        For each unique ``(resource_class, operation, req_out, req_in)`` group
        found in the DFG, count how many unique physical FUs are compatible —
        i.e., support the specific operation, share the resource class, **and**
        satisfy both output and input bitwidth checks (``fu_bw >= required_bw``
        on each side independently).
        """
        groups = set(self._node_resource_keys.values())
        supply: Dict[_ResourceKey, int] = {}

        for group in groups:
            rc, op_type, req_out, req_in = group
            count = 0
            for fu in self._unique_fus.values():
                if not fu.can_execute(op_type):
                    continue
                if fu.get_resource_class_for_operation(op_type) != rc:
                    continue
                if not self._fu_bw_compatible(fu, op_type, req_out, req_in):
                    continue
                count += 1
            supply[group] = count

        return supply

    def compute_cycle_resource_capacities(self) -> Dict[int, Dict[str, int]]:
        """Return available FU capacity by (cycle, resource class)."""
        capacities: Dict[int, Dict[str, int]] = {}
        for cycle in range(self._mrrg.II):
            class_counter: Counter[str] = Counter()
            for fu in self._mrrg.get_nodes_at_cycle(cycle):
                if fu.node_type != NodeType.FUNCTION:
                    continue
                seen_for_fu: Set[str] = set()
                for op in fu.supported_operations:
                    resource_class = fu.get_resource_class_for_operation(op)
                    if resource_class in seen_for_fu:
                        continue
                    seen_for_fu.add(resource_class)
                    class_counter[resource_class] += fu.capacity
            capacities[cycle] = dict(class_counter)
        return capacities

    def compute_res_mii(self) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute bitwidth-aware resource-based ResMII.

        Groups DFG nodes by ``(resource_class, required_output_bw,
        required_input_bw)`` and counts only FUs that are bitwidth-compatible
        for each group.  This correctly handles mixed-precision architectures
        where FUs of different bitwidths share the same resource class.
        """
        demand: Counter[_ResourceKey] = Counter(self._node_resource_keys.values())
        supply = self.compute_bitwidth_aware_supply()

        details: Dict[str, Dict[str, int]] = {}
        res_mii = 1
        for group, count in demand.items():
            rc, op_type, req_out, req_in = group
            available = supply.get(group, 0)
            required = 0 if available == 0 else ceil(count / available)
            if available > 0:
                res_mii = max(res_mii, required)
            # Use a descriptive label for the details key
            if req_out is None and req_in is None:
                label = rc
            else:
                label = f"{rc}:{op_type.value}(out={req_out},in={req_in})"
            details[label] = {
                "demand": count,
                "supply": available,
                "required_ii": required,
            }
        return res_mii, details

    def _zero_distance_path_latency(
        self,
        source_id: str,
        destination_id: str,
        allowed_nodes: Set[str],
    ) -> int:
        """Return the longest zero-distance path latency between two nodes."""
        if source_id == destination_id:
            return 0

        memo: Dict[str, Optional[int]] = {}
        visiting: Set[str] = set()

        def _dfs(node_id: str) -> Optional[int]:
            if node_id == destination_id:
                return 0
            if node_id in memo:
                return memo[node_id]
            if node_id in visiting:
                return None

            visiting.add(node_id)
            best: Optional[int] = None
            for edge in self._dfg.get_outgoing_edges(node_id):
                if self._edge_iteration_distance(edge) != 0:
                    continue
                succ_id = edge.destination.id
                if succ_id not in allowed_nodes:
                    continue
                tail = _dfs(succ_id)
                if tail is None:
                    continue
                candidate = self._dependency_latency(edge) + tail
                if best is None or candidate > best:
                    best = candidate
            visiting.remove(node_id)
            memo[node_id] = best
            return best

        result = _dfs(source_id)
        return 0 if result is None else result

    def compute_rec_mii(self) -> Tuple[int, List[RecurrenceConstraint]]:
        """Compute recurrence-constrained minimum II."""
        sccs = find_strongly_connected_components(self._dfg)
        scc_by_node: Dict[str, Tuple[str, ...]] = {}
        for scc in sccs:
            scc_tuple = tuple(sorted(scc))
            for node_id in scc:
                scc_by_node[node_id] = scc_tuple

        constraints: List[RecurrenceConstraint] = []
        rec_mii = 1
        for edge in self._dfg.get_edges():
            dist = self._edge_iteration_distance(edge)
            if dist <= 0:
                continue

            scc_nodes = set(scc_by_node.get(edge.source.id, (edge.source.id,)))
            latency = self._dependency_latency(edge)
            path_latency = self._zero_distance_path_latency(
                edge.destination.id,
                edge.source.id,
                scc_nodes,
            )
            total_latency = latency + path_latency
            required_ii = max(1, ceil(total_latency / dist))
            constraint = RecurrenceConstraint(
                source_id=edge.source.id,
                destination_id=edge.destination.id,
                iteration_distance=dist,
                latency=total_latency,
                recurrence_mii=required_ii,
                scc_nodes=tuple(sorted(scc_nodes)),
            )
            constraints.append(constraint)
            rec_mii = max(rec_mii, required_ii)

        return rec_mii, constraints

    def analyze_ii(self, ii: int) -> Dict[str, Any]:
        """Return a consolidated II analysis dictionary."""
        res_mii, resource_pressure = self.compute_res_mii()
        rec_mii, recurrence_constraints = self.compute_rec_mii()
        unsupported_nodes = [
            node.id
            for node in self._dfg.get_nodes()
            if not any(fu.can_execute(node.operation) for fu in self._unique_fus.values())
        ]

        limiting_rec = None
        if recurrence_constraints:
            limiting_rec = max(
                recurrence_constraints,
                key=lambda item: item.recurrence_mii,
            )

        limiting_classes = [
            resource_class
            for resource_class, details in resource_pressure.items()
            if details["required_ii"] == res_mii and details["supply"] > 0
        ]

        return {
            "requested_ii": ii,
            "operation_resource_classes": dict(self._resource_classes),
            "resource_pressure": resource_pressure,
            "cycle_resource_capacities": self.compute_cycle_resource_capacities(),
            "recurrence_constraints": [
                {
                    "source_id": constraint.source_id,
                    "destination_id": constraint.destination_id,
                    "iteration_distance": constraint.iteration_distance,
                    "latency": constraint.latency,
                    "recurrence_mii": constraint.recurrence_mii,
                    "scc_nodes": list(constraint.scc_nodes),
                }
                for constraint in recurrence_constraints
            ],
            "res_mii": res_mii,
            "rec_mii": rec_mii,
            "min_ii": max(res_mii, rec_mii),
            "unsupported_nodes": unsupported_nodes,
            "limiting_resource_classes": limiting_classes,
            "limiting_recurrence": (
                None
                if limiting_rec is None
                else {
                    "source_id": limiting_rec.source_id,
                    "destination_id": limiting_rec.destination_id,
                    "iteration_distance": limiting_rec.iteration_distance,
                    "latency": limiting_rec.latency,
                    "recurrence_mii": limiting_rec.recurrence_mii,
                    "scc_nodes": list(limiting_rec.scc_nodes),
                }
            ),
            "ii_is_feasible_by_bounds": ii >= max(res_mii, rec_mii),
        }
