from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cgrasolve.algorithms.router.route_plans import RoutePlan, SharedCarrierPlan
from cgrasolve.algorithms.router.routing_demands import RoutingDemand, build_routing_demands
from cgrasolve.algorithms.router.routing_state import RoutingState
from cgrasolve.algorithms.router.types import RouteBundle, TransportMode, TransportProfile, extract_predecessors
from cgrasolve.graph.hyperdfg import HyperVal
from cgrasolve.graph.mrrg import MRRGNode


@dataclass(frozen=True)
class SharedCarrierCandidate:
    """Temporary selection candidate while greedily building a shared carrier."""

    demand: RoutingDemand
    plan: RoutePlan
    solo_score: float
    improvement: float
    shared_node_ids: Tuple[str, ...]


class CandidateGenerator:
    """Generate route plans from normalized routing demands."""

    MAX_SHARED_LANES = 4
    MAX_ROOT_CANDIDATES = 4
    MAX_COMPANION_WINDOW = 12
    SHARED_NODE_BONUS = 0.35
    SHARED_ROUTE_BONUS = 0.5

    def __init__(self, router: "PathFinder") -> None:
        self._router = router
        self._state = RoutingState(router)

    def build_hyperval_plan(self, hyperval: HyperVal) -> Optional[RoutePlan]:
        """Build but do not commit the best plan for all sinks of one HyperVal."""
        demands = build_routing_demands(self._router, hyperval)
        if not demands:
            return None

        claimed_nodes: Set[MRRGNode] = set()
        claimed_predecessors: Dict[MRRGNode, MRRGNode] = {}
        claimed_lane_assignments: Dict[str, int] = {}
        claimed_full_width_nodes: Set[str] = set()
        sink_plans: List[RoutePlan] = []

        for demand in demands:
            best_plan = self._pick_best_singleton_plan(
                demand,
                claimed_nodes=claimed_nodes,
                claimed_predecessors=claimed_predecessors,
                claimed_lane_assignments=claimed_lane_assignments,
                claimed_full_width_nodes=claimed_full_width_nodes,
            )
            if best_plan is None:
                self._router._last_route_failure = {
                    "source_id": hyperval.source_id,
                    "dest_id": hyperval.destination_ids[demand.sink.dest_idx],
                    "sink_idx": demand.sink.dest_idx,
                    "token_bitwidth": demand.sink.bitwidth,
                    "cycles_to_sink": demand.sink.cycles_to_sink,
                    "fanout_candidates": len(demand.source_ports),
                    "flat_candidates": 0,
                    "packed_candidates": 0,
                    "fractured_candidates": 0,
                    "flat_stats": {},
                    "packed_stats": {},
                    "fractured_stats": {},
                }
                return None

            sink_plans.append(best_plan)
            bundle = next(iter(best_plan.bundles.values()))
            if bundle.transport_profile.mode == TransportMode.PACKED:
                for node_id, lane_index in bundle.lane_assignments.items():
                    existing_lane = claimed_lane_assignments.get(node_id)
                    if existing_lane is None:
                        claimed_lane_assignments[node_id] = lane_index
                        continue
                    if existing_lane != lane_index:
                        return None
            else:
                for node in bundle.all_nodes:
                    if node.num_lanes > 1:
                        claimed_full_width_nodes.add(node.id)
            claimed_nodes.update(bundle.all_nodes)
            claimed_predecessors.update(extract_predecessors(bundle))

        return RoutePlan.merge(sink_plans)

    def build_shared_carrier_plan(
        self,
        root_hyperval: HyperVal,
        companion_hypervals: Sequence[HyperVal],
    ) -> Optional[SharedCarrierPlan]:
        """Build a co-optimized packed plan rooted at one single-sink HyperVal."""
        if root_hyperval.cardinality != 1:
            return None

        root_demands = build_routing_demands(self._router, root_hyperval)
        if len(root_demands) != 1:
            return None
        root_demand = root_demands[0]

        root_candidates = [
            plan
            for plan in self._build_singleton_plans(root_demand)
            if next(iter(plan.bundles.values())).transport_profile.mode == TransportMode.PACKED
        ]
        if not root_candidates:
            return None

        best_shared_plan: Optional[SharedCarrierPlan] = None
        best_shared_score = float("inf")

        for root_plan in sorted(root_candidates, key=lambda plan: plan.score)[: self.MAX_ROOT_CANDIDATES]:
            root_bundle = next(iter(root_plan.bundles.values()))
            max_shared_routes = min(
                self.MAX_SHARED_LANES,
                max((node.num_lanes for node in root_bundle.flat_path if node.num_lanes > 1), default=1),
            )
            selected_plans: List[RoutePlan] = [root_plan]
            selected_shared_nodes: Set[str] = set()
            companion_solo_scores: Dict[RouteKey, float] = {}
            for hyperval in companion_hypervals[: self.MAX_COMPANION_WINDOW]:
                if hyperval.cardinality != 1:
                    continue
                companion_demands = build_routing_demands(self._router, hyperval)
                if len(companion_demands) != 1:
                    continue
                solo_plan = self._pick_best_singleton_plan(companion_demands[0])
                if solo_plan is None:
                    continue
                companion_solo_scores[companion_demands[0].route_key] = solo_plan.score

            with self._state.temporary_plan(root_plan):
                companions: List[RoutingDemand] = []
                for hyperval in companion_hypervals[: self.MAX_COMPANION_WINDOW]:
                    if hyperval.cardinality != 1:
                        continue
                    companion_demands = build_routing_demands(self._router, hyperval)
                    if len(companion_demands) != 1:
                        continue
                    if companion_demands[0].route_key not in companion_solo_scores:
                        continue
                    companions.append(companion_demands[0])
                remaining = list(companions)
                while remaining and len(selected_plans) < max_shared_routes:
                    best_companion = self._pick_best_shared_companion(
                        remaining,
                        selected_plans,
                        companion_solo_scores,
                    )
                    if best_companion is None:
                        break
                    selected_plans.append(best_companion.plan)
                    selected_shared_nodes.update(best_companion.shared_node_ids)
                    remaining = [
                        demand
                        for demand in remaining
                        if demand.route_key != best_companion.demand.route_key
                    ]
                    self._state.commit_plan(best_companion.plan)

                for plan in reversed(selected_plans[1:]):
                    self._state.rip_up_plan(plan)

            if len(selected_plans) <= 1:
                continue

            score_adjustment = -(
                self.SHARED_NODE_BONUS * len(selected_shared_nodes)
                + self.SHARED_ROUTE_BONUS * (len(selected_plans) - 1)
            )
            merged = SharedCarrierPlan.merge(
                selected_plans,
                score_adjustment=score_adjustment,
                shared_carrier=True,
                shared_node_ids=selected_shared_nodes,
            )
            shared_plan = SharedCarrierPlan(
                routes=merged.routes,
                score=merged.score,
                shared_carrier=True,
                shared_node_ids=merged.shared_node_ids,
                root_route_key=root_demand.route_key,
                lane_map={
                    route_key: planned.bundle.transport_profile.lane_index or 0
                    for route_key, planned in merged.routes.items()
                },
            )
            if shared_plan.score < best_shared_score:
                best_shared_score = shared_plan.score
                best_shared_plan = shared_plan

        return best_shared_plan

    def _pick_best_shared_companion(
        self,
        demands: Sequence[RoutingDemand],
        selected_plans: Sequence[RoutePlan],
        solo_scores: Dict[RouteKey, float],
    ) -> Optional[SharedCarrierCandidate]:
        """Return the best additional demand to attach to the current shared carrier."""
        best: Optional[SharedCarrierCandidate] = None
        for demand in demands:
            solo_score = solo_scores.get(demand.route_key)
            if solo_score is None:
                continue

            shared_candidates = [
                self._to_shared_candidate(demand, plan, solo_score, selected_plans)
                for plan in self._build_singleton_plans(demand)
            ]
            shared_candidates = [candidate for candidate in shared_candidates if candidate is not None]
            if not shared_candidates:
                continue
            candidate = min(shared_candidates, key=lambda item: item.plan.score)
            if best is None or candidate.improvement > best.improvement:
                best = candidate
        return best

    def _to_shared_candidate(
        self,
        demand: RoutingDemand,
        plan: RoutePlan,
        solo_score: float,
        selected_plans: Sequence[RoutePlan],
    ) -> Optional[SharedCarrierCandidate]:
        """Convert a singleton plan into a shared-carrier candidate when legal."""
        bundle = next(iter(plan.bundles.values()))
        if bundle.transport_profile.mode != TransportMode.PACKED:
            return None

        shared_node_ids = self._extract_shared_carrier_nodes(bundle, selected_plans)
        if not shared_node_ids:
            return None

        improvement = solo_score - (
            plan.score
            - self.SHARED_NODE_BONUS * len(shared_node_ids)
            - self.SHARED_ROUTE_BONUS
        )
        if improvement <= 0:
            return None

        return SharedCarrierCandidate(
            demand=demand,
            plan=plan,
            solo_score=solo_score,
            improvement=improvement,
            shared_node_ids=shared_node_ids,
        )

    def _extract_shared_carrier_nodes(
        self,
        bundle: RouteBundle,
        selected_plans: Sequence[RoutePlan],
    ) -> Tuple[str, ...]:
        """Return contiguous shared-carrier nodes when a bundle piggybacks legally."""
        if not bundle.active_pack_nodes:
            return ()
        pack_node_id = min(bundle.active_pack_nodes)
        pack_node = self._router._mrrg.get_node(pack_node_id)
        if pack_node is None or not pack_node.packing_requires_carrier:
            return ()

        candidate_path = bundle.flat_path
        try:
            candidate_start = next(
                idx for idx, node in enumerate(candidate_path) if node.id == pack_node_id
            )
        except StopIteration:
            return ()

        for selected in selected_plans:
            for existing_bundle in selected.bundles.values():
                existing_path = existing_bundle.flat_path
                try:
                    existing_start = next(
                        idx for idx, node in enumerate(existing_path) if node.id == pack_node_id
                    )
                except StopIteration:
                    continue
                shared_ids: List[str] = []
                cand_idx = candidate_start
                exist_idx = existing_start
                while cand_idx < len(candidate_path) and exist_idx < len(existing_path):
                    cand_node = candidate_path[cand_idx]
                    exist_node = existing_path[exist_idx]
                    if cand_node.id != exist_node.id:
                        break
                    if cand_node.num_lanes > 1:
                        shared_ids.append(cand_node.id)
                    cand_idx += 1
                    exist_idx += 1
                if len(shared_ids) >= 2:
                    return tuple(shared_ids)
        return ()

    def _pick_best_singleton_plan(
        self,
        demand: RoutingDemand,
        *,
        claimed_nodes: Optional[Set[MRRGNode]] = None,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        claimed_lane_assignments: Optional[Dict[str, int]] = None,
        claimed_full_width_nodes: Optional[Set[str]] = None,
    ) -> Optional[RoutePlan]:
        """Return the best singleton plan for one demand."""
        best_plan: Optional[RoutePlan] = None
        best_score = float("inf")
        for plan in self._build_singleton_plans(
            demand,
            claimed_nodes=claimed_nodes,
            claimed_predecessors=claimed_predecessors,
            claimed_lane_assignments=claimed_lane_assignments,
            claimed_full_width_nodes=claimed_full_width_nodes,
        ):
            bundle = next(iter(plan.bundles.values()))
            if self._router._candidate_beats_best(
                candidate_bundle=bundle,
                candidate_cost=plan.score,
                best_bundle=(
                    None if best_plan is None else next(iter(best_plan.bundles.values()))
                ),
                best_cost=best_score,
            ):
                best_plan = plan
                best_score = plan.score
        return best_plan

    def _build_singleton_plans(
        self,
        demand: RoutingDemand,
        *,
        claimed_nodes: Optional[Set[MRRGNode]] = None,
        claimed_predecessors: Optional[Dict[MRRGNode, MRRGNode]] = None,
        claimed_lane_assignments: Optional[Dict[str, int]] = None,
        claimed_full_width_nodes: Optional[Set[str]] = None,
    ) -> List[RoutePlan]:
        """Generate flat, fractured, and packed singleton plans for one demand."""
        claimed_nodes = set() if claimed_nodes is None else set(claimed_nodes)
        claimed_lane_assignments = {} if claimed_lane_assignments is None else dict(claimed_lane_assignments)
        claimed_full_width_nodes = set() if claimed_full_width_nodes is None else set(claimed_full_width_nodes)

        sink_info = demand.sink.to_sink_and_latency()
        route_key = demand.route_key
        plans: List[RoutePlan] = []
        flat_debug: Dict[str, int] = {}
        packed_debug: Dict[str, int] = {}
        fractured_debug: Dict[str, int] = {}

        flat_profile = TransportProfile.flat(demand.sink.bitwidth)
        for source_port in demand.source_ports:
            if flat_profile.rejects_node_width(source_port):
                self._router._bump_debug_counter(flat_debug, "reject_root_too_narrow")
                continue
            candidate_route = self._router._dijkstra_pathfinder(
                source=source_port,
                sink=demand.sink.sink_fu,
                tag_to_find=demand.sink.operand_tag,
                hyperval=demand.hyperval,
                routes=claimed_nodes,
                cycles_to_sink=demand.sink.cycles_to_sink,
                latency_of_src=demand.latency_of_src,
                transport_profile=flat_profile,
                dest_op=demand.sink.dest_dfg_node.operation,
                claimed_predecessors=claimed_predecessors,
                debug_stats=flat_debug,
            )
            if not candidate_route:
                continue
            if self._router._route_conflicts_with_speculative_transport(
                {source_port, *candidate_route},
                flat_profile,
                claimed_lane_assignments,
                claimed_full_width_nodes,
            ):
                self._router._bump_debug_counter(
                    flat_debug,
                    "reject_speculative_transport_conflict",
                )
                continue
            bundle = RouteBundle.from_flat_path(candidate_route, transport_profile=flat_profile)
            score = self._router._score_candidate_route(
                fanout=source_port,
                route=candidate_route,
                transport_profile=flat_profile,
                hyperval=demand.hyperval,
                speculative_hv_nodes=claimed_nodes,
            )
            plans.append(RoutePlan.from_bundle(route_key, bundle, score))

        for bundle, score in self._router._build_packed_candidates(
            hyperval=demand.hyperval,
            src_fu=demand.source_fu,
            src_operation=demand.source_dfg_node.operation,
            sink_info=sink_info,
            token_bitwidth=demand.sink.bitwidth,
            cycles_to_sink=demand.sink.cycles_to_sink,
            speculative_hv_nodes=claimed_nodes,
            latency_of_src=demand.latency_of_src,
            speculative_lane_assignments=claimed_lane_assignments,
            speculative_full_width_nodes=claimed_full_width_nodes,
            claimed_predecessors=claimed_predecessors,
            debug_reasons=packed_debug,
        ):
            plans.append(RoutePlan.from_bundle(route_key, bundle, score))

        for bundle, score in self._router._build_fractured_candidates(
            hyperval=demand.hyperval,
            src_fu=demand.source_fu,
            src_operation=demand.source_dfg_node.operation,
            sink_info=sink_info,
            token_bitwidth=demand.sink.bitwidth,
            cycles_to_sink=demand.sink.cycles_to_sink,
            speculative_hv_nodes=claimed_nodes,
            latency_of_src=demand.latency_of_src,
            claimed_predecessors=claimed_predecessors,
            debug_reasons=fractured_debug,
        ):
            plans.append(RoutePlan.from_bundle(route_key, bundle, score))

        return plans


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cgrasolve.algorithms.router.pathfinder import PathFinder
