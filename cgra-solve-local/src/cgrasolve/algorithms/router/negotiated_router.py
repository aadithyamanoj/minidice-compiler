from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Set

from cgrasolve.algorithms.router.candidate_generator import CandidateGenerator
from cgrasolve.algorithms.router.route_plans import RoutePlan
from cgrasolve.algorithms.router.routing_demands import build_routing_demands
from cgrasolve.algorithms.router.routing_state import RoutingState
from cgrasolve.algorithms.router.types import HyperValNetInfo, RoutingResult
from cgrasolve.graph.hyperdfg import HyperVal


class NegotiatedRouter:
    """Route orchestration built on RoutePlan generation and plan-level commits."""

    COOPT_LOOKAHEAD = 12

    def __init__(self, router: "PathFinder") -> None:
        self._router = router
        self._candidates = CandidateGenerator(router)
        self._state = RoutingState(router)
        self._root_eligibility_cache: Dict[HyperVal, bool] = {}
        self._packing_eligibility_cache: Dict[HyperVal, bool] = {}
        self._cohort_by_hyperval: Dict[HyperVal, FrozenSet[HyperVal]] = {}

    def route_initial_hypervals(self, hypervals: Sequence[HyperValNetInfo]) -> None:
        """Run the initial routing pass, including shared packed cohorts."""
        self._route_hypervals(hypervals)

    def reroute_hypervals(self, hypervals: Sequence[HyperValNetInfo]) -> None:
        """Rip up and reroute only dirty HyperVals (and their full cohorts).

        A HyperVal is dirty when it has uncovered sinks or conflicts with
        other routes.  When a dirty HyperVal belongs to a shared packed-carrier
        cohort, the entire cohort is ripped up so the shared plan can be
        rebuilt together.  Legal, non-conflicted HyperVals are left in place.
        """
        dirty_hypervals: Set[HyperVal] = set()
        for hv_info in hypervals:
            if hv_info.uncovered_sinks > 0 or hv_info.conflict_score > 0:
                cohort = self._cohort_by_hyperval.get(
                    hv_info.hyperval, frozenset({hv_info.hyperval}),
                )
                dirty_hypervals.update(cohort)

        if not dirty_hypervals:
            return

        reroute_list = [
            hv_info for hv_info in hypervals
            if hv_info.hyperval in dirty_hypervals
        ]

        ripped_up: Set[HyperVal] = set()
        for hv_info in reroute_list:
            hyperval = hv_info.hyperval
            if hyperval in ripped_up:
                continue
            if self._deadline_exceeded():
                return
            cohort = self._cohort_by_hyperval.get(
                hyperval, frozenset({hyperval}),
            )
            self._clear_cohort_metadata(cohort)
            for member in cohort:
                self._router._rip_up_hyperval(member)
                ripped_up.add(member)

        self._route_hypervals(reroute_list)

    def _route_hypervals(self, hypervals: Sequence[HyperValNetInfo]) -> None:
        """Route HyperVals in order, co-optimizing shared carriers when profitable."""
        handled: Set[HyperVal] = set()
        ordered = list(hypervals)
        for index, hyperval_info in enumerate(ordered):
            if self._deadline_exceeded():
                return
            hyperval = hyperval_info.hyperval
            if hyperval in handled:
                continue
            if all((hyperval, dest_idx) in self._router._routing_solution for dest_idx in range(hyperval.cardinality)):
                handled.add(hyperval)
                continue

            plan = self._maybe_build_shared_carrier_plan(hyperval, ordered[index + 1 :])
            if plan is None:
                plan = self._candidates.build_hyperval_plan(hyperval)
            if plan is None:
                continue

            self._state.commit_plan(plan)
            self._record_committed_plan(plan)
            handled.update(plan.hypervals())

    def _record_committed_plan(self, plan: RoutePlan) -> None:
        """Record which HyperVals were committed together as a cohort."""
        cohort = frozenset(plan.hypervals())
        for hyperval in cohort:
            self._cohort_by_hyperval[hyperval] = cohort

    def _clear_cohort_metadata(self, cohort: FrozenSet[HyperVal]) -> None:
        """Remove cohort tracking for all members of a cohort."""
        for member in cohort:
            self._cohort_by_hyperval.pop(member, None)

    def _deadline_exceeded(self) -> bool:
        """Return True when the wrapped router exposes and reports a deadline expiry."""
        deadline_check = getattr(self._router, "_deadline_exceeded", None)
        if deadline_check is None:
            return False
        return bool(deadline_check())

    def _maybe_build_shared_carrier_plan(
        self,
        hyperval: HyperVal,
        remaining: Sequence[HyperValNetInfo],
    ) -> Optional[RoutePlan]:
        """Try to build a shared packed plan for the current HyperVal."""
        if hyperval.cardinality != 1:
            return None

        if not self._is_packed_share_eligible(hyperval):
            return None

        candidate_pool = [
            hv_info.hyperval
            for hv_info in remaining
            if hv_info.hyperval.cardinality == 1
            and (hv_info.hyperval, 0) not in self._router._routing_solution
            and self._is_packed_share_eligible(hv_info.hyperval)
        ]
        candidate_pool = candidate_pool[: max(0, self.COOPT_LOOKAHEAD - 1)]

        root_candidates: List[HyperVal] = []
        if self._is_root_eligible(hyperval):
            root_candidates.append(hyperval)
        for candidate in candidate_pool:
            if candidate in root_candidates:
                continue
            if not self._is_root_eligible(candidate):
                continue
            root_candidates.append(candidate)
            if len(root_candidates) >= 2:
                break
        if not root_candidates:
            return None

        best_plan: Optional[RoutePlan] = None
        best_score = float("inf")
        for root_hyperval in root_candidates:
            companions = [
                candidate
                for candidate in [hyperval, *candidate_pool]
                if candidate != root_hyperval
            ]
            shared_plan = self._candidates.build_shared_carrier_plan(
                root_hyperval,
                companions,
            )
            if shared_plan is None or hyperval not in shared_plan.hypervals():
                continue
            if shared_plan.score < best_score:
                best_score = shared_plan.score
                best_plan = shared_plan
        return best_plan

    def _is_packed_share_eligible(self, hyperval: HyperVal) -> bool:
        """Return True when a HyperVal could participate in packed cohabitation."""
        cached = self._packing_eligibility_cache.get(hyperval)
        if cached is not None:
            return cached

        demands = build_routing_demands(self._router, hyperval)
        if len(demands) != 1:
            self._packing_eligibility_cache[hyperval] = False
            return False
        demand = demands[0]
        bitwidth = demand.sink.bitwidth
        if bitwidth is None or bitwidth <= 0:
            self._packing_eligibility_cache[hyperval] = False
            return False
        eligible = any(
            self._router._mrrg.get_packing_endpoints_for_output_port(source_port)
            for source_port in demand.source_ports
        )
        self._packing_eligibility_cache[hyperval] = eligible
        return eligible

    def _is_root_eligible(self, hyperval: HyperVal) -> bool:
        """Return True when a HyperVal can originate a shared packed carrier."""
        cached = self._root_eligibility_cache.get(hyperval)
        if cached is not None:
            return cached

        demands = build_routing_demands(self._router, hyperval)
        if len(demands) != 1:
            self._root_eligibility_cache[hyperval] = False
            return False
        demand = demands[0]
        bitwidth = demand.sink.bitwidth
        if bitwidth is None or bitwidth <= 0:
            self._root_eligibility_cache[hyperval] = False
            return False
        eligible = any(
            not endpoint.requires_existing_carrier and endpoint.lane_width >= bitwidth
            for source_port in demand.source_ports
            for endpoint in self._router._mrrg.get_packing_endpoints_for_output_port(source_port)
        )
        self._root_eligibility_cache[hyperval] = eligible
        return eligible


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cgrasolve.algorithms.router.pathfinder import PathFinder
