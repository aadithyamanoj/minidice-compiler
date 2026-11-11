from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

from cgrasolve.algorithms.router.route_plans import RouteKey, RoutePlan
from cgrasolve.graph.mrrg import NodeType


class RoutingState:
    """Compatibility wrapper around PathFinder's mutable routing state."""

    def __init__(self, router: "PathFinder") -> None:
        self._router = router

    def commit_plan(self, plan: RoutePlan) -> None:
        """Commit every route bundle in a plan."""
        for route_key, planned in plan.routes.items():
            bundle = planned.bundle
            for node in bundle.all_nodes:
                if node.node_type in (NodeType.ROUTING, NodeType.ROUTING_FUNCTION):
                    self._router._commit_node(route_key, node, bundle.transport_profile)
            self._router._routing_solution[route_key] = bundle

    def rip_up_plan(self, plan: RoutePlan) -> None:
        """Rip up every route bundle in a plan."""
        for route_key in reversed(plan.route_keys):
            if route_key in self._router._routing_solution:
                self._router._rip_up_hyperval_net(route_key)

    @contextmanager
    def temporary_plan(self, plan: RoutePlan) -> Iterator[None]:
        """Temporarily commit a plan, then restore any overwritten routes."""
        backups: Dict[RouteKey, object] = {}
        for route_key in plan.route_keys:
            if route_key in self._router._routing_solution:
                backups[route_key] = self._router._routing_solution[route_key]
                self._router._rip_up_hyperval_net(route_key)
        self.commit_plan(plan)
        try:
            yield
        finally:
            self.rip_up_plan(plan)
            for route_key, bundle in backups.items():
                restored_plan = RoutePlan.from_bundle(route_key, bundle, score=0.0)
                self.commit_plan(restored_plan)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cgrasolve.algorithms.router.pathfinder import PathFinder
