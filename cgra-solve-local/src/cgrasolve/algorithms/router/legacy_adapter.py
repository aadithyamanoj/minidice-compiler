from __future__ import annotations

from typing import Dict

from cgrasolve.algorithms.router.route_plans import RouteKey, RoutePlan
from cgrasolve.algorithms.router.types import RouteBundle


def plan_to_route_bundles(plan: RoutePlan) -> Dict[RouteKey, RouteBundle]:
    """Lower a RoutePlan into the legacy routing-solution representation."""
    return {route_key: planned.bundle for route_key, planned in plan.routes.items()}
