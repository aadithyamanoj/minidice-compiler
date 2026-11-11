from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from cgrasolve.algorithms.router.types import RouteBundle
from cgrasolve.graph.hyperdfg import HyperVal
from cgrasolve.graph.mrrg import MRRGNode

RouteKey = Tuple[HyperVal, int]


@dataclass(frozen=True)
class PlannedRoute:
    """One routed demand inside a plan."""

    route_key: RouteKey
    bundle: RouteBundle
    score: float

    @property
    def hyperval(self) -> HyperVal:
        """Return the HyperVal that owns this route."""
        return self.route_key[0]


@dataclass
class RoutePlan:
    """A commit-ready collection of route bundles plus its composite score."""

    routes: Dict[RouteKey, PlannedRoute] = field(default_factory=dict)
    score: float = 0.0
    shared_carrier: bool = False
    shared_node_ids: Tuple[str, ...] = ()

    @classmethod
    def from_bundle(
        cls,
        route_key: RouteKey,
        bundle: RouteBundle,
        score: float,
        *,
        shared_carrier: bool = False,
        shared_node_ids: Iterable[str] = (),
    ) -> "RoutePlan":
        """Create a single-route plan from one bundle."""
        return cls(
            routes={
                route_key: PlannedRoute(
                    route_key=route_key,
                    bundle=bundle,
                    score=score,
                )
            },
            score=score,
            shared_carrier=shared_carrier,
            shared_node_ids=tuple(sorted(set(shared_node_ids))),
        )

    @classmethod
    def merge(
        cls,
        plans: Iterable["RoutePlan"],
        *,
        score_adjustment: float = 0.0,
        shared_carrier: bool = False,
        shared_node_ids: Iterable[str] = (),
    ) -> "RoutePlan":
        """Merge several plans into one composite plan."""
        merged_routes: Dict[RouteKey, PlannedRoute] = {}
        total_score = 0.0
        for plan in plans:
            total_score += plan.score
            for route_key, planned in plan.routes.items():
                merged_routes[route_key] = planned
        return cls(
            routes=merged_routes,
            score=total_score + score_adjustment,
            shared_carrier=shared_carrier,
            shared_node_ids=tuple(sorted(set(shared_node_ids))),
        )

    @property
    def route_keys(self) -> Tuple[RouteKey, ...]:
        """Return route keys in a stable order."""
        return tuple(sorted(self.routes.keys(), key=lambda item: (item[0].source_id, item[1])))

    @property
    def bundles(self) -> Mapping[RouteKey, RouteBundle]:
        """Expose bundles by route key."""
        return {route_key: planned.bundle for route_key, planned in self.routes.items()}

    def all_nodes(self) -> Set[MRRGNode]:
        """Return all nodes touched by any route in the plan."""
        nodes: Set[MRRGNode] = set()
        for planned in self.routes.values():
            nodes.update(planned.bundle.all_nodes)
        return nodes

    def hypervals(self) -> Set[HyperVal]:
        """Return all HyperVals participating in this plan."""
        return {route_key[0] for route_key in self.routes}

    def get_bundle(self, route_key: RouteKey) -> Optional[RouteBundle]:
        """Return the bundle for one route key."""
        planned = self.routes.get(route_key)
        return None if planned is None else planned.bundle


@dataclass
class SharedCarrierPlan(RoutePlan):
    """Route plan that intentionally co-optimizes several packed routes."""

    root_route_key: Optional[RouteKey] = None
    lane_map: Dict[RouteKey, int] = field(default_factory=dict)
