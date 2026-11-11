from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

from cgrasolve.algorithms.common.routing_validation import (
    resolve_sink_data_bitwidth,
    resolve_source_output_bitwidth_info,
)
from cgrasolve.algorithms.router.types import SinkAndLatency
from cgrasolve.graph.dfg import DFGNode
from cgrasolve.graph.hyperdfg import HyperVal
from cgrasolve.graph.mrrg import MRRGNode, OperandTag

if TYPE_CHECKING:
    from cgrasolve.algorithms.router.pathfinder import PathFinder


@dataclass(frozen=True)
class DemandSink:
    """One routed sink requirement derived from a HyperVal destination."""

    dest_idx: int
    dest_dfg_node: DFGNode
    sink_fu: MRRGNode
    operand_tag: Optional[OperandTag]
    cycles_to_sink: int
    bitwidth: int

    def to_sink_and_latency(self) -> SinkAndLatency:
        """Expose the legacy sink descriptor used by the search kernel."""
        return SinkAndLatency(
            cycles_to_sink=self.cycles_to_sink,
            dest_idx=self.dest_idx,
            dest_dfg_node=self.dest_dfg_node,
            sink_fu=self.sink_fu,
            operand_tag=self.operand_tag,
        )


@dataclass(frozen=True)
class RoutingDemand:
    """Normalized routing demand for one source-to-sink requirement."""

    hyperval: HyperVal
    source_dfg_node: DFGNode
    source_fu: MRRGNode
    source_ports: Tuple[MRRGNode, ...]
    produced_bitwidth: int
    source_width_is_explicit: bool
    latency_of_src: int
    sink: DemandSink

    @property
    def route_key(self) -> tuple[HyperVal, int]:
        """Return the canonical routing-solution key for this demand."""
        return (self.hyperval, self.sink.dest_idx)


def build_routing_demands(router: "PathFinder", hyperval: HyperVal) -> Tuple[RoutingDemand, ...]:
    """Build normalized sink demands for one HyperVal."""
    src_hyper_node = router._hyperdfg.get_node(hyperval.source_id)
    if src_hyper_node is None:
        return ()

    src_dfg_node = router._dfg.get_node(src_hyper_node.original_dfg_node_id)
    if src_dfg_node is None or src_dfg_node not in router._placement:
        return ()

    src_fu = router._placement[src_dfg_node]
    latency_of_src = src_fu.latency
    produced_bitwidth, source_width_is_explicit = resolve_source_output_bitwidth_info(
        src_dfg_node,
        src_fu,
        hyperval,
    )
    hyperval.source_output_bitwidth = produced_bitwidth

    candidate_source_ports = tuple(
        router._mrrg.get_output_ports_for_operation(
            src_fu,
            src_dfg_node.operation,
        )
    )
    if not candidate_source_ports:
        return ()

    sinks = []
    for dest_idx in range(hyperval.cardinality):
        dest_id = hyperval.destination_ids[dest_idx]
        dest_hyper_node = router._hyperdfg.get_node(dest_id)
        if dest_hyper_node is None:
            return ()

        dest_dfg_node = router._dfg.get_node(dest_hyper_node.original_dfg_node_id)
        if dest_dfg_node is None or dest_dfg_node not in router._placement:
            return ()

        sink_fu = router._placement[dest_dfg_node]
        edge_dist = hyperval.dists[dest_idx] if hyperval.dists[dest_idx] is not None else 0
        cycles_to_sink = router._compute_cycles_source_to_sink(
            src_dfg_node,
            dest_dfg_node,
            edge_dist,
        )
        operand_tag = router._map_operand_to_tag(hyperval.operands[dest_idx])
        sink_bitwidth = resolve_sink_data_bitwidth(
            produced_bitwidth,
            hyperval,
            dest_idx,
            produced_bitwidth,
            source_width_is_explicit=source_width_is_explicit,
        )
        sinks.append(
            RoutingDemand(
                hyperval=hyperval,
                source_dfg_node=src_dfg_node,
                source_fu=src_fu,
                source_ports=candidate_source_ports,
                produced_bitwidth=produced_bitwidth,
                source_width_is_explicit=source_width_is_explicit,
                latency_of_src=latency_of_src,
                sink=DemandSink(
                    dest_idx=dest_idx,
                    dest_dfg_node=dest_dfg_node,
                    sink_fu=sink_fu,
                    operand_tag=operand_tag,
                    cycles_to_sink=cycles_to_sink,
                    bitwidth=sink_bitwidth,
                ),
            )
        )

    return tuple(sorted(sinks, key=lambda demand: demand.sink.to_sink_and_latency()))
