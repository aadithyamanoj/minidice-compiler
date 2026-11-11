"""Cost models for evaluating mapping quality."""

from typing import Dict
from cgrasolve.graph.dfg import DFG
from cgrasolve.graph.mrrg import MRRG, MRRGNode
from cgrasolve.graph.hyperdfg import HyperDFG
from cgrasolve.algorithms.mapper_base import MappingResult, Placement, Routing


class CostModel:
    """
    Cost model for evaluating mapping quality.

    Computes various costs including routing distance, congestion, and resource usage.
    """

    def __init__(
        self,
        routing_weight: float = 1.0,
        congestion_weight: float = 2.0,
        resource_weight: float = 0.5
    ) -> None:
        """
        Initialize the cost model.

        Args:
            routing_weight: Weight for routing cost
            congestion_weight: Weight for congestion cost
            resource_weight: Weight for resource utilization cost
        """
        self.routing_weight = routing_weight
        self.congestion_weight = congestion_weight
        self.resource_weight = resource_weight

    def compute_total_cost(
        self,
        dfg: DFG,
        mrrg: MRRG,
        mapping: MappingResult
    ) -> float:
        """
        Compute total mapping cost.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            mapping: Mapping result

        Returns:
            Total cost value
        """
        routing_cost = self.compute_routing_cost(dfg, mrrg, mapping.routing)
        congestion_cost = self.compute_congestion_cost(dfg, mrrg, mapping.routing)
        resource_cost = self.compute_resource_cost(dfg, mrrg, mapping.placement)

        total = (
            self.routing_weight * routing_cost +
            self.congestion_weight * congestion_cost +
            self.resource_weight * resource_cost
        )

        return total

    def compute_routing_cost(
        self,
        dfg: DFG,
        mrrg: MRRG,
        routing: Routing
    ) -> float:
        """
        Compute routing cost based on total wire length.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            routing: Routing solution

        Returns:
            Routing cost
        """
        if routing is None:
            return float('inf')

        total_distance = 0.0

        for edge in dfg.get_edges():
            path = routing.get_route(edge.id)
            if path is None:
                return float('inf')

            # Compute path length
            path_length = len(path) - 1  # Number of hops

            # Add Manhattan distance if nodes have coordinates
            for i in range(len(path) - 1):
                src_node = mrrg.get_node(path[i])
                dst_node = mrrg.get_node(path[i + 1])

                if (src_node and dst_node and
                    hasattr(src_node, 'coordinates') and
                    hasattr(dst_node, 'coordinates') and
                    src_node.coordinates and dst_node.coordinates):
                    manhattan = (
                        abs(src_node.coordinates[0] - dst_node.coordinates[0]) +
                        abs(src_node.coordinates[1] - dst_node.coordinates[1])
                    )
                    total_distance += manhattan

            # If no coordinates, use hop count
            if total_distance == 0:
                total_distance = path_length

        return total_distance

    def compute_congestion_cost(
        self,
        dfg: DFG,
        mrrg: MRRG,
        routing: Routing
    ) -> float:
        """
        Compute congestion cost based on shared routing resources.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            routing: Routing solution

        Returns:
            Congestion cost
        """
        if routing is None:
            return 0.0

        # Count usage of each MRRG node
        usage_count: Dict[str, int] = {}

        for edge in dfg.get_edges():
            path = routing.get_route(edge.id)
            if path:
                for node_id in path:
                    usage_count[node_id] = usage_count.get(node_id, 0) + 1

        # Compute congestion penalty (quadratic for high usage)
        congestion = 0.0
        for node_id, count in usage_count.items():
            if count > 1:
                congestion += (count - 1) ** 2

        return congestion

    def compute_resource_cost(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Placement
    ) -> float:
        """
        Compute resource utilization cost.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            placement: Placement solution

        Returns:
            Resource cost
        """
        if placement is None:
            return 0.0

        # Simple metric: spread of used resources
        used_fus = set(placement.node_mapping.values())
        num_fus = len(mrrg.get_fu_nodes())

        if num_fus == 0:
            return 0.0

        utilization = len(used_fus) / num_fus
        return utilization

    def compute_performance_metrics(
        self,
        dfg: DFG,
        mapping: MappingResult
    ) -> Dict[str, float]:
        """
        Compute performance metrics.

        Args:
            dfg: Data Flow Graph
            mapping: Mapping result

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        if mapping.placement and mapping.placement.schedule:
            # Compute latency
            max_time = max(mapping.placement.schedule.values())
            metrics['latency'] = max_time

            # Compute throughput
            if mapping.initiation_interval:
                metrics['throughput'] = 1.0 / mapping.initiation_interval
            else:
                metrics['throughput'] = 1.0 / max(max_time, 1)

        return metrics


class WirelengthLatencyCostModel:
    """
    Cost model implementing wirelength and latency-aware placement cost.

    Based on the bounding box wirelength estimation with Hanan expansion
    factor for multi-fanout nets, combined with latency cost based on
    the difference between Manhattan distance and scheduled timing.
    """

    def compute_total_cost(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement_map: Dict[str, MRRGNode],
        schedule: Dict[str, int],
        II: int,
        cost_function: int = 0
    ) -> float:
        """
        Compute total placement cost.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            placement_map: Mapping from DFG node ID to placed MRRG node
            schedule: Mapping from DFG node ID to scheduled time
            II: Initiation Interval
            cost_function: 0 for wire cost only, 1 for combined (0.5*wire + 0.5*latency)

        Returns:
            Total cost value
        """
        wire_cost = 0.0
        latency_cost = 0.0

        # Convert DFG to HyperDFG for multi-fanout edge handling
        hyperdfg = HyperDFG.from_dfg(dfg)

        # Iterate over all hyperedges (values/edges in the DFG)
        for edge in hyperdfg.get_edges():
            # Get source node and its placement
            source_hyper_node = hyperdfg.get_node(edge.source_id)
            if source_hyper_node is None:
                continue

            source_dfg_id = source_hyper_node.original_dfg_node_id
            if source_dfg_id not in placement_map:
                continue

            source_mrrg_node = placement_map[source_dfg_id]
            if not source_mrrg_node.coordinates:
                continue

            # Get fanout (number of destinations)
            fanout = edge.cardinality

            # Hanan expansion factor for multi-fanout wirelength quality
            # q = 2.79 + 0.33 * (fanout - 3) for fanout >= 3
            if fanout >= 3:
                q = 2.79 + 0.33 * (fanout - 3)
            else:
                q = 1.0  # For fanout < 3, use standard bounding box

            # Initialize bounding box with source coordinates
            # coordinates are (x, y) where x=col, y=row
            src_x, src_y = source_mrrg_node.coordinates
            x_min = src_x
            x_max = src_x
            y_min = src_y
            y_max = src_y

            # Expand bounding box over all destination nodes
            for dest_id in edge.destination_ids:
                dest_hyper_node = hyperdfg.get_node(dest_id)
                if dest_hyper_node is None:
                    continue

                dest_dfg_id = dest_hyper_node.original_dfg_node_id
                if dest_dfg_id not in placement_map:
                    continue

                dest_mrrg_node = placement_map[dest_dfg_id]
                if not dest_mrrg_node.coordinates:
                    continue

                dest_x, dest_y = dest_mrrg_node.coordinates

                # Expand bounding box
                x_min = min(x_min, dest_x)
                x_max = max(x_max, dest_x)
                y_min = min(y_min, dest_y)
                y_max = max(y_max, dest_y)

                # Compute latency cost for this source-destination pair
                if cost_function == 1:  # Only compute if needed
                    # Spatial distance (Manhattan)
                    manhattan_dist = abs(src_x - dest_x) + abs(src_y - dest_y)

                    # Temporal distance with cycle information
                    src_time = schedule.get(source_dfg_id, 0)
                    dest_time = schedule.get(dest_dfg_id, 0)

                    # Compute cycles to sink with modulo-II wraparound
                    cycles_to_sink = dest_time - src_time
                    if cycles_to_sink < 0:
                        cycles_to_sink += II  # Handle wraparound

                    # Ensure minimum of 1 cycle
                    if cycles_to_sink == 0:
                        cycles_to_sink = 1

                    # Latency cost: penalize when distance doesn't match timing
                    # cost = |manhattan_dist - cycles_to_sink| * cycles_to_sink
                    cost = abs(manhattan_dist - cycles_to_sink) * cycles_to_sink
                    latency_cost += cost

            # Wire cost: Hanan-weighted bounding box perimeter
            # x spans columns, y spans rows
            bbox_cost = q * ((x_max - x_min) + (y_max - y_min))
            wire_cost += bbox_cost

        # Return cost based on cost_function mode
        if cost_function == 0:
            return wire_cost
        else:  # cost_function == 1
            return 0.5 * wire_cost + 0.5 * latency_cost
