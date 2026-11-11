from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random
import math
from cgrasolve.graph.dfg import DFG, DFGNode
from cgrasolve.graph.mrrg import MRRG, MRRGNode
from cgrasolve.graph.hyperdfg import HyperDFG, HyperNode, HyperVal

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