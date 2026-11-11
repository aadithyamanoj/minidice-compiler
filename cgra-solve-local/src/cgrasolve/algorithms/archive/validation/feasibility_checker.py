"""Feasibility checker for mapping validation."""

from typing import List, Tuple, Optional
from cgrasolve.graph.dfg import DFG
from cgrasolve.graph.mrrg import MRRG
from cgrasolve.algorithms.mapper_base import MappingResult, Placement, Routing


class ValidationError:
    """Represents a validation error."""

    def __init__(self, error_type: str, message: str, details: Optional[dict] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.error_type}] {self.message}"


class FeasibilityChecker:
    """
    Checks the feasibility and correctness of a mapping.

    Validates:
    - Functional compatibility (operations can execute on assigned FUs)
    - Resource conflicts (no double-booking of resources)
    - Routing validity (paths exist and are valid)
    - Timing constraints (dependencies are satisfied)
    """

    def __init__(self) -> None:
        """Initialize the feasibility checker."""
        pass

    def validate(
        self,
        dfg: DFG,
        mrrg: MRRG,
        mapping: MappingResult
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a complete mapping.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            mapping: Mapping result to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: List[ValidationError] = []

        if mapping.placement is None:
            errors.append(ValidationError(
                "MISSING_PLACEMENT",
                "Mapping has no placement"
            ))
            return False, errors

        # Validate placement
        placement_errors = self.validate_placement(dfg, mrrg, mapping.placement)
        errors.extend(placement_errors)

        # Validate routing if present
        if mapping.routing:
            routing_errors = self.validate_routing(
                dfg,
                mrrg,
                mapping.placement,
                mapping.routing
            )
            errors.extend(routing_errors)

        return len(errors) == 0, errors

    def validate_placement(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Placement
    ) -> List[ValidationError]:
        """
        Validate placement.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            placement: Placement to validate

        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []

        # Check completeness
        for node in dfg.get_nodes():
            if node.id not in placement.node_mapping:
                errors.append(ValidationError(
                    "INCOMPLETE_PLACEMENT",
                    f"Node {node.id} is not placed",
                    {"node_id": node.id}
                ))

        # Check functional compatibility
        for dfg_node_id, mrrg_node_id in placement.node_mapping.items():
            dfg_node = dfg.get_node(dfg_node_id)
            mrrg_node = mrrg.get_node(mrrg_node_id)

            if dfg_node is None:
                errors.append(ValidationError(
                    "INVALID_DFG_NODE",
                    f"DFG node {dfg_node_id} not found",
                    {"node_id": dfg_node_id}
                ))
                continue

            if mrrg_node is None:
                errors.append(ValidationError(
                    "INVALID_MRRG_NODE",
                    f"MRRG node {mrrg_node_id} not found",
                    {"node_id": mrrg_node_id}
                ))
                continue

            # Check if FU can execute the operation
            if hasattr(mrrg_node, 'can_execute'):
                if not mrrg_node.can_execute(dfg_node.operation):
                    errors.append(ValidationError(
                        "INCOMPATIBLE_OPERATION",
                        f"FU {mrrg_node_id} cannot execute operation {dfg_node.operation.value}",
                        {
                            "dfg_node": dfg_node_id,
                            "mrrg_node": mrrg_node_id,
                            "operation": dfg_node.operation.value
                        }
                    ))

        # Check resource conflicts
        resource_usage = placement.get_resource_usage()
        for mrrg_node_id, usage_list in resource_usage.items():
            # Check for time conflicts (same resource used at same time modulo II)
            if dfg.initiation_interval:
                time_map = {}
                for dfg_id, time in usage_list:
                    time_mod = time % dfg.initiation_interval
                    if time_mod in time_map:
                        errors.append(ValidationError(
                            "RESOURCE_CONFLICT",
                            f"Resource {mrrg_node_id} used by multiple nodes at time {time_mod}",
                            {
                                "resource": mrrg_node_id,
                                "time": time_mod,
                                "nodes": [time_map[time_mod], dfg_id]
                            }
                        ))
                    time_map[time_mod] = dfg_id

        return errors

    def validate_routing(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Placement,
        routing: Routing
    ) -> List[ValidationError]:
        """
        Validate routing.

        Args:
            dfg: Data Flow Graph
            mrrg: MRRG
            placement: Node placement
            routing: Routing to validate

        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []

        # Check completeness
        for edge in dfg.get_edges():
            if edge.id not in routing.edge_routing:
                errors.append(ValidationError(
                    "INCOMPLETE_ROUTING",
                    f"Edge {edge.id} is not routed",
                    {"edge_id": edge.id}
                ))

        # Check path validity
        for dfg_edge_id, path in routing.edge_routing.items():
            dfg_edge = dfg.get_edge(dfg_edge_id)

            if dfg_edge is None:
                errors.append(ValidationError(
                    "INVALID_DFG_EDGE",
                    f"DFG edge {dfg_edge_id} not found",
                    {"edge_id": dfg_edge_id}
                ))
                continue

            # Check path validity
            if len(path) < 2:
                errors.append(ValidationError(
                    "INVALID_PATH",
                    f"Path for edge {dfg_edge_id} is too short",
                    {"edge_id": dfg_edge_id, "path_length": len(path)}
                ))
                continue

            # Check that path starts at source FU
            src_fu = placement.get_placement(dfg_edge.source.id)
            if path[0] != src_fu:
                errors.append(ValidationError(
                    "PATH_START_MISMATCH",
                    f"Path for edge {dfg_edge_id} doesn't start at source FU",
                    {
                        "edge_id": dfg_edge_id,
                        "expected": src_fu,
                        "actual": path[0]
                    }
                ))

            # Check that path ends at destination FU
            dst_fu = placement.get_placement(dfg_edge.destination.id)
            if path[-1] != dst_fu:
                errors.append(ValidationError(
                    "PATH_END_MISMATCH",
                    f"Path for edge {dfg_edge_id} doesn't end at destination FU",
                    {
                        "edge_id": dfg_edge_id,
                        "expected": dst_fu,
                        "actual": path[-1]
                    }
                ))

            # Check that consecutive nodes in path are connected
            for i in range(len(path) - 1):
                src_id = path[i]
                dst_id = path[i + 1]

                # Check if edge exists in MRRG
                has_edge = any(
                    e.destination.id == dst_id
                    for e in mrrg.get_outgoing_edges(src_id)
                )

                if not has_edge:
                    errors.append(ValidationError(
                        "DISCONNECTED_PATH",
                        f"No edge from {src_id} to {dst_id} in path for {dfg_edge_id}",
                        {
                            "edge_id": dfg_edge_id,
                            "from": src_id,
                            "to": dst_id
                        }
                    ))

        return errors
