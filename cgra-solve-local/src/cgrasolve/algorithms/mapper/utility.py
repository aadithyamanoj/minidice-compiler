"""Utility class for extracting and validating ILP mapping solutions."""

import warnings
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class ILPMapperUtility:
    """Utility class for extracting and validating ILP mapping solutions.

    All methods are classmethods that accept parameters explicitly for maximum
    flexibility and testability.
    """

    # -- Version 1 ----------------------------------------------------------------
    @classmethod
    def extract_placements_v1(cls, F_pq: Any, hdfg_nodes: List[Any], fu_nodes: List[Any]) -> Dict[str, str]:
        """Return {dfg_node_id: fu_node_id} mapping for solved model.

        Args:
            F_pq: Gurobi binary variable array indexed by (fu_idx, dfg_idx)
            hdfg_nodes: List of DFG HyperNodes
            fu_nodes: List of MRRG FU nodes

        Returns:
            Dictionary mapping DFG node IDs to FU node IDs
        """
        placement = {}
        num_fu_nodes = len(fu_nodes)
        num_hdfg_nodes = len(hdfg_nodes)

        for p in range(num_fu_nodes):
            for q in range(num_hdfg_nodes):
                if F_pq[p, q].X > 0.5:
                    placement[hdfg_nodes[q].id] = fu_nodes[p].id
        return placement

    @classmethod
    def reconstruct_path_v1(cls, active_node_ids: List[str], mrrg: Any) -> List[str]:
        """Reconstruct correct routing path order by following MRRG edges.

        Args:
            active_node_ids: List of routing node IDs that are active for this route
            mrrg: MRRG instance with get_incoming_edges and get_outgoing_edges methods

        Returns:
            List of node IDs in correct edge-connected order
        """
        if len(active_node_ids) <= 1:
            return active_node_ids

        active_set = set(active_node_ids)

        # Find start node: node with no active predecessors
        start = None
        for node_id in active_set:
            has_active_pred = False
            incoming = mrrg.get_incoming_edges(node_id)
            for edge in incoming:
                if edge and edge.source.id in active_set:
                    has_active_pred = True
                    break

            if not has_active_pred:
                start = node_id
                break

        if start is None:
            # Fallback: use node with earliest cycle
            start = min(active_set, key=lambda nid: (
                int(nid.split(':')[0]) if ':' in nid else 0,
                nid
            ))

        # Build path by following edges
        path = [start]
        current = start
        visited = {start}

        while len(visited) < len(active_set):
            # Find next active node connected by an edge
            found = False
            outgoing = mrrg.get_outgoing_edges(current)
            for edge in outgoing:
                if edge and edge.destination.id in active_set and edge.destination.id not in visited:
                    next_node = edge.destination.id
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    found = True
                    break

            if not found:
                # Dead end - try to find any unvisited active node
                remaining = active_set - visited
                if remaining:
                    # Add disconnected nodes sorted by cycle
                    disconnected = sorted(remaining, key=lambda nid: (
                        int(nid.split(':')[0]) if ':' in nid else 0,
                        nid
                    ))
                    path.extend(disconnected)
                break

        return path

    @classmethod
    def extract_routes_v1(cls, V_ijk: Any, hypervals: List[Any], routing_nodes: List[Any],
                          mrrg: Any, placement: Dict[str, str] = None,
                          hdfg_nodes: List[Any] = None) -> Dict[Tuple[str, int], List[str]]:
        """Return {(source_id, sink_k): [routing_node_ids]} dictionary.

        Maps routes using the original DFG source node ID and reconstructs
        paths in correct edge-connected order.

        Args:
            V_ijk: Gurobi binary variable array indexed by (routing_idx, val_idx, sink_idx)
            hypervals: List of HyperVal objects
            routing_nodes: List of MRRG routing nodes
            mrrg: MRRG instance for path reconstruction
            placement: Optional dictionary mapping DFG node IDs to FU node IDs (for pretty printing)
            hdfg_nodes: Optional list of DFG HyperNodes (for pretty printing)

        Returns:
            Dictionary mapping (source_id, sink_k) to ordered routing paths
        """
        routes = {}
        num_vals = len(hypervals)
        num_routing_nodes = len(routing_nodes)

        # First, collect all active nodes for each (source_id, sink_k) pair
        active_nodes_map = {}  # (source_id, sink_k) -> [node_ids]

        for j in range(num_vals):
            source_id = hypervals[j].source_id
            for k in range(hypervals[j].cardinality):
                route_key = (source_id, k)
                active_nodes = []

                for i in range(num_routing_nodes):
                    if V_ijk[i, j, k].X > 0.5:
                        active_nodes.append(routing_nodes[i].id)

                if active_nodes:
                    active_nodes_map[route_key] = active_nodes

        # Now reconstruct proper path order for each route
        print("\n" + "="*70)
        print("ROUTING PATHS (Source MRRG -> Sink MRRG)")
        print("="*70)

        for route_key, active_nodes in active_nodes_map.items():
            source_id, sink_k = route_key
            ordered_path = cls.reconstruct_path_v1(active_nodes, mrrg)
            routes[route_key] = ordered_path

            # Get destination ID from hyperval
            dest_id = None
            for hv in hypervals:
                if hv.source_id == source_id and sink_k < len(hv.destination_ids):
                    dest_id = hv.destination_ids[sink_k]
                    break

            # Pretty print the complete path
            if ordered_path:
                # Get source and sink FU nodes from placement if available
                source_fu = placement.get(source_id, "?") if placement else "?"
                sink_fu = placement.get(dest_id, "?") if placement and dest_id else "?"

                print(f"\nDFG Edge: {source_id} -> {dest_id}")
                print(f"  Source DFG node '{source_id}' mapped to MRRG node: {source_fu}")
                print(f"  Sink DFG node '{dest_id}' mapped to MRRG node: {sink_fu}")
                complete_path = [source_fu] + ordered_path + [sink_fu]
                print(f"  Complete Path: {' -> '.join(complete_path)}")
            else:
                warnings.warn(f"DFG Edge ({source_id} -> {dest_id}): Path not available")

        # Check for missing routes and warn
        for j in range(num_vals):
            source_id = hypervals[j].source_id
            for k in range(hypervals[j].cardinality):
                route_key = (source_id, k)
                if route_key not in active_nodes_map:
                    dest_id = hypervals[j].destination_ids[k] if k < len(hypervals[j].destination_ids) else f"sink[{k}]"
                    warnings.warn(f"DFG Edge ({source_id} -> {dest_id}): Path not available (no active routing nodes)")

        print("="*70)
        return routes

    @classmethod
    def print_solution_v1(cls, placement: Dict[str, str], routes: Dict[Tuple[str, int], List[str]],
                          hdfg_nodes: List[Any], hypervals: List[Any]) -> None:
        """Pretty-print placements and routes.

        Args:
            placement: Dictionary mapping DFG node IDs to FU node IDs
            routes: Dictionary mapping (source_id, sink_k) to routing paths
            hdfg_nodes: List of DFG HyperNodes (for operation types)
            hypervals: List of HyperVal objects (for destination names)
        """
        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        print(f"\nPlacement ({len(placement)} operations):")
        for qid, pid in sorted(placement.items()):
            # Find operation type
            op_type = next((n.operation.name for n in hdfg_nodes if n.id == qid), "UNKNOWN")
            print(f"  {qid:15s} ({op_type:10s}) → FU {pid}")
        print(f"\nConnection Mapping Result:")

        # Group routes by source_id to avoid duplicate headers
        routes_by_source = defaultdict(list)
        for (source_id, sink_k), path in routes.items():
            routes_by_source[source_id].append((sink_k, path))

        # Print routes grouped by source
        for source_id in sorted(routes_by_source.keys()):
            route_name = f"{source_id}_out" if not source_id.endswith("_out") else source_id
            print(f"{route_name}:")

            # Group by unique path and collect which sinks use each path
            path_to_sinks = defaultdict(list)
            for sink_k, path in routes_by_source[source_id]:
                path_tuple = tuple(path)
                path_to_sinks[path_tuple].append(sink_k)

            # Print each unique path with its sink indices
            for path_tuple, sink_indices in sorted(path_to_sinks.items()):
                # Show which sinks use this path (if multiple)
                if len(sink_indices) > 1:
                    sink_list = ','.join(map(str, sorted(sink_indices)))

                    # Get destination names for these sinks
                    dest_names = []
                    for hv in hypervals:
                        if hv.source_id == source_id:
                            for idx in sorted(sink_indices):
                                if idx < len(hv.destination_ids):
                                    dest_names.append(hv.destination_ids[idx])
                            break

                    if dest_names:
                        dest_str = ', '.join(dest_names)
                        print(f"  [sinks {sink_list}: {dest_str}]")
                    else:
                        print(f"  [sinks {sink_list}]")

                # Print each routing node with time-step prefix
                for routing_node_id in path_tuple:
                    # Extract time-step from node ID (format: "time:node_id")
                    if ':' in routing_node_id:
                        parts = routing_node_id.split(':', 1)
                        time_step = parts[0]
                        node_name = parts[1]
                        print(f"  {time_step}:{node_name}")
                    else:
                        print(f"  {routing_node_id}")
        print("="*70)

    @classmethod
    def validate_solution_v1(cls, placement: Dict[str, str], routes: Dict[Tuple[str, int], List[str]],
                             hdfg_nodes: List[Any], hypervals: List[Any], mrrg: Any) -> Tuple[bool, List[str]]:
        """Validate the ILP solution for correctness.

        Args:
            placement: Dictionary mapping DFG node IDs to FU node IDs
            routes: Dictionary mapping (source_id, sink_k) to routing paths
            hdfg_nodes: List of DFG HyperNodes
            hypervals: List of HyperVal objects
            mrrg: MRRG instance for edge validation

        Returns:
            Tuple of (is_valid: bool, errors: list of error messages)
        """
        errors = []

        print("\n" + "="*70)
        print("SOLUTION VALIDATION")
        print("="*70)

        # 1. Check all DFG nodes are placed
        placed_nodes = set(placement.keys())
        all_nodes = set(n.id for n in hdfg_nodes)
        missing = all_nodes - placed_nodes
        if missing:
            error = f"Missing placements for nodes: {missing}"
            errors.append(error)
            print(f"❌ {error}")
        else:
            print(f"✓ All {len(all_nodes)} DFG nodes placed")

        # 2. Check no resource conflicts (same FU used twice in same cycle)
        fu_usage = {}  # (fu_base_id, cycle) -> node_id
        for node_id, fu_id in placement.items():
            mrrg_node = mrrg.get_node(fu_id) if mrrg else None
            if mrrg_node is not None:
                key = (mrrg_node.get_base_name(), mrrg_node.cycle)
            elif ':' in fu_id:
                cycle_str, fu_base = fu_id.split(':', 1)
                key = (fu_base, int(cycle_str))
            else:
                continue

            if key in fu_usage:
                error = f"Resource conflict: {node_id} and {fu_usage[key]} both use {key[0]} at cycle {key[1]}"
                errors.append(error)
                print(f"❌ {error}")
            else:
                fu_usage[key] = node_id

        if not errors or len(errors) == len(missing) if missing else True:
            print(f"✓ No resource conflicts ({len(fu_usage)} FU usages)")

        # 3. Check routing path connectivity
        invalid_routes = []
        for (source_id, sink_k), path in routes.items():
            if len(path) == 0:
                error = f"Empty route: {source_id} → sink[{sink_k}]"
                errors.append(error)
                invalid_routes.append((source_id, sink_k))
                continue

            if len(path) == 1:
                # Single-node path (direct connection)
                continue

            # Check consecutive edges exist in MRRG
            for i in range(len(path) - 1):
                src_node_id = path[i]
                dst_node_id = path[i + 1]

                # Check if edge exists
                outgoing = mrrg.get_outgoing_edges(src_node_id)
                edge_exists = any(e.destination.id == dst_node_id for e in outgoing)

                if not edge_exists:
                    error = f"Missing edge in route {source_id}→sink[{sink_k}]: {src_node_id} → {dst_node_id}"
                    errors.append(error)
                    invalid_routes.append((source_id, sink_k))
                    break

        if invalid_routes:
            print(f"❌ Invalid routing paths: {len(invalid_routes)} routes")
            # Print all invalid routes with detailed context
            for source_id, sink_k in invalid_routes:
                # Find the corresponding HyperVal
                hyperval = None
                for hv in hypervals:
                    if hv.source_id == source_id:
                        hyperval = hv
                        break

                if hyperval and sink_k < len(hyperval.destination_ids):
                    dest_id = hyperval.destination_ids[sink_k]

                    # Get source and dest nodes for operation info
                    src_node = next((n for n in hdfg_nodes if n.id == source_id), None)
                    dst_node = next((n for n in hdfg_nodes if n.id == dest_id), None)

                    src_op = f" ({src_node.operation.value})" if src_node else ""
                    dst_op = f" ({dst_node.operation.value})" if dst_node else ""

                    # Get placement info if available
                    src_place = f" @ {placement[source_id]}" if source_id in placement else ""
                    dst_place = f" @ {placement[dest_id]}" if dest_id in placement else ""

                    print(f"   ✗ {source_id}{src_op}{src_place} → {dest_id}{dst_op}{dst_place} [sink {sink_k}]")
                else:
                    print(f"   ✗ {source_id} → sink[{sink_k}]")
        else:
            print(f"✓ All {len(routes)} routing paths valid")

        # 4. Check all HyperVal destinations have routes
        expected_routes = set()
        for j, hv in enumerate(hypervals):
            for k in range(hv.cardinality):
                expected_routes.add((hv.source_id, k))

        actual_routes = set(routes.keys())
        missing_routes = expected_routes - actual_routes

        if missing_routes:
            error = f"Missing routes for {len(missing_routes)} destinations"
            errors.append(error)
            print(f"❌ {error}")
            # Print all missing routes with detailed context
            for source_id, sink_k in list(missing_routes):
                # Find the corresponding HyperVal
                hyperval = None
                for hv in hypervals:
                    if hv.source_id == source_id:
                        hyperval = hv
                        break

                if hyperval and sink_k < len(hyperval.destination_ids):
                    dest_id = hyperval.destination_ids[sink_k]

                    # Get source and dest nodes for operation info
                    src_node = next((n for n in hdfg_nodes if n.id == source_id), None)
                    dst_node = next((n for n in hdfg_nodes if n.id == dest_id), None)

                    src_op = f" ({src_node.operation.value})" if src_node else ""
                    dst_op = f" ({dst_node.operation.value})" if dst_node else ""

                    # Get placement info if available
                    src_place = f" @ {placement[source_id]}" if source_id in placement else ""
                    dst_place = f" @ {placement[dest_id]}" if dest_id in placement else ""

                    print(f"   ✗ {source_id}{src_op}{src_place} → {dest_id}{dst_op}{dst_place} [sink {sink_k}]")
                else:
                    print(f"   ✗ {source_id} → sink[{sink_k}]")
        else:
            print(f"✓ All {len(expected_routes)} required routes present")

        print("="*70)

        is_valid = len(errors) == 0
        if is_valid:
            print("✓✓✓ SOLUTION VALIDATION PASSED ✓✓✓")
        else:
            print(f"❌❌❌ SOLUTION VALIDATION FAILED ({len(errors)} errors) ❌❌❌")
        print("="*70)

        return is_valid, errors

    # -- Version 2 ----------------------------------------------------------------
    @classmethod
    def extract_placements_v2(cls, F_ou: Any, dfg_nodes: List[Any], fu_nodes: List[Any]) -> Dict[str, str]:
        """Return {dfg_node_id: fu_node_id} mapping for solved model (v2).

        Args:
            F_ou: Gurobi binary variable array indexed by (dfg_idx, fu_idx)
            dfg_nodes: List of DFG nodes
            fu_nodes: List of MRRG FU nodes

        Returns:
            Dictionary mapping DFG node IDs to FU node IDs
        """
        placement = {}
        num_dfg_nodes = len(dfg_nodes)
        num_fu_nodes = len(fu_nodes)

        for o in range(num_dfg_nodes):
            for u in range(num_fu_nodes):
                if F_ou[o, u].X > 0.5:
                    placement[dfg_nodes[o].id] = fu_nodes[u].id
        return placement

    @classmethod
    def extract_routes_v2(cls, E_oupv: Any, P_uvq: Any, dfg_nodes: List[Any],
                          dfg_edges: List[Any], fu_nodes: List[Any],
                          path_map: Dict[Tuple[int, int], List[List[Any]]],
                          mrrg: Any, placement: Dict[str, str] = None) -> Dict[Tuple[str, str], List[str]]:
        """Return {(source_id, dest_id): [routing_node_ids]} dictionary for v2.

        Maps routes using DFG source and destination node IDs and reconstructs
        paths in correct edge-connected order.

        Args:
            E_oupv: Gurobi binary variable array for edge mapping
            P_uvq: Gurobi binary variable array for path selection
            dfg_nodes: List of DFG nodes
            dfg_edges: List of DFG edges
            fu_nodes: List of MRRG FU nodes
            path_map: Dictionary mapping (u, v) FU pairs to lists of paths
            mrrg: MRRG instance for path reconstruction

        Returns:
            Dictionary mapping (source_id, dest_id) to ordered routing paths
        """
        routes = {}
        num_dfg_nodes = len(dfg_nodes)
        num_fu_nodes = len(fu_nodes)

        # Build index maps
        dfg_id2idx = {n.id: i for i, n in enumerate(dfg_nodes)}

        # For each DFG edge, find the selected FU pair and path
        for edge in dfg_edges:
            o = dfg_id2idx[edge.source.id]
            p = dfg_id2idx[edge.destination.id]

            # Find which FU pair (u, v) was selected for this edge
            selected_path = None
            for u in range(num_fu_nodes):
                for v in range(num_fu_nodes):
                    try:
                        if E_oupv[o, u, p, v].X > 0.5:
                            # Found the FU pair, now find the path
                            if (u, v) in path_map:
                                for q in range(len(path_map[(u, v)])):
                                    try:
                                        if P_uvq[u, v, q].X > 0.5:
                                            selected_path = path_map[(u, v)][q]
                                            break
                                    except KeyError:
                                        continue
                            break
                    except (KeyError, AttributeError):
                        continue
                if selected_path is not None:
                    break

            # Store the path if found
            if selected_path is not None:
                # Convert MRRGNode objects to IDs
                # Handle both MRRGNode objects and string IDs
                if selected_path and hasattr(selected_path[0], 'id'):
                    path_ids = [node.id for node in selected_path]
                else:
                    # Already strings
                    path_ids = selected_path

                route_key = (edge.source.id, edge.destination.id)
                routes[route_key] = path_ids

        # Print routing paths similar to v1
        print("\n" + "="*70)
        print("ROUTING PATHS (Source MRRG -> Sink MRRG)")
        print("="*70)

        for (source_id, dest_id), path in routes.items():
            # Get source and sink FU nodes from placement if available
            source_fu = placement.get(source_id, "?") if placement else "?"
            sink_fu = placement.get(dest_id, "?") if placement else "?"

            print(f"\nDFG Edge: {source_id} -> {dest_id}")
            print(f"  Source DFG node '{source_id}' mapped to MRRG node: {source_fu}")
            print(f"  Sink DFG node '{dest_id}' mapped to MRRG node: {sink_fu}")

            if path:
                # Build complete path: source FU -> routing nodes -> sink FU
                # The path from path_map already includes source and sink FUs
                complete_path = [str(node_id) for node_id in path]
                print(f"  Complete Path: {' -> '.join(complete_path)}")
            else:
                print("  Complete Path: (empty)")

        # Check for missing routes and warn
        for edge in dfg_edges:
            route_key = (edge.source.id, edge.destination.id)
            if route_key not in routes:
                warnings.warn(f"DFG Edge ({edge.source.id} -> {edge.destination.id}): Path not available")

        print("="*70)

        return routes

    @classmethod
    def print_solution_v2(cls, placement: Dict[str, str], routes: Dict[Tuple[str, str], List[str]],
                          dfg_nodes: List[Any], dfg_edges: List[Any]) -> None:
        """Pretty-print placements and routes for v2.

        Args:
            placement: Dictionary mapping DFG node IDs to FU node IDs
            routes: Dictionary mapping (source_id, dest_id) to routing paths
            dfg_nodes: List of DFG nodes (for operation types)
            dfg_edges: List of DFG edges
        """
        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        print(f"\nPlacement ({len(placement)} operations):")
        for node_id, fu_id in sorted(placement.items()):
            # Find operation type
            op_type = next((n.operation.name for n in dfg_nodes if n.id == node_id), "UNKNOWN")
            print(f"  {node_id:15s} ({op_type:10s}) → FU {fu_id}")

        print(f"\nRouting ({len(routes)} connections):")
        for (source_id, dest_id), path in sorted(routes.items()):
            print(f"\n  {source_id} → {dest_id}:")
            if len(path) == 0:
                print("    (empty path)")
            else:
                for routing_node_id in path:
                    # Extract time-step from node ID (format: "time:node_id")
                    if ':' in routing_node_id:
                        parts = routing_node_id.split(':', 1)
                        time_step = parts[0]
                        node_name = parts[1]
                        print(f"    {time_step}:{node_name}")
                    else:
                        print(f"    {routing_node_id}")
        print("="*70)

    @classmethod
    def validate_solution_v2(cls, placement: Dict[str, str], routes: Dict[Tuple[str, str], List[str]],
                             dfg_nodes: List[Any], dfg_edges: List[Any], mrrg: Any) -> Tuple[bool, List[str]]:
        """Validate the ILP solution for correctness (v2).

        Args:
            placement: Dictionary mapping DFG node IDs to FU node IDs
            routes: Dictionary mapping (source_id, dest_id) to routing paths
            dfg_nodes: List of DFG nodes
            dfg_edges: List of DFG edges
            mrrg: MRRG instance for edge validation

        Returns:
            Tuple of (is_valid: bool, errors: list of error messages)
        """
        errors = []

        print("\n" + "="*70)
        print("SOLUTION VALIDATION")
        print("="*70)

        # 1. Check all DFG nodes are placed
        placed_nodes = set(placement.keys())
        all_nodes = set(n.id for n in dfg_nodes)
        missing = all_nodes - placed_nodes
        if missing:
            error = f"Missing placements for nodes: {missing}"
            errors.append(error)
            print(f"❌ {error}")
        else:
            print(f"✓ All {len(all_nodes)} DFG nodes placed")

        # 2. Check no resource conflicts (same FU used twice in same cycle)
        fu_usage = {}  # (fu_base_id, cycle) -> node_id
        for node_id, fu_id in placement.items():
            mrrg_node = mrrg.get_node(fu_id) if mrrg else None
            if mrrg_node is not None:
                key = (mrrg_node.get_base_name(), mrrg_node.cycle)
            elif ':' in fu_id:
                cycle_str, fu_base = fu_id.split(':', 1)
                key = (fu_base, int(cycle_str))
            else:
                continue

            if key in fu_usage:
                error = f"Resource conflict: {node_id} and {fu_usage[key]} both use {key[0]} at cycle {key[1]}"
                errors.append(error)
                print(f"❌ {error}")
            else:
                fu_usage[key] = node_id

        if not errors or len(errors) == len(missing) if missing else True:
            print(f"✓ No resource conflicts ({len(fu_usage)} FU usages)")

        # 3. Check routing path connectivity
        invalid_routes = []
        for (source_id, dest_id), path in routes.items():
            if len(path) == 0:
                error = f"Empty route: {source_id} → {dest_id}"
                errors.append(error)
                invalid_routes.append((source_id, dest_id))
                continue

            if len(path) == 1:
                # Single-node path (direct connection)
                continue

            # Check consecutive edges exist in MRRG
            for i in range(len(path) - 1):
                src_node_id = path[i]
                dst_node_id = path[i + 1]

                # Check if edge exists
                outgoing = mrrg.get_outgoing_edges(src_node_id)
                edge_exists = any(e.destination.id == dst_node_id for e in outgoing)

                if not edge_exists:
                    error = f"Missing edge in route {source_id}→{dest_id}: {src_node_id} → {dst_node_id}"
                    errors.append(error)
                    invalid_routes.append((source_id, dest_id))
                    break

        if invalid_routes:
            print(f"❌ Invalid routing paths: {len(invalid_routes)} routes")
            # Print first few examples
            for source_id, dest_id in invalid_routes[:3]:
                print(f"   - {source_id} → {dest_id}")
            if len(invalid_routes) > 3:
                print(f"   ... and {len(invalid_routes) - 3} more")
        else:
            print(f"✓ All {len(routes)} routing paths valid")

        # 4. Check all DFG edges have routes
        expected_routes = set()
        for edge in dfg_edges:
            expected_routes.add((edge.source.id, edge.destination.id))

        actual_routes = set(routes.keys())
        missing_routes = expected_routes - actual_routes

        if missing_routes:
            error = f"Missing routes for {len(missing_routes)} edges"
            errors.append(error)
            print(f"❌ {error}")
            # Print first few examples
            for source_id, dest_id in list(missing_routes)[:3]:
                print(f"   - {source_id} → {dest_id}")
            if len(missing_routes) > 3:
                print(f"   ... and {len(missing_routes) - 3} more")
        else:
            print(f"✓ All {len(expected_routes)} required routes present")

        print("="*70)

        is_valid = len(errors) == 0
        if is_valid:
            print("✓✓✓ SOLUTION VALIDATION PASSED ✓✓✓")
        else:
            print(f"❌❌❌ SOLUTION VALIDATION FAILED ({len(errors)} errors) ❌❌❌")
        print("="*70)

        return is_valid, errors
