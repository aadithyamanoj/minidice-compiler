"""Shared routing validation used by both PathFinder and HeuristicMapper.

All legality checks live here so the router and mapper agree on what
constitutes a valid solution.
"""

from typing import Dict, List, Optional, Set, Tuple

from cgrasolve.graph.dfg import DFG, DFGNode, OperationType
from cgrasolve.graph.mrrg import MRRG, MRRGNode, NodeType, OperandTag, HWEntityType
from cgrasolve.graph.hyperdfg import HyperDFG, HyperVal
from cgrasolve.algorithms.router.types import RouteBundle, TransportMode, TransportProfile
from cgrasolve.algorithms.scheduler.types import get_node_schedule_time


def resolve_source_output_bitwidth_info(
    src_dfg_node: DFGNode,
    src_fu: MRRGNode,
    hyperval: HyperVal,
) -> Tuple[Optional[int], bool]:
    """Resolve the source transport width and whether it is source-authoritative.

    The DFG node bitwidth represents the produced value width, while Dora FU
    metadata represents the physical port width. For mixed-precision kernels we
    must route the narrower of the two, then fall back to edge metadata only
    when no source-side width information exists.

    Returns:
        A tuple of:
          - resolved bitwidth, or ``None`` if unknown
          - ``True`` when the width came from DFG/FU source metadata
    """
    source_widths: List[int] = []

    node_bw = src_dfg_node.get_attribute("bitwidth")
    if isinstance(node_bw, int) and node_bw > 0:
        source_widths.append(node_bw)

    fu_bw = None
    if src_fu.operation_output_bitwidths:
        fu_bw = src_fu.operation_output_bitwidths.get(src_dfg_node.operation)
    if isinstance(fu_bw, int) and fu_bw > 0:
        source_widths.append(fu_bw)

    if source_widths:
        return min(source_widths), True

    # Fall back to max of non-None edge bitwidths
    if hyperval.bitwidths:
        non_none = [bw for bw in hyperval.bitwidths if bw is not None]
        if non_none:
            return max(non_none), False

    return None, False


def resolve_source_output_bitwidth(
    src_dfg_node: DFGNode,
    src_fu: MRRGNode,
    hyperval: HyperVal,
) -> Optional[int]:
    """Resolve the source transport width for a routed value."""
    bitwidth, _ = resolve_source_output_bitwidth_info(
        src_dfg_node,
        src_fu,
        hyperval,
    )
    return bitwidth


def resolve_sink_data_bitwidth(
    source_token_bitwidth: Optional[int],
    hyperval: HyperVal,
    dest_idx: int,
    token_bitwidth: Optional[int],
    source_width_is_explicit: bool = False,
) -> Optional[int]:
    """Resolve the transport width required for a specific sink.

    When the producer width comes from explicit DFG/FU metadata, that width is
    authoritative for every sink of the HyperVal. Per-edge widths only specialize
    sinks when the source width had to be inferred from edge metadata.

    Returns:
        Resolved per-sink bitwidth, or None if unknown.
    """
    if source_width_is_explicit and source_token_bitwidth is not None:
        return source_token_bitwidth

    edge_bw = hyperval.bitwidths[dest_idx]
    if source_token_bitwidth is not None and edge_bw is not None:
        return min(source_token_bitwidth, edge_bw)
    if source_token_bitwidth is not None:
        return source_token_bitwidth
    if edge_bw is not None:
        return edge_bw
    return token_bitwidth


def resolve_route_transport_profile(bundle: RouteBundle) -> TransportProfile:
    """Return the explicit transport profile attached to ``bundle``."""
    profile = bundle.transport_profile
    if profile.mode == TransportMode.UNKNOWN:
        raise ValueError("RouteBundle is missing an explicit transport_profile")
    return profile


def compute_route_latency(
    source_fu: MRRGNode,
    route: List[MRRGNode],
    mrrg: MRRG,
) -> int:
    """Compute route latency: source_fu.latency + sum(node.latency for node in route).

    Route includes routing nodes between source FU and sink FU, excludes FUs.
    """
    total = source_fu.latency
    for node in route:
        total += node.latency
    return total


def _derive_reserved_routing_function_ids(
    placement: Dict[DFGNode, MRRGNode],
) -> Set[str]:
    """Return MRRG node IDs of ROUTING_FUNCTION nodes reserved for compute."""
    reserved = set()
    for dfg_node, mrrg_node in placement.items():
        if mrrg_node.node_type == NodeType.ROUTING_FUNCTION:
            reserved.add(mrrg_node.id)
    return reserved


def collect_routing_usage(
    routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
) -> Dict[MRRGNode, Set[str]]:
    """Build a map from routing node -> set of HyperVal source_ids using it."""
    usage: Dict[MRRGNode, Set[str]] = {}
    for (hv, dest_idx), entry in routing_solution.items():
        bundle = _normalize_route_bundle(entry)
        for node in bundle.all_nodes:
            if node not in usage:
                usage[node] = set()
            usage[node].add(hv.source_id)
    return usage


def _normalize_route_bundle(
    entry: RouteBundle,
) -> RouteBundle:
    """Validate that route entries already use the RouteBundle contract."""
    if not isinstance(entry, RouteBundle):
        raise TypeError(
            f"Expected RouteBundle routing entries, got {type(entry).__name__}"
        )
    return entry


def _format_route_ref(
    source_id: str,
    sink_node_id: Optional[str],
    sink_idx: int,
    *,
    sub_path_index: int,
    num_sub_paths: int,
) -> str:
    """Return one compact route label for diagnostics."""
    sink_label = sink_node_id if sink_node_id is not None else "?"
    suffix = f" (sub-path {sub_path_index})" if num_sub_paths > 1 else ""
    return f"{source_id}->{sink_label} [sink {sink_idx}]{suffix}"


def _ordered_unique_path_ids(bundle: RouteBundle) -> List[str]:
    """Return route node IDs with stable first-seen ordering across sub-paths."""
    ordered: List[str] = []
    seen: Set[str] = set()
    for sub_path in bundle.to_node_id_paths():
        for node_id in sub_path:
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered.append(node_id)
    return ordered


def _extract_fracture_span(
    path_ids: List[str],
    mrrg: MRRG,
) -> Optional[Tuple[int, int, MRRGNode, MRRGNode]]:
    """Return the chunk-span boundaries inside a fractured sub-path.

    The span starts at the first node carrying fracture chunk metadata and ends
    at the last such node.
    """
    chunk_positions: List[Tuple[int, MRRGNode]] = []
    for idx, node_id in enumerate(path_ids):
        node = mrrg.get_node(node_id)
        if node and node.fracture_chunk_index is not None:
            chunk_positions.append((idx, node))

    if len(chunk_positions) < 2:
        return None

    start_idx, start_node = chunk_positions[0]
    end_idx, end_node = chunk_positions[-1]
    return start_idx, end_idx, start_node, end_node


def _effective_node_capacity(node: MRRGNode) -> int:
    """Return the effective routing capacity for ``node``."""
    if node.num_lanes > 1:
        return node.num_lanes
    return node.capacity


def validate_routing_solution(
    dfg: DFG,
    mrrg: MRRG,
    placement: Dict[DFGNode, MRRGNode],
    routing_solution: Dict[Tuple[HyperVal, int], RouteBundle],
    debug: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate a routing solution for correctness.

    Checks:
        1. Route connectivity (consecutive nodes have MRRG edges)
        2. All required routes present
        3. Source/sink FU connectivity
        4. Routing capacity (no overuse by distinct HyperVals)
        5. Operand pin validation
        6. Loop-back feedback paths
        7. Route timing
        8. Bitwidth legality
        9. No route traverses a compute-reserved ROUTING_FUNCTION node
        10. Fracture completeness (for fractured routes)
        11. Multi-fanout predecessor consistency (no crossbar mux conflicts)

    Args:
        dfg: Data flow graph
        mrrg: Modulo routing resource graph
        placement: DFGNode -> MRRGNode placement
        routing_solution: (HyperVal, dest_idx) -> RouteBundle or list of MRRGNode path
        debug: Print verbose output

    Returns:
        (is_valid, errors) tuple
    """
    from cgrasolve.algorithms.router.pathfinder import PathFinder

    errors: List[str] = []
    hyperdfg = HyperDFG.from_dfg(dfg)
    dfg_node_by_id = {node.id: node for node in dfg.get_nodes()}

    # Convert to ID-based dicts for lookups
    placement_ids = {dn.id: mn.id for dn, mn in placement.items()}

    # Track which routes are fractured for special validation
    fractured_routes: Dict[Tuple[str, int], RouteBundle] = {}

    # Build routes_ids: for fractured routes, validate each sub-path separately
    # but use the first sub-path for source/sink FU connectivity checks.
    # For compatibility with existing checks, routes_ids maps to all sub-path IDs.
    routes_ids: Dict[Tuple[str, int], List[str]] = {}
    routes_sub_paths: Dict[Tuple[str, int], List[List[str]]] = {}
    for (hv, dest_idx), entry in routing_solution.items():
        key = (hv.source_id, dest_idx)
        bundle = _normalize_route_bundle(entry)
        profile = resolve_route_transport_profile(bundle)
        if profile.mode == TransportMode.FRACTURED or bundle.fractured:
            fractured_routes[key] = bundle
        routes_sub_paths[key] = bundle.to_node_id_paths()
        routes_ids[key] = _ordered_unique_path_ids(bundle)

    hyperval_by_key: Dict[Tuple[str, int], HyperVal] = {}
    for hv in hyperdfg.get_edges():
        for k in range(hv.cardinality):
            hyperval_by_key[(hv.source_id, k)] = hv

    # 1. Route connectivity (validate each sub-path independently)
    for (source_id, sink_idx), sub_paths in routes_sub_paths.items():
        for sp_idx, path in enumerate(sub_paths):
            if len(path) < 2:
                continue
            for i in range(len(path) - 1):
                outgoing = mrrg.get_outgoing_edges(path[i])
                if not any(e.destination.id == path[i + 1] for e in outgoing):
                    suffix = f" (sub-path {sp_idx})" if len(sub_paths) > 1 else ""
                    errors.append(
                        f"Missing edge in route {source_id}->sink[{sink_idx}]{suffix}: "
                        f"{path[i]} -> {path[i+1]}"
                    )
                    break

    # 2. All required routes present
    expected = set()
    for hv in hyperdfg.get_edges():
        for k in range(hv.cardinality):
            expected.add((hv.source_id, k))
    missing = expected - set(routes_ids.keys())
    if missing:
        errors.append(f"Missing routes for {len(missing)} destinations")

    # 3. Source/sink FU connectivity
    for (source_id, sink_idx), sub_paths in routes_sub_paths.items():
        if not sub_paths:
            continue
        source_fu_id = placement_ids.get(source_id)
        hv = hyperval_by_key.get((source_id, sink_idx))
        sink_node_id = hv.destination_ids[sink_idx] if hv is not None else None
        sink_fu_id = placement_ids.get(sink_node_id) if sink_node_id else None
        source_dfg_node = dfg_node_by_id.get(source_id)
        sink_dfg_node = dfg_node_by_id.get(sink_node_id) if sink_node_id else None

        legal_source_ports: Set[str] = set()
        if source_fu_id and source_dfg_node:
            source_fu = mrrg.get_node(source_fu_id)
            if source_fu is not None:
                legal_source_ports = {
                    port.id for port in mrrg.get_output_ports_for_operation(
                        source_fu, source_dfg_node.operation
                    )
                }

        legal_sink_ports: Set[str] = set()
        if sink_fu_id and sink_dfg_node:
            sink_fu = mrrg.get_node(sink_fu_id)
            if sink_fu is not None:
                operand_str = hv.operands[sink_idx] if hv is not None else None
                operand_tag = PathFinder._map_operand_to_tag(None, operand_str)
                legal_sink_ports = {
                    port.id for port in mrrg.get_input_ports_for_operand(
                        sink_fu, sink_dfg_node.operation, operand_tag
                    )
                }

        for sp_idx, path_ids in enumerate(sub_paths):
            if not path_ids:
                continue
            suffix = f" (sub-path {sp_idx})" if len(sub_paths) > 1 else ""
            if legal_source_ports and path_ids[0] not in legal_source_ports:
                errors.append(
                    f"Route start {path_ids[0]} is not a legal source port for "
                    f"{source_id}{suffix}"
                )
            if source_fu_id:
                outgoing = mrrg.get_outgoing_edges(source_fu_id)
                if not any(e.destination.id == path_ids[0] for e in outgoing):
                    errors.append(
                        f"Source FU {source_fu_id} not connected to route start {path_ids[0]}{suffix}"
                    )
            if legal_sink_ports and path_ids[-1] not in legal_sink_ports:
                errors.append(
                    f"Route end {path_ids[-1]} is not a legal sink port for "
                    f"{sink_node_id}{suffix}"
                )
            if sink_fu_id:
                last_out = mrrg.get_outgoing_edges(path_ids[-1])
                if not any(e.destination.id == sink_fu_id for e in last_out):
                    errors.append(
                        f"Route end {path_ids[-1]} not connected to sink FU {sink_fu_id}{suffix}"
                    )

    # 4. Routing capacity
    routing_usage: Dict[str, Set[str]] = {}
    # Track distinct logical values by HyperVal source ID, not by sink route key.
    # A single HyperVal may legally fan out through the same node/lane to multiple
    # sinks without consuming extra capacity.
    full_width_lane_usage: Dict[str, Set[str]] = {}
    packed_lane_usage: Dict[str, Dict[int, Set[str]]] = {}
    for (hv, dest_idx), bundle in routing_solution.items():
        profile = resolve_route_transport_profile(bundle)
        if profile.mode == TransportMode.PACKED:
            lane_idx = profile.lane_index
            if lane_idx is None:
                errors.append(
                    f"Packed route {hv.source_id}->sink[{dest_idx}] is missing lane_index"
                )
                continue
            for node_id, assigned_lane in bundle.lane_assignments.items():
                packed_lane_usage.setdefault(node_id, {}).setdefault(assigned_lane, set()).add(
                    hv.source_id
                )
            continue

        route_key = (hv.source_id, dest_idx)
        for node_id in routes_ids.get(route_key, []):
            node = mrrg.get_node(node_id)
            if node is None:
                continue
            if node.num_lanes > 1:
                full_width_lane_usage.setdefault(node_id, set()).add(hv.source_id)
            else:
                routing_usage.setdefault(node_id, set()).add(hv.source_id)

    for node_id, hv_set in routing_usage.items():
        node = mrrg.get_node(node_id)
        if node and len(hv_set) > _effective_node_capacity(node):
            errors.append(
                f"Routing node {node_id} overused: {len(hv_set)}/{_effective_node_capacity(node)}"
            )

    for node_id, hypervals_using in full_width_lane_usage.items():
        node = mrrg.get_node(node_id)
        if node is None:
            continue
        if len(hypervals_using) > 1:
            errors.append(
                f"Packed-capable node {node_id} is overused by multiple full-width routes: "
                f"{sorted(hypervals_using)}"
            )
        if node_id in packed_lane_usage and packed_lane_usage[node_id]:
            errors.append(
                f"Packed-capable node {node_id} mixes full-width and packed routes"
            )

    for node_id, lane_map in packed_lane_usage.items():
        node = mrrg.get_node(node_id)
        if node is None:
            continue
        if node.num_lanes <= 1:
            errors.append(
                f"Packed route assigns lane usage on non-lane-capable node {node_id}"
            )
            continue
        if len(lane_map) > node.num_lanes:
            errors.append(
                f"Packed-capable node {node_id} uses {len(lane_map)}/{node.num_lanes} lanes"
            )
        for lane_idx, hypervals_using in lane_map.items():
            if lane_idx < 0 or lane_idx >= node.num_lanes:
                errors.append(
                    f"Packed-capable node {node_id} uses invalid lane index {lane_idx}"
                )
            elif not node.allows_lane_index(lane_idx):
                errors.append(
                    f"Packed-capable node {node_id} uses disallowed lane index {lane_idx}"
                )
            if len(hypervals_using) > 1:
                errors.append(
                    f"Packed-capable node {node_id} lane {lane_idx} is shared by multiple routes: "
                    f"{sorted(hypervals_using)}"
                )

    # 5. Operand pin validation
    for (source_id, sink_idx), sub_paths in routes_sub_paths.items():
        if not sub_paths:
            continue
        hv_match = hyperval_by_key.get((source_id, sink_idx))
        if hv_match is None:
            continue
        operand_str = hv_match.operands[sink_idx]
        if operand_str is None:
            continue
        dest_node_id = hv_match.destination_ids[sink_idx]
        dest_dfg_node = next((n for n in dfg.nodes if n.id == dest_node_id), None)
        if dest_dfg_node is None:
            continue
        dest_op = dest_dfg_node.operation
        operand_tag = PathFinder._map_operand_to_tag(None, operand_str)
        if operand_tag is None:
            continue
        for sp_idx, path_ids in enumerate(sub_paths):
            if not path_ids:
                continue
            last_node = mrrg.get_node(path_ids[-1])
            if last_node is None:
                continue
            if last_node.operation_supported_operand_tags:
                pin_tags = last_node.get_supported_operand_tags_for_operation(dest_op)
                suffix = f" (sub-path {sp_idx})" if len(sub_paths) > 1 else ""
                if not pin_tags:
                    errors.append(
                        f"Operand pin error: route {source_id}->sink[{sink_idx}] "
                        f"ends at pin '{last_node.id}' which has no valid tags "
                        f"for operation {dest_op.value}{suffix}"
                    )
                elif not PathFinder._operand_tag_matches_request(operand_tag, pin_tags):
                    errors.append(
                        f"Operand pin error: route {source_id}->sink[{sink_idx}] "
                        f"operand '{operand_str}' routed to pin '{last_node.id}' "
                        f"which supports {pin_tags} for {dest_op.value}{suffix}"
                    )

    # 6. Loop-back feedback
    for hv in hyperdfg.get_edges():
        for k in range(hv.cardinality):
            if not hv.is_loop_backs[k]:
                continue
            dist = hv.dists[k] if hv.dists[k] is not None else 0
            if dist <= 0:
                continue
            path = routes_ids.get((hv.source_id, k), [])
            if not path:
                dest_str = "self" if hv.source_id == hv.destination_ids[k] else hv.destination_ids[k]
                errors.append(
                    f"Loop-back error: {hv.source_id}->{dest_str} [sink {k}] "
                    f"has empty feedback path"
                )

    # 7. Route timing (validate each sub-path)
    for hv in hyperdfg.get_edges():
        for k in range(hv.cardinality):
            route_key = (hv.source_id, k)
            sub_paths_for_key = routes_sub_paths.get(route_key, [])
            if not sub_paths_for_key or not sub_paths_for_key[0]:
                continue
            src_dfg = next((n for n in dfg.nodes if n.id == hv.source_id), None)
            dst_dfg = next((n for n in dfg.nodes if n.id == hv.destination_ids[k]), None)
            if not src_dfg or not dst_dfg:
                continue
            src_time = get_node_schedule_time(src_dfg)
            dst_time = get_node_schedule_time(dst_dfg)
            if src_time is None or dst_time is None:
                continue
            edge_dist = hv.dists[k] if hv.dists[k] is not None else 0
            # Expected cycles (same as PathFinder._compute_cycles_source_to_sink)
            II = mrrg.II
            if II is not None:
                expected = (dst_time + edge_dist * II) - src_time
            else:
                expected = dst_time - src_time
            if II is not None and src_dfg.id == dst_dfg.id and edge_dist == 0:
                expected = II
            elif II is not None and expected < 0:
                expected += II
            # Actual latency: source FU + routing path
            source_fu_id = placement_ids.get(hv.source_id)
            source_fu = mrrg.get_node(source_fu_id) if source_fu_id else None
            src_lat = source_fu.latency if source_fu else 0
            # Validate each sub-path has matching latency
            for sp_idx, path_ids in enumerate(sub_paths_for_key):
                total_latency = src_lat
                for nid in path_ids:
                    node = mrrg.get_node(nid)
                    if node:
                        total_latency += node.latency
                if total_latency != expected:
                    suffix = f" (sub-path {sp_idx})" if len(sub_paths_for_key) > 1 else ""
                    errors.append(
                        f"Route timing mismatch: {hv.source_id}->{hv.destination_ids[k]} "
                        f"[sink {k}]{suffix}: got {total_latency} cycles, expected {expected}"
                    )

    # 8. Bitwidth legality
    for hv in hyperdfg.get_edges():
        src_fu_id = placement_ids.get(hv.source_id)
        src_fu = mrrg.get_node(src_fu_id) if src_fu_id else None
        src_dfg = next((n for n in dfg.nodes if n.id == hv.source_id), None)
        source_token_bitwidth = None
        if src_fu is not None and src_dfg is not None:
            source_token_bitwidth, source_width_is_explicit = resolve_source_output_bitwidth_info(
                src_dfg,
                src_fu,
                hv,
            )
        elif hv.bitwidths:
            non_none_bws = [bw for bw in hv.bitwidths if bw is not None]
            if non_none_bws:
                source_token_bitwidth = max(non_none_bws)
            source_width_is_explicit = False
        else:
            source_width_is_explicit = False
        for k in range(hv.cardinality):
            route_key = (hv.source_id, k)
            sub_paths_for_key = routes_sub_paths.get(route_key, [])
            if not sub_paths_for_key:
                continue
            data_bw = resolve_sink_data_bitwidth(
                source_token_bitwidth,
                hv,
                k,
                source_token_bitwidth,
                source_width_is_explicit=source_width_is_explicit,
            )
            entry = routing_solution.get((hv, k))
            if entry is None:
                raise ValueError(
                    f"Missing RouteBundle entry for route {hv.source_id}->{hv.destination_ids[k]} [sink {k}]"
                )
            bundle = _normalize_route_bundle(entry)
            profile = resolve_route_transport_profile(bundle)
            is_fractured = profile.mode == TransportMode.FRACTURED or route_key in fractured_routes
            for sp_idx, path_ids in enumerate(sub_paths_for_key):
                if is_fractured:
                    span = _extract_fracture_span(path_ids, mrrg)
                    if span is None:
                        errors.append(
                            f"Fracture span missing: {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] (sub-path {sp_idx})"
                        )
                        continue
                    start_idx, end_idx, start_node, end_node = span
                    chunk_bw = start_node.bitwidth
                    if end_node.bitwidth != chunk_bw:
                        errors.append(
                            f"Chunk width mismatch: {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] (sub-path {sp_idx}) uses {chunk_bw}-bit source chunk "
                            f"but {end_node.bitwidth}-bit sink chunk"
                        )
                    if bundle and sp_idx < len(bundle.chunk_widths) and bundle.chunk_widths[sp_idx] != chunk_bw:
                        errors.append(
                            f"Chunk metadata mismatch: {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] (sub-path {sp_idx}) expected "
                            f"{bundle.chunk_widths[sp_idx]}-bit chunk but routed {chunk_bw}-bit chunk"
                        )
                    for nid in path_ids[start_idx:end_idx + 1]:
                        node = mrrg.get_node(nid)
                        if node and node.get_attribute("is_instance", False):
                            continue
                        if node and node.bitwidth != chunk_bw:
                            errors.append(
                                f"Bitwidth violation: fractured chunk {chunk_bw}-bit path uses "
                                f"{node.bitwidth}-bit node {nid} (sub-path {sp_idx})"
                            )
                            break
                elif profile.mode == TransportMode.PACKED:
                    lane_idx = profile.lane_index
                    if lane_idx is None:
                        errors.append(
                            f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] is missing lane_index"
                        )
                        continue
                    assigned_positions = [
                        idx for idx, nid in enumerate(path_ids) if nid in bundle.lane_assignments
                    ]
                    if not assigned_positions:
                        errors.append(
                            f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] has no lane assignments"
                        )
                        continue

                    first_lane_idx = assigned_positions[0]
                    last_lane_idx = assigned_positions[-1]
                    pack_positions = [
                        idx
                        for idx in assigned_positions
                        if (node := mrrg.get_node(path_ids[idx])) is not None and node.pack_capable
                    ]
                    unpack_positions = [
                        idx
                        for idx in assigned_positions
                        if (node := mrrg.get_node(path_ids[idx])) is not None and node.unpack_capable
                    ]
                    first_pack_idx = pack_positions[0] if pack_positions else None
                    last_unpack_idx = unpack_positions[-1] if unpack_positions else None
                    first_pack_node = (
                        mrrg.get_node(path_ids[first_pack_idx])
                        if first_pack_idx is not None else None
                    )
                    last_unpack_node = (
                        mrrg.get_node(path_ids[last_unpack_idx])
                        if last_unpack_idx is not None else None
                    )
                    if first_pack_node is None:
                        errors.append(
                            f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] does not start on a pack-capable node"
                        )
                    if last_unpack_node is None:
                        errors.append(
                            f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                            f"[sink {k}] does not end on an unpack-capable node"
                        )

                    if bundle.active_pack_nodes:
                        active_pack_positions = [
                            idx
                            for idx, nid in enumerate(path_ids)
                            if nid in bundle.active_pack_nodes
                        ]
                        if not active_pack_positions:
                            errors.append(
                                f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                f"[sink {k}] active pack nodes are missing from the path"
                            )
                        elif (
                            first_pack_idx is not None
                            and min(active_pack_positions) != first_pack_idx
                        ):
                            errors.append(
                                f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                f"[sink {k}] has mismatched active pack node metadata"
                            )
                        for node_id in bundle.active_pack_nodes:
                            node = mrrg.get_node(node_id)
                            if node_id not in bundle.lane_assignments:
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] marks pack node {node_id} active without a lane"
                                )
                            elif node is None or not node.pack_capable:
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] marks non-pack node {node_id} active"
                                )
                    if bundle.active_unpack_nodes:
                        active_unpack_positions = [
                            idx
                            for idx, nid in enumerate(path_ids)
                            if nid in bundle.active_unpack_nodes
                        ]
                        if not active_unpack_positions:
                            errors.append(
                                f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                f"[sink {k}] active unpack nodes are missing from the path"
                            )
                        elif (
                            last_unpack_idx is not None
                            and max(active_unpack_positions) != last_unpack_idx
                        ):
                            errors.append(
                                f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                f"[sink {k}] has mismatched active unpack node metadata"
                            )
                        for node_id in bundle.active_unpack_nodes:
                            node = mrrg.get_node(node_id)
                            if node_id not in bundle.lane_assignments:
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] marks unpack node {node_id} active without a lane"
                                )
                            elif node is None or not node.unpack_capable:
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] marks non-unpack node {node_id} active"
                                )

                    for idx, nid in enumerate(path_ids):
                        node = mrrg.get_node(nid)
                        if node is None:
                            continue
                        assigned_lane = bundle.lane_assignments.get(nid)
                        if assigned_lane is not None:
                            if assigned_lane != lane_idx:
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] uses inconsistent lane indices"
                                )
                            if node.num_lanes <= 1 or node.lane_width < (data_bw or 0):
                                errors.append(
                                    f"Bitwidth violation: packed transport on {nid} "
                                    f"cannot carry {data_bw}-bit data"
                                )
                            if assigned_lane < 0 or assigned_lane >= node.num_lanes:
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] uses invalid lane {assigned_lane} on {nid}"
                                )
                            elif not node.allows_lane_index(assigned_lane):
                                errors.append(
                                    f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                    f"[sink {k}] uses disallowed lane {assigned_lane} on {nid}"
                                )
                        elif first_lane_idx <= idx <= last_lane_idx and node.num_lanes > 1:
                            errors.append(
                                f"Packed route {hv.source_id}->{hv.destination_ids[k]} "
                                f"[sink {k}] leaves lane-capable node {nid} unassigned "
                                f"inside packed span"
                            )
                else:
                    for nid in path_ids:
                        node = mrrg.get_node(nid)
                        if node and profile.rejects_node_width(node):
                            suffix = f" (sub-path {sp_idx})" if len(sub_paths_for_key) > 1 else ""
                            errors.append(
                                f"Bitwidth violation: {profile.mode.value} transport on "
                                f"{node.bitwidth}-bit wire {nid}{suffix}"
                            )
                            break

    # 9. No route traverses a compute-reserved ROUTING_FUNCTION node
    reserved_ids = _derive_reserved_routing_function_ids(placement)
    for (source_id, sink_idx), path_ids in routes_ids.items():
        for nid in path_ids:
            if nid in reserved_ids:
                errors.append(
                    f"Route {source_id}->sink[{sink_idx}] traverses compute-reserved "
                    f"ROUTING_FUNCTION node {nid}"
                )
                break

    # 10. Fracture completeness
    for route_key, bundle in fractured_routes.items():
        source_id, sink_idx = route_key
        num_chunks = bundle.num_chunks
        if len(bundle.sub_paths) != num_chunks:
            errors.append(
                f"Fracture incomplete: {source_id}->sink[{sink_idx}] has "
                f"{len(bundle.sub_paths)} sub-paths, expected {num_chunks}"
            )
            continue

        hv = hyperval_by_key.get(route_key)
        if hv is None:
            continue

        src_dfg = next((n for n in dfg.nodes if n.id == source_id), None)
        src_fu_id = placement_ids.get(source_id)
        src_fu = mrrg.get_node(src_fu_id) if src_fu_id else None
        source_token_bitwidth = None
        if src_fu and src_dfg:
            source_token_bitwidth, source_width_is_explicit = resolve_source_output_bitwidth_info(
                src_dfg,
                src_fu,
                hv,
            )
        elif hv.bitwidths:
            non_none_bws = [bw for bw in hv.bitwidths if bw is not None]
            if non_none_bws:
                source_token_bitwidth = max(non_none_bws)
            source_width_is_explicit = False
        else:
            source_width_is_explicit = False
        data_bw = resolve_sink_data_bitwidth(
            source_token_bitwidth,
            hv,
            sink_idx,
            source_token_bitwidth,
            source_width_is_explicit=source_width_is_explicit,
        )

        if bundle.chunk_indices != list(range(num_chunks)):
            errors.append(
                f"Fracture chunk indices invalid: {source_id}->sink[{sink_idx}] has "
                f"{bundle.chunk_indices}, expected {list(range(num_chunks))}"
            )

        if len(bundle.chunk_widths) != num_chunks or len(bundle.chunk_offsets) != num_chunks:
            errors.append(
                f"Fracture metadata incomplete: {source_id}->sink[{sink_idx}] missing "
                f"width/offset info for all {num_chunks} chunks"
            )
            continue

        covered_bits: Set[int] = set()
        coverage_overlap = False
        for width, offset in zip(bundle.chunk_widths, bundle.chunk_offsets):
            for bit in range(offset, offset + width):
                if bit in covered_bits:
                    coverage_overlap = True
                covered_bits.add(bit)
        if coverage_overlap:
            errors.append(
                f"Fracture coverage overlap: {source_id}->sink[{sink_idx}] has overlapping chunk ranges"
            )
        if data_bw is not None and covered_bits != set(range(data_bw)):
            errors.append(
                f"Fracture coverage mismatch: {source_id}->sink[{sink_idx}] covers "
                f"{sorted(covered_bits)}, expected bits 0..{data_bw - 1}"
            )

        seen_interior_ids: Set[str] = set()
        sub_paths_for_key = routes_sub_paths.get(route_key, [])
        for sp_idx, path_ids in enumerate(sub_paths_for_key):
            span = _extract_fracture_span(path_ids, mrrg)
            if span is None:
                errors.append(
                    f"Fracture span missing: {source_id}->sink[{sink_idx}] "
                    f"(sub-path {sp_idx})"
                )
                continue

            start_idx, end_idx, start_node, end_node = span
            expected_idx = bundle.chunk_indices[sp_idx]
            expected_width = bundle.chunk_widths[sp_idx]
            expected_offset = bundle.chunk_offsets[sp_idx]

            if start_node.fracture_chunk_index != end_node.fracture_chunk_index:
                errors.append(
                    f"Fracture chunk mismatch: {source_id}->sink[{sink_idx}] "
                    f"(sub-path {sp_idx}) starts on chunk {start_node.fracture_chunk_index} "
                    f"and ends on chunk {end_node.fracture_chunk_index}"
                )
            if start_node.fracture_chunk_index != expected_idx:
                errors.append(
                    f"Fracture chunk ordering mismatch: {source_id}->sink[{sink_idx}] "
                    f"(sub-path {sp_idx}) routed chunk {start_node.fracture_chunk_index}, "
                    f"expected {expected_idx}"
                )
            if start_node.fracture_bit_offset != expected_offset or end_node.fracture_bit_offset != expected_offset:
                errors.append(
                    f"Fracture offset mismatch: {source_id}->sink[{sink_idx}] "
                    f"(sub-path {sp_idx}) does not preserve bit offset {expected_offset}"
                )
            if start_node.bitwidth != expected_width or end_node.bitwidth != expected_width:
                errors.append(
                    f"Fracture width mismatch: {source_id}->sink[{sink_idx}] "
                    f"(sub-path {sp_idx}) does not preserve chunk width {expected_width}"
                )

            interior_ids = set(path_ids[start_idx:end_idx + 1])
            overlap = seen_interior_ids & interior_ids
            if overlap:
                errors.append(
                    f"Fracture chunk overlap: {source_id}->sink[{sink_idx}] "
                    f"(sub-path {sp_idx}) reuses chunk-routing nodes {sorted(overlap)}"
                )
            seen_interior_ids |= interior_ids

    # 11. Multi-fanout predecessor consistency
    # For each HyperVal with multiple sinks, verify that shared routing nodes
    # are entered from the same predecessor in all sub-routes.  Two routes
    # entering a crossbar mux output via different input ports would require
    # conflicting mux select values in the generated FASM.
    for hv in hyperdfg.get_edges():
        if hv.cardinality <= 1:
            continue
        pred_map: Dict[str, str] = {}  # node_id -> required predecessor_id
        for k in range(hv.cardinality):
            route_key = (hv.source_id, k)
            sub_paths_for_key = routes_sub_paths.get(route_key, [])
            src_fu_id = placement_ids.get(hv.source_id)
            for path_ids in sub_paths_for_key:
                chain = ([src_fu_id] if src_fu_id else []) + list(path_ids)
                for i in range(1, len(chain)):
                    node_id = chain[i]
                    pred_id = chain[i - 1]
                    if node_id in pred_map:
                        if pred_map[node_id] != pred_id:
                            errors.append(
                                f"Predecessor conflict: node '{node_id}' in "
                                f"HyperVal from '{hv.source_id}' is reached "
                                f"via '{pred_map[node_id]}' (earlier sink) "
                                f"and '{pred_id}' (sink {k}) — crossbar mux "
                                f"select would conflict"
                            )
                    else:
                        pred_map[node_id] = pred_id

    return len(errors) == 0, errors
