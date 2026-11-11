"""Graph analysis algorithms for DFG scheduling and analysis."""

from typing import Dict, List, Set, Optional, Tuple
from cgrasolve.graph.dfg import DFG, DFGNode, DFGEdge


def compute_asap_schedule(dfg: DFG) -> Dict[str, int]:
    """
    Compute As-Soon-As-Possible (ASAP) schedule.

    Args:
        dfg: Data Flow Graph to schedule

    Returns:
        Dictionary mapping node IDs to their ASAP times
    """
    asap_times: Dict[str, int] = {}
    visited: Set[str] = set()

    # Get loop-back edge IDs to ignore during scheduling
    loop_edges = {e.id for e in dfg.get_loop_back_edges()}

    def _compute_asap(node_id: str) -> int:
        if node_id in asap_times:
            return asap_times[node_id]

        if node_id in visited:
            raise ValueError(f"Cycle detected at node {node_id}")

        visited.add(node_id)
        node = dfg.get_node(node_id)

        if not node:
            raise ValueError(f"Node {node_id} not found")

        # Get predecessors (excluding loop-back edges)
        pred_edges = [e for e in dfg.get_incoming_edges(node_id) if e.id not in loop_edges]

        if not pred_edges:
            # Input node
            asap_time = 0
        else:
            # ASAP time is max of predecessor ASAP times + 1
            asap_time = max(
                _compute_asap(e.source.id) + 1
                for e in pred_edges
            )

        visited.remove(node_id)
        asap_times[node_id] = asap_time
        node.asap_time = asap_time

        return asap_time

    for node in dfg.get_nodes():
        if node.id not in asap_times:
            _compute_asap(node.id)

    return asap_times


def compute_alap_schedule(
    dfg: DFG,
    max_latency: Optional[int] = None
) -> Dict[str, int]:
    """
    Compute As-Late-As-Possible (ALAP) schedule.

    Args:
        dfg: Data Flow Graph to schedule
        max_latency: Maximum latency constraint (if None, uses max ASAP time)

    Returns:
        Dictionary mapping node IDs to their ALAP times
    """
    # First compute ASAP times
    asap_times = compute_asap_schedule(dfg)

    if max_latency is None:
        max_latency = max(asap_times.values()) if asap_times else 0

    alap_times: Dict[str, int] = {}
    visited: Set[str] = set()

    # Get loop-back edge IDs
    loop_edges = {e.id for e in dfg.get_loop_back_edges()}

    def _compute_alap(node_id: str) -> int:
        if node_id in alap_times:
            return alap_times[node_id]

        if node_id in visited:
            raise ValueError(f"Cycle detected at node {node_id}")

        visited.add(node_id)
        node = dfg.get_node(node_id)

        if not node:
            raise ValueError(f"Node {node_id} not found")

        # Get successors (excluding loop-back edges)
        succ_edges = [e for e in dfg.get_outgoing_edges(node_id) if e.id not in loop_edges]

        if not succ_edges:
            # Output node
            alap_time = max_latency
        else:
            # ALAP time is min of (successor ALAP - 1)
            alap_time = min(
                _compute_alap(e.destination.id) - 1
                for e in succ_edges
            )

        visited.remove(node_id)
        alap_times[node_id] = alap_time
        node.alap_time = alap_time

        return alap_time

    for node in dfg.get_nodes():
        if node.id not in alap_times:
            _compute_alap(node.id)

    return alap_times


def compute_mobility(dfg: DFG) -> Dict[str, int]:
    """
    Compute scheduling mobility (ALAP - ASAP) for each node.

    Args:
        dfg: Data Flow Graph

    Returns:
        Dictionary mapping node IDs to their mobility values
    """
    asap_times = compute_asap_schedule(dfg)
    alap_times = compute_alap_schedule(dfg)

    mobility: Dict[str, int] = {}
    for node in dfg.get_nodes():
        mob = alap_times[node.id] - asap_times[node.id]
        mobility[node.id] = mob
        node.mobility = mob

    return mobility


def find_critical_path(dfg: DFG) -> Tuple[List[DFGNode], int]:
    """
    Find the critical path in the DFG (longest path from inputs to outputs).

    Args:
        dfg: Data Flow Graph

    Returns:
        Tuple of (critical path nodes, critical path length)
    """
    # Compute ASAP times
    asap_times = compute_asap_schedule(dfg)

    # Find the node with maximum ASAP time
    max_time = 0
    critical_end_node = None

    for node in dfg.get_nodes():
        total_time = asap_times[node.id]
        if total_time > max_time:
            max_time = total_time
            critical_end_node = node

    if not critical_end_node:
        return [], 0

    # Backtrack to find the critical path
    critical_path: List[DFGNode] = []
    current_node = critical_end_node
    loop_edges = {e.id for e in dfg.get_loop_back_edges()}

    while current_node:
        critical_path.append(current_node)

        # Find predecessor with maximum ASAP time
        pred_edges = [
            e for e in dfg.get_incoming_edges(current_node.id)
            if e.id not in loop_edges
        ]

        if not pred_edges:
            break

        # Choose predecessor that led to this node's ASAP time
        current_asap = asap_times[current_node.id]
        next_node = None

        for edge in pred_edges:
            pred_node = edge.source
            if asap_times[pred_node.id] + 1 == current_asap:
                next_node = pred_node
                break

        current_node = next_node

    critical_path.reverse()

    # Store critical path length in DFG
    dfg.set_critical_path_length(max_time)

    return critical_path, max_time


def detect_cycles(dfg: DFG, ignore_loop_edges: bool = True) -> List[List[str]]:
    """
    Detect cycles in the DFG.

    Args:
        dfg: Data Flow Graph
        ignore_loop_edges: If True, ignore marked loop-back edges

    Returns:
        List of cycles, where each cycle is a list of node IDs
    """
    loop_edges = {e.id for e in dfg.get_loop_back_edges()} if ignore_loop_edges else set()

    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    cycles: List[List[str]] = []
    current_path: List[str] = []

    def _visit(node_id: str) -> None:
        visited.add(node_id)
        rec_stack.add(node_id)
        current_path.append(node_id)

        for edge in dfg.get_outgoing_edges(node_id):
            if edge.id in loop_edges:
                continue

            succ_id = edge.destination.id

            if succ_id not in visited:
                _visit(succ_id)
            elif succ_id in rec_stack:
                # Found a cycle
                cycle_start = current_path.index(succ_id)
                cycle = current_path[cycle_start:] + [succ_id]
                cycles.append(cycle)

        current_path.pop()
        rec_stack.remove(node_id)

    for node in dfg.get_nodes():
        if node.id not in visited:
            _visit(node.id)

    return cycles


def find_strongly_connected_components(dfg: DFG) -> List[List[str]]:
    """
    Find strongly connected components using Tarjan's algorithm.

    Args:
        dfg: Data Flow Graph

    Returns:
        List of SCCs, where each SCC is a list of node IDs
    """
    index_counter = [0]
    stack: List[str] = []
    lowlinks: Dict[str, int] = {}
    index: Dict[str, int] = {}
    on_stack: Set[str] = set()
    sccs: List[List[str]] = []

    def _strongconnect(node_id: str) -> None:
        index[node_id] = index_counter[0]
        lowlinks[node_id] = index_counter[0]
        index_counter[0] += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for successor in dfg.get_successors(node_id):
            succ_id = successor.id
            if succ_id not in index:
                _strongconnect(succ_id)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[succ_id])
            elif succ_id in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], index[succ_id])

        if lowlinks[node_id] == index[node_id]:
            scc: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node_id:
                    break
            sccs.append(scc)

    for node in dfg.get_nodes():
        if node.id not in index:
            _strongconnect(node.id)

    return sccs


def compute_resource_pressure(
    dfg: DFG,
    schedule: Optional[Dict[str, int]] = None
) -> Dict[int, int]:
    """
    Compute resource pressure (number of operations at each time step).

    Args:
        dfg: Data Flow Graph
        schedule: Node schedule (if None, uses ASAP schedule)

    Returns:
        Dictionary mapping time steps to number of operations
    """
    if schedule is None:
        schedule = compute_asap_schedule(dfg)

    pressure: Dict[int, int] = {}

    for node in dfg.get_nodes():
        time = schedule.get(node.id, 0)
        pressure[time] = pressure.get(time, 0) + 1

    return pressure
