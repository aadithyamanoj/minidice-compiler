"""Compute the physical pipeline depth (in cycles) of a mapped p-graph.

Mini_Dice's dispatcher tags each thread's CGRA output with `tid + lat`, where
`lat` tells it how many clock edges the CGRA fabric takes to propagate a
single thread's inputs to its outputs. If `lat` is too small the writeback
gets credited to the wrong thread (thread N's value lands in thread
N + (real - lat)'s register slot).

Mini_Dice's pipeline model -- verified against the static-build RTL:
  * Each switch box (SB) has exactly one register (`reg_16b out_reg_<dir>`)
    on every output direction. A route that traverses N SBs passes through
    N flops. The path `*.out_reg_<dir>` MRRG node names this flop.
  * Every other element is purely combinational:
      - The xbars (`xbar_data_in`, `xbar_data_out`, `xbar_mem_addr`,
        `xbar_mem_data`) are `always_comb` muxes.
      - The PE (`pe.sv`) has no `always_ff` and no `reg_16b`; its input
        muxes, ALU control, and output muxes all settle within one cycle.
      - The chip-boundary pins (`regS_i_0`, `csrX_i_*`, `ext_data_o_*`,
        `mem_addr_o_*`, `mem_data_o_*`) are wired straight to the xbars.

So the physical pipeline depth equals the **count of SB flops on the longest
end-to-end DFG path** from a chip-input pin to a chip-output pin. PEs in the
path contribute zero cycles. We compute this by topological-DP across the
DFG, accumulating per-edge SB-flop counts through every chain of placements.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple


# Match the bare `*.out_reg_<dir>` flop node -- not its `.data_i` / `.data_o`
# neighbors (which would double-count one SB hop's single flop). Anchored to
# the end of the node id.
_OUT_REG_RE = re.compile(r'\.out_reg_[A-Z]+$')

# Chip-boundary pin name patterns (sources and sinks). Anything else is a
# PE FU placement (we don't actually care which PE -- PEs add 0 cycles).
_INPUT_PIN_RE  = re.compile(r'^(csrX_i_\d+|regS_i_\d+|ext_data_i_\d+|ext_pred_i_\d+)$')
_OUTPUT_PIN_RE = re.compile(r'^(mem_addr_o_\d+|mem_data_o_\d+|ext_data_o_\d+|ext_pred_o_\d+)$')


def _count_route_register_stages(path: List[str]) -> int:
    """Count `*.out_reg_<dir>` flops traversed by one MRRG path. One flop
    per SB hop."""
    return sum(1 for n in path if _OUT_REG_RE.search(n))


def _is_chip_input(placement_id: str) -> bool:
    return placement_id is not None and _INPUT_PIN_RE.match(placement_id) is not None


def _is_chip_output(placement_id: str) -> bool:
    return placement_id is not None and _OUTPUT_PIN_RE.match(placement_id) is not None


def compute_pipeline_depth(
    routes: Dict[Tuple[str, int], List[List[str]]],
    placement: Dict[str, str],
    hyperdfg=None,
) -> int:
    """Return the longest input-pin → output-pin pipeline depth in cycles.

    Topological DP over the placed DFG:
      depth[input_dfg]  = 0
      depth[N]          = max over predecessors P of
                            depth[P] + flops_on_route(P -> N)
      result            = max depth at any chip-output DFG node

    PEs contribute 0 cycles (combinational); only SB hops count.

    Args:
        routes:    mapper result["routes"] -- {(src_dfg, dest_idx): [paths]}
        placement: mapper result["placement"] -- {dfg_id: mrrg_id}
        hyperdfg:  unused (kept for API symmetry with existing call sites);
                   the edge set is recovered from `routes` directly.

    Returns:
        Number of clock cycles in the longest source-to-sink path; floor 1.
    """
    # Per-(src_dfg, dest_dfg) max route flop count. We index by
    # dest-dfg-id by looking up the route's terminating MRRG node and
    # mapping it back through the placement.
    placement_inv: Dict[str, str] = {}
    for dfg_id, mrrg_id in placement.items():
        placement_inv[mrrg_id] = dfg_id

    edge_flops: Dict[Tuple[str, str], int] = defaultdict(int)
    successors: Dict[str, Set[str]]        = defaultdict(set)

    for (src_dfg, _dest_idx), sub_paths in routes.items():
        if src_dfg not in placement:
            continue
        for path in sub_paths:
            if not path:
                continue
            flops = _count_route_register_stages(path)
            # Find the destination DFG node by walking the path tail.
            # The route terminates at one of:
            #   * a chip-output pin (mem_addr_o_*, ext_data_o_*, etc.)
            #   * a PE ALU port (`<pe>.alu_inst.A` / `.B`)
            # In the first case the placement-inverse hits an OUTPUT dfg
            # node directly. In the second we walk the path tail to find
            # the bare `*.alu_inst` and look up which DFG op landed there.
            dest_dfg = None
            tail = path[-1]
            # Try three forms of the tail: the raw node id, the FU bare id
            # (strip `.A`/`.B`/`.out` port suffix), and the canonical chip-pin
            # name (resolve `xbar_*.data_o_N` back to its chip-pin alias).
            for candidate in (tail, _strip_port(tail), _canonical_pin(tail)):
                if candidate in placement_inv:
                    dest_dfg = placement_inv[candidate]
                    break
            if dest_dfg is None:
                continue
            key = (src_dfg, dest_dfg)
            if flops > edge_flops[key]:
                edge_flops[key] = flops
            successors[src_dfg].add(dest_dfg)

    # Topological DP. depth defaults to 0 for chip-input sources.
    depth: Dict[str, int]  = {dfg: 0 for dfg in placement
                              if _is_chip_input(placement[dfg])}
    in_degree: Dict[str, int] = defaultdict(int)
    for (src, dst) in edge_flops:
        in_degree[dst] += 1

    queue = deque([n for n in placement if _is_chip_input(placement[n])])
    while queue:
        node = queue.popleft()
        for succ in successors.get(node, ()):
            cand = depth[node] + edge_flops[(node, succ)]
            if cand > depth.get(succ, -1):
                depth[succ] = cand
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # Max depth at any chip-output DFG node.
    best = 1
    for dfg, mrrg in placement.items():
        if _is_chip_output(mrrg) and dfg in depth:
            if depth[dfg] > best:
                best = depth[dfg]
    return best


def _strip_port(mrrg_node: str) -> str:
    """`...alu_inst.A` -> `...alu_inst`; `...alu_inst.out` -> `...alu_inst`."""
    if '.' not in mrrg_node:
        return mrrg_node
    head, _, tail = mrrg_node.rpartition('.')
    if head.endswith('.alu_inst') or head.endswith('alu_inst'):
        return head
    return mrrg_node


# Routes that drive a chip output don't terminate at the chip pin -- they
# terminate at the xbar's internal output net (the xbar's `data_o_N` wire).
# dice_top wires that net straight to the chip pin via a simple `assign`,
# so for pipeline-depth purposes they're the same node.
_XBAR_OUT_RE = re.compile(
    r'^xbar_(?:data_out|mem_addr|mem_data)_inst\.data_o_(\d+)$'
)
_XBAR_TO_PIN_PREFIX = {
    "xbar_data_out_inst": "ext_data_o_",
    "xbar_mem_addr_inst": "mem_addr_o_",
    "xbar_mem_data_inst": "mem_data_o_",
}


def _canonical_pin(mrrg_node: str) -> str:
    """If `mrrg_node` is an xbar output net feeding a chip pin, return the
    chip-pin name (the form that appears in `placement.values()`)."""
    m = _XBAR_OUT_RE.match(mrrg_node)
    if not m:
        return mrrg_node
    xbar_inst = mrrg_node.split('.', 1)[0]
    idx       = m.group(1)
    prefix    = _XBAR_TO_PIN_PREFIX.get(xbar_inst)
    return f"{prefix}{idx}" if prefix else mrrg_node
