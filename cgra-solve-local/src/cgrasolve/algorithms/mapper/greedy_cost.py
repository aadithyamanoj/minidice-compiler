"""HyCUBE-style cost functions for the greedy mapper.

Implements the four cost functions from the HyCUBE DAC'17 paper
(Karunaratne et al.) plus an additional width-based cost:

- SRC  (Static Routing Cost)       -- rewards multicast link reuse
- UARC (Used Adjacent Resource Cost) -- penalises congested regions
- MRC  (Memory Resource Cost)      -- preserves scarce memory FUs
- Width                            -- penalises bitwidth mismatches
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from cgrasolve.graph.dfg import DFG, DFGNode, OperationType
from cgrasolve.graph.mrrg import MRRG, MRRGNode

# Operations that require memory-capable FUs.
_MEMORY_OPS: frozenset[OperationType] = frozenset({
    OperationType.LOAD,
    OperationType.STORE,
})


class GreedyCostEvaluator:
    """Evaluate HyCUBE-inspired placement cost functions.

    Each cost function is exposed as a standalone method so that unit
    tests can exercise them individually.  The ``evaluate`` entry-point
    returns the weighted sum.
    """

    def __init__(
        self,
        mrrg: MRRG,
        dfg: DFG,
        *,
        w_src: float = 1.0,
        w_uarc: float = 1.0,
        w_mrc: float = 1.0,
        w_width: float = 1.0,
    ) -> None:
        self._mrrg = mrrg
        self._dfg = dfg
        self._w_src = w_src
        self._w_uarc = w_uarc
        self._w_mrc = w_mrc
        self._w_width = w_width

        # Pre-compute the set of memory-capable FU IDs in the MRRG.
        self._memory_fu_ids: Set[str] = set()
        for fu in mrrg.get_fu_nodes():
            if any(fu.can_execute(op) for op in _MEMORY_OPS):
                self._memory_fu_ids.add(fu.id)

        # Pre-compute the set of memory DFG node IDs.
        self._memory_dfg_ids: Set[str] = {
            n.id for n in dfg.get_nodes() if n.operation in _MEMORY_OPS
        }

    # ------------------------------------------------------------------
    # Individual cost functions
    # ------------------------------------------------------------------

    def compute_src(
        self,
        parent_link_sets: List[Set[str]],
        sibling_link_sets: List[Set[str]],
    ) -> float:
        """Static Routing Cost.

        For each parent, the cost is the number of *new* links required
        (i.e. links not already shared with a sibling route for the same
        parent value).  Sibling overlap reduces cost because the same
        data can be multicast through shared links.

        Args:
            parent_link_sets: One set of MRRG node IDs per parent route.
            sibling_link_sets: Link sets already committed for siblings
                of the same parent values.
        """
        all_sibling_links: Set[str] = set()
        for s in sibling_link_sets:
            all_sibling_links |= s

        cost = 0.0
        for link_set in parent_link_sets:
            new_links = link_set - all_sibling_links
            cost += len(new_links)
        return cost

    def compute_uarc(
        self,
        parent_link_sets: List[Set[str]],
        used_resources: Set[str],
    ) -> float:
        """Used Adjacent Resources Cost.

        For each *new* (not yet committed) link needed to route from
        parents to the candidate FU, count how many adjacent MRRG nodes
        are already used.  Higher counts indicate a more congested area.
        """
        cost = 0.0
        for link_set in parent_link_sets:
            for node_id in link_set:
                if node_id in used_resources:
                    continue  # already committed -- no incremental cost
                for neighbour in self._mrrg.get_successors(node_id):
                    if neighbour.id in used_resources:
                        cost += 1.0
                for neighbour in self._mrrg.get_predecessors(node_id):
                    if neighbour.id in used_resources:
                        cost += 1.0
        return cost

    def compute_mrc(
        self,
        candidate_fu: MRRGNode,
        dfg_node: DFGNode,
        placed_dfg_ids: Set[str],
        placed_fu_ids: Set[str],
    ) -> float:
        """Memory Resource Cost.

        Placing a non-memory DFG operation on a memory-capable FU wastes
        a scarce resource.  The cost is the ratio of remaining unmapped
        memory DFG nodes to remaining unused memory FUs.
        """
        if dfg_node.operation in _MEMORY_OPS:
            return 0.0
        if candidate_fu.id not in self._memory_fu_ids:
            return 0.0

        unmapped_mem = len(self._memory_dfg_ids - placed_dfg_ids)
        unused_mem_fus = len(self._memory_fu_ids - placed_fu_ids)
        if unused_mem_fus == 0:
            return 0.0
        return unmapped_mem / unused_mem_fus

    def compute_width_cost(
        self,
        candidate_fu: MRRGNode,
        parent_link_sets: List[Set[str]],
        edge_bitwidths: List[Optional[int]],
    ) -> float:
        """Width mismatch cost.

        Penalises routing narrow data on wide buses and placing narrow
        operations on wide FUs.  For each incoming edge, the penalty is
        the fractional over-provisioning of the routing nodes.
        """
        cost = 0.0
        for link_set, bw in zip(parent_link_sets, edge_bitwidths):
            if bw is None or bw <= 0:
                continue
            for node_id in link_set:
                node = self._mrrg.get_node(node_id)
                if node is None:
                    continue
                node_bw = getattr(node, "bitwidth", None)
                if node_bw is not None and node_bw > bw:
                    cost += (node_bw - bw) / bw
        return cost

    # ------------------------------------------------------------------
    # Aggregate entry-point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate_fu: MRRGNode,
        dfg_node: DFGNode,
        parent_link_sets: List[Set[str]],
        sibling_link_sets: List[Set[str]],
        edge_bitwidths: List[Optional[int]],
        used_resources: Set[str],
        placed_dfg_ids: Set[str],
        placed_fu_ids: Set[str],
    ) -> float:
        """Return the weighted sum of all four cost functions."""
        src = self._w_src * self.compute_src(parent_link_sets, sibling_link_sets)
        uarc = self._w_uarc * self.compute_uarc(parent_link_sets, used_resources)
        mrc = self._w_mrc * self.compute_mrc(
            candidate_fu, dfg_node, placed_dfg_ids, placed_fu_ids,
        )
        width = self._w_width * self.compute_width_cost(
            candidate_fu, parent_link_sets, edge_bitwidths,
        )
        return src + uarc + mrc + width
