"""HyperDFG: Condensed representation of DFG with vectorized edges for ILP formulation."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from cgrasolve.graph.graph_base import Graph, Node, Edge
from cgrasolve.graph.dfg import DFG, DFGNode, OperationType


@dataclass
class HyperNode(Node):
    """
    HyperNode represents a node in the HyperDFG.

    Maintains the same attributes as DFGNode for scheduling and operation info.
    """

    operation: OperationType
    original_dfg_node_id: str

    # Scheduling attributes (same as DFGNode)
    asap_time: Optional[int] = None
    alap_time: Optional[int] = None
    scheduled_time: Optional[int] = None
    mobility: Optional[int] = None

    def __init__(
        self,
        id: str,
        operation: OperationType,
        original_dfg_node_id: str,
        asap_time: Optional[int] = None,
        alap_time: Optional[int] = None,
        scheduled_time: Optional[int] = None,
        mobility: Optional[int] = None,
        **attributes: Any
    ):
        """Initialize a HyperNode."""
        super().__init__(id, **attributes)
        self.operation = operation
        self.original_dfg_node_id = original_dfg_node_id
        self.asap_time = asap_time
        self.alap_time = alap_time
        self.scheduled_time = scheduled_time
        self.mobility = mobility

    def is_scheduled(self) -> bool:
        """Check if the node has been scheduled."""
        return self.scheduled_time is not None

    def to_string(self) -> str:
        """Return a detailed string representation."""
        base = f"HyperNode(id={self.id}, op={self.operation.value}, orig={self.original_dfg_node_id}"
        if self.is_scheduled():
            base += f", t={self.scheduled_time}"
        if self.asap_time is not None:
            base += f", asap={self.asap_time}, alap={self.alap_time}, mob={self.mobility}"
        return base + ")"


@dataclass
class HyperVal(Edge):
    """
    HyperVal represents condensed outgoing edges from a single source node.

    All attributes are vectorized - each index corresponds to one condensed edge.
    The cardinality indicates how many edges are condensed into this HyperVal.
    """

    source_id: str
    destination_ids: List[str]
    cardinality: int

    # Vectorized edge attributes (all lists of same length = cardinality)
    latencies: List[int]
    is_loop_backs: List[bool]
    dists: List[Optional[int]]
    operands: List[Optional[str]]
    bitwidths: List[Optional[int]]
    kinds: List[Optional[str]]
    predicates: List[Optional[bool]]

    def __init__(
        self,
        id: str,
        source_id: str,
        destination_ids: List[str],
        latencies: List[int],
        is_loop_backs: List[bool],
        dists: List[Optional[int]],
        operands: List[Optional[str]],
        bitwidths: List[Optional[int]],
        kinds: List[Optional[str]],
        predicates: List[Optional[bool]],
        **attributes: Any
    ):
        """
        Initialize a HyperVal.

        Args:
            id: Unique identifier for this HyperVal
            source_id: ID of the source HyperNode
            destination_ids: List of destination HyperNode IDs
            latencies: List of communication latencies for each edge
            is_loop_backs: List of loop-back flags for each edge
            dists: List of iteration distances for each edge
            operands: List of operand positions for each edge (e.g., "LHS", "RHS")
            bitwidths: List of data widths for each edge
            kinds: List of edge types for each edge
            predicates: List of predicate flags for each edge
            **attributes: Additional custom attributes
        """
        # For Edge base class, we use a dummy source/destination
        # The actual source/dest tracking is in source_id and destination_ids
        super().__init__(id, None, None, **attributes)  # type: ignore

        self.source_id = source_id
        self.destination_ids = destination_ids
        self.cardinality = len(destination_ids)

        # Validate all vectors have same length
        vectors = [latencies, is_loop_backs, dists, operands, bitwidths, kinds, predicates]
        if not all(len(v) == self.cardinality for v in vectors):
            raise ValueError(
                f"All attribute vectors must have same length as destination_ids "
                f"(cardinality={self.cardinality})"
            )

        self.latencies = latencies
        self.is_loop_backs = is_loop_backs
        self.dists = dists
        self.operands = operands
        self.bitwidths = bitwidths
        self.kinds = kinds
        self.predicates = predicates

        # Resolved source operation output bitwidth (set during routing
        # from the placed FU's port map metadata, not the DFG edge attribute)
        self.source_output_bitwidth: Optional[int] = None

    def __hash__(self):
        """Make HyperVal hashable for use in dictionaries."""
        return hash(self.id)

    def __eq__(self, other):
        """Compare HyperVals by ID."""
        if not isinstance(other, HyperVal):
            return False
        return self.id == other.id

    def get_edge_at(self, index: int) -> Dict[str, Any]:
        """
        Get attributes for a specific edge by index.

        Args:
            index: Index of the edge (0 to cardinality-1)

        Returns:
            Dictionary containing all attributes for that edge
        """
        if index < 0 or index >= self.cardinality:
            raise IndexError(f"Edge index {index} out of range [0, {self.cardinality})")

        return {
            "source_id": self.source_id,
            "destination_id": self.destination_ids[index],
            "latency": self.latencies[index],
            "is_loop_back": self.is_loop_backs[index],
            "dist": self.dists[index],
            "operand": self.operands[index],
            "bitwidth": self.bitwidths[index],
            "kind": self.kinds[index],
            "predicate": self.predicates[index],
        }

    def get_edges_to(self, destination_id: str) -> List[Dict[str, Any]]:
        """
        Get all edges going to a specific destination.

        Args:
            destination_id: ID of the destination node

        Returns:
            List of edge attribute dictionaries for edges to that destination
        """
        edges = []
        for i, dest_id in enumerate(self.destination_ids):
            if dest_id == destination_id:
                edges.append(self.get_edge_at(i))
        return edges

    def filter_by_operand(self, operand: str) -> List[int]:
        """
        Get indices of edges with a specific operand position.

        Args:
            operand: Operand position to filter by (e.g., "LHS", "RHS")

        Returns:
            List of indices where operand matches
        """
        return [i for i, op in enumerate(self.operands) if op == operand]

    def filter_by_loop_back(self, is_loop_back: bool = True) -> List[int]:
        """
        Get indices of loop-back (or non-loop-back) edges.

        Args:
            is_loop_back: Whether to filter for loop-back edges (True) or regular edges (False)

        Returns:
            List of indices matching the loop-back criteria
        """
        return [i for i, lb in enumerate(self.is_loop_backs) if lb == is_loop_back]

    def to_string(self) -> str:
        """Return a detailed string representation."""
        return (
            f"HyperVal(id={self.id}, src={self.source_id}, "
            f"cardinality={self.cardinality}, dests={self.destination_ids}), "
            f"latencies={self.latencies}, is_loop_backs={self.is_loop_backs}, dists={self.dists}, operands={self.operands}, bitwidths={self.bitwidths}, kinds={self.kinds}, predicates={self.predicates}"
        )


class HyperDFG(Graph[HyperNode, HyperVal]):
    """
    HyperDFG: Condensed representation of DFG with vectorized edges.

    In HyperDFG, all outgoing edges from a given node are condensed into a single
    HyperVal with vectorized attributes. This representation is optimized for
    ILP formulation and batch constraint generation.
    """

    __slots__ = (
        "name",
        "initiation_interval",
        "_input_nodes",
        "_output_nodes",
        "_critical_path_length",
    )

    def __init__(self, name: str = "hyperdfg"):
        """Initialize an empty HyperDFG."""
        super().__init__()
        self.name = name
        self.initiation_interval: Optional[int] = None
        self._input_nodes: List[str] = []
        self._output_nodes: List[str] = []
        self._critical_path_length: Optional[int] = None

    def add_input_node(self, node: HyperNode) -> None:
        """Add an input node to the HyperDFG."""
        self.add_node(node)
        if node.id not in self._input_nodes:
            self._input_nodes.append(node.id)

    def add_output_node(self, node: HyperNode) -> None:
        """Add an output node to the HyperDFG."""
        self.add_node(node)
        if node.id not in self._output_nodes:
            self._output_nodes.append(node.id)

    def add_edge(self, edge: HyperVal) -> None:
        """
        Add a HyperVal to the graph.

        Overrides base class method to handle HyperVal's unique structure
        where source/destination are stored as IDs, not Node objects.
        """
        if edge.id in self._edges:
            raise ValueError(f"HyperVal {edge.id} already exists in graph")

        # Ensure source node exists
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist in graph")

        # Ensure all destination nodes exist
        for dest_id in edge.destination_ids:
            if dest_id not in self._nodes:
                raise ValueError(f"Destination node {dest_id} does not exist in graph")

        # Add edge to storage
        self._edges[edge.id] = edge

        # Update adjacency dicts
        # For HyperVal, create multiple entries (one per destination) with the same edge_id
        edge_data = {
            'edge_id': edge.id,
            'weight': getattr(edge, 'latency', 1)
        }
        # Add any other edge attributes
        for key, value in edge.attributes.items():
            if key not in edge_data:
                edge_data[key] = value

        # Add to multigraph adjacency (src -> dst -> {edge_id -> edge_data}).
        # A single HyperVal can have multiple destination_ids (one per
        # condensed underlying DFG edge); each contributes one entry under
        # the same edge_id, which is fine since the inner dict is keyed by
        # edge_id and a single HyperVal IS one logical edge object.
        for dest_id in edge.destination_ids:
            self._adjacency_list[edge.source_id].setdefault(dest_id, {})[edge.id] = edge_data
            self._reverse_adjacency_list[dest_id].setdefault(edge.source_id, {})[edge.id] = edge_data

    def get_input_nodes(self) -> List[HyperNode]:
        """Get all input nodes."""
        return [self.get_node(node_id) for node_id in self._input_nodes]

    def get_output_nodes(self) -> List[HyperNode]:
        """Get all output nodes."""
        return [self.get_node(node_id) for node_id in self._output_nodes]

    def get_hyperval_from_node(self, node_id: str) -> Optional[HyperVal]:
        """
        Get the HyperVal (condensed outgoing edges) from a specific node.

        Args:
            node_id: ID of the source node

        Returns:
            HyperVal if the node has outgoing edges, None otherwise
        """
        outgoing = self.get_outgoing_edges(node_id)
        return outgoing[0] if outgoing else None

    def validate(self) -> bool:
        """
        Validate the HyperDFG structure.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check that all destination IDs in HyperVals point to existing nodes
        for hyperval in self.get_edges():
            if not self.has_node(hyperval.source_id):
                raise ValueError(f"HyperVal {hyperval.id} references non-existent source {hyperval.source_id}")

            for dest_id in hyperval.destination_ids:
                if not self.has_node(dest_id):
                    raise ValueError(
                        f"HyperVal {hyperval.id} references non-existent destination {dest_id}"
                    )

        # Check that input/output nodes exist
        for input_id in self._input_nodes:
            if not self.has_node(input_id):
                raise ValueError(f"Input node {input_id} does not exist")

        for output_id in self._output_nodes:
            if not self.has_node(output_id):
                raise ValueError(f"Output node {output_id} does not exist")

        return True

    @staticmethod
    def from_dfg(dfg: DFG) -> "HyperDFG":
        """
        Create a HyperDFG from a DFG by condensing all outgoing edges per node.

        Args:
            dfg: The DFG to convert

        Returns:
            A new HyperDFG with condensed edges
        """
        hyperdfg = HyperDFG(name=f"hyper_{dfg.name}")
        hyperdfg.initiation_interval = dfg.initiation_interval
        hyperdfg._critical_path_length = dfg._critical_path_length

        # Create HyperNodes from DFG nodes
        for dfg_node in dfg.get_nodes():
            hyper_node = HyperNode(
                id=dfg_node.id,
                operation=dfg_node.operation,
                original_dfg_node_id=dfg_node.id,
                asap_time=dfg_node.asap_time,
                alap_time=dfg_node.alap_time,
                scheduled_time=dfg_node.scheduled_time,
                mobility=dfg_node.mobility,
                **dfg_node.attributes
            )
            hyperdfg.add_node(hyper_node)

            # Track input/output nodes
            if dfg_node.id in dfg._input_nodes:
                hyperdfg._input_nodes.append(dfg_node.id)
            if dfg_node.id in dfg._output_nodes:
                hyperdfg._output_nodes.append(dfg_node.id)

        # Create HyperVals by condensing outgoing edges per node
        hyperval_id = 0
        for dfg_node in dfg.get_nodes():
            outgoing_edges = dfg.get_outgoing_edges(dfg_node.id)

            if not outgoing_edges:
                continue  # Skip nodes with no outgoing edges

            # Collect vectorized attributes from all outgoing edges
            destination_ids: List[str] = []
            latencies: List[int] = []
            is_loop_backs: List[bool] = []
            dists: List[Optional[int]] = []
            operands: List[Optional[str]] = []
            bitwidths: List[Optional[int]] = []
            kinds: List[Optional[str]] = []
            predicates: List[Optional[bool]] = []

            for edge in outgoing_edges:
                destination_ids.append(edge.destination.id)
                latencies.append(edge.latency)
                is_loop_backs.append(edge.is_loop_back)
                dists.append(edge.get_attribute("dist"))
                operands.append(edge.get_attribute("operand"))
                bitwidths.append(edge.get_attribute("bitwidth"))
                kinds.append(edge.get_attribute("kind"))
                predicates.append(edge.get_attribute("predicate"))

            # Create the condensed HyperVal
            hyperval = HyperVal(
                id=f"hv_{hyperval_id}",
                source_id=dfg_node.id,
                destination_ids=destination_ids,
                latencies=latencies,
                is_loop_backs=is_loop_backs,
                dists=dists,
                operands=operands,
                bitwidths=bitwidths,
                kinds=kinds,
                predicates=predicates,
            )
            hyperdfg.add_edge(hyperval)
            hyperval_id += 1

        return hyperdfg

    def __repr__(self) -> str:
        """Return a string representation of the HyperDFG."""
        return (
            f"HyperDFG(name={self.name}, nodes={self.num_nodes()}, "
            f"hypervals={self.num_edges()}, II={self.initiation_interval})"
        )

    # Helper methods
    # Shwet TODO: Re-org later.

    def get_vals(self) -> List[HyperVal]:
        """Get all values."""
        return [self.get_edge(edge_id) for edge_id in self._edges]
    
    def get_nodes(self) -> List[HyperNode]:
        """Get all nodes."""
        return [self.get_node(node_id) for node_id in self._nodes]

    def create_hyperval_index_map(self) -> Dict[Tuple[int, int], str]:
        """Create a mapping from (HyperVal index, destination index) to destination node ID."""
        index_map = {}
        for j, hyperval in enumerate(self.get_vals()):
            for k in range(hyperval.cardinality):
                index_map[(j, k)] = hyperval.destination_ids[k]
        return index_map

    def create_hyperval_fanout_map(self) -> Dict[int, List[int]]:
        """Create a mapping from HyperVal index to list of destination node indices."""
        fanout_map: Dict[int, List[int]] = {}
        for j, hyperval in enumerate(self.get_vals()):
            for k in range(hyperval.cardinality):
                fanout_map.setdefault(j, []).append(k)
        return fanout_map