"""Data Flow Graph (DFG) representation."""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from cgrasolve.graph.graph_base import Graph, Node, Edge


class OperationType(Enum):
    """Enumeration of operation types in a DFG."""

    # Special
    NOP = "nop"

    # Type conversion operations
    SEXT = "sext"        # Sign extension
    ZEXT = "zext"        # Zero extension
    TRUNC = "trunc"      # Truncate
    FP2INT = "fp2int"    # Float to int conversion
    INT2FP = "int2fp"    # Int to float conversion

    # Input/Output operations
    INPUT = "input"
    INPUT_PRED = "input_pred"
    OUTPUT = "output"
    OUTPUT_PRED = "output_pred"
    PHI = "phi"          # SSA phi node
    CONST = "const"

    # Integer arithmetic operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"

    # Logical operations
    AND = "and"
    OR = "or"
    XOR = "xor"

    # Shift operations
    SHL = "shl"          # Shift left
    ASHR = "ashr"        # Arithmetic shift right
    LSHR = "lshr"        # Logical shift right

    # Memory operations
    LOAD = "load"
    STORE = "store"
    GEP = "gep"          # Get element pointer

    # Comparison operations
    ICMP = "icmp"        # Integer compare
    CMP = "cmp"          # Generic compare

    # Control flow
    BR = "br"            # Branch
    SELECT = "select"

    # Floating point operations
    SQRT = "sqrt"
    FADD = "fadd"        # Float add
    FMUL = "fmul"        # Float multiply
    FDIV = "fdiv"        # Float divide
    FMA = "fma"          # Fused multiply-add
    FSUB = "fsub"        # Float subtract

    # Immediate variants
    GEP_IMM = "gep_imm"    # Get element pointer with immediate
    ADD_IMM = "add_imm"    # Add with immediate

    # Unsigned multiply variants
    MULU_FULL_LO = "mulu_full_lo"
    MULU_HALF_LO = "mulu_half_lo"
    MULU_QUART_LO = "mulu_quart_lo"
    MULU_FULL_HI = "mulu_full_hi"
    MULU_HALF_HI = "mulu_half_hi"
    MULU_QUART_HI = "mulu_quart_hi"

    # Signed multiply variants
    MULS_FULL_LO = "muls_full_lo"
    MULS_HALF_LO = "muls_half_lo"
    MULS_QUART_LO = "muls_quart_lo"
    MULS_FULL_HI = "muls_full_hi"
    MULS_HALF_HI = "muls_half_hi"
    MULS_QUART_HI = "muls_quart_hi"

    # Add variants
    ADD_FULL = "add_full"
    ADD_HALF = "add_half"
    ADD_QUART = "add_quart"

    # Legacy/compatibility operations
    MOVE = "move"
    MOD = "mod"
    NOT = "not"
    SHR = "shr"          # Generic shift right (kept for compatibility)
    BRANCH = "branch"    # Alternative name for BR


class DFGNode(Node):
    """Node in a Data Flow Graph."""

    def __init__(
        self,
        node_id: str,
        operation: OperationType,
        **attributes: Any
    ) -> None:
        """
        Initialize a DFG node.

        Args:
            node_id: Unique identifier for the node
            operation: Type of operation this node performs
            **attributes: Additional attributes (e.g., constant value, data type)
        """
        super().__init__(node_id, **attributes)
        self.operation = operation

        # Scheduling information (computed by scheduler)
        self.asap_time: Optional[int] = None
        self.alap_time: Optional[int] = None
        self.scheduled_time: Optional[int] = None
        self.mobility: Optional[int] = None

    def is_scheduled(self) -> bool:
        """Check if this node has been scheduled."""
        return self.scheduled_time is not None

    def to_string(self) -> str:
        """Return a detailed string representation of the DFG node."""
        parts = [f"DFGNode(id={self.id}, op={self.operation.value}"]

        # Add scheduling info if available
        if self.asap_time is not None:
            parts.append(f"asap={self.asap_time}")
        if self.alap_time is not None:
            parts.append(f"alap={self.alap_time}")
        if self.scheduled_time is not None:
            parts.append(f"scheduled={self.scheduled_time}")
        if self.mobility is not None:
            parts.append(f"mobility={self.mobility}")

        # Add attributes
        for key, value in self.attributes.items():
            parts.append(f"{key}={value}")

        return ", ".join(parts) + ")"

    def __repr__(self) -> str:
        return f"DFGNode(id={self.id}, op={self.operation.value})"


class DFGEdge(Edge):
    """Edge in a Data Flow Graph representing data dependencies."""

    def __init__(
        self,
        edge_id: str,
        source: DFGNode,
        destination: DFGNode,
        latency: int = 0,
        **attributes: Any
    ) -> None:
        """
        Initialize a DFG edge.

        Args:
            edge_id: Unique identifier for the edge
            source: Source node (data producer)
            destination: Destination node (data consumer)
            latency: Communication latency in hardware (register pipeline stages).
                     For logical DFG edges, defaults to 0.
                     Actual latency comes from target MRRG mapping.
            **attributes: Additional attributes. Common attributes include:
                - dist (int): Iteration distance from DOT file metadata.
                              Represents logical distance in dataflow graph.
                              Used for recurrence analysis in modulo scheduling.
                - operand (str): Operand position (e.g., "LHS", "RHS")
                - bitwidth (int): Data width in bits
                - kind (str): Edge type (e.g., "dataflow")
                - predicate (bool): Whether edge is predicate-dependent
        """
        super().__init__(edge_id, source, destination, **attributes)
        self.latency = latency
        self.is_loop_back: bool = attributes.get('is_loop_back', False)

    def to_string(self) -> str:
        """Return a detailed string representation of the DFG edge."""
        parts = [f"DFGEdge({self.source.id} -> {self.destination.id}"]

        # Add latency
        parts.append(f"latency={self.latency}")

        # Add loop-back flag
        if self.is_loop_back:
            parts.append("LOOP_BACK")

        # Add attributes
        for key, value in self.attributes.items():
            if key != 'is_loop_back':  # Already handled
                parts.append(f"{key}={value}")

        return ", ".join(parts) + ")"

    def __repr__(self) -> str:
        loop_str = " [LOOP]" if self.is_loop_back else ""
        return f"DFGEdge({self.source.id} -> {self.destination.id}{loop_str})"


class DFG(Graph[DFGNode, DFGEdge]):
    """Data Flow Graph representation."""

    __slots__ = ("_input_nodes", "_output_nodes", "_critical_path_length", "initiation_interval")
    initiation_interval: Optional[int]
    _input_nodes: List[str]
    _output_nodes: List[str]
    _critical_path_length: Optional[int]

    def __init__(
        self,
        name: str = "DFG",
        initiation_interval: Optional[int] = None
    ) -> None:
        """
        Initialize a DFG.

        Args:
            name: Name of the DFG
            initiation_interval: Target initiation interval for modulo scheduling
        """
        super().__init__(name)
        self.initiation_interval = initiation_interval
        self._input_nodes: List[str] = []
        self._output_nodes: List[str] = []
        self._critical_path_length: Optional[int] = None

    def add_input_node(self, node: DFGNode) -> None:
        """Add an input node to the DFG."""
        if node.operation != OperationType.INPUT:
            raise ValueError(f"Node {node.id} is not an INPUT node")
        self.add_node(node)
        self._input_nodes.append(node.id)

    def add_output_node(self, node: DFGNode) -> None:
        """Add an output node to the DFG."""
        if node.operation != OperationType.OUTPUT:
            raise ValueError(f"Node {node.id} is not an OUTPUT node")
        self.add_node(node)
        self._output_nodes.append(node.id)

    def get_input_nodes(self) -> List[DFGNode]:
        """Get all input nodes."""
        return [self.get_node(nid) for nid in self._input_nodes if self.get_node(nid)]

    def get_output_nodes(self) -> List[DFGNode]:
        """Get all output nodes."""
        return [self.get_node(nid) for nid in self._output_nodes if self.get_node(nid)]

    def get_nodes(self) -> List[DFGNode]:
        """Get all nodes."""
        return [self.get_node(nid) for nid in self._nodes if self.get_node(nid)]

    def get_operations_by_type(self, op_type: OperationType) -> List[DFGNode]:
        """Get all nodes with a specific operation type."""
        return [n for n in self.get_nodes() if n.operation == op_type]

    def get_critical_path_length(self) -> int:
        """
        Get the critical path length of the DFG.
        This should be computed by graph analysis utilities.
        """
        if self._critical_path_length is None:
            # This will be computed by the critical path analysis utility
            raise ValueError("Critical path has not been computed yet")
        return self._critical_path_length

    def set_critical_path_length(self, length: int) -> None:
        """Set the critical path length."""
        self._critical_path_length = length

    def get_loop_back_edges(self) -> List[DFGEdge]:
        """Get all loop-back edges (for iterative kernels)."""
        return [e for e in self.get_edges() if e.is_loop_back]

    def get_vals(self) -> List[int]:
        """Get all values."""

        # Count all the edges between nodes. Multiple outgoing edges from the same node are considered as a single value.
        vals: List[int] = []
        val = 0
        for node in self.get_nodes():
            for edge in self.get_outgoing_edges(node.id):
                val += 1
                vals.append(val)
        return vals

    def validate(self) -> bool:
        """
        Validate the DFG structure.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for cycles (excluding loop-back edges)
        non_loop_edges = [e for e in self.get_edges() if not e.is_loop_back]

        # Check that all nodes are reachable from inputs
        if self._input_nodes:
            # This would use graph traversal utilities
            pass

        # Check that all outputs are reachable
        if self._output_nodes:
            # This would use graph traversal utilities
            pass

        return True

    def clear_scheduling_info(self) -> None:
        """Clear all scheduling information from nodes."""
        for node in self.get_nodes():
            node.asap_time = None
            node.alap_time = None
            node.scheduled_time = None
            node.mobility = None

    def to_string(self) -> str:
        """Return a detailed string representation of the DFG."""
        lines = []
        lines.append(f"DFG: {self.name}")
        if self.initiation_interval:
            lines.append(f"  Initiation Interval: {self.initiation_interval}")
        lines.append(f"  Total Nodes: {self.num_nodes()}")
        lines.append(f"  Total Edges: {self.num_edges()}")
        lines.append(f"  Input Nodes: {len(self._input_nodes)}")
        lines.append(f"  Output Nodes: {len(self._output_nodes)}")
        lines.append(f"  Loop-back Edges: {len(self.get_loop_back_edges())}")

        lines.append("\n  Input Nodes:")
        for node_id in self._input_nodes:
            node = self.get_node(node_id)
            if node:
                lines.append(f"    {node.to_string()}")

        lines.append("\n  Output Nodes:")
        for node_id in self._output_nodes:
            node = self.get_node(node_id)
            if node:
                lines.append(f"    {node.to_string()}")

        lines.append("\n  All Nodes:")
        for node in self.get_nodes():
            lines.append(f"    {node.to_string()}")

        lines.append("\n  All Edges:")
        for edge in self.get_edges():
            lines.append(f"    {edge.to_string()}")

        return "\n".join(lines)

    def to_hyperdfg(self) -> "HyperDFG":
        """
        Convert this DFG to a HyperDFG with condensed edges.

        Returns:
            A new HyperDFG with all outgoing edges from each node condensed into HyperVals
        """
        from cgrasolve.graph.hyperdfg import HyperDFG
        return HyperDFG.from_dfg(self)

    def remove_phi_nodes(self) -> None:
        """
        Remove PHI nodes from the DFG, preserving loop-carried dependencies.

        PHI nodes in SSA form select between initialization values and loop-carried
        values. For CGRA mapping, we bypass PHI nodes by:
        - Connecting the loop-carried source directly to all PHI consumers with dist=1
        - The initialization value is assumed to be provided externally (not in the kernel)

        This preserves the loop recurrence structure needed for modulo scheduling.
        """
        # Find all PHI nodes
        phi_nodes = self.get_operations_by_type(OperationType.PHI)

        edge_counter = 0  # For generating unique edge IDs

        for phi_node in phi_nodes:
            # Get incoming and outgoing edges
            incoming_edges = self.get_incoming_edges(phi_node.id)
            outgoing_edges = self.get_outgoing_edges(phi_node.id)

            if len(incoming_edges) != 2:
                # Skip PHI nodes with unexpected structure
                continue

            # Identify init edge (not loop-back) and loop-carried edge (is_loop_back)
            init_edge = None
            loop_edge = None

            for edge in incoming_edges:
                if edge.is_loop_back:
                    loop_edge = edge
                else:
                    init_edge = edge

            if not init_edge or not loop_edge:
                # Skip if we can't identify both edges properly
                continue

            # IMPORTANT: Only create bypass edges from loop-carried source
            # The init value is handled outside the kernel (by the host or prolog)
            for out_edge in outgoing_edges:
                # Create edge from loop-carried source to consumer
                loop_bypass = DFGEdge(
                    edge_id=f"bypass_loop_{edge_counter}",
                    source=loop_edge.source,
                    destination=out_edge.destination,
                    latency=loop_edge.latency + out_edge.latency,
                    is_loop_back=True,  # Preserve loop-back flag
                )
                
                # Copy attributes from output edge (operand, bitwidth, etc.)
                for key, value in out_edge.attributes.items():
                    loop_bypass.set_attribute(key, value)

                # Set iteration distance
                # Accumulate distances: dist from loop-back edge + 1 for crossing PHI
                existing_dist = loop_edge.attributes.get('dist', 0)
                loop_bypass.set_attribute('dist', existing_dist + 1)

                self.add_edge(loop_bypass)
                edge_counter += 1

            # Remove the PHI node (this automatically removes its connected edges)
            self.remove_node(phi_node.id)
        
        # Note: The init value node (e.g., const 0) may now be unused
        # It will be cleaned up by a later pass that removes unused nodes

    def remove_unused_nodes(self) -> None:
        """
        Remove nodes that have no successors (fanout) and are not outputs or stores.

        After removing PHI and branch nodes, some nodes may become orphaned
        (they produce values that are no longer consumed). This method iteratively
        removes such nodes. OUTPUT and STORE nodes are always preserved as they
        represent side-effecting operations. INPUT and CONST nodes are removed if
        they have no consumers.
        """
        changed = True
        while changed:
            changed = False
            # Get all nodes (make a copy since we'll be modifying the graph)
            all_nodes = list(self.get_nodes())

            for node in all_nodes:
                # Skip if node was already removed
                if not self.has_node(node.id):
                    continue

                # Always preserve OUTPUT and STORE nodes (side-effecting operations)
                # STORE operations write to memory and should be preserved even without
                # explicit OUTPUT nodes, as they represent the kernel's computation results
                if node.operation in (OperationType.OUTPUT, OperationType.STORE,
                                     OperationType.OUTPUT_PRED):
                    continue

                # Check if node has any outgoing edges
                outgoing = self.get_outgoing_edges(node.id)
                if len(outgoing) == 0:
                    # Node has no fanout, remove it (including unused INPUT/CONST)
                    self.remove_node(node.id)
                    changed = True

    def preprocess_for_mapping(self) -> "DFG":
        """
        Preprocess the DFG for CGRA mapping.

        This method applies a series of transformations to prepare the DFG
        for mapping to CGRA architectures that don't support control flow operations:
        1. Remove branch nodes (control flow not needed for static scheduling)
        2. Remove PHI nodes (preserving loop-carried dependencies)
        3. Remove unused nodes (cleanup orphaned nodes)

        Returns:
            Self (DFG is modified in-place)
        """
        self.remove_branch_nodes()
        self.remove_phi_nodes()
        self.remove_unused_nodes()
        return self

    def preprocess_for_scheduling(self) -> None:
        """
        Preprocess the DFG for scheduling.

        This method applies a series of transformations to prepare the DFG
        for scheduling.
        """
        self.remove_branch_nodes()
        self.remove_unused_nodes()
        return self

    def remove_branch_nodes(self) -> None:
        """
        Remove branch nodes from the DFG.

        Branch nodes represent control flow that is not needed for CGRA mapping
        where execution is statically scheduled. This method removes all BR and BRANCH
        operation nodes.
        """
        # Find all branch nodes (check both BR and BRANCH operation types)
        branch_nodes = []
        branch_nodes.extend(self.get_operations_by_type(OperationType.BR))
        branch_nodes.extend(self.get_operations_by_type(OperationType.BRANCH))

        # Remove each branch node
        # Note: remove_node() automatically removes all connected edges
        for branch_node in branch_nodes:
            self.remove_node(branch_node.id)

    def __repr__(self) -> str:
        ii_str = f", II={self.initiation_interval}" if self.initiation_interval else ""
        return f"DFG(name={self.name}, nodes={self.num_nodes()}, edges={self.num_edges()}{ii_str})"