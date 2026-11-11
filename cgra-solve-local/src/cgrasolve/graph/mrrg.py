"""Modulo Routing Resource Graph (MRRG) representation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from cgrasolve.graph.graph_base import Graph, Node, Edge
from cgrasolve.graph.dfg import OperationType
import cgrasolve.graph.utils.traversal as traversal


class NodeType(Enum):
    """
    Types of nodes in an MRRG (CGRA-ME classification).

    Note: Use HWEntityType to distinguish between different hardware resources
    (registers, wires, muxes, etc.), not NodeType.
    """
    ROUTING = "routing"              # MRRG_NODE_ROUTING: Pure routing/data movement
    FUNCTION = "function"             # MRRG_NODE_FUNCTION: Computational operations
    ROUTING_FUNCTION = "route_fu"    # MRRG_NODE_ROUTING_FUNCTION: Hybrid (PHI, SELECT)


class OperandTag(Enum):
    """Operand tags for FU input binding constraints (CGRA-ME specification)."""
    UNTAGGED = ""  # Untagged/default
    PREDICATE = "pred"  # Predicate/condition
    BINARY_LHS = "LHS"  # Left-hand side operand
    BINARY_RHS = "RHS"  # Right-hand side operand
    BINARY_ANY = "any2input"  # Either operand
    TERNARY_ANY = "any3input"  # Any of three operands
    ADDR = "addr"  # Address operand
    DATA = "data"  # Data operand
    BR_TRUE = "branch_true"  # Branch true
    BR_FALSE = "branch_false"  # Branch false


class HWEntityType(Enum):
    """Hardware entity types for MRRG nodes."""
    HW_WIRE = "wire"  # Combinational wire (latency=0)
    HW_REG = "reg"  # Register (latency>=1)
    HW_MUX = "mux"  # Multiplexer
    HW_COMB = "comb"  # Combinational logic (FU, memory)
    HW_UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class FractureEndpoint:
    """Resolved fracture-capable endpoint for a specific FU/IO port."""

    port: "MRRGNode"
    chunk_nodes: Tuple["MRRGNode", ...]
    chunk_indices: Tuple[int, ...]
    chunk_widths: Tuple[int, ...]
    chunk_offsets: Tuple[int, ...]
    total_width: int

    @property
    def num_chunks(self) -> int:
        """Number of chunks in this fracture endpoint."""
        return len(self.chunk_nodes)


@dataclass(frozen=True)
class PackingEndpoint:
    """Resolved pack/unpack-capable routing endpoint for a specific port."""

    port: "MRRGNode"
    boundary_node: "MRRGNode"
    boundary_path: Tuple["MRRGNode", ...]
    lane_width: int
    num_lanes: int
    allowed_lane_indices: Tuple[int, ...] = ()
    requires_existing_carrier: bool = False


class MRRGNode(Node):
    """Unified MRRG node with all attributes."""

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        cycle: int,
        coordinates: Optional[Tuple[int, int]] = None,
        hw_entity_type: HWEntityType = HWEntityType.HW_UNSPECIFIED,
        latency: int = 0,
        bitwidth: int = 32,
        capacity: int = 1,
        # FU-specific attributes
        supported_operations: Optional[Set[OperationType]] = None,
        supported_operand_tags: Optional[Set[OperandTag]] = None,
        num_inputs: int = 2,
        num_outputs: int = 1,
        # Routing-specific attributes
        routing_type: str = "wire",
        # Register-specific attributes
        bank_id: Optional[int] = None,
        read_ports: int = 1,
        write_ports: int = 1,
        **attributes: Any
    ) -> None:
        """
        Initialize an MRRG node.

        Args:
            node_id: Unique identifier (format: "cycle:name" for time-expanded nodes)
            node_type: Type of MRRG node (ROUTING, FUNCTION, ROUTING_FUNCTION)
            cycle: Time slot within II (0 to II-1)
            coordinates: (x, y) position in the CGRA array
            hw_entity_type: Hardware entity type (WIRE, REG, MUX, COMB)
            latency: Cycles through this node
            bitwidth: Data width in bits
            capacity: Multi-use capacity
            supported_operations: Set of operations this node can execute (for FUNCTION nodes)
            supported_operand_tags: Set of operand tags for input binding (for FUNCTION nodes)
            num_inputs: Number of input ports (for FUNCTION nodes)
            num_outputs: Number of output ports (for FUNCTION nodes)
            routing_type: Type of routing resource - wire, mux, crossbar (for ROUTING nodes)
            bank_id: Register bank identifier (for register nodes)
            read_ports: Number of read ports (for register nodes)
            write_ports: Number of write ports (for register nodes)
            **attributes: Additional attributes
        """
        super().__init__(node_id, **attributes)
        self.node_type = node_type
        self.cycle = cycle
        self.coordinates = coordinates
        self.hw_entity_type = hw_entity_type
        self.latency = latency
        self.bitwidth = bitwidth
        self.capacity = capacity

        # FU attributes
        self.supported_operations = supported_operations or set()
        self.supported_operand_tags = supported_operand_tags or set()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Routing attributes
        self.routing_type = routing_type

        # Register attributes
        self.bank_id = bank_id
        self.read_ports = read_ports
        self.write_ports = write_ports

        # Per-operation output bitwidths (from Dora port map metadata).
        # Maps OperationType -> output bitwidth for that operation on this FU.
        # Populated by DoraJsonParser when port map entries include "bitwidth".
        self.operation_output_bitwidths: Dict[OperationType, int] = {}

        # Per-operation output port bindings from Dora output_port_map.
        # Maps OperationType -> ordered list of raw/base MRRG node IDs
        # for legal output ports of that operation (e.g. "pe.alu.out_o").
        self.operation_output_ports: Dict[OperationType, List[str]] = {}

        # Per-operation output-port bitwidths from Dora output_port_map.
        # Maps OperationType -> {raw/base port ID -> bitwidth}.
        self.operation_output_port_bitwidths: Dict[OperationType, Dict[str, int]] = {}

        # Per-operation maximum input bitwidth (from Dora input_port_map).
        # For sink operations (OUTPUT, STORE), this is the primary bitwidth
        # constraint since they have no output ports.
        # Maps OperationType -> max input port bitwidth for that operation.
        self.operation_max_input_bitwidth: Dict[OperationType, int] = {}

        # Per-operation scheduler resource classes from Dora scheduling metadata.
        # Maps OperationType -> resource class string (for example "alu", "lsu").
        # When absent, scheduling falls back to the operation opcode.
        self.operation_resource_classes: Dict[OperationType, str] = {}

        # Per-operation operand tags for input pin nodes (from Dora input_port_map).
        # Maps OperationType -> Set[OperandTag] for this specific pin.
        # A single pin may have different tags for different operations
        # (e.g., A_i is LHS for ADD but RHS for SUB on some architectures).
        # When empty, falls back to the global supported_operand_tags.
        self.operation_supported_operand_tags: Dict[OperationType, Set['OperandTag']] = {}

        # Fracture metadata — instance nodes (split/join primitives)
        self.fracture_type: Optional[str] = None           # "split" / "join"
        self.fracture_parent_inst: Optional[str] = None
        self.fracture_chunk_width: Optional[int] = None

        # Fracture metadata — chunk net nodes
        self.fracture_parent_port: Optional[str] = None
        self.fracture_chunk_index: Optional[int] = None
        self.fracture_bit_offset: Optional[int] = None

        # Shared fracture metadata
        self.fracture_num_chunks: Optional[int] = None

        # Packed-transport metadata
        self.num_lanes: int = 1
        self.lane_width: int = bitwidth
        self.pack_capable: bool = False
        self.unpack_capable: bool = False
        self.packing_parent_port: Optional[str] = None
        self.pack_config: Optional[Dict[str, Any]] = None
        self.unpack_config: Optional[Dict[str, Any]] = None
        self.packing_requires_carrier: bool = False
        self.packing_inject_only: bool = False
        self.allowed_lane_indices: Tuple[int, ...] = ()

    def can_execute(self, operation: OperationType) -> bool:
        """Check if this node can execute the given operation."""
        return operation in self.supported_operations

    def supports_operand_tag(self, tag: OperandTag) -> bool:
        """Check if this node supports a specific operand tag."""
        return tag in self.supported_operand_tags

    def get_supported_operand_tags_for_operation(self, op: OperationType) -> Set[OperandTag]:
        """Get operand tags for a specific operation, falling back to global tags."""
        if op in self.operation_supported_operand_tags:
            return self.operation_supported_operand_tags[op]
        return self.supported_operand_tags

    def get_resource_class_for_operation(self, op: OperationType) -> str:
        """Return the scheduler resource class for an operation."""
        return self.operation_resource_classes.get(op, op.value)

    @staticmethod
    def operand_tag_matches_request(
        requested: "OperandTag", supported: Set["OperandTag"]
    ) -> bool:
        """Check if a requested operand tag is satisfied by a node's supported set."""
        if not supported:
            return False
        if requested == OperandTag.BINARY_ANY:
            return bool(
                supported & {OperandTag.BINARY_LHS, OperandTag.BINARY_RHS, OperandTag.BINARY_ANY}
            )
        if requested in (OperandTag.BINARY_LHS, OperandTag.BINARY_RHS):
            return requested in supported or OperandTag.BINARY_ANY in supported
        if requested == OperandTag.TERNARY_ANY:
            # Wildcard for any non-control data slot on a 3-input FU.
            return bool(
                supported & {
                    OperandTag.TERNARY_ANY, OperandTag.DATA,
                    OperandTag.BINARY_LHS, OperandTag.BINARY_RHS, OperandTag.BINARY_ANY,
                }
            )
        if requested == OperandTag.DATA:
            # Data edges may also land on TERNARY_ANY-tagged ports.
            return OperandTag.DATA in supported or OperandTag.TERNARY_ANY in supported
        return requested in supported

    def get_base_name(self) -> str:
        """Get the base name without cycle prefix."""
        if ':' in self.id:
            return self.id.split(':', 1)[1]
        return self.id

    def get_allowed_lane_indices(self) -> Tuple[int, ...]:
        """Return the normalized admissible packed-lane indices for this node."""
        if self.num_lanes <= 1:
            return ()
        if not self.allowed_lane_indices:
            return tuple(range(self.num_lanes))
        return tuple(
            lane_idx
            for lane_idx in self.allowed_lane_indices
            if 0 <= lane_idx < self.num_lanes
        )

    def allows_lane_index(self, lane_index: int) -> bool:
        """Return True when this node can carry the requested packed lane."""
        if self.num_lanes <= 1:
            return False
        return lane_index in self.get_allowed_lane_indices()

    def get_full_name(self) -> str:
        """Get full name in CGRA-ME format: 'cycle:name'."""
        base_name = self.get_base_name()
        return f"{self.cycle}:{base_name}"

    def to_string(self) -> str:
        """
        Return detailed string representation of the node.

        Returns:
            Multi-line string with all node attributes
        """
        lines = [
            f"Node: {self.id}",
            f"  Node Type: {self.node_type.value}",
            f"  HW Entity Type: {self.hw_entity_type.value}",
            f"  Cycle: {self.cycle}",
            f"  Coordinates: {self.coordinates}",
            f"  Latency: {self.latency}",
            f"  Bitwidth: {self.bitwidth}",
            f"  Capacity: {self.capacity}",
        ]

        # Add FUNCTION-specific attributes
        if self.supported_operations:
            ops = ', '.join(sorted([op.value for op in self.supported_operations]))
            lines.append(f"  Operations: {ops}")

        if self.supported_operand_tags:
            tags = ', '.join(sorted([tag.value for tag in self.supported_operand_tags]))
            lines.append(f"  Operand Tags: {tags}")

        if self.num_inputs != 2:  # Only show if non-default
            lines.append(f"  Num Inputs: {self.num_inputs}")

        if self.num_outputs != 1:  # Only show if non-default
            lines.append(f"  Num Outputs: {self.num_outputs}")

        # Add ROUTING-specific attributes
        if self.routing_type != "wire":
            lines.append(f"  Routing Type: {self.routing_type}")

        # Add REGISTER-specific attributes
        if self.bank_id is not None:
            lines.append(f"  Bank ID: {self.bank_id}")
            if self.read_ports != 1:
                lines.append(f"  Read Ports: {self.read_ports}")
            if self.write_ports != 1:
                lines.append(f"  Write Ports: {self.write_ports}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        coord_str = f", pos={self.coordinates}" if self.coordinates else ""
        type_str = self.node_type.value
        if self.node_type == NodeType.FUNCTION and self.supported_operations:
            type_str += f"({len(self.supported_operations)} ops)"
        elif self.hw_entity_type == HWEntityType.HW_REG:
            type_str += "(reg)"
        return f"MRRGNode(id={self.id}, cycle={self.cycle}, type={type_str}{coord_str})"


class MRRGEdge(Edge):
    """Edge in an MRRG representing routing connections."""

    def __init__(
        self,
        edge_id: str,
        source: MRRGNode,
        destination: MRRGNode,
        latency: int = 0,
        **attributes: Any
    ) -> None:
        """
        Initialize an MRRG edge.

        Args:
            edge_id: Unique identifier
            source: Source node
            destination: Destination node
            latency: Routing latency
            **attributes: Additional attributes
        """
        super().__init__(edge_id, source, destination, **attributes)
        self.latency = latency
        self.capacity: int = attributes.get('capacity', 1)
        self.conflict_free: bool = attributes.get('conflict_free', True)

    def to_string(self) -> str:
        """
        Return detailed string representation of the edge.

        Returns:
            Multi-line string with all edge attributes
        """
        lines = [
            f"Edge: {self.id}",
            f"  Source: {self.source.id}",
            f"  Destination: {self.destination.id}",
            f"  Latency: {self.latency}",
            f"  Capacity: {self.capacity}",
            f"  Conflict Free: {self.conflict_free}",
        ]

        # Add cross-cycle information if applicable
        if self.source.cycle != self.destination.cycle:
            lines.append(f"  Cross-Cycle: {self.source.cycle} -> {self.destination.cycle}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"MRRGEdge({self.source.id} -> {self.destination.id}, lat={self.latency})"


class MRRG(Graph[MRRGNode, MRRGEdge]):
    """Modulo Routing Resource Graph representation."""

    def __init__(
        self,
        name: str = "MRRG",
        II: int = 1,
        rows: int = 0,
        cols: int = 0
    ) -> None:
        """
        Initialize an MRRG.

        Args:
            name: Name of the MRRG
            II: Initiation Interval (number of time slots for time expansion)
            rows: Number of rows in the CGRA
            cols: Number of columns in the CGRA
        """
        super().__init__(name)
        self.II = II
        self.rows = rows
        self.cols = cols

        # Cycle-indexed storage (CGRA-ME style)
        # nodes_by_cycle[cycle][base_name] -> MRRGNode
        self.nodes_by_cycle: List[Dict[str, MRRGNode]] = [
            {} for _ in range(II)
        ]

        # Spatial indexing for quick lookup
        self._fu_nodes: Dict[Tuple[int, int], List[str]] = {}  # (x,y) -> list of FU node IDs
        self._routing_nodes: Dict[Tuple[int, int], List[str]] = {}  # (x,y) -> list of route node IDs
        self._register_nodes: Dict[Tuple[int, int], List[str]] = {}  # (x,y) -> list of reg node IDs

    def get_node_by_cycle(self, cycle: int, base_name: str) -> Optional[MRRGNode]:
        """
        Get node by cycle and base name (CGRA-ME style accessor).

        Args:
            cycle: Time slot (0 to II-1)
            base_name: Base name without cycle prefix

        Returns:
            MRRGNode if found, None otherwise
        """
        if 0 <= cycle < self.II:
            return self.nodes_by_cycle[cycle].get(base_name)
        return None

    def get_nodes_at_cycle(self, cycle: int) -> List[MRRGNode]:
        """Get all nodes at a specific cycle."""
        if 0 <= cycle < self.II:
            return list(self.nodes_by_cycle[cycle].values())
        return []

    def add_node(self, node: MRRGNode) -> None:
        """Override to add cycle-indexed and spatial storage."""
        super().add_node(node)

        # Add to cycle-indexed storage
        if 0 <= node.cycle < self.II:
            base_name = node.get_base_name()
            self.nodes_by_cycle[node.cycle][base_name] = node

        # Add to spatial indexing based on node type
        if node.coordinates:
            if node.node_type == NodeType.FUNCTION:
                if node.coordinates not in self._fu_nodes:
                    self._fu_nodes[node.coordinates] = []
                self._fu_nodes[node.coordinates].append(node.id)
            elif node.hw_entity_type == HWEntityType.HW_REG:
                if node.coordinates not in self._register_nodes:
                    self._register_nodes[node.coordinates] = []
                self._register_nodes[node.coordinates].append(node.id)
            elif node.node_type == NodeType.ROUTING:
                if node.coordinates not in self._routing_nodes:
                    self._routing_nodes[node.coordinates] = []
                self._routing_nodes[node.coordinates].append(node.id)

    def get_fu_nodes(self) -> List[MRRGNode]:
        """Get all functional unit nodes."""
        return [n for n in self.get_nodes() if n.node_type == NodeType.FUNCTION]

    def get_fus_at_position(self, x: int, y: int) -> List[MRRGNode]:
        """Get all FUs at a specific position."""
        node_ids = self._fu_nodes.get((x, y), [])
        return [self.get_node(nid) for nid in node_ids if self.get_node(nid)]

    def get_routing_nodes(self) -> List[MRRGNode]:
        """Get all routing nodes (excluding registers)."""
        return [n for n in self.get_nodes()
                if n.node_type == NodeType.ROUTING and n.hw_entity_type != HWEntityType.HW_REG]

    def get_register_nodes(self) -> List[MRRGNode]:
        """Get all register nodes."""
        return [n for n in self.get_nodes() if n.hw_entity_type == HWEntityType.HW_REG]

    def get_compatible_fus(self, operation: OperationType) -> List[MRRGNode]:
        """Get all FUs that can execute a specific operation."""
        return [fu for fu in self.get_fu_nodes() if fu.can_execute(operation)]

    def get_nodes_at_position(self, x: int, y: int) -> List[MRRGNode]:
        """Get all nodes at a specific position."""
        nodes = []
        for node_ids in [
            self._fu_nodes.get((x, y), []),
            self._routing_nodes.get((x, y), []),
            self._register_nodes.get((x, y), [])
        ]:
            nodes.extend([self.get_node(nid) for nid in node_ids if self.get_node(nid)])
        return nodes

    def validate(self) -> bool:
        """
        Validate the MRRG structure.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check that all FU nodes have valid coordinates
        for fu in self.get_fu_nodes():
            if fu.coordinates is None:
                raise ValueError(f"FU node {fu.id} has no coordinates")
            x, y = fu.coordinates
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                raise ValueError(f"FU node {fu.id} has invalid coordinates {fu.coordinates}")

        # Check that edges connect valid nodes
        for edge in self.get_edges():
            if not self.has_node(edge.source.id) or not self.has_node(edge.destination.id):
                raise ValueError(f"Edge {edge.id} connects non-existent nodes")

        return True

    # ------------------------------------------------------------------
    # Sub-II slicing
    # ------------------------------------------------------------------

    def slice_to_ii(self, target_ii: int) -> "MRRG":
        """Create a new MRRG containing only cycles 0 .. *target_ii* - 1.

        Cross-cycle register edges at the slice boundary (from cycle
        ``target_ii - 1`` to cycle ``target_ii``) are remapped to wrap
        back to cycle 0.  All other out-of-range edges are dropped.

        Node objects are **shared** with the original MRRG for memory
        efficiency; only boundary wrap edges get new ``MRRGEdge`` objects.

        Results are cached so repeated calls with the same *target_ii*
        return the same object.
        """
        if target_ii < 1:
            raise ValueError(f"target_ii must be >= 1, got {target_ii}")
        if target_ii > self.II:
            raise ValueError(
                f"target_ii ({target_ii}) exceeds native II ({self.II})"
            )
        if target_ii == self.II:
            return self

        if not hasattr(self, "_slice_cache"):
            self._slice_cache: Dict[int, "MRRG"] = {}
        if target_ii in self._slice_cache:
            return self._slice_cache[target_ii]

        sliced = MRRG(
            name=f"{self.name}_ii{target_ii}",
            II=target_ii,
            rows=self.rows,
            cols=self.cols,
        )

        # Phase 1: add nodes from cycles 0 .. target_ii-1 (shared refs).
        for cycle in range(target_ii):
            for _base, node in self.nodes_by_cycle[cycle].items():
                sliced.add_node(node)

        valid_ids = set(sliced._nodes)

        # Phase 2: classify and copy edges.
        wrap_counter = 0
        for edge in self._edges.values():
            src_in = edge.source.id in valid_ids
            dst_in = edge.destination.id in valid_ids

            if src_in and dst_in:
                # Both endpoints inside the slice — share edge object.
                sliced.add_edge(edge)
                continue

            if (
                src_in
                and edge.source.cycle == target_ii - 1
                and edge.destination.cycle == target_ii
            ):
                # Boundary register edge — remap destination to cycle 0.
                dst_base = edge.destination.get_base_name()
                new_dst = sliced.get_node_by_cycle(0, dst_base)
                if new_dst is None:
                    continue  # defensive: skip if cycle-0 counterpart missing
                wrap_edge = MRRGEdge(
                    f"wrap_{wrap_counter}",
                    edge.source,
                    new_dst,
                    latency=edge.latency,
                )
                wrap_edge.capacity = edge.capacity
                wrap_edge.conflict_free = edge.conflict_free
                sliced.add_edge(wrap_edge)
                wrap_counter += 1
                continue

            # Edge outside slice — drop.

        self._slice_cache[target_ii] = sliced
        return sliced

    def get_output_ports_for_operation(
        self,
        fu_node: MRRGNode,
        operation: OperationType,
        required_bitwidth: Optional[int] = None,
    ) -> List[MRRGNode]:
        """Return legal source output ports for an operation on a specific FU/IO node.

        If Dora output-port metadata is available, use it directly.
        Otherwise fall back to scanning routing successors for backward compatibility.
        """
        ports: List[MRRGNode] = []
        seen_ids: Set[str] = set()
        outgoing_ports = [
            self.get_node(edge.destination.id)
            for edge in self.get_outgoing_edges(fu_node.id)
        ]
        outgoing_ports = [
            port for port in outgoing_ports
            if port is not None and port.node_type != NodeType.FUNCTION
        ]

        explicit_ports = fu_node.operation_output_ports.get(operation, [])
        if explicit_ports:
            port_bws = fu_node.operation_output_port_bitwidths.get(operation, {})
            for port_base in explicit_ports:
                matched = False
                for port in outgoing_ports:
                    if port.get_base_name() != port_base:
                        continue
                    matched = True
                    port_bw = port_bws.get(port_base, port.bitwidth)
                    if required_bitwidth is not None and port_bw < required_bitwidth:
                        continue
                    if port.id not in seen_ids:
                        ports.append(port)
                        seen_ids.add(port.id)
                if matched:
                    continue

                # Fallback for architectures that store explicit port metadata
                # but do not expose the port as a direct FU successor.
                port = self.get_node_by_cycle(fu_node.cycle, port_base)
                if port is None:
                    continue
                port_bw = port_bws.get(port_base, port.bitwidth)
                if required_bitwidth is not None and port_bw < required_bitwidth:
                    continue
                if port.id not in seen_ids:
                    ports.append(port)
                    seen_ids.add(port.id)
        else:
            for port in outgoing_ports:
                if required_bitwidth is not None and port.bitwidth < required_bitwidth:
                    continue
                if port.id not in seen_ids:
                    ports.append(port)
                    seen_ids.add(port.id)

        ports.sort(key=lambda n: (-n.bitwidth, n.id))
        return ports

    def get_input_ports_for_operand(
        self,
        fu_node: MRRGNode,
        operation: OperationType,
        operand_tag: Optional[OperandTag],
    ) -> List[MRRGNode]:
        """Return legal sink input ports for an operation/operand on a placed FU/IO."""
        ports: List[MRRGNode] = []
        seen_ids: Set[str] = set()

        for edge in self.get_incoming_edges(fu_node.id):
            port = self.get_node(edge.source.id)
            if port is None or port.node_type not in (NodeType.ROUTING, NodeType.ROUTING_FUNCTION):
                continue

            pin_tags = port.get_supported_operand_tags_for_operation(operation)
            has_binding_metadata = bool(
                port.operation_supported_operand_tags or port.supported_operand_tags
            )

            if operand_tag is None:
                if has_binding_metadata and not pin_tags:
                    continue
            else:
                if pin_tags:
                    if not MRRGNode.operand_tag_matches_request(operand_tag, pin_tags):
                        continue
                elif has_binding_metadata:
                    continue

            if port.id not in seen_ids:
                ports.append(port)
                seen_ids.add(port.id)

        ports.sort(key=lambda n: (-n.bitwidth, n.id))
        return ports

    def get_array_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the CGRA array."""
        return (self.rows, self.cols)

    def _parse_coordinates(self, attrs: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Parse coordinates from node attributes.

        Args:
            attrs: Dictionary of node attributes

        Returns:
            (x, y) tuple if coordinates found, None otherwise

        Notes:
            - Tries 'x'/'y' attributes first
            - Falls back to 'col'/'row' attributes
            - Returns None if coordinates cannot be parsed
        """
        x_str = attrs.get('x', attrs.get('col', ''))
        y_str = attrs.get('y', attrs.get('row', ''))

        if x_str and y_str:
            try:
                return (int(x_str), int(y_str))
            except ValueError:
                pass

        return None

    def _parse_coordinates_from_name(self, node_name: str, add_edge_offset: bool = True) -> Optional[Tuple[int, int]]:
        """
        Extract coordinates from node name patterns.

        Args:
            node_name: Node name or base name (without cycle prefix)
            add_edge_offset: If True, add 1 to coordinates to account for edge PEs

        Returns:
            (x, y) tuple if pattern matches, None otherwise

        Notes:
            Supports common naming patterns:
            - pe_cX_rY (e.g., pe_c2_r1 -> (3, 2) with edge offset)
            - pe_X_Y (e.g., pe_2_1 -> (3, 2) with edge offset)
            - cXrY (e.g., c2r1 -> (3, 2) with edge offset)

            The add_edge_offset parameter accounts for architectures where
            edge PEs are numbered separately (e.g., CGRA-ME convention where
            pe_c0_r0 is at position (1,1) in the actual array).
        """
        import re

        # Remove cycle prefix if present
        if ':' in node_name:
            node_name = node_name.split(':', 1)[1]

        # Remove port suffixes (.in, .out, .in_a, etc.)
        base = node_name.split('.')[0]

        col, row = None, None

        # Pattern 1: pe_cX_rY or fu_cX_rY or reg_cX_rY
        match = re.search(r'[_]c(\d+)_r(\d+)', base)
        if match:
            col = int(match.group(1))
            row = int(match.group(2))

        # Pattern 2: pe_X_Y or fu_X_Y (generic underscore-separated)
        if col is None:
            match = re.search(r'[_](\d+)_(\d+)', base)
            if match:
                col = int(match.group(1))
                row = int(match.group(2))

        # Pattern 3: cXrY
        if col is None:
            match = re.search(r'c(\d+)r(\d+)', base)
            if match:
                col = int(match.group(1))
                row = int(match.group(2))

        # Apply edge offset if requested and coordinates found
        if col is not None and row is not None:
            if add_edge_offset:
                return (col + 1, row + 1)
            return (col, row)

        return None

    def create_time_expanded_fu(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        supported_operations: Set[OperationType],
        fu_latency: int = 1,
        bitwidth: int = 32,
        fu_ii: int = 1
    ) -> None:
        """
        Create time-expanded FU nodes following CGRA-ME formulation.

        Creates input operand pins, FU node, and output node for each time slot,
        with latency edges connecting FU to output at (cycle + latency) % II.

        Args:
            base_name: Base name for the FU (e.g., "fu_0_0")
            coordinates: (x, y) position in CGRA
            supported_operations: Set of operations this FU can execute
            fu_latency: Execution latency
            bitwidth: Data width in bits
            fu_ii: FU initiation interval (for pipelined FUs)
        """
        # Create nodes at each time slot (stepping by FU II)
        for cycle in range(0, self.II, fu_ii):
            # Input operand pins
            in_a_id = f"{cycle}:{base_name}.in_a"
            in_a = MRRGNode(
                in_a_id,
                NodeType.ROUTING,  # Input pins are routing nodes
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_WIRE,
                latency=0,
                bitwidth=bitwidth,
                supported_operand_tags={OperandTag.BINARY_LHS, OperandTag.BINARY_ANY}
            )

            in_b_id = f"{cycle}:{base_name}.in_b"
            in_b = MRRGNode(
                in_b_id,
                NodeType.ROUTING,
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_WIRE,
                latency=0,
                bitwidth=bitwidth,
                supported_operand_tags={OperandTag.BINARY_RHS, OperandTag.BINARY_ANY}
            )

            # Function node
            fu_id = f"{cycle}:{base_name}"
            fu = MRRGNode(
                fu_id,
                NodeType.FUNCTION,
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_COMB,
                latency=fu_latency,
                bitwidth=bitwidth,
                supported_operations=supported_operations
            )

            # Output node (will connect to future cycle)
            out_id = f"{cycle}:{base_name}.out"
            out = MRRGNode(
                out_id,
                NodeType.ROUTING,
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_WIRE,
                latency=0,
                bitwidth=bitwidth,
                routing_type="wire"
            )

            # Add nodes
            self.add_node(in_a)
            self.add_node(in_b)
            self.add_node(fu)
            self.add_node(out)

            # Connect inputs to FU (within same cycle)
            self.add_edge(MRRGEdge(f"{in_a_id}_to_{fu_id}", in_a, fu, latency=0))
            self.add_edge(MRRGEdge(f"{in_b_id}_to_{fu_id}", in_b, fu, latency=0))

        # Connect FU to output across cycles (latency edges)
        for cycle in range(0, self.II, fu_ii):
            fu_id = f"{cycle}:{base_name}"
            fu = self.get_node(fu_id)

            out_cycle = (cycle + fu_latency) % self.II  # Modulo wrap
            out_id = f"{out_cycle}:{base_name}.out"
            out = self.get_node(out_id)

            if fu and out:
                self.add_edge(MRRGEdge(f"{fu_id}_to_{out_id}", fu, out, latency=fu_latency))

    def create_time_expanded_register(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        bank_id: Optional[int] = None,
        reg_latency: int = 1,
        bitwidth: int = 32
    ) -> None:
        """
        Create time-expanded register nodes following CGRA-ME formulation.

        Creates register feedback loop across II boundary for pipelined operation.
        Structure per cycle:
            in -> m_in -> reg -[next_cycle]-> m_out -> out -> m_enable -|
                                 ^                                      |
                                 |--------------------------------------|

        Args:
            base_name: Base name for the register (e.g., "reg_0_0")
            coordinates: (x, y) position in CGRA
            bank_id: Register bank identifier
            reg_latency: Register latency (typically 1)
            bitwidth: Data width in bits
        """
        # Create nodes for all cycles
        for cycle in range(self.II):
            # Input
            in_id = f"{cycle}:{base_name}.in"
            in_node = MRRGNode(in_id, NodeType.ROUTING, cycle, coordinates,
                              HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")

            # Multiplexer input
            m_in_id = f"{cycle}:{base_name}.m_in"
            m_in = MRRGNode(m_in_id, NodeType.ROUTING, cycle, coordinates,
                           HWEntityType.HW_MUX, 0, bitwidth, routing_type="mux")

            # Register (with latency)
            reg_id = f"{cycle}:{base_name}.reg"
            reg = MRRGNode(reg_id, NodeType.ROUTING, cycle, coordinates,
                          HWEntityType.HW_REG, reg_latency, bitwidth, bank_id=bank_id)

            # Multiplexer output
            m_out_id = f"{cycle}:{base_name}.m_out"
            m_out = MRRGNode(m_out_id, NodeType.ROUTING, cycle, coordinates,
                            HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")

            # Enable multiplexer
            m_enable_id = f"{cycle}:{base_name}.m_enable"
            m_enable = MRRGNode(m_enable_id, NodeType.ROUTING, cycle, coordinates,
                               HWEntityType.HW_MUX, 0, bitwidth, routing_type="mux")

            # Output
            out_id = f"{cycle}:{base_name}.out"
            out_node = MRRGNode(out_id, NodeType.ROUTING, cycle, coordinates,
                               HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")

            # Add nodes
            self.add_node(in_node)
            self.add_node(m_in)
            self.add_node(reg)
            self.add_node(m_out)
            self.add_node(m_enable)
            self.add_node(out_node)

        # Connect across cycles
        for cycle in range(self.II):
            next_cycle = (cycle + reg_latency) % self.II  # Modulo wrapping

            # Get nodes for this cycle
            in_node = self.get_node(f"{cycle}:{base_name}.in")
            m_in = self.get_node(f"{cycle}:{base_name}.m_in")
            reg = self.get_node(f"{cycle}:{base_name}.reg")
            m_enable = self.get_node(f"{cycle}:{base_name}.m_enable")
            out_node = self.get_node(f"{cycle}:{base_name}.out")

            # Get nodes for next cycle
            m_out_next = self.get_node(f"{next_cycle}:{base_name}.m_out")

            # Within same cycle
            if in_node and m_in:
                self.add_edge(MRRGEdge(f"{in_node.id}_to_{m_in.id}", in_node, m_in, 0))
            if m_in and reg:
                self.add_edge(MRRGEdge(f"{m_in.id}_to_{reg.id}", m_in, reg, 0))
            if m_enable and reg:
                self.add_edge(MRRGEdge(f"{m_enable.id}_to_{reg.id}", m_enable, reg, 0))
            if out_node and m_enable:
                self.add_edge(MRRGEdge(f"{out_node.id}_to_{m_enable.id}", out_node, m_enable, 0))

            # Cross-cycle for latency (register to next cycle's output mux)
            if reg and m_out_next:
                self.add_edge(MRRGEdge(f"{reg.id}_to_{m_out_next.id}", reg, m_out_next, reg_latency))

            # Connect m_out to out within next cycle
            if m_out_next:
                out_next = self.get_node(f"{next_cycle}:{base_name}.out")
                if out_next:
                    self.add_edge(MRRGEdge(f"{m_out_next.id}_to_{out_next.id}", m_out_next, out_next, 0))

    def create_time_expanded_mux(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        num_inputs: int,
        bitwidth: int = 32
    ) -> None:
        """
        Create time-expanded multiplexer nodes following CGRA-ME formulation.

        Args:
            base_name: Base name for the mux (e.g., "mux_0_0")
            coordinates: (x, y) position in CGRA
            num_inputs: Number of input ports
            bitwidth: Data width in bits
        """
        for cycle in range(self.II):
            # Output
            out_id = f"{cycle}:{base_name}.out"
            out = MRRGNode(out_id, NodeType.ROUTING, cycle, coordinates,
                          HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")
            self.add_node(out)

            # Mux node
            mux_id = f"{cycle}:{base_name}.mux"
            mux = MRRGNode(mux_id, NodeType.ROUTING, cycle, coordinates,
                          HWEntityType.HW_MUX, 0, bitwidth, routing_type="mux")
            self.add_node(mux)

            # Connect mux to output
            self.add_edge(MRRGEdge(f"{mux_id}_to_{out_id}", mux, out, 0))

            # Input ports
            for i in range(num_inputs):
                in_id = f"{cycle}:{base_name}.in{i}"
                in_node = MRRGNode(in_id, NodeType.ROUTING, cycle, coordinates,
                                  HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")
                self.add_node(in_node)

                # Connect input to mux
                self.add_edge(MRRGEdge(f"{in_id}_to_{mux_id}", in_node, mux, 0))

    def create_time_expanded_routing_wire(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        wire_latency: int = 0,
        bitwidth: int = 32
    ) -> None:
        """
        Create time-expanded routing wire nodes.

        Args:
            base_name: Base name for the wire (e.g., "wire_n_0_0")
            coordinates: (x, y) position in CGRA
            wire_latency: Wire latency (0 for combinational)
            bitwidth: Data width in bits
        """
        for cycle in range(self.II):
            wire_id = f"{cycle}:{base_name}"
            wire = MRRGNode(wire_id, NodeType.ROUTING, cycle, coordinates,
                           HWEntityType.HW_WIRE, wire_latency, bitwidth, routing_type="wire")
            self.add_node(wire)

    def to_string(self) -> str:
        """
        Return detailed string representation of the MRRG.

        Returns:
            Multi-line string with MRRG statistics and breakdown
        """
        # Count nodes by type
        fu_nodes = self.get_fu_nodes()
        routing_nodes = self.get_routing_nodes()
        register_nodes = self.get_register_nodes()

        # Count by hardware entity type
        hw_types = {}
        for node in self.get_nodes():
            hw_type = node.hw_entity_type.value
            hw_types[hw_type] = hw_types.get(hw_type, 0) + 1

        # Build detailed string
        lines = [
            f"MRRG: {self.name}",
            f"  Initiation Interval (II): {self.II}",
            f"  Array Dimensions: {self.rows} x {self.cols}",
            f"  Total Nodes: {self.num_nodes()}",
            f"  Total Edges: {self.num_edges()}",
            "",
            "  Node Breakdown by NodeType:",
            f"    FUNCTION nodes:  {len(fu_nodes):5d}",
            f"    ROUTING nodes:   {len(routing_nodes):5d}",
            f"    REGISTER nodes:  {len(register_nodes):5d}",
            "",
            "  Node Breakdown by HWEntityType:",
        ]

        for hw_type, count in sorted(hw_types.items()):
            lines.append(f"    {hw_type.upper():15s}: {count:5d}")

        # Add cycle information
        if self.II > 0:
            lines.append("")
            lines.append("  Nodes per Cycle:")
            for cycle in range(self.II):
                nodes_at_cycle = self.get_nodes_at_cycle(cycle)
                fu_at_cycle = [n for n in nodes_at_cycle if n.node_type == NodeType.FUNCTION]
                reg_at_cycle = [n for n in nodes_at_cycle if n.hw_entity_type == HWEntityType.HW_REG]
                lines.append(f"    Cycle {cycle}: {len(nodes_at_cycle):4d} nodes "
                           f"({len(fu_at_cycle)} FUs, {len(reg_at_cycle)} regs)")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f"MRRG(name={self.name}, II={self.II}, dim={self.rows}x{self.cols}, "
                f"nodes={self.num_nodes()}, edges={self.num_edges()})")

    # -- Fracture helpers -----------------------------------------------------------

    def _build_packing_endpoint_index(self) -> Dict[str, List[MRRGNode]]:
        """Build lazy index: {parent_port_base -> [pack/unpack-capable nodes]}."""
        if hasattr(self, "_packing_endpoint_idx"):
            return self._packing_endpoint_idx

        idx: Dict[str, List[MRRGNode]] = {}
        for node in self.get_nodes():
            if node.packing_parent_port is None:
                continue
            if not (node.pack_capable or node.unpack_capable):
                continue
            idx.setdefault(node.packing_parent_port, []).append(node)

        self._packing_endpoint_idx = idx
        return idx

    def get_pack_nodes_for_port(
        self,
        port_base_name: str,
        cycle: int,
    ) -> List[MRRGNode]:
        """Return pack-capable routing nodes associated with an output port."""
        idx = self._build_packing_endpoint_index()
        candidates = idx.get(port_base_name, [])
        return sorted(
            (
                node
                for node in candidates
                if node.cycle == cycle and node.pack_capable
            ),
            key=lambda node: node.id,
        )

    def get_unpack_nodes_for_port(
        self,
        port_base_name: str,
        cycle: int,
    ) -> List[MRRGNode]:
        """Return unpack-capable routing nodes associated with an input port."""
        idx = self._build_packing_endpoint_index()
        candidates = idx.get(port_base_name, [])
        return sorted(
            (
                node
                for node in candidates
                if node.cycle == cycle and node.unpack_capable
            ),
            key=lambda node: node.id,
        )

    def get_packing_endpoints_for_output_port(
        self,
        port: MRRGNode,
    ) -> List[PackingEndpoint]:
        """Return pack-capable endpoints reachable from an output port."""
        endpoints: List[PackingEndpoint] = []
        for node in self.get_pack_nodes_for_port(port.get_base_name(), port.cycle):
            boundary_path = self.find_routing_path(port, node, max_depth=12)
            if not boundary_path:
                continue
            endpoints.append(
                PackingEndpoint(
                    port=port,
                    boundary_node=node,
                    boundary_path=tuple(boundary_path),
                    lane_width=node.lane_width,
                    num_lanes=node.num_lanes,
                    allowed_lane_indices=node.get_allowed_lane_indices(),
                    requires_existing_carrier=node.packing_requires_carrier,
                )
            )
        return endpoints

    def get_packing_endpoints_for_input_port(
        self,
        port: MRRGNode,
    ) -> List[PackingEndpoint]:
        """Return unpack-capable endpoints attached to an input port."""
        endpoints: List[PackingEndpoint] = []
        for node in self.get_unpack_nodes_for_port(port.get_base_name(), port.cycle):
            boundary_path = self.find_routing_path(node, port, max_depth=12)
            if not boundary_path:
                continue
            endpoints.append(
                PackingEndpoint(
                    port=port,
                    boundary_node=node,
                    boundary_path=tuple(boundary_path),
                    lane_width=node.lane_width,
                    num_lanes=node.num_lanes,
                    allowed_lane_indices=node.get_allowed_lane_indices(),
                    requires_existing_carrier=False,
                )
            )
        return endpoints

    def _build_fracture_chunk_index(self) -> Dict[str, List[MRRGNode]]:
        """Build lazy index: {parent_port_base -> [chunk_nodes]}.

        Only actual chunk net nodes are indexed. Split/join instance nodes may
        carry ``fracture_parent_port`` metadata as well, but they are not chunk
        endpoints and should not appear in this lookup.
        """
        if hasattr(self, '_fracture_chunk_idx'):
            return self._fracture_chunk_idx

        idx: Dict[str, List[MRRGNode]] = {}
        for node in self.get_nodes():
            if node.fracture_parent_port is None or node.fracture_chunk_index is None:
                continue
            base = node.fracture_parent_port
            if base not in idx:
                idx[base] = []
            idx[base].append(node)
        self._fracture_chunk_idx = idx
        return idx

    def get_fracture_chunks_for_port(
        self, port_base_name: str, cycle: int
    ) -> List[MRRGNode]:
        """Return chunk nodes whose fracture_parent_port matches, filtered to cycle.

        Sorted by fracture_chunk_index. Works for ALL fracturable endpoints
        (ALU, IO, const, LSU) because it's driven by fracture_parent_port metadata.

        Args:
            port_base_name: Base name of the FU port (e.g., "pe_0_0.alu_inst.out_o")
            cycle: Time slot to filter on

        Returns:
            List of MRRGNode sorted by fracture_chunk_index, empty if none found.
        """
        idx = self._build_fracture_chunk_index()
        candidates = idx.get(port_base_name, [])
        filtered = [n for n in candidates if n.cycle == cycle]
        filtered.sort(key=lambda n: n.fracture_chunk_index)
        return filtered

    def get_fracture_chunk_groups_reaching_port(
        self,
        sink_port: MRRGNode,
        *,
        max_depth: int = 4,
    ) -> List[List[MRRGNode]]:
        """Return chunk groups that can reach a sink port via join framing.

        This is needed for architectures where the chunk metadata is attached
        to a shared join tree rather than directly to the FU input pin's
        ``parent_port``. A common example is the fracturable LSU address/data
        inputs, where multiple ``join_data_*`` instances fan out to both
        ``addr_i`` and ``data_i``.
        """
        groups: List[List[MRRGNode]] = []
        queue = deque([(sink_port, 0)])
        visited: Set[str] = {sink_port.id}
        seen_joins: Set[str] = set()

        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                continue

            if node.fracture_type == "join":
                if node.id in seen_joins:
                    continue
                seen_joins.add(node.id)

                chunks = []
                for edge in self.get_incoming_edges(node.id):
                    pred = self.get_node(edge.source.id)
                    if pred is None or pred.cycle != sink_port.cycle:
                        continue
                    if pred.fracture_chunk_index is None:
                        continue
                    chunks.append(pred)

                chunks.sort(key=lambda n: n.fracture_chunk_index)
                if chunks:
                    groups.append(chunks)
                continue

            for edge in self.get_incoming_edges(node.id):
                pred = self.get_node(edge.source.id)
                if pred is None or pred.id in visited:
                    continue
                if pred.node_type == NodeType.FUNCTION:
                    continue
                visited.add(pred.id)
                queue.append((pred, depth + 1))

        return groups

    @staticmethod
    def compute_fracture_chunk_layout(
        chunks: List[MRRGNode],
    ) -> Optional[Tuple[List[int], List[int], List[int], int]]:
        """Return validated chunk layout metadata.

        Returns ``(indices, widths, offsets, total_width)`` when the chunk list
        forms a complete contiguous layout beginning at bit offset 0. Returns
        ``None`` if the metadata is missing, duplicated, or leaves gaps.
        """
        if not chunks:
            return None

        indexed: List[Tuple[int, int, int]] = []
        seen_indices: Set[int] = set()
        for node in chunks:
            chunk_index = node.fracture_chunk_index
            bit_offset = node.fracture_bit_offset
            if chunk_index is None or bit_offset is None:
                return None
            if chunk_index in seen_indices:
                return None
            seen_indices.add(chunk_index)
            indexed.append((chunk_index, node.bitwidth, bit_offset))

        indexed.sort(key=lambda item: item[0])
        indices = [idx for idx, _, _ in indexed]
        if indices != list(range(len(indexed))):
            return None

        offsets_by_index = [offset for _, _, offset in indexed]
        widths_by_index = [width for _, width, _ in indexed]

        expected_offset = 0
        for (_, width, offset) in sorted(indexed, key=lambda item: item[2]):
            if offset != expected_offset:
                return None
            expected_offset += width

        return indices, widths_by_index, offsets_by_index, expected_offset

    def get_fracture_endpoints_for_output_port(
        self,
        port: MRRGNode,
    ) -> List[FractureEndpoint]:
        """Return fracture-capable source endpoints attached to an output port."""
        chunks = self.get_fracture_chunks_for_port(port.get_base_name(), port.cycle)
        layout = self.compute_fracture_chunk_layout(chunks)
        if layout is None:
            return []
        return [
            FractureEndpoint(
                port=port,
                chunk_nodes=tuple(chunks),
                chunk_indices=tuple(layout[0]),
                chunk_widths=tuple(layout[1]),
                chunk_offsets=tuple(layout[2]),
                total_width=layout[3],
            )
        ]

    def get_fracture_endpoints_for_input_port(
        self,
        port: MRRGNode,
    ) -> List[FractureEndpoint]:
        """Return fracture-capable sink endpoints attached to or reaching an input port."""
        chunk_groups = self.get_fracture_chunk_groups_reaching_port(port)
        if not chunk_groups:
            direct_chunks = self.get_fracture_chunks_for_port(port.get_base_name(), port.cycle)
            if direct_chunks:
                chunk_groups = [direct_chunks]

        endpoints: List[FractureEndpoint] = []
        for chunks in chunk_groups:
            layout = self.compute_fracture_chunk_layout(chunks)
            if layout is None:
                continue
            endpoints.append(
                FractureEndpoint(
                    port=port,
                    chunk_nodes=tuple(chunks),
                    chunk_indices=tuple(layout[0]),
                    chunk_widths=tuple(layout[1]),
                    chunk_offsets=tuple(layout[2]),
                    total_width=layout[3],
                )
            )
        return endpoints

    def find_routing_path(
        self,
        start: MRRGNode,
        end: MRRGNode,
        *,
        exact_bitwidth: Optional[int] = None,
        max_depth: int = 16,
    ) -> List[MRRGNode]:
        """Find a local routing-only path from ``start`` to ``end`` inclusive.

        This is used for deterministic split/join boundary framing, not global
        congestion-aware routing. FUNCTION nodes are not traversed unless they
        are the explicit destination.
        """
        if start == end:
            return [start]

        queue = deque([(start, [start])])
        visited = {start.id}

        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue

            for edge in self.get_outgoing_edges(node.id):
                nxt = self.get_node(edge.destination.id)
                if nxt is None:
                    continue
                if nxt.node_type == NodeType.FUNCTION and nxt != end:
                    continue
                if (
                    exact_bitwidth is not None
                    and nxt != end
                    and not nxt.get_attribute("is_instance", False)
                    and nxt.bitwidth != exact_bitwidth
                ):
                    continue
                if nxt.id in visited:
                    continue

                next_path = path + [nxt]
                if nxt == end:
                    return next_path

                visited.add(nxt.id)
                queue.append((nxt, next_path))

        return []

    # -- Helper methods for Version 1 ---------------------------------------------
    def get_fu_nodes(self) -> List[MRRGNode]:
        """Get all FU nodes."""
        return [n for n in self.get_nodes() if n.node_type == NodeType.FUNCTION]

    def get_routing_nodes(self) -> List[MRRGNode]:
        """Get all routing nodes."""
        return [n for n in self.get_nodes() if n.node_type == NodeType.ROUTING]

    def create_routing_fanout_map(self) -> Dict[int, List[int]]:
        """Create a mapping from routing node index to list of destination node indices."""
        routing_nodes = self.get_routing_nodes()
        fanout_map = {i: [] for i in range(len(routing_nodes))}
        routing_node_id_reverse_map = {node.id: i for i, node in enumerate(routing_nodes)}

        for node in routing_nodes:
            # get_outgoing_edges returns EDGE OBJECTS, not IDs!
            for edge_obj in self.get_outgoing_edges(node.id):
                if edge_obj:  # Changed from get_edge(edge_id)
                    dest_id = edge_obj.destination.id
                    if dest_id in routing_node_id_reverse_map:
                        src_idx = routing_node_id_reverse_map[node.id]
                        dst_idx = routing_node_id_reverse_map[dest_id]
                        fanout_map[src_idx].append(dst_idx)
        
        return fanout_map

    def create_routing_fanin_map(self) -> Dict[int, List[int]]:
        """Create a mapping from routing node index to list of source (predecessor) node indices.

        This is the inverse of create_routing_fanout_map - for each routing node,
        it returns the list of routing nodes that feed into it.
        """
        routing_nodes = self.get_routing_nodes()
        fanin_map = {i: [] for i in range(len(routing_nodes))}
        routing_node_id_reverse_map = {node.id: i for i, node in enumerate(routing_nodes)}

        for node in routing_nodes:
            # Get incoming edges to this node
            for edge_obj in self.get_incoming_edges(node.id):
                if edge_obj:
                    src_id = edge_obj.source.id
                    if src_id in routing_node_id_reverse_map:
                        dst_idx = routing_node_id_reverse_map[node.id]
                        src_idx = routing_node_id_reverse_map[src_id]
                        fanin_map[dst_idx].append(src_idx)

        return fanin_map

    def get_sink_fus_connected_to_src_routing_node(self, routing_node_id: str) -> List[MRRGNode]:
        """Get all FUs connected to a routing node."""

        # Make sure the entered routing node id is actually a routing node.
        if self.get_node(routing_node_id).node_type != NodeType.ROUTING:
            raise ValueError(f"The entered routing node id {routing_node_id} is not a routing node.")

        return [edge.destination for edge in self.get_edges() if edge.destination.node_type == NodeType.FUNCTION and edge.source.id == routing_node_id]

    def get_sink_routing_nodes_connected_to_src_fu(self, fu_node_id: str) -> List[MRRGNode]:
        """Get all routing nodes (sinks) connected to a FU node."""

        # Make sure the entered FU node id is actually a FU node.
        if self.get_node(fu_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered FU node id {fu_node_id} is not a FU node.")

        return [edge.destination for edge in self.get_edges() if edge.destination.node_type == NodeType.ROUTING and edge.source.id == fu_node_id]

    def get_fan_in_nodes(self, node_id: str) -> List[MRRGNode]:
        """Get all nodes that are fan-in to this node."""

        # Make sure the entered node id is actually a node.
        if self.get_node(node_id) is None:
            raise ValueError(f"The entered node id {node_id} is not a node.")

        return [edge.source for edge in self.get_edges() if edge.destination.id == node_id]

    # -- Helper methods for Version 2 ---------------------------------------------
    def get_sink_fu_neighbors(self, fu_node_id: str, neighbor_count: int = 20) -> List[MRRGNode]:
        """Get the specified number of FU nodes that are neighbors to the given FU node."""

        # Make sure the entered FU node id is actually a FU node.
        if self.get_node(fu_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered FU node id {fu_node_id} is not a FU node.")

        # Perform a breadth-first search to find the specified number of FU nodes that are neighbors to the given FU node.
        all_neighbors = traversal.bfs(graph=self, start_node_id=fu_node_id)

        # Prune out the nodes that are not FU nodes.
        fu_neighbors = [n for n in all_neighbors if n.node_type == NodeType.FUNCTION]

        # Return the specified number of FU nodes.
        return fu_neighbors[:neighbor_count]

    def get_all_paths_between_fu_nodes(self, start_node_id: str, end_node_id: str, max_paths: int = 20) -> List[List[MRRGNode]]:
        """Get the shortest path between two nodes."""

        # Make sure the entered start node id is actually a node.
        if self.get_node(start_node_id) is None or self.get_node(start_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered start node id {start_node_id} is not a node.")

        # Make sure the entered end node id is actually a node.
        if self.get_node(end_node_id) is None or self.get_node(end_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered end node id {end_node_id} is not a node.")

        return traversal.get_all_paths(graph=self, start_node_id=start_node_id, end_node_id=end_node_id, max_paths=20)

    def get_all_paths_in_mrrg_fu_space(self) -> Dict[Tuple[int, int], List[List[MRRGNode]]]:
        """Get a map of all paths in the MRRG FU space."""

        # Get all FU nodes.
        fu_nodes = self.get_fu_nodes()

        # Create a map of all paths between all FU nodes both ways.
        path_map = {}
        for i in range(len(fu_nodes)):
            for j in range(len(fu_nodes)):
                path_map[(i, j)] = self.get_all_paths_between_fu_nodes(fu_nodes[i].id, fu_nodes[j].id)
        return path_map

    def get_k_shortest_paths_between_fu_nodes(self, start_node_id: str, end_node_id: str, k: int) -> List[List[str]]:
        """Get the k shortest paths between two FU nodes.

        Only returns paths where all intermediate nodes are routing nodes.
        The start and end nodes are FU nodes, but internal nodes must be routing nodes.

        Returns:
            List of paths, where each path is a list of node IDs (strings), not MRRGNode objects.
        """

        # Make sure the entered start node id is actually a node.
        if self.get_node(start_node_id) is None or self.get_node(start_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered start node id {start_node_id} is not a node.")

        # Make sure the entered end node id is actually a node.
        if self.get_node(end_node_id) is None or self.get_node(end_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered end node id {end_node_id} is not a node.")

        # Get more paths than requested since we'll filter some out
        # Use a multiplier to account for paths that go through intermediate FUs
        candidate_paths = traversal.get_k_shortest_paths(
            graph=self, start_node_id=start_node_id, end_node_id=end_node_id, k=k * 5
        )

        # Filter to only keep paths where intermediate nodes are routing nodes
        valid_paths = []
        for path in candidate_paths:
            # Check intermediate nodes (exclude first and last which are the FU endpoints)
            is_valid = True
            for node_id in path[1:-1]:
                node = self.get_node(node_id)
                if node.node_type == NodeType.FUNCTION:
                    is_valid = False
                    break
            if is_valid:
                valid_paths.append(path)
                if len(valid_paths) >= k:
                    break

        return valid_paths

    def get_k_shortest_paths_between_fu_nodes_optimized(
        self, start_node_id: str, end_node_id: str, k: int
    ) -> List[List[str]]:
        """Optimized version using subgraph filtering before path enumeration.

        Converts the MRRG to a NetworkX DiGraph, excludes intermediate FU nodes,
        then runs Yen's algorithm. This eliminates the need for the k*5 multiplier
        and post-hoc filtering.

        Args:
            start_node_id: ID of the source FU node
            end_node_id: ID of the destination FU node
            k: Number of shortest paths to find

        Returns:
            List of paths, where each path is a list of node IDs (strings).
        """
        import itertools
        import networkx as nx

        # Validate inputs
        start_node = self.get_node(start_node_id)
        end_node = self.get_node(end_node_id)

        if start_node is None or start_node.node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered start node id {start_node_id} is not a valid FU node.")
        if end_node is None or end_node.node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered end node id {end_node_id} is not a valid FU node.")

        # Get all FU nodes and build exclusion set
        fu_node_ids = {n.id for n in self.get_fu_nodes()}
        nodes_to_exclude = fu_node_ids - {start_node_id, end_node_id}

        # Build a NetworkX DiGraph with only non-excluded nodes
        # This is more compatible than using subgraph_view on our custom graph
        nx_graph = nx.DiGraph()

        # Add nodes (excluding intermediate FUs)
        for node_id in self:
            if node_id not in nodes_to_exclude:
                nx_graph.add_node(node_id)

        # Add edges (only between included nodes)
        for src_id in self:
            if src_id in nodes_to_exclude:
                continue
            for dst_id, edge_data in self[src_id].items():
                if dst_id not in nodes_to_exclude:
                    nx_graph.add_edge(src_id, dst_id, weight=edge_data.get('weight', 1))

        # Request exactly k paths - no multiplier needed since we pre-filtered
        try:
            gen = nx.shortest_simple_paths(nx_graph, start_node_id, end_node_id, weight='weight')
            return list(itertools.islice(gen, k))
        except nx.NetworkXNoPath:
            return []

    def get_k_shortest_paths_in_mrrg_fu_space(
        self, k: int, max_manhattan_dist: Optional[int] = None
    ) -> Dict[Tuple[int, int], List[List[str]]]:
        """Get a map of the k shortest paths in the MRRG FU space.

        Uses spatial pruning to skip FU pairs that are too far apart,
        and uses the optimized path computation that pre-filters intermediate FUs.

        Args:
            k: Number of shortest paths to find per FU pair
            max_manhattan_dist: Maximum Manhattan distance between FU pairs.
                               If None, defaults to max(rows, cols).
                               Pairs beyond this distance return empty paths.

        Returns:
            Dictionary mapping (FU_index_i, FU_index_j) to list of paths,
            where each path is a list of node IDs (strings), not MRRGNode objects.
        """
        fu_nodes = self.get_fu_nodes()

        # Use max(rows, cols) as default - safer than sum (avoids over-pruning)
        if max_manhattan_dist is None:
            max_manhattan_dist = max(self.rows, self.cols)

        path_map = {}
        for i, fu_i in enumerate(fu_nodes):
            for j, fu_j in enumerate(fu_nodes):
                # Spatial pruning with safe fallback for missing coordinates
                if fu_i.coordinates is not None and fu_j.coordinates is not None:
                    x1, y1 = fu_i.coordinates
                    x2, y2 = fu_j.coordinates
                    if abs(x1 - x2) + abs(y1 - y2) > max_manhattan_dist:
                        path_map[(i, j)] = []
                        continue

                # Use optimized version with subgraph filtering
                path_map[(i, j)] = self.get_k_shortest_paths_between_fu_nodes_optimized(
                    fu_i.id, fu_j.id, k
                )

        return path_map
