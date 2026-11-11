"""DOT parser for CGRA-ME MRRG files."""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from cgrasolve.graph.mrrg import (
    MRRG, MRRGNode, MRRGEdge,
    NodeType, OperandTag, HWEntityType
)
from cgrasolve.graph.dfg import OperationType


class MRRGDotParser:
    """
    Parser for CGRA-ME MRRG DOT files (new format).

    Expected DOT format attributes:
    - type: "ROUTING" | "FUNCTION" | "ROUTING_FUNCTION"
    - et: "WIRE" | "REG" | "MUX" | "COMB"
    - supportedOps: semicolon-separated operations (e.g., "add;sub;mul;")
    - supportedOperands: semicolon-separated operand tags (optional)
    - bitwidth, latency, delay, capacity, cycle, parent

    Parses Modulo Routing Resource Graphs extracting:
    - Time-expanded nodes with cycle:name format
    - Unified MRRGNode instances with all attributes
    - Edges with latency information
    """

    def __init__(self):
        """Initialize the MRRG DOT parser."""
        self.node_pattern = re.compile(r'"?([^"\s]+)"?\s*\[([^\]]+)\]')
        self.edge_pattern = re.compile(r'"?([^"\s]+)"?\s*->\s*"?([^"\s]+)"?(?:\s*\[([^\]]+)\])?')
        self.attr_pattern = re.compile(r'(\w+)="?([^",\]]*)"?')

    def parse(self, file_path: Path) -> MRRG:
        """
        Parse an MRRG from a CGRA-ME DOT file.

        Args:
            file_path: Path to the DOT file

        Returns:
            MRRG object
        """
        print(f"Parsing MRRG from {file_path}...")

        # Read file
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract II from comment or infer from max cycle
        II = self._extract_ii(content)

        # Extract nodes and edges
        nodes_dict = self._extract_nodes(content)
        edges_list = self._extract_edges(content)

        # Create MRRG
        mrrg = self._create_mrrg(nodes_dict, edges_list, II, file_path.stem)

        print(f"✓ Parsed MRRG: II={mrrg.II}, {mrrg.num_nodes()} nodes, {mrrg.num_edges()} edges")
        return mrrg

    def _extract_ii(self, content: str) -> int:
        """
        Extract Initiation Interval from DOT content.

        Looks for comment like "// II = 4" or infers from max cycle number.
        """
        # Try to find II in comments
        ii_match = re.search(r'//\s*II\s*=\s*(\d+)', content)
        if ii_match:
            return int(ii_match.group(1))

        # Otherwise, infer from node IDs
        max_cycle = 0
        for line in content.split('\n'):
            # Look for cycle:name pattern
            cycle_match = re.search(r'"?(\d+):', line)
            if cycle_match:
                cycle = int(cycle_match.group(1))
                max_cycle = max(max_cycle, cycle)

        # II is max_cycle + 1
        return max_cycle + 1 if max_cycle > 0 else 1

    def _extract_nodes(self, content: str) -> Dict[str, Dict]:
        """Extract all node declarations from DOT content."""
        nodes = {}

        for line in content.split('\n'):
            line = line.strip()
            if '->' not in line and '[' in line:
                match = self.node_pattern.search(line)
                if match:
                    node_id = match.group(1)
                    attrs_str = match.group(2)

                    # Parse attributes
                    attrs = {}
                    for attr_match in self.attr_pattern.finditer(attrs_str):
                        key = attr_match.group(1)
                        value = attr_match.group(2).strip('"')
                        attrs[key] = value

                    nodes[node_id] = attrs

        return nodes

    def _extract_edges(self, content: str) -> List[Tuple[str, str, Dict]]:
        """Extract all edges from DOT content."""
        edges = []

        for line in content.split('\n'):
            line = line.strip()
            if '->' in line:
                match = self.edge_pattern.search(line)
                if match:
                    src = match.group(1)
                    dst = match.group(2)
                    attrs_str = match.group(3) if match.group(3) else ""

                    # Parse edge attributes
                    attrs = {}
                    for attr_match in self.attr_pattern.finditer(attrs_str):
                        key = attr_match.group(1)
                        value = attr_match.group(2).strip('"')
                        attrs[key] = value

                    edges.append((src, dst, attrs))

        return edges

    def _parse_cycle_from_id(self, node_id: str) -> Tuple[int, str]:
        """
        Parse cycle and base name from node ID.

        Args:
            node_id: Node ID in format "cycle:name" or just "name"

        Returns:
            (cycle, base_name) tuple
        """
        if ':' in node_id:
            parts = node_id.split(':', 1)
            cycle = int(parts[0])
            base_name = parts[1]
            return cycle, base_name
        else:
            # No cycle prefix, assume cycle 0
            return 0, node_id

    def _parse_node_type(self, attrs: Dict) -> NodeType:
        """
        Parse node type from 'type' attribute.

        Expected values: "ROUTING", "FUNCTION", "ROUTING_FUNCTION"
        Default: ROUTING
        """
        node_type_str = attrs.get('type', 'ROUTING')

        type_map = {
            'function': NodeType.FUNCTION,
            'routing': NodeType.ROUTING,
            'routing_function': NodeType.ROUTING_FUNCTION,
        }

        return type_map.get(node_type_str.lower(), NodeType.ROUTING)

    def _parse_hw_entity_type(self, attrs: Dict) -> HWEntityType:
        """
        Parse hardware entity type from 'et' attribute.

        Expected values: "WIRE", "REG", "MUX", "COMB"
        Default: HW_UNSPECIFIED
        """
        hw_type_str = attrs.get('et', 'unspecified')

        hw_map = {
            'wire': HWEntityType.HW_WIRE,
            'reg': HWEntityType.HW_REG,
            'mux': HWEntityType.HW_MUX,
            'comb': HWEntityType.HW_COMB,
        }

        return hw_map.get(hw_type_str.lower(), HWEntityType.HW_UNSPECIFIED)

    def _parse_operations(self, attrs: Dict) -> Set[OperationType]:
        """
        Parse supported operations from 'supportedOps' attribute.

        Expected format: semicolon-separated operations (e.g., "add;sub;mul;")
        """
        ops_str = attrs.get('supportedOps', '')
        if not ops_str:
            return set()

        operations = set()
        for op_name in ops_str.split(';'):
            op_name = op_name.strip()
            if not op_name:  # Skip empty strings (from trailing semicolons)
                continue
            try:
                # Convert to uppercase for enum name lookup
                operations.add(OperationType[op_name.upper()])
            except KeyError:
                # Unknown operation, skip
                pass

        return operations

    def _parse_operand_tags(self, attrs: Dict) -> Set[OperandTag]:
        """
        Parse operand tags from 'supportedOperands' attribute.

        Expected format: semicolon-separated tags (e.g., "binary_lhs;binary_rhs;")
        """
        tags_str = attrs.get('supportedOperands', '')
        if not tags_str:
            return set()

        tags = set()
        for tag_name in tags_str.split(';'):
            tag_name = tag_name.strip()
            if not tag_name:  # Skip empty strings (from trailing semicolons)
                continue
            try:
                # Convert to uppercase for enum name lookup
                tags.add(OperandTag[tag_name.upper()])
            except KeyError:
                # Unknown tag, skip
                pass

        return tags

    def _parse_coordinates(self, attrs: Dict, node_id: str = '') -> Optional[Tuple[int, int]]:
        """
        Parse coordinates from attributes or node name.

        Args:
            attrs: Dictionary of node attributes
            node_id: Node ID to extract coordinates from if attributes are missing

        Returns:
            (x, y) tuple if coordinates found, None otherwise
        """
        # Try attributes first - check multiple naming conventions
        x_str = attrs.get('x_pos', attrs.get('x', attrs.get('col', '')))
        y_str = attrs.get('y_pos', attrs.get('y', attrs.get('row', '')))

        if x_str and y_str:
            try:
                return (int(x_str), int(y_str))
            except ValueError:
                pass

        # Fallback: try to extract from node name
        if node_id:
            coords = self._parse_coordinates_from_name(node_id, add_edge_offset=True)
            if coords:
                return coords

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

    def _create_mrrg(
        self,
        nodes_dict: Dict,
        edges_list: List,
        II: int,
        name: str
    ) -> MRRG:
        """Create MRRG from parsed node and edge information."""
        # Determine array dimensions
        max_x, max_y = 0, 0
        for node_id, attrs in nodes_dict.items():
            coords = self._parse_coordinates(attrs, node_id)
            if coords:
                max_x = max(max_x, coords[0])
                max_y = max(max_y, coords[1])

        rows = max_y + 1
        cols = max_x + 1

        mrrg = MRRG(name=name, II=II, rows=rows, cols=cols)

        # Create MRRG nodes
        node_map = {}
        for node_id, attrs in nodes_dict.items():
            cycle, base_name = self._parse_cycle_from_id(node_id)
            node_type = self._parse_node_type(attrs)
            hw_entity_type = self._parse_hw_entity_type(attrs)
            coordinates = self._parse_coordinates(attrs, node_id)

            # Parse common attributes
            latency = int(attrs.get('latency', 0))
            bitwidth = int(attrs.get('bitwidth', 32))
            capacity = int(attrs.get('capacity', 1))

            # Parse FU-specific attributes
            operations = self._parse_operations(attrs)
            operand_tags = self._parse_operand_tags(attrs)

            # Parse routing-specific attributes
            routing_type = attrs.get('routing_type', 'wire')

            # Parse register-specific attributes
            bank_id_str = attrs.get('bank_id', attrs.get('bank', None))
            bank_id = int(bank_id_str) if bank_id_str is not None else None

            # Create unified MRRGNode with all attributes
            mrrg_node = MRRGNode(
                node_id=node_id,
                node_type=node_type,
                cycle=cycle,
                coordinates=coordinates or (0, 0),
                hw_entity_type=hw_entity_type,
                latency=latency,
                bitwidth=bitwidth,
                capacity=capacity,
                supported_operations=operations,
                supported_operand_tags=operand_tags,
                routing_type=routing_type,
                bank_id=bank_id
            )

            mrrg.add_node(mrrg_node)
            node_map[node_id] = mrrg_node

        # Create edges
        edge_counter = 0
        for src_id, dst_id, attrs in edges_list:
            if src_id in node_map and dst_id in node_map:
                src_node = node_map[src_id]
                dst_node = node_map[dst_id]

                latency = int(attrs.get('latency', 0))
                capacity = int(attrs.get('capacity', 1))
                conflict_free = attrs.get('conflict_free', 'true').lower() == 'true'

                edge = MRRGEdge(
                    f"edge_{edge_counter}",
                    src_node,
                    dst_node,
                    latency=latency,
                    capacity=capacity,
                    conflict_free=conflict_free
                )

                mrrg.add_edge(edge)
                edge_counter += 1

        return mrrg
