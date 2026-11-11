"""DOT parser for CGRA-ME MRRG files."""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from cgrasolve.graph.mrrg import MRRG, MRRGNode, MRRGEdge, NodeType, HWEntityType
from cgrasolve.graph.dfg import OperationType


class MRRGDotParser:
    """
    Parser for CGRA-ME MRRG DOT files.

    Parses modulo routing resource graphs in DOT format, extracting:
    - Functional units (PE nodes)
    - Routing resources (multiplexers, crossbars)
    - Registers
    - Interconnect topology
    - Timing information (latencies from CGRAME_latency attribute)
    """

    def __init__(self):
        """Initialize the DOT parser."""
        self.node_pattern = re.compile(r'"([^"]+)"\s*(?:\[([^\]]+)\])?')
        self.edge_pattern = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"')
        self.attr_pattern = re.compile(r'(\w+)=([^,\]]+)')

    def parse(self, file_path: Path) -> MRRG:
        """
        Parse an MRRG from a DOT file.

        Args:
            file_path: Path to the DOT file

        Returns:
            MRRG object
        """
        print(f"Parsing MRRG from {file_path}...")

        # Read file
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract nodes and edges
        nodes_dict = self._extract_nodes(content)
        edges_list = self._extract_edges(content)

        # Analyze node types and structure
        node_info = self._analyze_nodes(nodes_dict, edges_list)

        # Create MRRG
        mrrg = self._create_mrrg(node_info, edges_list, file_path.stem)

        print(f"✓ Parsed MRRG: {mrrg.num_nodes()} nodes, {mrrg.num_edges()} edges")
        return mrrg

    def _extract_nodes(self, content: str) -> Dict[str, Dict]:
        """Extract all node declarations from DOT content."""
        nodes = {}

        for line in content.split('\n'):
            line = line.strip()
            if '->' not in line and line and not line.startswith('digraph'):
                match = self.node_pattern.search(line)
                if match:
                    node_id = match.group(1)
                    attrs_str = match.group(2) if match.group(2) else ""

                    # Parse attributes
                    attrs = {}
                    for attr_match in self.attr_pattern.finditer(attrs_str):
                        key = attr_match.group(1)
                        value = attr_match.group(2).strip('"')
                        attrs[key] = value

                    nodes[node_id] = attrs

        return nodes

    def _extract_edges(self, content: str) -> List[Tuple[str, str]]:
        """Extract all edges from DOT content."""
        edges = []

        for line in content.split('\n'):
            line = line.strip()
            if '->' in line:
                match = self.edge_pattern.search(line)
                if match:
                    src = match.group(1)
                    dst = match.group(2)
                    edges.append((src, dst))

        return edges

    def _analyze_nodes(self, nodes_dict: Dict, edges_list: List) -> Dict:
        """
        Analyze nodes to determine their types and properties.

        CGRA-ME node naming conventions:
        - Nodes with timestamps: "0:pe_c0_r0.alu.out"
        - Format: "timestamp:component.subcomponent.port"
        - PEs: contain "pe_" prefix
        - Registers: contain "reg" in name
        - IO: contain "io_" prefix
        """
        node_info = {}

        for node_id, attrs in nodes_dict.items():
            info = {
                'id': node_id,
                'attrs': attrs,
                'type': None,
                'coordinates': None,
                'supported_ops': set(),
                'latency': int(attrs.get('CGRAME_latency', 0))
            }

            # Extract timestamp and component
            parts = node_id.split(':')
            if len(parts) == 2:
                timestamp, component = parts
                info['timestamp'] = int(timestamp)
                info['component'] = component
            else:
                info['timestamp'] = 0
                info['component'] = node_id

            # Extract coordinates from DOT attributes if available
            if 'x_pos' in attrs and 'y_pos' in attrs:
                info['coordinates'] = (int(attrs['x_pos']), int(attrs['y_pos']))

            # Determine node type based on naming
            component = info['component'].lower()

            if 'pe_c' in component:
                # Processing element - extract coordinates
                info['type'] = NodeType.FUNCTION  # CGRA-ME: MRRG_NODE_FUNCTION
                # Fall back to parsing from name if not in attributes
                if info['coordinates'] is None:
                    info['coordinates'] = self._extract_pe_coordinates(component)
                # Assume PEs support basic operations
                info['supported_ops'] = {
                    OperationType.ADD, OperationType.SUB,
                    OperationType.MUL, OperationType.AND,
                    OperationType.OR, OperationType.XOR
                }
            elif 'reg' in component:
                # Registers are ROUTING nodes in CGRA-ME (distinguished by HWEntityType.HW_REG)
                info['type'] = NodeType.ROUTING
                # Fall back to parsing from name if not in attributes
                if info['coordinates'] is None:
                    info['coordinates'] = self._extract_coordinates(component)
            elif 'io_' in component:
                # I/O ports are ROUTING nodes in CGRA-ME (distinguished by HWEntityType)
                info['type'] = NodeType.ROUTING
                # Fall back to parsing from name if not in attributes
                if info['coordinates'] is None:
                    info['coordinates'] = self._extract_coordinates(component)
            else:
                # Default to routing node
                info['type'] = NodeType.ROUTING
                # Fall back to parsing from name if not in attributes
                if info['coordinates'] is None:
                    info['coordinates'] = self._extract_coordinates(component)

            node_info[node_id] = info

        return node_info

    def _extract_pe_coordinates(self, component: str) -> Optional[Tuple[int, int]]:
        """Extract (col, row) coordinates from PE names like 'pe_c2_r3'."""
        match = re.search(r'pe_c(\d+)_r(\d+)', component)
        if match:
            col = int(match.group(1))
            row = int(match.group(2))
            return (col, row)
        return None

    def _extract_coordinates(self, component: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates from various component naming schemes."""
        # Try PE format first
        coords = self._extract_pe_coordinates(component)
        if coords:
            return coords

        # Try general numeric patterns
        match = re.search(r'_(\d+)_(\d+)', component)
        if match:
            return (int(match.group(1)), int(match.group(2)))

        # Try single numeric suffix
        match = re.search(r'_(\d+)$', component)
        if match:
            idx = int(match.group(1))
            return (idx, 0)

        return None

    def _create_mrrg(self, node_info: Dict, edges_list: List, name: str) -> MRRG:
        """Create MRRG from parsed node and edge information."""
        # Determine array dimensions
        max_row = 0
        max_col = 0
        max_timestamp = 0
        for info in node_info.values():
            if info['coordinates']:
                col, row = info['coordinates']
                max_col = max(max_col, col)
                max_row = max(max_row, row)
            max_timestamp = max(max_timestamp, info.get('timestamp', 0))

        # II is max_timestamp + 1 (if time-expanded)
        II = max_timestamp + 1 if max_timestamp > 0 else 1

        mrrg = MRRG(name=name, II=II, rows=max_row + 1, cols=max_col + 1)

        # Group nodes by timestamp = 0 (we'll use just the initial configuration)
        # CGRA-ME creates time-expanded graphs, but for FastMap we want the base architecture
        base_nodes = {nid: info for nid, info in node_info.items()
                     if info['timestamp'] == 0}

        # Create MRRG nodes
        node_map = {}
        for node_id, info in base_nodes.items():
            mrrg_node = self._create_mrrg_node(info)
            if mrrg_node:
                mrrg.add_node(mrrg_node)
                node_map[node_id] = mrrg_node

        # Create edges (only between timestamp 0 nodes)
        for src_id, dst_id in edges_list:
            if src_id in node_map and dst_id in node_map:
                src_node = node_map[src_id]
                dst_node = node_map[dst_id]

                # Get latency from destination node if it's a register
                latency = 0
                if dst_id in node_info:
                    latency = node_info[dst_id]['latency']

                edge = MRRGEdge(
                    f"edge_{src_id}_to_{dst_id}",
                    src_node,
                    dst_node,
                    latency=latency
                )
                mrrg.add_edge(edge)

        return mrrg

    def _create_mrrg_node(self, info: Dict) -> Optional[MRRGNode]:
        """Create appropriate MRRG node based on node info."""
        node_id = info['id']
        node_type = info['type']
        coords = info['coordinates'] or (0, 0)
        timestamp = info.get('timestamp', 0)

        if node_type == NodeType.FUNCTION:
            return MRRGNode(
                node_id=node_id,
                node_type=NodeType.FUNCTION,
                cycle=timestamp,
                coordinates=coords,
                hw_entity_type=HWEntityType.HW_COMB,
                latency=1,
                supported_operations=info['supported_ops'],
            )
        elif 'reg' in info['component'].lower():
            return MRRGNode(
                node_id=node_id,
                node_type=NodeType.ROUTING,
                cycle=timestamp,
                coordinates=coords,
                hw_entity_type=HWEntityType.HW_REG,
                latency=1,
            )
        elif node_type == NodeType.ROUTING:
            return MRRGNode(
                node_id=node_id,
                node_type=NodeType.ROUTING,
                cycle=timestamp,
                coordinates=coords,
                hw_entity_type=HWEntityType.HW_WIRE,
                routing_type="mux",
            )

        return None
