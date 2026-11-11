"""DOT parser for CGRA-ME DFG files."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from cgrasolve.graph.dfg import DFG, DFGNode, DFGEdge, OperationType


class DFGDotParser:
    """
    Parser for CGRA-ME DFG DOT files.

    Parses data flow graphs in DOT format, extracting:
    - Operations (nodes with opcodes)
    - Data dependencies (edges with operand labels)
    - Loop-back edges (phi nodes indicate loops)
    - Constants and inputs/outputs
    """

    def __init__(self):
        """Initialize the DOT parser."""
        # Pattern to match node declarations (handles quoted and unquoted IDs)
        # Matches: node_id [...] or "node.id" [...]
        self.node_pattern = re.compile(r'"?([^"\s\[]+)"?\s*\[([^\]]+)\]')
        # Pattern to match edges (handles quoted and unquoted IDs)
        # Matches: src -> dst [...] or "src.id" -> "dst.id" [...]
        self.edge_pattern = re.compile(r'"?([^"\s\-]+)"?\s*->\s*"?([^"\s\[]+)"?(?:\s*\[([^\]]+)\])?')
        self.attr_pattern = re.compile(r'(\w+)="?([^",\]]*)"?')

    def parse(self, file_path: Path) -> DFG:
        """
        Parse a DFG from a DOT file.

        Args:
            file_path: Path to the DOT file

        Returns:
            DFG object
        """
        print(f"Parsing DFG from {file_path}...")

        # Read file
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract nodes and edges
        nodes_dict = self._extract_nodes(content)
        edges_list = self._extract_edges(content)

        # Detect loop-back edges (edges to phi nodes from within the loop)
        loop_edges = self._detect_loop_edges(nodes_dict, edges_list)

        # Create DFG
        dfg = self._create_dfg(nodes_dict, edges_list, loop_edges, file_path.stem)

        print(f"✓ Parsed DFG: {dfg.num_nodes()} nodes, {dfg.num_edges()} edges")
        return dfg

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

    def _detect_loop_edges(self, nodes_dict: Dict, edges_list: List) -> set:
        """
        Detect loop-back edges.

        Loop-back edges are edges that go to phi nodes from later operations,
        indicating loop-carried dependencies. This method uses two detection strategies:
        1. Heuristic: Edges to PHI nodes with RHS operand
        2. Explicit: Edges marked with is_loop_back=true attribute

        If both are present, it verifies they match and warns about mismatches.
        """
        # Strategy 1: Heuristic detection (PHI nodes with RHS operand)
        heuristic_loop_edges = set()
        phi_nodes = {nid for nid, attrs in nodes_dict.items()
                     if attrs.get('opcode') == 'phi'}

        for src, dst, attrs in edges_list:
            # Edges TO phi nodes with RHS operand (loop-carried input)
            if dst in phi_nodes and attrs.get('operand') == 'RHS':
                heuristic_loop_edges.add((src, dst))

        # Strategy 2: Explicit marking (is_loop_back attribute)
        explicit_loop_edges = set()
        for src, dst, attrs in edges_list:
            is_loop_back = attrs.get('is_loop_back', '').lower() in ('true', '1', 'yes')
            if is_loop_back:
                explicit_loop_edges.add((src, dst))

        # Verify consistency if explicit markings exist
        if explicit_loop_edges:
            # Check for mismatches
            only_heuristic = heuristic_loop_edges - explicit_loop_edges
            only_explicit = explicit_loop_edges - heuristic_loop_edges

            if only_heuristic or only_explicit:
                print("Warning: Loop-back edge detection mismatch!")
                if only_heuristic:
                    print(f"  Detected by heuristic but not marked explicitly: {only_heuristic}")
                if only_explicit:
                    print(f"  Marked explicitly but not detected by heuristic: {only_explicit}")
                print("  Using union of both detection methods.")
            else:
                print(f"✓ Loop-back edge detection verified: {len(explicit_loop_edges)} edges match")

            # Use union of both methods to be safe
            return heuristic_loop_edges | explicit_loop_edges
        else:
            # No explicit markings, use heuristic only
            if heuristic_loop_edges:
                print(f"  Using heuristic loop-back detection: {len(heuristic_loop_edges)} edges")
            return heuristic_loop_edges

    def _parse_bool(self, value: str) -> bool:
        """Parse boolean value from string."""
        if isinstance(value, bool):
            return value
        return value.lower() in ('true', '1', 'yes')

    def _map_opcode_to_operation(self, opcode: str) -> OperationType:
        """Map DOT opcode to FastMap OperationType."""
        opcode_map = {
            # Special
            'nop': OperationType.NOP,

            # Type conversions
            'sext': OperationType.SEXT,
            'zext': OperationType.ZEXT,
            'trunc': OperationType.TRUNC,
            'fp2int': OperationType.FP2INT,
            'int2fp': OperationType.INT2FP,

            # Input/Output
            'input': OperationType.INPUT,
            'input_pred': OperationType.INPUT_PRED,
            'output': OperationType.OUTPUT,
            'output_pred': OperationType.OUTPUT_PRED,
            'phi': OperationType.PHI,
            'const': OperationType.CONST,

            # Integer arithmetic
            'add': OperationType.ADD,
            'sub': OperationType.SUB,
            'mul': OperationType.MUL,
            'div': OperationType.DIV,

            # Logical operations
            'and': OperationType.AND,
            'or': OperationType.OR,
            'xor': OperationType.XOR,

            # Shift operations
            'shl': OperationType.SHL,
            'ashr': OperationType.ASHR,
            'lshr': OperationType.LSHR,

            # Memory operations
            'load': OperationType.LOAD,
            'store': OperationType.STORE,
            'gep': OperationType.GEP,

            # Comparison operations
            'icmp': OperationType.ICMP,
            'cmp': OperationType.CMP,

            # Control flow
            'br': OperationType.BR,
            'select': OperationType.SELECT,

            # Floating point operations
            'sqrt': OperationType.SQRT,
            'fadd': OperationType.FADD,
            'fmul': OperationType.FMUL,
            'fdiv': OperationType.FDIV,
            'fma': OperationType.FMA,
            'fsub': OperationType.FSUB,
            'gep_imm': OperationType.GEP_IMM,
            'add_imm': OperationType.ADD_IMM,

            # Unsigned multiply variants
            'mulu_full_lo': OperationType.MULU_FULL_LO,
            'mulu_half_lo': OperationType.MULU_HALF_LO,
            'mulu_quart_lo': OperationType.MULU_QUART_LO,
            'mulu_full_hi': OperationType.MULU_FULL_HI,
            'mulu_half_hi': OperationType.MULU_HALF_HI,
            'mulu_quart_hi': OperationType.MULU_QUART_HI,

            # Signed multiply variants
            'muls_full_lo': OperationType.MULS_FULL_LO,
            'muls_half_lo': OperationType.MULS_HALF_LO,
            'muls_quart_lo': OperationType.MULS_QUART_LO,
            'muls_full_hi': OperationType.MULS_FULL_HI,
            'muls_half_hi': OperationType.MULS_HALF_HI,
            'muls_quart_hi': OperationType.MULS_QUART_HI,

            # Add variants
            'add_full': OperationType.ADD_FULL,
            'add_half': OperationType.ADD_HALF,
            'add_quart': OperationType.ADD_QUART,
        }

        return opcode_map.get(opcode.lower(), OperationType.NOP)

    def _create_dfg(
        self,
        nodes_dict: Dict,
        edges_list: List,
        loop_edges: set,
        name: str
    ) -> DFG:
        """Create DFG from parsed node and edge information."""
        dfg = DFG(name=name)

        # Create DFG nodes
        node_map = {}
        for node_id, attrs in nodes_dict.items():
            opcode = attrs.get('opcode', 'nop')
            operation = self._map_opcode_to_operation(opcode)

            # Create node
            dfg_node = DFGNode(
                node_id,
                operation
            )

            # Store const value if present
            if 'constVal' in attrs:
                dfg_node.set_attribute('const_value', attrs['constVal'])

            # Store bitwidth if present
            if 'bitwidth' in attrs:
                try:
                    dfg_node.set_attribute('bitwidth', int(attrs['bitwidth']))
                except ValueError:
                    pass  # Skip if not a valid integer

            # Store memory name if present (for load/store operations)
            if 'memName' in attrs:
                dfg_node.set_attribute('mem_name', attrs['memName'])

            # Add to DFG
            if operation == OperationType.INPUT:
                dfg.add_input_node(dfg_node)
            elif operation == OperationType.OUTPUT:
                dfg.add_output_node(dfg_node)
            else:
                dfg.add_node(dfg_node)

            node_map[node_id] = dfg_node

        # Create edges
        edge_counter = 0
        for src_id, dst_id, attrs in edges_list:
            if src_id in node_map and dst_id in node_map:
                src_node = node_map[src_id]
                dst_node = node_map[dst_id]

                # Check if this is a loop-back edge
                is_loop_back = (src_id, dst_id) in loop_edges

                # DFG edges have no inherent latency (logical graph)
                # Latency comes from target hardware (MRRG) during mapping
                edge = DFGEdge(
                    f"edge_{edge_counter}",
                    src_node,
                    dst_node,
                    latency=0,  # Default to 0 for DFG edges
                    is_loop_back=is_loop_back
                )

                # Store operand info
                if 'operand' in attrs:
                    edge.set_attribute('operand', attrs['operand'])

                # Store dist attribute (iteration distance, separate from latency)
                if 'dist' in attrs:
                    try:
                        edge.set_attribute('dist', int(attrs['dist']))
                    except ValueError:
                        pass  # Skip if not a valid integer

                # Store bitwidth if present
                if 'bitwidth' in attrs:
                    try:
                        edge.set_attribute('bitwidth', int(attrs['bitwidth']))
                    except ValueError:
                        pass  # Skip if not a valid integer

                # Store kind if present (e.g., "dataflow")
                if 'kind' in attrs:
                    edge.set_attribute('kind', attrs['kind'])

                # Store predicate flag if present
                if 'predicate' in attrs:
                    edge.set_attribute('predicate', self._parse_bool(attrs['predicate']))

                dfg.add_edge(edge)
                edge_counter += 1

        return dfg
