"""DOT parser for latency specification files."""

import re
from pathlib import Path
from typing import Dict, Tuple

from cgrasolve.graph.dfg import OperationType
from cgrasolve.schedules.latency_spec import LatencySpecification, OperationLatencyEdge


class LatencyDotParser:
    """Parser for latency specification DOT files.

    Parses latency specifications in DOT format, extracting:
    - Operation latencies (OP_LATENCY attribute on nodes)
    - Network latencies (LOWER_BOUND_NETWORK_LATENCY, UPPER_BOUND_NETWORK_LATENCY on edges)
    """

    def __init__(self):
        """Initialize the latency DOT parser."""
        # Pattern to match node declarations with OP_LATENCY
        # Matches: node_id[OP_LATENCY = X] or "node_id"[OP_LATENCY = X]
        self.node_pattern = re.compile(r'"?([^"\s\[]+)"?\s*\[([^\]]+)\]')

        # Pattern to match edges with network latency
        # Matches: "src"->"sink" [LOWER_BOUND_NETWORK_LATENCY = X, UPPER_BOUND_NETWORK_LATENCY = Y]
        self.edge_pattern = re.compile(r'"?([^"\s\-]+)"?\s*->\s*"?([^"\s\[]+)"?\s*\[([^\]]+)\]')

        # Pattern to match attributes
        self.attr_pattern = re.compile(r'(\w+)\s*=\s*(\d+)')

    def parse(self, file_path: Path) -> LatencySpecification:
        """Parse a latency specification from a DOT file.

        Args:
            file_path: Path to the DOT file

        Returns:
            LatencySpecification object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        print(f"Parsing latency specification from {file_path}...")

        # Read file
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract operation latencies and network latencies
        op_latencies = self._extract_op_latencies(content)
        network_latencies = self._extract_network_latencies(content)

        # Create latency specification
        spec = LatencySpecification(
            op_latencies=op_latencies,
            network_latencies=network_latencies
        )

        print(f"✓ Parsed latency spec: {spec.num_operations()} operations, {spec.num_edges()} edges")
        return spec

    def _extract_op_latencies(self, content: str) -> Dict[OperationType, int]:
        """Extract operation latencies from DOT content.

        Args:
            content: DOT file content

        Returns:
            Dictionary mapping operation types to latencies
        """
        op_latencies = {}

        for line in content.split('\n'):
            line = line.strip()
            # Skip lines with edges
            if '->' in line:
                continue

            # Look for node declarations with OP_LATENCY
            if '[' in line and 'OP_LATENCY' in line:
                match = self.node_pattern.search(line)
                if match:
                    node_id = match.group(1)
                    attrs_str = match.group(2)

                    # Extract OP_LATENCY attribute
                    latency_match = re.search(r'OP_LATENCY\s*=\s*(\d+)', attrs_str)
                    if latency_match:
                        latency = int(latency_match.group(1))

                        # Map node name to OperationType
                        try:
                            op_type = self._map_to_operation_type(node_id)
                            op_latencies[op_type] = latency
                        except ValueError as e:
                            print(f"  Warning: {e}")

        return op_latencies

    def _extract_network_latencies(self, content: str) -> Dict[OperationLatencyEdge, Tuple[int, int]]:
        """Extract network latencies from DOT content.

        Args:
            content: DOT file content

        Returns:
            Dictionary mapping operation edges to (lower_bound, upper_bound) latency tuples
        """
        network_latencies = {}

        for line in content.split('\n'):
            line = line.strip()

            # Look for edge declarations with network latency
            if '->' in line and 'NETWORK_LATENCY' in line:
                match = self.edge_pattern.search(line)
                if match:
                    src_name = match.group(1)
                    sink_name = match.group(2)
                    attrs_str = match.group(3)

                    # Extract latency bounds
                    lower_match = re.search(r'LOWER_BOUND_NETWORK_LATENCY\s*=\s*(\d+)', attrs_str)
                    upper_match = re.search(r'UPPER_BOUND_NETWORK_LATENCY\s*=\s*(\d+)', attrs_str)

                    if lower_match and upper_match:
                        lower_bound = int(lower_match.group(1))
                        upper_bound = int(upper_match.group(1))

                        # Map node names to OperationTypes
                        try:
                            src_type = self._map_to_operation_type(src_name)
                            sink_type = self._map_to_operation_type(sink_name)

                            edge = OperationLatencyEdge(src_type, sink_type)
                            network_latencies[edge] = (lower_bound, upper_bound)
                        except ValueError as e:
                            print(f"  Warning: {e}")

        return network_latencies

    def _map_to_operation_type(self, op_name: str) -> OperationType:
        """Map operation name string to OperationType enum.

        Args:
            op_name: Operation name from DOT file

        Returns:
            OperationType enum value

        Raises:
            ValueError: If operation name is not recognized
        """
        # Normalize the operation name (lowercase, strip quotes)
        normalized = op_name.strip('"').lower()

        # Try to find matching OperationType
        for op_type in OperationType:
            if op_type.value == normalized:
                return op_type

        # If not found, raise error
        raise ValueError(f"Unknown operation type: {op_name}")
