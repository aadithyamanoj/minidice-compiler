"""Latency specification data structures for scheduling."""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple, Optional
from cgrasolve.graph.dfg import OperationType


class OperationLatencyEdge(NamedTuple):
    """Edge representing network latency between two operation types.

    Attributes:
        src: Source operation type
        sink: Sink (destination) operation type
    """
    src: OperationType
    sink: OperationType


_PREDICATE_DATA_EQUIVALENT: Dict[OperationType, OperationType] = {
    OperationType.INPUT_PRED: OperationType.INPUT,
    OperationType.OUTPUT_PRED: OperationType.OUTPUT,
}


def _data_equiv(op: OperationType) -> OperationType:
    return _PREDICATE_DATA_EQUIVALENT.get(op, op)


@dataclass
class LatencySpecification:
    """Specification of operation and network latencies for scheduling.

    This class encapsulates latency information needed for scheduling:
    - Operation latencies: execution time for each operation type
    - Network latencies: communication delay between operation types

    Predicate IO ops (INPUT_PRED / OUTPUT_PRED) fall back to the corresponding
    data IO entries (INPUT / OUTPUT) when no explicit predicate entry is present.
    This matches architectures where predicate signals share the data routing
    fabric. Architectures with a dedicated predicate NoC can override by adding
    explicit predicate entries to the spec.

    Attributes:
        op_latencies: Map from operation type to execution latency
        network_latencies: Map from (src, sink) operation edge to (lower_bound, upper_bound) latency tuple
    """

    op_latencies: Dict[OperationType, int]
    network_latencies: Dict[OperationLatencyEdge, Tuple[int, int]]

    def get_op_latency(self, op_type: OperationType) -> int:
        """Get the execution latency for an operation type.

        Args:
            op_type: The operation type

        Returns:
            Execution latency in cycles

        Raises:
            ValueError: If operation type not found in specification
        """
        if op_type in self.op_latencies:
            return self.op_latencies[op_type]
        equiv = _data_equiv(op_type)
        if equiv is not op_type and equiv in self.op_latencies:
            return self.op_latencies[equiv]
        raise ValueError(f"No operation latency found for {op_type}")

    def get_network_latency(self, src: OperationType, sink: OperationType) -> Tuple[int, int]:
        """Get the network latency between two operation types.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            Tuple of (lower_bound, upper_bound) network latency in cycles

        Raises:
            ValueError: If edge not found in specification
        """

        # Check if the edge is in the specification, with predicate→data fallback.
        edge = OperationLatencyEdge(src, sink)
        if edge in self.network_latencies:
            return self.network_latencies[edge]
        fallback = OperationLatencyEdge(_data_equiv(src), _data_equiv(sink))
        if fallback != edge and fallback in self.network_latencies:
            return self.network_latencies[fallback]
        raise ValueError(f"No network latency found for {src} to {sink}")

    def get_network_latency_lower(self, src: OperationType, sink: OperationType) -> int:
        """Get the lower bound network latency between two operation types.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            Lower bound network latency in cycles

        Raises:
            ValueError: If edge not found in specification
        """
        return self.get_network_latency(src, sink)[0]

    def get_network_latency_upper(self, src: OperationType, sink: OperationType) -> int:
        """Get the upper bound network latency between two operation types.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            Upper bound network latency in cycles

        Raises:
            ValueError: If edge not found in specification
        """
        return self.get_network_latency(src, sink)[1]

    def has_op_latency(self, op_type: OperationType) -> bool:
        """Check if operation type has a defined latency.

        Args:
            op_type: The operation type

        Returns:
            True if operation latency is defined, False otherwise
        """
        return op_type in self.op_latencies

    def has_network_latency(self, src: OperationType, sink: OperationType) -> bool:
        """Check if network latency is defined for an edge.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            True if network latency is defined, False otherwise
        """
        edge = OperationLatencyEdge(src, sink)
        if edge in self.network_latencies:
            return True
        fallback = OperationLatencyEdge(_data_equiv(src), _data_equiv(sink))
        return fallback != edge and fallback in self.network_latencies

    def num_operations(self) -> int:
        """Get the number of operations with defined latencies.

        Returns:
            Number of operations
        """
        return len(self.op_latencies)

    def num_edges(self) -> int:
        """Get the number of edges with defined network latencies.

        Returns:
            Number of edges
        """
        return len(self.network_latencies)
