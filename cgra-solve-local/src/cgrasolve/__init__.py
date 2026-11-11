"""CGRA-Solve: A modular CGRA mapping framework."""

__version__ = "0.1.0"
__all__ = []

# Import and export common graph types
from cgrasolve.graph.dfg import DFG, DFGNode, DFGEdge, OperationType
from cgrasolve.graph.hyperdfg import HyperDFG, HyperNode, HyperVal
from cgrasolve.graph.mrrg import MRRG, MRRGNode, MRRGEdge, NodeType, HWEntityType

__all__.extend([
    "DFG", "DFGNode", "DFGEdge", "OperationType",
    "HyperDFG", "HyperNode", "HyperVal",
    "MRRG", "MRRGNode", "MRRGEdge", "NodeType", "HWEntityType",
])

# Import and export mapper classes
from cgrasolve.algorithms.mapper_base import MappingResult, Placement, Routing

__all__.extend([
    "MappingResult", "Placement", "Routing",
])

# Import and export architecture and workload loaders
from cgrasolve.architecture import ArchitectureLoader
from cgrasolve.workload import DFGLoader

__all__.extend([
    "ArchitectureLoader",
    "DFGLoader",
])

# Import and export schedule-related classes
from cgrasolve.schedules import LatencySpecification, OperationLatencyEdge, LatencyDotParser

__all__.extend([
    "LatencySpecification",
    "OperationLatencyEdge",
    "LatencyDotParser",
])
