"""Architecture loading and representation for CGRA-ME integration."""

from cgrasolve.architecture.architecture_loader import ArchitectureBundle, ArchitectureLoader
from cgrasolve.architecture.fasm_generator import FasmGenerator

__all__ = [
    "ArchitectureBundle",
    "ArchitectureLoader",
    "FasmGenerator",
]
