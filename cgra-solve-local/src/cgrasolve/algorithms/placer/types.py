from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MRRGNodePlacementState:
    """Node attributes for heuristic mapping."""

    # Number of DFG nodes placed on the MRRG node
    occupancy: int = 0   

    # Shwet TODO: Add more properties for exploration of other mappers.