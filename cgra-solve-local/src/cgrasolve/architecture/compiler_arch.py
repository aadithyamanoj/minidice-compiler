"""Shared compiler-arch.json loading and validation.

This module provides the single source of truth for compiler_arch.json
schema validation, used by both ArchitectureLoader and FasmGenerator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REQUIRED_KEYS = [
    "schema_version",
    "fabric_contexts",
    "features",
    "layout_hash",
    "routing_programming",
]


def load_and_validate_compiler_arch(path: str) -> Dict[str, Any]:
    """Load and validate a compiler_arch.json file.

    Args:
        path: Path to the compiler_arch.json file.

    Returns:
        Parsed and validated dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or the schema is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"compiler_arch.json not found: {path}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in compiler_arch.json: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("compiler_arch.json root must be an object")

    for key in REQUIRED_KEYS:
        if key not in payload:
            raise ValueError(f"compiler_arch.json missing required key '{key}'")

    if payload["fabric_contexts"] < 1:
        raise ValueError("fabric_contexts must be >= 1")

    return payload
