"""FASM generator for Dora bitstream integration.

Translates CGRA-Solve mapping results (placement + routing) into FASM files
that Dora can ingest for bitstream generation.  Requires compiler_arch.json
exported by Dora, which provides feature definitions, routing-edge
programming tables, and operation encoding hints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from cgrasolve.architecture.compiler_arch import load_and_validate_compiler_arch
from cgrasolve.architecture.parsers.dora_json_parser import DoraJsonParser
from cgrasolve.graph.dfg import DFG, DFGNode, OperationType
from cgrasolve.graph.hyperdfg import HyperDFG


# Invert DoraJsonParser.OPTYPE_MAP so we can go OperationType -> Dora string.
_OP_TO_DORA: Dict[OperationType, str] = {
    v: k for k, v in DoraJsonParser.OPTYPE_MAP.items()
}


# Workaround for missing encoding_hint values in dora's compiler_arch.json
# export for no-pred-NoC mini_dice (the bindings list optypes but every
# encoding_hint is null). Recovered from the SV control modules:
#   fu_int16_add_int16_sub_int16_mul_control.sv
# When the compiler_arch's encoding_hint is present we use it; otherwise fall
# back to this table. Key: (module_name, dora_optype) -> (cw_value, cw_width).
_FU_OPCODE_FALLBACK: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("fu_int16_add_int16_sub_int16_mul", "ADD"): (0, 2),  # OPCODE_ADD = 2'b00
    ("fu_int16_add_int16_sub_int16_mul", "SUB"): (1, 2),  # OPCODE_SUB = 2'b01
    ("fu_int16_add_int16_sub_int16_mul", "MUL"): (2, 2),  # OPCODE_MUL = 2'b10
}


# Static map of mini_dice's flat instance crossbars: feature_name -> sel_width
# per output channel. The full control word is `output_idx * sel_width` bits
# wide; for each output, bits [output_idx*sel_width +: sel_width] pick which
# input drives that output. Sourced from the RTL (xbar_*.sv localparams) and
# verified against compiler_arch.features widths:
#   xbar_data_in_inst   width=50  (10 outputs x 5-bit selects)
#   xbar_data_out_inst  width=72  (18 outputs x 4-bit selects)  -- inferred
#   xbar_mem_addr_inst  width=16  ( 4 outputs x 4-bit selects)
#   xbar_mem_data_inst  width=16  ( 4 outputs x 4-bit selects)
_FLAT_XBAR_SEL_WIDTH: Dict[str, int] = {
    "xbar_data_in_inst":  5,
    "xbar_data_out_inst": 4,
    "xbar_mem_addr_inst": 4,
    "xbar_mem_data_inst": 4,
}


@dataclass(frozen=True)
class _PackedFeatureDescriptor:
    """Dynamic pack/unpack feature encoding attached to one MRRG node."""

    feature_name: str
    bypass_bit: Optional[int] = None
    enable_bit: Optional[int] = None
    lane_lsb: Optional[int] = None
    lane_width_bits: int = 0
    sign_extend_bit: Optional[int] = None
    sign_extend_value: int = 1

    def encode(self, *, lane_index: int = 0, active: bool) -> int:
        """Encode one feature value for the requested lane state."""
        value = 0

        if self.bypass_bit is not None and not active:
            value |= 1 << self.bypass_bit
        if self.enable_bit is not None and active:
            value |= 1 << self.enable_bit

        if active and self.lane_lsb is not None and self.lane_width_bits > 0:
            max_lane = 1 << self.lane_width_bits
            if lane_index < 0 or lane_index >= max_lane:
                raise ValueError(
                    f"lane_index={lane_index} does not fit in {self.lane_width_bits} bits"
                )
            value |= lane_index << self.lane_lsb

        if active and self.sign_extend_bit is not None and self.sign_extend_value:
            value |= 1 << self.sign_extend_bit

        return value


@dataclass(frozen=True)
class _PackedNodeDescriptor:
    """Pack/unpack metadata for one MRRG routing node."""

    node_id: str
    pack_capable: bool
    unpack_capable: bool
    allowed_lane_indices: Tuple[int, ...] = ()
    pack_config: Optional[_PackedFeatureDescriptor] = None
    unpack_config: Optional[_PackedFeatureDescriptor] = None


@dataclass
class FasmAssignment:
    """A single FASM feature assignment."""

    feature_name: str
    width: int
    value: int
    context: int = 0


@dataclass
class FasmResult:
    """Result of FASM generation."""

    fasm_path: str
    assignment_count: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


class FasmGenerator:
    """Generate FASM from CGRA-Solve mapping results + compiler_arch.json."""

    def __init__(self, compiler_arch_path: str) -> None:
        self._compiler_arch_path = compiler_arch_path
        self._payload = load_and_validate_compiler_arch(compiler_arch_path)

        # feature_name -> width
        self.features: Dict[str, int] = {}
        for feat in self._payload["features"]:
            self.features[feat["name"]] = feat["width"]

        # (src_mrrg_bare, dst_mrrg_bare) -> [{feature_name, value}, ...]
        self.routing_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for edge in self._payload["routing_programming"]["edges"]:
            key = (edge["source"], edge["sink"])
            self.routing_map[key] = edge["required_assignments"]

        # (module_name, Dora optype string) -> (control_word_value, control_word_width)
        self.optype_encoding_by_module: Dict[Tuple[str, str], Tuple[int, int]] = {}
        # Backward-compatible fallback for architectures where the encoding is
        # globally unique per optype.
        self.optype_encoding: Dict[str, Tuple[int, int]] = {}
        for module in self._payload.get("module_operation_capabilities", []):
            module_name = module.get("module_name")
            if not isinstance(module_name, str) or module_name == "":
                continue
            for binding in module.get("bindings", []):
                hint = binding.get("encoding_hint")
                if hint is None:
                    continue
                optype = binding["optype"]
                encoding = (
                    hint["control_word_value"],
                    hint["control_word_width"],
                )
                self.optype_encoding_by_module[(module_name, optype)] = encoding
                self.optype_encoding[optype] = encoding

        self.layout_hash: str = self._payload["layout_hash"]
        self.fabric_contexts: int = self._payload["fabric_contexts"]
        self._node_model_by_id: Dict[str, str] = {}
        self._packed_node_by_id: Dict[str, _PackedNodeDescriptor] = {}
        self._packed_nodes: List[_PackedNodeDescriptor] = []
        self._load_mrrg_sidecar()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        dfg: DFG,
        placement: Dict[str, str],
        routes: Dict[Tuple[str, int], List[List[str]]],
        output_path: str,
        ii: int = 1,
        route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    ) -> FasmResult:
        """Generate a FASM file from a mapping result.

        Args:
            dfg: The data-flow graph that was mapped.
            placement: ``{dfg_node_id: mrrg_fu_node_id}`` (with cycle prefix).
            routes: ``{(source_dfg_id, dest_idx): [[mrrg_node_id, ...], ...]}``
                    where each inner list is a sub-path (length 1 for unfractured).
            route_metadata: Optional exported RouteBundle metadata from the
                    mapper. Required for packed-route FASM emission.
            output_path: Destination path for the .fasm file.
            ii: Initiation interval (number of MRRG time slots).

        Returns:
            :class:`FasmResult` with path, counts, warnings and errors.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Validate context capacity
        if ii > self.fabric_contexts:
            errors.append(
                f"II={ii} exceeds fabric_contexts={self.fabric_contexts} in "
                f"compiler_arch.json.  Regenerate the Dora build with "
                f"fabric_contexts >= {ii}."
            )
            return FasmResult(fasm_path=output_path, assignment_count=0, errors=errors)

        # Build node lookup
        dfg_nodes: Dict[str, DFGNode] = {n.id: n for n in dfg.get_nodes()}

        # Reconstruct HyperDFG for dest-index resolution
        hyperdfg = HyperDFG.from_dfg(dfg)

        # ---- FU assignments ----
        fu_assignments = self._generate_fu_assignments(dfg_nodes, placement, errors, warnings)

        # ---- Input/output xbar assignments ----
        xbar_assignments = self._generate_xbar_assignments(
            dfg_nodes, placement, routes, errors, warnings
        )

        # ---- Routing assignments ----
        routing_assignments = self._generate_routing_assignments(
            dfg_nodes, placement, routes, hyperdfg, errors, warnings
        )
        packed_defaults = self._generate_packed_default_assignments(errors)
        packed_assignments = self._generate_packed_route_assignments(
            route_metadata=route_metadata,
            errors=errors,
            warnings=warnings,
        )

        # ---- Merge & deduplicate ----
        all_assignments: Dict[Tuple[str, int], FasmAssignment] = {}
        default_keys: Set[Tuple[str, int]] = set()

        for a in packed_defaults:
            key = (a.feature_name, a.context)
            if not self._validate_assignment(a, errors):
                continue
            if key in all_assignments and all_assignments[key].value != a.value:
                errors.append(
                    f"Conflict: feature '{a.feature_name}' context {a.context} "
                    f"has multiple packed-default values ({all_assignments[key].value}, {a.value})"
                )
                continue
            all_assignments[key] = a
            default_keys.add(key)

        for a in fu_assignments + xbar_assignments + routing_assignments + packed_assignments:
            key = (a.feature_name, a.context)
            if key in all_assignments:
                existing = all_assignments[key]
                if existing.value == a.value:
                    continue
                if key in default_keys:
                    if not self._validate_assignment(a, errors):
                        continue
                    all_assignments[key] = a
                    default_keys.discard(key)
                    continue
                if existing.value != a.value:
                    errors.append(
                        f"Conflict: feature '{a.feature_name}' context {a.context} "
                        f"assigned both {existing.value} and {a.value}"
                    )
            else:
                if not self._validate_assignment(a, errors):
                    continue
                all_assignments[key] = a

        if errors:
            return FasmResult(
                fasm_path=output_path,
                assignment_count=0,
                warnings=warnings,
                errors=errors,
            )

        # ---- Write FASM ----
        sorted_assignments = sorted(
            all_assignments.values(), key=lambda a: (a.context, a.feature_name)
        )
        self._write_fasm(sorted_assignments, output_path)

        return FasmResult(
            fasm_path=output_path,
            assignment_count=len(sorted_assignments),
            warnings=warnings,
        )

    def _load_mrrg_sidecar(self) -> None:
        """Load sibling ``mrrg.json`` to recover module and packing metadata."""
        mrrg_path = Path(self._compiler_arch_path).with_name("mrrg.json")
        if not mrrg_path.exists():
            return

        with open(mrrg_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)

        nodes = payload.get("nodes", [])
        ii = max((node.get("time", 0) for node in nodes), default=0) + 1
        prefix_cycles = ii > 1

        for node in nodes:
            raw_id = node.get("node_id")
            if not isinstance(raw_id, str) or raw_id == "":
                continue
            time = int(node.get("time", 0))
            full_id = f"{time}:{raw_id}" if prefix_cycles else raw_id
            prefixed_id = f"{time}:{raw_id}"

            model = node.get("model")
            if isinstance(model, str) and model != "":
                self._node_model_by_id[full_id] = model
                self._node_model_by_id.setdefault(raw_id, model)
                self._node_model_by_id.setdefault(prefixed_id, model)

            packing = (
                node.get("extensions", {})
                .get("node_attrs", {})
                .get("sub_word_packing")
            )
            if not isinstance(packing, dict):
                continue

            descriptor = _PackedNodeDescriptor(
                node_id=full_id,
                pack_capable=bool(packing.get("pack_capable", False)),
                unpack_capable=bool(packing.get("unpack_capable", False)),
                allowed_lane_indices=self._parse_allowed_lane_indices(
                    packing.get("allowed_lane_indices"),
                    num_lanes=int(packing.get("num_lanes", 1) or 1),
                    node_id=full_id,
                ),
                pack_config=self._parse_packed_feature_descriptor(packing.get("pack_config")),
                unpack_config=self._parse_packed_feature_descriptor(packing.get("unpack_config")),
            )
            self._packed_nodes.append(descriptor)
            self._packed_node_by_id[full_id] = descriptor
            self._packed_node_by_id.setdefault(raw_id, descriptor)
            self._packed_node_by_id.setdefault(prefixed_id, descriptor)

    # ------------------------------------------------------------------
    # FU assignment generation
    # ------------------------------------------------------------------

    def _generate_xbar_assignments(
        self,
        dfg_nodes: Dict[str, DFGNode],
        placement: Dict[str, str],
        routes: Dict[Tuple[str, int], List[List[str]]],
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        """Emit packed control words for the flat instance crossbars.

        Two complementary strategies coexist here:

        1. **Route-walk pass (no-pred-NoC arch).** Each flat xbar instance
           (``xbar_data_in_inst`` / ``xbar_data_out_inst`` / ``xbar_mem_addr_inst``
           / ``xbar_mem_data_inst``) is programmed by a single wide control word
           whose bits ``[output_idx*sel_width +: sel_width]`` select which input
           drives ``data_o_<output_idx>``. The per-edge ``routing_programming``
           list in compiler_arch.json does not cover these internal selectors
           (the dora export omits them), so we recover the selector value by
           walking each route's path looking for the triple
           ``<inst>.data_i_X -> <inst> -> <inst>.data_o_Y``  and OR'ing the
           bits for ``(Y, X)`` into the per-instance accumulator.

        2. **iopad pass (legacy predicated-NoC arch).** When DFG INPUT nodes
           land on an ``ext_(data|pred)_iopad_inst_N`` FU, the route ends at
           ``xbar_(data|pred)_in_inst.data_o_K`` and we set bits
           ``[5*(K+1)-1 : 5*K]`` to N. Kept for backward compatibility with
           the older iopad-wrapped builds.
        """
        import re
        assignments: List[FasmAssignment] = []

        # ----- (1) Route-walk pass: program each flat xbar control word -----
        # Per-(feature_name, ctx) accumulator. We OR each output-channel's
        # sel_width bits into the running value.
        xbar_values: Dict[Tuple[str, int], int] = {}
        for (src_dfg, _dest_idx), sub_paths in routes.items():
            for path in sub_paths:
                if len(path) < 3:
                    continue
                ctx = self._get_context(path[0])
                for i in range(len(path) - 2):
                    a_full, b_full, c_full = path[i], path[i + 1], path[i + 2]
                    a = self._strip_cycle_prefix(a_full)
                    b = self._strip_cycle_prefix(b_full)
                    c = self._strip_cycle_prefix(c_full)
                    sel_width = _FLAT_XBAR_SEL_WIDTH.get(b)
                    if sel_width is None:
                        continue
                    in_prefix  = f"{b}.data_i_"
                    out_prefix = f"{b}.data_o_"
                    if not a.startswith(in_prefix) or not c.startswith(out_prefix):
                        continue
                    try:
                        input_idx  = int(a[len(in_prefix):])
                        output_idx = int(c[len(out_prefix):])
                    except ValueError:
                        continue
                    mask = (1 << sel_width) - 1
                    bit_offset = output_idx * sel_width
                    key = (b, ctx)
                    existing = xbar_values.get(key, 0)
                    existing_field = (existing >> bit_offset) & mask
                    if existing_field not in (0, input_idx):
                        warnings.append(
                            f"xbar '{b}' output {output_idx} given conflicting "
                            f"sources {existing_field} and {input_idx} by routes"
                        )
                        continue
                    xbar_values[key] = existing | ((input_idx & mask) << bit_offset)

        for (fname, ctx), val in xbar_values.items():
            fwidth = self.features.get(fname)
            if fwidth is None:
                errors.append(
                    f"flat xbar feature '{fname}' was used by a route but is "
                    f"absent from compiler_arch.features"
                )
                continue
            assignments.append(FasmAssignment(fname, fwidth, val, ctx))

        # ----- (2) iopad pass: legacy predicated-NoC arch -----

        # Resolve which xbar features exist in compiler_arch
        xbar_data_in_feat = next(
            (f for f in self._payload["features"] if f.get("name") == "xbar_data_in_inst"), None
        )
        xbar_pred_in_feat = next(
            (f for f in self._payload["features"] if f.get("name") == "xbar_pred_in_inst"), None
        )

        # Accumulate per-output-channel selects (channel_idx -> source_idx)
        data_in_selects: Dict[int, int] = {}
        pred_in_selects: Dict[int, int] = {}

        for dfg_id, mrrg_id in placement.items():
            dfg_node = dfg_nodes.get(dfg_id)
            if dfg_node is None:
                continue
            bare = self._strip_cycle_prefix(mrrg_id)

            # Data input iopad: ext_data_iopad_inst_N
            m = re.fullmatch(r"ext_data_iopad_inst_(\d+)", bare)
            if m and dfg_node.operation == OperationType.INPUT:
                src_idx = int(m.group(1))
                # Find the xbar output channel used in the route for this input
                for (src_dfg, _), sub_paths in routes.items():
                    if src_dfg != dfg_id:
                        continue
                    for path in sub_paths:
                        for node in path:
                            ch_m = re.fullmatch(r"xbar_data_in_inst\.data_o_(\d+)", node)
                            if ch_m:
                                ch_idx = int(ch_m.group(1))
                                if ch_idx in data_in_selects and data_in_selects[ch_idx] != src_idx:
                                    warnings.append(
                                        f"xbar_data_in channel {ch_idx} assigned"
                                        f" conflicting sources {data_in_selects[ch_idx]} and {src_idx}"
                                    )
                                data_in_selects[ch_idx] = src_idx

            # Pred input iopad: ext_pred_iopad_inst_N
            p = re.fullmatch(r"ext_pred_iopad_inst_(\d+)", bare)
            if p and dfg_node.operation in (OperationType.INPUT, OperationType.INPUT_PRED):
                src_idx = int(p.group(1))
                for (src_dfg, _), sub_paths in routes.items():
                    if src_dfg != dfg_id:
                        continue
                    for path in sub_paths:
                        for node in path:
                            ch_m = re.fullmatch(r"xbar_pred_in_inst\.data_o_(\d+)", node)
                            if ch_m:
                                ch_idx = int(ch_m.group(1))
                                pred_in_selects[ch_idx] = src_idx

        # Build packed select value for xbar_data_in_inst
        if data_in_selects and xbar_data_in_feat is not None:
            width = xbar_data_in_feat.get("width", 50)
            num_channels = 10
            bits_per_chan = width // num_channels  # 5
            packed = 0
            for ch, src in data_in_selects.items():
                packed |= (src & ((1 << bits_per_chan) - 1)) << (ch * bits_per_chan)
            assignments.append(
                FasmAssignment(
                    feature_name="xbar_data_in_inst",
                    width=width,
                    value=packed,
                    context=0,
                )
            )

        # Build packed select value for xbar_pred_in_inst
        if pred_in_selects and xbar_pred_in_feat is not None:
            width = xbar_pred_in_feat.get("width", 10)
            num_channels = 10
            bits_per_chan = width // num_channels  # 1
            packed = 0
            for ch, src in pred_in_selects.items():
                packed |= (src & ((1 << bits_per_chan) - 1)) << (ch * bits_per_chan)
            assignments.append(
                FasmAssignment(
                    feature_name="xbar_pred_in_inst",
                    width=width,
                    value=packed,
                    context=0,
                )
            )

        return assignments

    def _generate_fu_assignments(
        self,
        dfg_nodes: Dict[str, DFGNode],
        placement: Dict[str, str],
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        assignments: List[FasmAssignment] = []
        warned_load_addr_data_i: Set[Tuple[str, int]] = set()

        for dfg_id, mrrg_id in placement.items():
            dfg_node = dfg_nodes.get(dfg_id)
            if dfg_node is None:
                errors.append(f"DFG node '{dfg_id}' not found in DFG")
                continue

            bare = self._strip_cycle_prefix(mrrg_id)
            ctx = self._get_context(mrrg_id)
            op = dfg_node.operation

            # Map OperationType -> Dora optype string
            dora_optype = _OP_TO_DORA.get(op)
            if dora_optype is None:
                warnings.append(
                    f"No Dora optype mapping for {op.value} "
                    f"(DFG node '{dfg_id}' placed at '{mrrg_id}')"
                )
                continue

            module_name = self._resolve_module_name(mrrg_id)
            encoding = None
            if module_name is not None:
                encoding = self.optype_encoding_by_module.get((module_name, dora_optype))
            if encoding is None:
                encoding = self.optype_encoding.get(dora_optype)
            if encoding is None and module_name is not None:
                # dora's no-pred-NoC compiler_arch export leaves
                # encoding_hint null on every binding; fall back to the
                # encoding values we recovered from the SV control modules.
                encoding = _FU_OPCODE_FALLBACK.get((module_name, dora_optype))
            if encoding is None:
                # INPUT/OUTPUT are not "operations" with control words; their
                # FU lookup is expected to miss without any warning noise.
                if dora_optype in ("INPUT", "OUTPUT", "INPUT_PRED", "OUTPUT_PRED"):
                    continue
                warnings.append(
                    f"No encoding for Dora optype '{dora_optype}' "
                    f"(DFG node '{dfg_id}', module={module_name})"
                )
                continue

            cw_value, cw_width = encoding

            # For CONST operations, the feature value is the constant itself
            if op == OperationType.CONST:
                const_val = dfg_node.get_attribute("const_value")
                if const_val is None:
                    const_val = dfg_node.get_attribute("value")
                if const_val is None:
                    errors.append(
                        f"CONST node '{dfg_id}' has no const_value attribute"
                    )
                    continue
                const_int = int(const_val)
                # Handle negative values via two's complement
                if const_int < 0:
                    const_int = const_int & ((1 << cw_width) - 1)
                assignments.append(
                    FasmAssignment(bare, cw_width, const_int, ctx)
                )
            else:
                # Validate feature name exists
                if bare not in self.features:
                    warnings.append(
                        f"FU feature '{bare}' not found in compiler_arch.json "
                        f"(DFG node '{dfg_id}'); skipping opcode assignment"
                    )
                    continue
                assignments.append(
                    FasmAssignment(bare, cw_width, cw_value, ctx)
                )

        return assignments

    def _generate_packed_default_assignments(
        self,
        errors: List[str],
    ) -> List[FasmAssignment]:
        """Emit deterministic disabled defaults for all pack/unpack features."""
        assignments: List[FasmAssignment] = []
        seen: Set[Tuple[str, int]] = set()

        for node in self._packed_nodes:
            ctx = self._get_context(node.node_id)
            for descriptor in (node.pack_config, node.unpack_config):
                if descriptor is None:
                    continue
                key = (descriptor.feature_name, ctx)
                if key in seen:
                    continue
                width = self.features.get(descriptor.feature_name)
                if width is None:
                    errors.append(
                        f"Packed feature '{descriptor.feature_name}' from mrrg.json "
                        f"is missing from compiler_arch.json"
                    )
                    continue
                assignments.append(
                    FasmAssignment(
                        descriptor.feature_name,
                        width,
                        descriptor.encode(active=False),
                        ctx,
                    )
                )
                seen.add(key)

        return assignments

    def _generate_packed_route_assignments(
        self,
        *,
        route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]],
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        """Generate dynamic pack/unpack assignments for packed routes."""
        if not route_metadata:
            return []

        assignments: List[FasmAssignment] = []
        for route_key, metadata in route_metadata.items():
            if not isinstance(metadata, dict):
                errors.append(f"route_metadata[{route_key!r}] must be a dict")
                continue
            if metadata.get("transport_mode") != "packed":
                continue

            lane_assignments = metadata.get("lane_assignments", {})
            if not isinstance(lane_assignments, dict) or not lane_assignments:
                errors.append(
                    f"Packed route {route_key!r} is missing lane_assignments metadata"
                )
                continue

            active_pack_nodes = metadata.get("active_pack_nodes")
            active_unpack_nodes = metadata.get("active_unpack_nodes")

            def _normalize_active_nodes(payload: Any, field_name: str) -> Optional[Set[str]]:
                if payload is None:
                    return None
                if not isinstance(payload, list) or not all(
                    isinstance(node_id, str) and node_id != "" for node_id in payload
                ):
                    errors.append(
                        f"Packed route {route_key!r} has invalid {field_name}: {payload!r}"
                    )
                    return set()
                return set(payload)

            active_pack_set = _normalize_active_nodes(active_pack_nodes, "active_pack_nodes")
            active_unpack_set = _normalize_active_nodes(
                active_unpack_nodes,
                "active_unpack_nodes",
            )
            if active_pack_set is None:
                active_pack_set = {
                    node_id
                    for node_id in lane_assignments
                    if (
                        (self._packed_node_by_id.get(node_id) or self._packed_node_by_id.get(
                            self._strip_cycle_prefix(node_id)
                        ))
                        and (
                            self._packed_node_by_id.get(node_id)
                            or self._packed_node_by_id.get(self._strip_cycle_prefix(node_id))
                        ).pack_config is not None
                    )
                }
            if active_unpack_set is None:
                active_unpack_set = {
                    node_id
                    for node_id in lane_assignments
                    if (
                        (self._packed_node_by_id.get(node_id) or self._packed_node_by_id.get(
                            self._strip_cycle_prefix(node_id)
                        ))
                        and (
                            self._packed_node_by_id.get(node_id)
                            or self._packed_node_by_id.get(self._strip_cycle_prefix(node_id))
                        ).unpack_config is not None
                    )
                }

            for field_name, node_ids in (
                ("active_pack_nodes", active_pack_set),
                ("active_unpack_nodes", active_unpack_set),
            ):
                for node_id in node_ids:
                    if node_id not in lane_assignments:
                        errors.append(
                            f"Packed route {route_key!r} marks node '{node_id}' in "
                            f"{field_name} but it has no lane assignment"
                        )

            for node_id, lane_index in lane_assignments.items():
                descriptor = self._packed_node_by_id.get(node_id)
                if descriptor is None:
                    descriptor = self._packed_node_by_id.get(
                        self._strip_cycle_prefix(node_id)
                    )
                if descriptor is None:
                    errors.append(
                        f"Packed route {route_key!r} references node '{node_id}' "
                        f"with no sub_word_packing metadata in mrrg.json"
                    )
                    continue

                if not isinstance(lane_index, int):
                    errors.append(
                        f"Packed route {route_key!r} has non-integer lane index "
                        f"for node '{node_id}': {lane_index!r}"
                    )
                    continue
                if descriptor.allowed_lane_indices and lane_index not in descriptor.allowed_lane_indices:
                    errors.append(
                        f"Packed route {route_key!r} assigns disallowed lane {lane_index} "
                        f"to node '{node_id}'; allowed lanes are "
                        f"{list(descriptor.allowed_lane_indices)}"
                    )
                    continue

                ctx = self._get_context(node_id)
                if descriptor.pack_config is not None and node_id in active_pack_set:
                    width = self.features.get(descriptor.pack_config.feature_name)
                    if width is None:
                        errors.append(
                            f"Packed feature '{descriptor.pack_config.feature_name}' "
                            f"is missing from compiler_arch.json"
                        )
                    else:
                        try:
                            value = descriptor.pack_config.encode(
                                lane_index=lane_index,
                                active=True,
                            )
                        except ValueError as exc:
                            errors.append(
                                f"Packed route {route_key!r} has invalid lane assignment "
                                f"for node '{node_id}': {exc}"
                            )
                        else:
                            assignments.append(
                                FasmAssignment(
                                    descriptor.pack_config.feature_name,
                                    width,
                                    value,
                                    ctx,
                                )
                            )

                if descriptor.unpack_config is not None and node_id in active_unpack_set:
                    width = self.features.get(descriptor.unpack_config.feature_name)
                    if width is None:
                        errors.append(
                            f"Packed feature '{descriptor.unpack_config.feature_name}' "
                            f"is missing from compiler_arch.json"
                        )
                    else:
                        try:
                            value = descriptor.unpack_config.encode(
                                lane_index=lane_index,
                                active=True,
                            )
                        except ValueError as exc:
                            errors.append(
                                f"Packed route {route_key!r} has invalid lane assignment "
                                f"for node '{node_id}': {exc}"
                            )
                        else:
                            assignments.append(
                                FasmAssignment(
                                    descriptor.unpack_config.feature_name,
                                    width,
                                    value,
                                    ctx,
                                )
                            )

                if descriptor.pack_config is None and descriptor.unpack_config is None:
                    warnings.append(
                        f"Packed carrier node '{node_id}' in route {route_key!r} "
                        f"has no programmable pack/unpack feature; skipping FASM emission"
                    )

        return assignments

    # ------------------------------------------------------------------
    # Routing assignment generation
    # ------------------------------------------------------------------

    def _generate_routing_assignments(
        self,
        dfg_nodes: Dict[str, DFGNode],
        placement: Dict[str, str],
        routes: Dict[Tuple[str, int], List[List[str]]],
        hyperdfg: HyperDFG,
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        assignments: List[FasmAssignment] = []
        warned_load_addr_data_i: Set[Tuple[str, int]] = set()

        # Build source_dfg_id -> HyperVal lookup
        hyperval_by_source: Dict[str, Any] = {}
        for hv in hyperdfg.get_edges():
            hyperval_by_source[hv.source_id] = hv

        for (source_dfg_id, dest_idx), sub_paths in routes.items():
            if not sub_paths:
                continue  # Empty path (self-loop)

            # Source FU
            src_fu_id = placement.get(source_dfg_id)
            if src_fu_id is None:
                errors.append(
                    f"Source DFG node '{source_dfg_id}' not in placement "
                    f"(route key ({source_dfg_id}, {dest_idx}))"
                )
                continue

            # Destination FU: look up via HyperVal
            hv = hyperval_by_source.get(source_dfg_id)
            if hv is None:
                errors.append(
                    f"No HyperVal for source '{source_dfg_id}'"
                )
                continue
            if dest_idx >= len(hv.destination_ids):
                errors.append(
                    f"dest_idx={dest_idx} out of range for HyperVal "
                    f"from '{source_dfg_id}' (cardinality={hv.cardinality})"
                )
                continue
            dest_dfg_id = hv.destination_ids[dest_idx]
            sink_fu_id = placement.get(dest_dfg_id)
            if sink_fu_id is None:
                errors.append(
                    f"Dest DFG node '{dest_dfg_id}' not in placement "
                    f"(route from '{source_dfg_id}' dest_idx={dest_idx})"
                )
                continue
            sink_fu_bare = self._strip_cycle_prefix(sink_fu_id)

            # Iterate over each sub-path (one for unfractured, multiple for fractured)
            for path in sub_paths:
                if not path:
                    continue

                # Defense-in-depth: warn if a LOAD route enters through data_i
                dest_dfg_node = dfg_nodes.get(dest_dfg_id)
                if dest_dfg_node and dest_dfg_node.operation == OperationType.LOAD:
                    operand = hv.operands[dest_idx] if dest_idx < len(hv.operands) else None
                    warn_key = (source_dfg_id, dest_idx)
                    if operand in ("addr", "ADDR") and warn_key not in warned_load_addr_data_i:
                        for node_id in path:
                            bare_node = self._strip_cycle_prefix(node_id)
                            if bare_node.startswith(f"{sink_fu_bare}.") and "data_i" in bare_node:
                                warnings.append(
                                    f"LOAD address operand routed through data_i "
                                    f"(route {source_dfg_id}->{dest_dfg_id}, "
                                    f"node '{node_id}'). This likely indicates a "
                                    f"router pin-binding bug."
                                )
                                warned_load_addr_data_i.add(warn_key)
                                break

                # Build full edge chain:
                # src_fu -> path[0] -> ... -> path[-1] -> sink_fu
                chain = [src_fu_id] + list(path) + [sink_fu_id]

                for i in range(len(chain) - 1):
                    a_bare = self._strip_cycle_prefix(chain[i])
                    b_bare = self._strip_cycle_prefix(chain[i + 1])
                    ctx = self._get_context(chain[i])

                    rp = self.routing_map.get((a_bare, b_bare))
                    if rp is None:
                        # Hard-wired edge — no programming needed
                        continue

                    for entry in rp:
                        fname = entry["feature_name"]
                        fvalue = entry["value"]
                        fwidth = self.features.get(fname)
                        if fwidth is None:
                            errors.append(
                                f"routing_programming references unknown feature "
                                f"'{fname}' (edge {a_bare} -> {b_bare})"
                            )
                            continue
                        assignments.append(
                            FasmAssignment(fname, fwidth, fvalue, ctx)
                        )

        return assignments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_assignment(
        self,
        assignment: FasmAssignment,
        errors: List[str],
    ) -> bool:
        """Return True when ``assignment`` matches the compiler-arch feature table."""
        if assignment.feature_name not in self.features:
            errors.append(
                f"Unknown feature '{assignment.feature_name}' (not in compiler_arch.json)"
            )
            return False
        expected_w = self.features[assignment.feature_name]
        if assignment.width != expected_w:
            errors.append(
                f"Width mismatch for '{assignment.feature_name}': "
                f"expected {expected_w}, got {assignment.width}"
            )
            return False
        return True

    def _resolve_module_name(self, node_id: str) -> Optional[str]:
        """Return the Dora module model for a placed FU node when available."""
        model = self._node_model_by_id.get(node_id)
        if model is not None:
            return model
        return self._node_model_by_id.get(self._strip_cycle_prefix(node_id))

    @staticmethod
    def _parse_packed_feature_descriptor(
        payload: Any,
    ) -> Optional[_PackedFeatureDescriptor]:
        """Parse one ``pack_config`` / ``unpack_config`` payload."""
        if not isinstance(payload, dict):
            return None
        feature_name = payload.get("feature_name")
        if not isinstance(feature_name, str) or feature_name == "":
            return None

        def _optional_int(key: str) -> Optional[int]:
            value = payload.get(key)
            return value if isinstance(value, int) else None

        sign_extend_value = payload.get("sign_extend_value", 1)
        if not isinstance(sign_extend_value, int):
            sign_extend_value = 1

        return _PackedFeatureDescriptor(
            feature_name=feature_name,
            bypass_bit=_optional_int("bypass_bit"),
            enable_bit=_optional_int("enable_bit"),
            lane_lsb=_optional_int("lane_lsb"),
            lane_width_bits=int(payload.get("lane_width_bits", 0) or 0),
            sign_extend_bit=_optional_int("sign_extend_bit"),
            sign_extend_value=sign_extend_value,
        )

    @staticmethod
    def _parse_allowed_lane_indices(
        payload: Any,
        *,
        num_lanes: int,
        node_id: str,
    ) -> Tuple[int, ...]:
        """Parse one ``allowed_lane_indices`` payload."""
        if payload is None:
            if num_lanes <= 1:
                return ()
            return tuple(range(num_lanes))
        if not isinstance(payload, list):
            raise ValueError(
                f"Packed node '{node_id}' has non-list allowed_lane_indices: {payload!r}"
            )
        normalized = sorted({lane_idx for lane_idx in payload if isinstance(lane_idx, int)})
        if len(normalized) != len(payload):
            raise ValueError(
                f"Packed node '{node_id}' has invalid allowed_lane_indices: {payload!r}"
            )
        if not normalized:
            raise ValueError(
                f"Packed node '{node_id}' declares empty allowed_lane_indices"
            )
        if any(lane_idx < 0 or lane_idx >= num_lanes for lane_idx in normalized):
            raise ValueError(
                f"Packed node '{node_id}' has allowed_lane_indices outside "
                f"[0, {num_lanes}): {payload!r}"
            )
        return tuple(normalized)

    @staticmethod
    def _strip_cycle_prefix(node_id: str) -> str:
        """``'0:pe_0_0.alu_inst'`` -> ``'pe_0_0.alu_inst'``."""
        if ":" in node_id:
            return node_id.split(":", 1)[1]
        return node_id

    @staticmethod
    def _get_context(node_id: str) -> int:
        """``'0:pe_0_0.alu_inst'`` -> ``0``."""
        if ":" in node_id:
            return int(node_id.split(":", 1)[0])
        return 0

    def _write_fasm(self, assignments: List[FasmAssignment], output_path: str) -> None:
        lines = [
            "# Dora CGRA FASM Configuration",
            f"# dora_layout_hash: {self.layout_hash}",
            "# Source: CGRA-Solve mapping",
            f"# Compiler architecture: {self._compiler_arch_path}",
            "",
        ]

        for a in assignments:
            prefix = f"{{ctx{a.context}}}" if a.context > 0 else ""
            bits = format(a.value, f"0{a.width}b")
            lines.append(f"{prefix}{a.feature_name}[{a.width - 1}:0] = {a.width}'b{bits}")

        lines.append("")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
