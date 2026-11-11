"""Parser for Dora MRRG JSON files (schema_version >= 3)."""

import json
from pathlib import Path
from typing import Dict, Set

from cgrasolve.graph.dfg import OperationType
from cgrasolve.graph.mrrg import MRRG, MRRGNode, MRRGEdge, NodeType, HWEntityType, OperandTag


class DoraJsonParser:
    """Parser for Dora MRRG JSON files with enriched node metadata.

    Reads mrrg.json files exported by Dora and constructs a CGRA-Solve MRRG.
    Uses structured fields (node_class, global_coordinates, optype) directly
    instead of regex-based name parsing.
    """

    _BASE_OPTYPE_MAP = {
        "ADD": OperationType.ADD,
        "SUB": OperationType.SUB,
        "MUL": OperationType.MUL,
        "AND": OperationType.AND,
        "OR": OperationType.OR,
        "XOR": OperationType.XOR,
        "ICMP": OperationType.ICMP,
        "LSL": OperationType.SHL,
        "LSR": OperationType.LSHR,
        "ASR": OperationType.ASHR,
        "LD": OperationType.LOAD,
        "ST": OperationType.STORE,
        "FADD": OperationType.FADD,
        "FMUL": OperationType.FMUL,
        "FMA": OperationType.FMA,
        "FSUB": OperationType.FSUB,
        "GEP": OperationType.GEP,
        "GEP_IMM": OperationType.GEP_IMM,
        "ADD_IMM": OperationType.ADD_IMM,
        # IO operations (emitted by Dora for iopad primitives)
        "INPUT": OperationType.INPUT,
        "OUTPUT": OperationType.OUTPUT,
        "INPUT_PRED": OperationType.INPUT_PRED,
        "OUTPUT_PRED": OperationType.OUTPUT_PRED,
        "CONST": OperationType.CONST,
    }

    # Optional extensions enabled per parser instance. PHI is the dora export
    # name for the 3-input data-flow mux (pred, true, false) -> out, which the
    # DFG calls SELECT. Promoting it to a FU lets the placer target phi_mux_inst
    # for select ops, but it changes the placer's reachability/cost landscape
    # and can hurt convergence on dense ALU-only kernels (e.g. mac4_int16).
    # Off by default; enable per-load when the kernel uses select.
    _SELECT_OPTYPE_MAP = {
        "PHI": OperationType.SELECT,
        "SELECT": OperationType.SELECT,
    }

    # Class-level union of all known optype mappings, including opt-in extensions.
    # Consumers that need to reverse-map (OperationType -> dora string), e.g. the
    # FASM generator, use this rather than the per-instance OPTYPE_MAP.
    OPTYPE_MAP = {**_BASE_OPTYPE_MAP, **_SELECT_OPTYPE_MAP}

    def __init__(self, *, enable_select: bool = False) -> None:
        """Create a parser.

        Args:
            enable_select: When True, promote dora `PHI` FU nodes (phi_mux_inst)
                to NodeType.FUNCTION with OperationType.SELECT support. Required
                for kernels that contain select ops; off by default to keep the
                placer/router behavior identical to ALU-only flows.
        """
        self.enable_select = enable_select
        self.OPTYPE_MAP = dict(self._BASE_OPTYPE_MAP)
        if enable_select:
            self.OPTYPE_MAP.update(self._SELECT_OPTYPE_MAP)

    NODE_CLASS_MAP = {
        "functional_unit": (NodeType.FUNCTION, HWEntityType.HW_COMB),
        "register": (NodeType.ROUTING, HWEntityType.HW_REG),
        "routing": (NodeType.ROUTING, HWEntityType.HW_WIRE),
        "io": (NodeType.FUNCTION, HWEntityType.HW_COMB),
    }

    # Top-level chip-IO net naming convention used by the dora no-pred-NoC
    # MRRG export, where IO instance wrappers are absent and the chip boundary
    # is exposed directly as routing nets.
    _CHIP_IO_PREFIXES = ("ext_data_", "ext_pred_", "csrX_", "regS_",
                         "mem_addr_", "mem_data_")

    def _effective_node_class(self, node_data: dict) -> str:
        """Return the effective node_class for a raw MRRG node entry.

        Some MRRG exports (e.g. the dora no-predicate-NoC build) tag every
        node as `routing`, including FU instances. Recover the FU classification
        by checking whether the node carries a non-empty `operations` list whose
        optypes we recognize. Also promote bare chip-boundary IO nets to the
        `io` class so they can serve as fixed_placement targets.
        """
        declared = node_data.get("node_class", "routing")
        if declared in ("functional_unit", "io", "infrastructure", "register"):
            return declared
        for op in node_data.get("operations") or ():
            if op.get("optype") in self.OPTYPE_MAP:
                return "functional_unit"
        if self._is_chip_boundary_io_net(node_data):
            return "io"
        return declared

    def _is_chip_boundary_io_net(self, node_data: dict) -> bool:
        """True if this node looks like a top-level chip IO net (no IO wrapper)."""
        if node_data.get("kind") != "net":
            return False
        if node_data.get("direction") not in ("input", "output"):
            return False
        node_id = node_data["node_id"]
        if "." in node_id:  # internal port like xbar_data_in_inst.data_i_0
            return False
        if not any(node_id.startswith(p) for p in self._CHIP_IO_PREFIXES):
            return False
        # Pattern: prefix_(i|o)_<index>
        parts = node_id.rsplit("_", 2)
        if len(parts) < 3:
            return False
        return parts[-2] in ("i", "o") and parts[-1].isdigit()

    def _operations_for(self, node_data: dict, effective_class: str) -> list:
        """Return the operations list for a node, synthesizing one for bare IO nets.

        The dora no-predicate-NoC MRRG omits IO instance wrappers, so chip-boundary
        nets carry no operations even though they are the entry/exit points of the
        fabric. Synthesize a minimal INPUT/OUTPUT operation (with no port_map, since
        the net itself IS the port) so they look like ordinary IO FUs to the mapper.
        """
        ops = node_data.get("operations") or []
        if ops or effective_class != "io":
            return ops
        if not self._is_chip_boundary_io_net(node_data):
            return ops
        node_id = node_data["node_id"]
        direction = node_data["direction"]
        is_pred = node_id.startswith("ext_pred_")
        if direction == "input":
            optype = "INPUT_PRED" if is_pred else "INPUT"
        else:
            optype = "OUTPUT_PRED" if is_pred else "OUTPUT"
        return [{
            "binding_id": f"_synth_{optype.lower()}_bind",
            "operation_id": f"_synth_{optype.lower()}",
            "optype": optype,
            "input_port_map": [],
            "output_port_map": [],
            "scheduling": {
                "latency_cycles": 0,
                "initiation_interval_cycles": 1,
                "throughput_per_cycle": 1,
                "resource_class": "io",
            },
            "encoding_hint": None,
        }]

    def parse(self, file_path: Path) -> MRRG:
        """Parse a Dora mrrg.json file into an MRRG.

        Args:
            file_path: Path to the mrrg.json file.

        Returns:
            MRRG object populated with nodes and edges.

        Raises:
            ValueError: If schema_version < 3 or required fields are missing.
        """
        with open(file_path) as f:
            data = json.load(f)

        schema_version = data.get("schema_version", 0)
        if schema_version < 3:
            raise ValueError(
                f"DoraJsonParser requires schema_version >= 3, got {schema_version}"
            )

        rows, cols = data["grid_dimensions"]
        II = max((n["time"] for n in data["nodes"]), default=0) + 1

        mrrg = MRRG(name=file_path.stem, II=II, rows=rows, cols=cols)

        # Pre-scan: build per-pin, per-operation operand tag map.
        # pin_node_id -> {OperationType -> Set[OperandTag]}
        pin_op_tags: Dict[str, Dict[OperationType, Set[OperandTag]]] = {}
        self._build_pin_operand_tags(data["nodes"], pin_op_tags)

        node_map = {}
        for node_data in data["nodes"]:
            node_class = self._effective_node_class(node_data)
            if node_class == "infrastructure":
                continue

            mapping = self.NODE_CLASS_MAP.get(node_class)
            if mapping is None:
                continue
            node_type, hw_type = mapping

            # Mux-class FUs (e.g. phi_mux_inst) are FUs functionally but their
            # data path is a routing mux. Tag them HW_MUX so downstream
            # reachability BFSes can treat them as transit when not bound.
            if node_class == "functional_unit":
                for op in node_data.get("operations") or ():
                    sched = op.get("scheduling") or {}
                    if sched.get("resource_class") == "mux":
                        hw_type = HWEntityType.HW_MUX
                        break

            coords = node_data.get("global_coordinates")
            coords = tuple(coords) if coords else (0, 0)

            bitwidth = node_data.get("bitwidth", 32) or 32
            cycle = node_data["time"]
            raw_id = node_data["node_id"]
            node_id = f"{cycle}:{raw_id}" if II > 1 else raw_id

            operations = set()
            operation_output_bitwidths = {}
            operation_output_ports = {}
            operation_output_port_bitwidths = {}
            operation_max_input_bitwidth = {}
            operation_resource_classes = {}
            if node_class in ("functional_unit", "io"):
                for op in self._operations_for(node_data, node_class):
                    optype = op.get("optype", "")
                    if optype in self.OPTYPE_MAP:
                        op_enum = self.OPTYPE_MAP[optype]
                        operations.add(op_enum)
                        resource_class = (
                            op.get("scheduling", {}).get("resource_class")
                            or op.get("resource_class")
                        )
                        if resource_class:
                            operation_resource_classes[op_enum] = resource_class
                        output_ports = []
                        output_port_widths = {}
                        for port_entry in op.get("output_port_map", []):
                            port_name = port_entry.get("port_name") or port_entry.get("port")
                            if not port_name:
                                continue
                            full_port_name = f"{raw_id}.{port_name}"
                            output_ports.append(full_port_name)
                            port_bw = port_entry.get("bitwidth")
                            if port_bw is not None:
                                output_port_widths[full_port_name] = port_bw
                        if output_ports:
                            operation_output_ports[op_enum] = output_ports
                        if output_port_widths:
                            operation_output_port_bitwidths[op_enum] = output_port_widths
                            operation_output_bitwidths[op_enum] = max(output_port_widths.values())
                        # Extract max input port bitwidth (for sink ops like OUTPUT/STORE)
                        input_port_widths = [
                            p.get("bitwidth") for p in op.get("input_port_map", [])
                            if p.get("bitwidth") is not None
                        ]
                        if input_port_widths:
                            operation_max_input_bitwidth[op_enum] = max(input_port_widths)

            latency = node_data.get("latency", 0) or 0
            capacity = node_data.get("capacity", 1)

            mrrg_node = MRRGNode(
                node_id=node_id,
                node_type=node_type,
                cycle=cycle,
                coordinates=coords,
                hw_entity_type=hw_type,
                latency=latency,
                bitwidth=bitwidth,
                supported_operations=operations if operations else None,
                is_instance=node_data.get("kind") == "instance",
                capacity=capacity,
            )
            if operation_output_bitwidths:
                mrrg_node.operation_output_bitwidths = operation_output_bitwidths
            if operation_output_ports:
                mrrg_node.operation_output_ports = operation_output_ports
            if operation_output_port_bitwidths:
                mrrg_node.operation_output_port_bitwidths = operation_output_port_bitwidths
            if operation_resource_classes:
                mrrg_node.operation_resource_classes = operation_resource_classes
            if operation_max_input_bitwidth:
                mrrg_node.operation_max_input_bitwidth = operation_max_input_bitwidth
            # Pin operand tags use raw IDs (pre-scan doesn't know II)
            if raw_id in pin_op_tags:
                mrrg_node.operation_supported_operand_tags = pin_op_tags[raw_id]

            # Parse fracture metadata from extensions.node_attrs.fracture
            node_attrs = node_data.get("extensions", {}).get("node_attrs", {})

            fracture = node_attrs.get("fracture")
            if fracture:
                # Instance nodes (split/join primitives)
                mrrg_node.fracture_type = fracture.get("type")
                mrrg_node.fracture_parent_inst = fracture.get("parent_inst")
                mrrg_node.fracture_chunk_width = fracture.get("chunk_width")
                # Chunk net nodes
                mrrg_node.fracture_parent_port = fracture.get("parent_port")
                mrrg_node.fracture_chunk_index = fracture.get("chunk_index")
                mrrg_node.fracture_bit_offset = fracture.get("bit_offset")
                # Shared
                mrrg_node.fracture_num_chunks = fracture.get("num_chunks")

            packing = node_attrs.get("sub_word_packing")
            if packing:
                lane_width = packing.get("lane_width")
                num_lanes = packing.get("num_lanes")
                if isinstance(lane_width, int) and lane_width > 0:
                    mrrg_node.lane_width = lane_width
                if isinstance(num_lanes, int) and num_lanes > 0:
                    mrrg_node.num_lanes = num_lanes
                mrrg_node.pack_capable = bool(packing.get("pack_capable", False))
                mrrg_node.unpack_capable = bool(packing.get("unpack_capable", False))
                parent_port = packing.get("parent_port")
                if isinstance(parent_port, str) and parent_port:
                    mrrg_node.packing_parent_port = parent_port
                pack_config = packing.get("pack_config")
                if isinstance(pack_config, dict):
                    mrrg_node.pack_config = pack_config
                unpack_config = packing.get("unpack_config")
                if isinstance(unpack_config, dict):
                    mrrg_node.unpack_config = unpack_config
                mrrg_node.packing_requires_carrier = bool(
                    packing.get("requires_carrier", False)
                )
                mrrg_node.packing_inject_only = bool(
                    packing.get("inject_only", False)
                )
                allowed_lane_indices = packing.get("allowed_lane_indices")
                if allowed_lane_indices is not None:
                    if not isinstance(allowed_lane_indices, list):
                        raise ValueError(
                            f"Node '{node_id}' has non-list allowed_lane_indices: "
                            f"{allowed_lane_indices!r}"
                        )
                    normalized_lanes = sorted(
                        {
                            lane_idx
                            for lane_idx in allowed_lane_indices
                            if isinstance(lane_idx, int)
                        }
                    )
                    if len(normalized_lanes) != len(allowed_lane_indices):
                        raise ValueError(
                            f"Node '{node_id}' has invalid allowed_lane_indices: "
                            f"{allowed_lane_indices!r}"
                        )
                    if not normalized_lanes:
                        raise ValueError(
                            f"Node '{node_id}' declares empty allowed_lane_indices"
                        )
                    if any(
                        lane_idx < 0 or lane_idx >= mrrg_node.num_lanes
                        for lane_idx in normalized_lanes
                    ):
                        raise ValueError(
                            f"Node '{node_id}' has allowed_lane_indices outside "
                            f"[0, {mrrg_node.num_lanes}): {allowed_lane_indices!r}"
                        )
                    mrrg_node.allowed_lane_indices = tuple(normalized_lanes)
            mrrg.add_node(mrrg_node)
            node_map[node_id] = mrrg_node

        # Helper: resolve raw node_id + time to the prefixed key used in node_map.
        def _resolve(raw: str, t: int) -> str:
            return f"{t}:{raw}" if II > 1 else raw

        # Keep FU/IO outgoing edges in wider-first order for deterministic
        # traversal and to prefer primary data ports ahead of auxiliary narrow
        # outputs in consumers that inspect outgoing-edge order.
        fu_ids = {nid for nid, n in node_map.items() if n.node_type == NodeType.FUNCTION}
        sorted_edges = sorted(
            data["edges"],
            key=lambda e: (
                0 if _resolve(e["source_node"], e.get("source_time", 0)) not in fu_ids else 1,
                -(node_map[_resolve(e["target_node"], e.get("target_time", 0))].bitwidth
                  if _resolve(e["target_node"], e.get("target_time", 0)) in node_map else 0),
            ),
        )

        edge_counter = 0
        for edge_data in sorted_edges:
            src_id = _resolve(edge_data["source_node"], edge_data.get("source_time", 0))
            dst_id = _resolve(edge_data["target_node"], edge_data.get("target_time", 0))
            if src_id in node_map and dst_id in node_map:
                src_node = node_map[src_id]
                dst_node = node_map[dst_id]
                latency = 0
                if dst_node.hw_entity_type == HWEntityType.HW_REG:
                    latency = dst_node.latency
                edge = MRRGEdge(
                    f"edge_{edge_counter}", src_node, dst_node, latency=latency
                )
                mrrg.add_edge(edge)
                edge_counter += 1

        print(f"Parsed Dora MRRG: {mrrg.num_nodes()} nodes, {mrrg.num_edges()} edges, "
              f"grid={rows}x{cols}, II={II}")
        return mrrg

    def _build_pin_operand_tags(
        self,
        nodes_data: list,
        pin_op_tags: Dict[str, Dict[OperationType, Set[OperandTag]]],
    ) -> None:
        """Pre-scan FU/IO nodes and build per-pin, per-operation operand tags.

        Iterates ALL operations on each FU (not just the first), since
        different operations may expose different input ports (e.g., LSU
        LD has only addr_i while ST has addr_i + data_i).

        Raises:
            ValueError: If two bindings for the same OperationType assign
                        different tags to the same pin (ambiguous collapse).
        """
        for node_data in nodes_data:
            node_class = self._effective_node_class(node_data)
            if node_class not in ("functional_unit", "io"):
                continue

            fu_node_id = node_data["node_id"]

            for op in self._operations_for(node_data, node_class):
                dora_optype = op.get("optype", "")
                if dora_optype not in self.OPTYPE_MAP:
                    continue
                op_enum = self.OPTYPE_MAP[dora_optype]

                ipm = op.get("input_port_map", [])
                num_inputs = len(ipm)

                for entry in ipm:
                    port_name = entry["port_name"]
                    operand_index = entry["operand_index"]
                    pin_id = f"{fu_node_id}.{port_name}"

                    role = entry.get("role")
                    if role is not None:
                        tags = self._operand_tags_from_role(role, num_inputs)
                    else:
                        tags = self._operand_tags_for_port(
                            port_name, operand_index, num_inputs
                        )
                    if not tags:
                        continue

                    if pin_id not in pin_op_tags:
                        pin_op_tags[pin_id] = {}

                    existing = pin_op_tags[pin_id].get(op_enum)
                    if existing is not None and existing != tags:
                        raise ValueError(
                            f"Ambiguous operand binding: pin '{pin_id}' has "
                            f"conflicting tags for {op_enum.value}: "
                            f"{existing} vs {tags}. "
                            f"This may require keying by binding_id instead "
                            f"of OperationType."
                        )
                    pin_op_tags[pin_id][op_enum] = tags


    @staticmethod
    def _operand_tags_from_role(role: str, num_inputs: int) -> Set[OperandTag]:
        """Map an explicit Dora OperandRole name to OperandTag set."""
        ROLE_MAP = {
            "lhs": {OperandTag.BINARY_LHS},
            "rhs": {OperandTag.BINARY_RHS},
            "addr": {OperandTag.ADDR},
            "data": {OperandTag.DATA},
            "predicate": {OperandTag.PREDICATE},
        }
        if role in ROLE_MAP:
            return ROLE_MAP[role]
        if role == "any":
            return {OperandTag.BINARY_ANY} if num_inputs == 2 else {OperandTag.TERNARY_ANY}
        # addend, multiplicand, accumulator → ternary-any for >=3 inputs
        return {OperandTag.TERNARY_ANY} if num_inputs >= 3 else set()

    @staticmethod
    def _operand_tags_for_port(
        port_name: str, operand_index: int, num_inputs: int
    ) -> Set[OperandTag]:
        """Map a port name / operand index to a set of OperandTags."""
        lower = port_name.lower()
        if "addr" in lower:
            return {OperandTag.ADDR}
        if "data" in lower:
            return {OperandTag.DATA}
        if "pred" in lower:
            return {OperandTag.PREDICATE}
        # Generic binary (2-input)
        if num_inputs == 2:
            if operand_index == 0:
                return {OperandTag.BINARY_LHS}
            elif operand_index == 1:
                return {OperandTag.BINARY_RHS}
        # Ternary+ (FMA, etc.)
        if num_inputs >= 3:
            return {OperandTag.TERNARY_ANY}
        return set()
