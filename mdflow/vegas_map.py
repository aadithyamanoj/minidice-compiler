"""Place and route Mini_Dice p-graphs with the VEGAS DICE mapper.

Boxman's `map_program` is the mapper VEGAS uses for its DICE fabric. It does
not know Mini_Dice. This module builds the architecture document that mapper
expects from the Mini_Dice routing graph (the MRRG) and the DORA feature
edges in compiler_arch.json, then asks the mapper to place each p-graph on
that fabric. The bitstream features come back off the routes: mux edges carry
the DORA assignment, the flat crossbars are recovered from which input drove
which output, and an ALU opcode is the add/sub/mul placed on that PE.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOXMAN = ROOT / "third_party" / "boxman"
ARCH_DIR = (ROOT / "cgra-solve-local" / "benchmarks" / "architectures"
            / "mini_dice_static" / "II_1")
sys.path.insert(0, str(BOXMAN))

from src.lib.architecture.model import Architecture  # noqa: E402
from src.lib.architecture.pnr import MappingFailure, map_program, route_configuration  # noqa: E402

ALU_OPCODE = {"add": 0, "sub": 1, "mul": 2}
XBAR_SEL_WIDTH = {
    "xbar_data_in_inst": 5,
    "xbar_data_out_inst": 4,
    "xbar_mem_addr_inst": 4,
    "xbar_mem_data_inst": 4,
}
# Chip pins the p-graphs name, and the MRRG net each one is.
PIN_NET = {}
for _index in range(8):
    PIN_NET[f"R{_index}"] = f"ext_data_i_{_index}"
    PIN_NET[f"CSRX{_index}"] = f"csrX_i_{_index}"
    PIN_NET[f"OUT{_index}"] = f"ext_data_o_{_index}"
PIN_NET["REGS0"] = "regS_i_0"
for _index in range(4):
    PIN_NET[f"MEM_ADDR{_index}"] = f"mem_addr_o_{_index}"
    PIN_NET[f"MEM_DATA{_index}"] = f"mem_data_o_{_index}"

_NODE = re.compile(r"(\S+)\s+\[opcode=(\w+)")
_EDGE = re.compile(r"(\S+)\s+->\s+(\S+)\s+\[(?:operand=(\w+),)?")


def _sid(name: str) -> str:
    """Architecture names cannot contain a dot."""
    return name.replace(".", "__")


def load_fabric():
    """The VEGAS architecture object for this Mini_Dice fabric, built once."""
    if getattr(load_fabric, "cached", None) is not None:
        return load_fabric.cached
    mrrg = json.loads((ARCH_DIR / "mrrg.json").read_text())
    catalog = json.loads((ARCH_DIR / "compiler_arch.json").read_text())
    arch = Architecture(_document(mrrg, catalog))
    widths = {feature["name"]: feature["width"] for feature in catalog["features"]}
    load_fabric.cached = (arch, widths, catalog["layout_hash"])
    return load_fabric.cached


def _data_instance(node: dict) -> bool:
    """Predicate and phi resources are not data wires.

    A phi mux follows its predicate, and that predicate comes out of reset as
    X. Routing a value through it makes the first thread's result X and can
    drop a later thread. The data path is the switch boxes, the ALU, and the
    flat crossbars.
    """
    model = node.get("model")
    if model in ("phi_mux_pred16_16b", "pred_to_data_16b", "pred_nonzero_16b"):
        return False
    if model == "reg_16b" and "pred_reg" in node["node_id"]:
        return False
    return True


def _document(mrrg: dict, catalog: dict) -> dict:
    all_instances = [node for node in mrrg["nodes"] if node["kind"] == "instance"]
    blocked = {node["node_id"] for node in all_instances if not _data_instance(node)}
    instances = [node for node in all_instances if node["node_id"] not in blocked]
    by_length = sorted((node["node_id"] for node in instances), key=len, reverse=True)
    by_length_all = sorted((node["node_id"] for node in all_instances), key=len, reverse=True)
    node_of = {node["node_id"]: node for node in mrrg["nodes"]}

    def owner(net: str):
        for instance in by_length:
            if net.startswith(instance + "."):
                return instance
        return None

    def blocked_net(net: str) -> bool:
        for instance in by_length_all:
            if net.startswith(instance + "."):
                return instance in blocked
        return False

    pins: dict[str, dict[str, str]] = {node["node_id"]: {} for node in instances}
    for edge in mrrg["edges"]:
        for end in ("source_node", "target_node"):
            net = edge[end]
            instance = owner(net)
            if instance is None:
                continue
            other = edge["target_node" if end == "source_node" else "source_node"]
            if other != instance:
                continue
            # A pin feeding the instance is an input; a pin the instance drives
            # is an output.
            direction = "in" if end == "source_node" else "out"
            pins[instance][net[len(instance) + 1:]] = direction

    resources = []
    for node in instances:
        model = node.get("model")
        resource = {
            "id": _sid(node["node_id"]),
            "kind": "compute" if model == "fu_int16_add_int16_sub_int16_mul" else "routing",
            "mrrg_id": node["node_id"],
            "ports": [
                {"name": port, "direction": direction, "type": "data", "width": 16}
                for port, direction in sorted(pins[node["node_id"]].items())
            ],
        }
        if model == "fu_int16_add_int16_sub_int16_mul":
            resource["operations"] = ["add", "sub", "mul"]
        resources.append(resource)

    endpoints = set(PIN_NET.values())
    for node in mrrg["nodes"]:
        if node["kind"] != "net" or owner(node["node_id"]) or node["node_id"] in endpoints:
            continue
        if blocked_net(node["node_id"]):
            continue
        resources.append({
            "id": _sid(node["node_id"]),
            "kind": "routing",
            "mrrg_id": node["node_id"],
            "ports": [
                {"name": "i", "direction": "in", "type": "data", "width": 16},
                {"name": "o", "direction": "out", "type": "data", "width": 16},
            ],
        })
    for name, net in PIN_NET.items():
        direction = node_of[net]["direction"]
        resources.append({
            "id": _sid(net),
            "kind": "endpoint",
            "mrrg_id": net,
            "endpoint": "source" if direction == "input" else "sink",
            "operations": ["input" if direction == "input" else "output"],
            "pin": name,
            "ports": [{
                "name": "out" if direction == "input" else "in",
                "direction": "out" if direction == "input" else "in",
                "type": "data",
                "width": 16,
            }],
        })

    def locate(net: str, as_source: bool):
        instance = owner(net)
        if instance is not None:
            return _sid(instance), net[len(instance) + 1:]
        node = node_of[net]
        if net in endpoints:
            return _sid(net), ("out" if node["direction"] == "input" else "in")
        return _sid(net), ("o" if as_source else "i")

    programmed = {
        (edge["source"], edge["sink"]): edge["required_assignments"]
        for edge in catalog["routing_programming"]["edges"]
    }
    connections = []
    seen = set()

    def add(source, sink, **extra):
        key = (source, sink, extra.get("select"), extra.get("setting"))
        if key in seen:
            return
        seen.add(key)
        connections.append(dict(source=source, sink=sink, **extra))

    for instance in instances:
        model = instance.get("model")
        rid = _sid(instance["node_id"])
        ports = pins[instance["node_id"]]
        if instance["node_id"] in XBAR_SEL_WIDTH:
            for src, src_dir in ports.items():
                if src_dir != "in":
                    continue
                for dst, dst_dir in ports.items():
                    if dst_dir != "out":
                        continue
                    add(f"{rid}.{src}", f"{rid}.{dst}", kind="switch",
                        select=f"{rid}__{dst}", setting=src.removeprefix("data_i_"),
                        xbar=instance["node_id"], xbar_input=int(src.removeprefix("data_i_")),
                        xbar_output=int(dst.removeprefix("data_o_")))
            continue
        ins = [port for port, direction in ports.items() if direction == "in"]
        outs = [port for port, direction in ports.items() if direction == "out"]
        if model == "fu_int16_add_int16_sub_int16_mul":
            continue
        latency = 1 if instance.get("latency") else 0
        for src in ins:
            for dst in outs:
                add(f"{rid}.{src}", f"{rid}.{dst}", kind="fixed", latency=latency)
    for edge in mrrg["edges"]:
        if owner(edge["source_node"]) and edge["target_node"] == owner(edge["source_node"]):
            continue
        if owner(edge["target_node"]) and edge["source_node"] == owner(edge["target_node"]):
            continue
        if edge["source_node"] not in node_of or node_of[edge["source_node"]]["kind"] != "net":
            continue
        if edge["target_node"] not in node_of or node_of[edge["target_node"]]["kind"] != "net":
            continue
        if blocked_net(edge["source_node"]) or blocked_net(edge["target_node"]):
            continue
        # A flat crossbar is crossed only by its switch edges. The routing
        # graph also has wires into an output pin and out of an input pin,
        # and taking those would program nothing for that crossbar.
        source_owner = owner(edge["source_node"])
        target_owner = owner(edge["target_node"])
        if source_owner in XBAR_SEL_WIDTH and edge["source_node"].rsplit(".", 1)[1].startswith("data_i"):
            continue
        if target_owner in XBAR_SEL_WIDTH and edge["target_node"].rsplit(".", 1)[1].startswith("data_o"):
            continue
        src = locate(edge["source_node"], True)
        dst = locate(edge["target_node"], False)
        features = programmed.get((edge["source_node"], edge["target_node"]), [])
        add(f"{src[0]}.{src[1]}", f"{dst[0]}.{dst[1]}",
            **({"features": features} if features else {}))
    for node in mrrg["nodes"]:
        if node["kind"] != "net" or owner(node["node_id"]) or node["node_id"] in endpoints:
            continue
        if blocked_net(node["node_id"]):
            continue
        rid = _sid(node["node_id"])
        add(f"{rid}.i", f"{rid}.o", kind="fixed")

    return {
        "format": "step-architecture",
        "version": 1,
        "target": "mini-dice",
        "top": "top",
        "contract": {
            "execution": "elastic-configurations",
            "load_writeback": "region_exit",
            "data_width": 16,
            "registers": 8,
            "register_banks": 8,
            "rf_entries_per_bank": 16,
            "pred_registers": 2,
            "register_layout": "independent",
        },
        "modules": {"top": {"resources": resources, "connections": connections}},
    }


def _program(dot: str) -> dict:
    opcodes = dict(_NODE.findall(dot))
    operations = []
    for name, opcode in opcodes.items():
        if opcode == "input":
            operations.append({
                "id": name, "op": "input", "endpoint": "source",
                "ports": ["out"], "inputs": [],
                "fixed_resource": _sid(PIN_NET[name]),
            })
        elif opcode == "output":
            operations.append({
                "id": name, "op": "output", "endpoint": "sink",
                "ports": ["in"], "inputs": ["in"],
                "fixed_resource": _sid(PIN_NET[name]),
            })
        elif opcode in ALU_OPCODE:
            operations.append({
                "id": name, "op": opcode,
                "ports": ["A", "B", "out"], "inputs": ["A", "B"],
            })
        else:
            raise ValueError(f"p-graph opcode {opcode} is not on the Mini_Dice fabric")
    nets: dict[tuple, dict] = {}
    for src, dst, operand in _EDGE.findall(dot):
        src_port = "out"
        if opcodes[dst] in ALU_OPCODE:
            dst_port = {"LHS": "A", "RHS": "B"}[operand]
        else:
            dst_port = "in"
        key = (src, src_port)
        net = nets.get(key)
        if net is None:
            net = {"id": f"n{len(nets)}", "source": [src, src_port], "sinks": []}
            nets[key] = net
        net["sinks"].append([dst, dst_port])
    return {
        "format": "step-pnr-program",
        "version": 1,
        "configurations": [{
            "id": "pg",
            "operations": operations,
            "nets": list(nets.values()),
        }],
    }


def _fasm(mapped: dict, widths: dict, layout_hash: str) -> tuple[str, int]:
    arch = Architecture(mapped["architecture"])
    settings: dict[str, int] = {}
    flops = []

    def assign(name: str, value: int) -> None:
        if name in settings and settings[name] != value:
            raise ValueError(f"feature {name} programmed as {settings[name]} and {value}")
        settings[name] = value

    config = mapped["configurations"][0]
    spec = mapped["program"]["configurations"][0]
    for operation in spec["operations"]:
        if operation["op"] not in ALU_OPCODE:
            continue
        placed = config["placement"][operation["id"]]
        feature = arch.resources[placed]["mrrg_id"]
        assign(feature, ALU_OPCODE[operation["op"]])
    # Flops are counted per net. An ALU is combinational, so its two inputs
    # have to arrive on the same cycle, and the p-graph latency is the cycle
    # on which every output arrives.
    arrival = {op["id"]: 0 for op in spec["operations"] if op["op"] == "input"}
    for net in spec["nets"]:
        paths = config["routes"][net["id"]]
        for path, sink in zip(paths, net["sinks"]):
            count = 0
            for edge_id in path:
                edge = arch.edges[edge_id]
                count += edge.get("latency", 0)
                for feature in edge.get("features", []):
                    assign(feature["feature_name"], feature["value"])
                if "xbar" in edge:
                    width = XBAR_SEL_WIDTH[edge["xbar"]]
                    packed = settings.get(edge["xbar"], 0)
                    shift = edge["xbar_output"] * width
                    mask = ((1 << width) - 1) << shift
                    field = (packed & mask) >> shift
                    if field not in (0, edge["xbar_input"]):
                        raise ValueError(
                            f"{edge['xbar']} output {edge['xbar_output']} "
                            f"driven by {field} and {edge['xbar_input']}")
                    settings[edge["xbar"]] = (packed & ~mask) | (edge["xbar_input"] << shift)
            reached = arrival[net["source"][0]] + count
            previous = arrival.get(sink[0])
            if previous is not None and previous != reached:
                raise ValueError(
                    f"{sink[0]} is reached on cycle {previous} and cycle {reached}")
            arrival[sink[0]] = reached
    outputs = [op["id"] for op in spec["operations"] if op["op"] == "output"]
    cycles = {arrival[name] for name in outputs}
    if len(cycles) != 1:
        raise ValueError(f"outputs arrive on different cycles {sorted(cycles)}")
    flops.append(max(1, cycles.pop()))
    lines = [
        "# Dora CGRA FASM Configuration",
        f"# dora_layout_hash: {layout_hash}",
        "# Source: VEGAS mapper on the Mini_Dice architecture",
        "",
    ]
    for name, value in sorted(settings.items()):
        width = widths[name]
        if value < 0 or value >= 1 << width:
            raise ValueError(f"feature {name} value {value} does not fit in {width} bits")
        lines.append(f"{name}[{width - 1}:0] = {width}'b{value:0{width}b}")
    lines.append("")
    return "\n".join(lines), max(flops or [1])


def _arrivals(arch, spec: dict, routes: dict) -> tuple[dict, dict]:
    """Cycle each operation's inputs arrive on, and the path that got them there.

    Producers are visited before their consumers because the lowerer emits
    edges in dataflow order. An operation's arrival is the later of its inputs.
    """
    arrival = {op["id"]: 0 for op in spec["operations"] if op["op"] == "input"}
    paths_at: dict[str, list] = {}
    for net in spec["nets"]:
        for path, sink in zip(routes[net["id"]], net["sinks"]):
            count = sum(arch.edges[edge].get("latency", 0) for edge in path)
            reached = arrival[net["source"][0]] + count
            paths_at.setdefault(sink[0], []).append((path, reached))
            arrival[sink[0]] = max(arrival.get(sink[0], reached), reached)
    return arrival, paths_at


def _balance(arch, spec: dict, mapped: dict) -> None:
    """Re-route inputs that arrive early so every operand meets on one cycle.

    The router prefers the shortest path, and a value computed by another PE
    has already passed through more switchboxes than a value read from a pin.
    The switchboxes are the only registers, so the short path is sent through
    extra ones by making its current edges expensive and routing again.
    """
    config = mapped["configurations"][0]
    bumped: list[tuple[dict, int]] = []
    try:
        for _ in range(24):
            _, paths_at = _arrivals(arch, spec, config["routes"])
            early = [path for paths in paths_at.values() for path, reached in paths
                     if reached < max(item[1] for item in paths)]
            if not early:
                return
            for path in early:
                for edge_id in path:
                    edge = arch.edges[edge_id]
                    bumped.append((edge, edge.get("cost", 1)))
                    edge["cost"] = edge.get("cost", 1) + 25
            config["routes"] = route_configuration(
                arch, spec, config["placement"])
    finally:
        for edge, cost in reversed(bumped):
            edge["cost"] = cost


def map_dots(dots: Path, out: Path, attempts: int = 40) -> list:
    """Map every p-graph dot. Returns the mapping records, in file order."""
    out.mkdir(parents=True, exist_ok=True)
    arch, widths, layout_hash = load_fabric()
    records = []
    for dot in sorted(dots.glob("*.dot")):
        started = time.time()
        try:
            program = _program(dot.read_text())
            mapped = map_program(program, arch, attempts=attempts)
            _balance(arch, program["configurations"][0], mapped)
            text, latency = _fasm(mapped, widths, layout_hash)
            (out / f"{dot.stem}.fasm").write_text(text)
            record = {"dot": str(dot), "status": "success",
                      "seconds": round(time.time() - started, 3),
                      "latency": latency, "fasm": f"{dot.stem}.fasm"}
        except (MappingFailure, ValueError) as error:
            record = {"dot": str(dot), "status": "failed",
                      "seconds": round(time.time() - started, 3),
                      "latency": None, "error": str(error)}
        records.append(record)
        print(f"  {dot.stem}: {record['status']}"
              + (f" lat {record['latency']}" if record['latency'] is not None else "")
              + (f" ({record.get('error')})" if record['status'] != 'success' else ""))
    (out / "mapping.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
