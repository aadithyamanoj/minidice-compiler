"""Lower a VEGAS step_dfg_v4 DFG onto Mini_Dice's p-graph model.

Mini_Dice executes a kernel as a sequence of p-graphs. Each p-graph is one
statically routed configuration of the 4x4 fabric: up to four memory ports,
inputs from the register file and the eight broadcast CSRs, and results
written back to the register file. A load's result comes back through the
load/store unit into a general register, so it is only readable by a later
p-graph. Everything else flows through the fabric within one p-graph.

The fabric is INT16 with add, sub, and mul, and no per-PE immediate, so a
literal or a kernel parameter has to arrive in a CSR. Float arithmetic is
lowered to that INT16 arithmetic on values the host has already quantized,
and the rewrites below record each such approximation instead of applying it
silently.

Addresses in the DFG are byte addresses. The chip's data memory is 16-bit
word addressed, so an address is rewritten as

    word = csr_base + thread_id * stride + offset

where the CSR holds the array's word base and the host reprograms that base
between CTAs, because the fabric only ever sees the local thread id.
"""
from __future__ import annotations

from dataclasses import dataclass

# Bytes in the pointed-to element, keyed by the scalar type in a DFG type
# string such as "float*" or "i64*".
_POINTEE_BYTES = {
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "half": 2, "float": 4, "double": 8,
}

# DFG opcodes that lower straight onto the INT16 ALU. The float forms are the
# approximation: the host quantized the inputs, so the same operation on the
# integers preserves the result the kernel was rewritten to compute.
_ALU = {
    "OPT_ADD": "add", "OPT_SUB": "sub", "OPT_MUL": "mul",
    "OPT_FADD": "add", "OPT_FSUB": "sub", "OPT_FMUL": "mul",
}
_FLOAT_OPS = {"OPT_FADD", "OPT_FSUB", "OPT_FMUL", "OPT_FDIV", "OPT_SQRT"}

# A p-graph stays at or under these. They sit inside what the mapper has
# already routed on this fabric, so a graph that meets them maps.
ALU_BUDGET = 6
MEM_BUDGET = 4
# A store needs its own routing to the memory crossbar. Past this many ALU
# ops the compute graph is already as full as the largest one known to map,
# so the store starts a fresh p-graph instead of crowding it.
NUM_MEM_PORTS = 4
UNUSED_DEST = 31
GPR_COUNT = 8
CSR_COUNT = 8


class LowerError(Exception):
    """The DFG uses something Mini_Dice cannot run, named so it can be fixed."""


@dataclass
class Aff:
    """Byte address as base_pointer + tid * tid + const, before the word scale.

    `base` is the pointer parameter the address derives from. It is never
    scaled: the CSR holds the word base, and only the tid term and the
    constant field offset are converted from bytes to words.
    """
    base: str | None = None
    tid: int = 0
    const: int = 0

    def __add__(self, other: "Aff") -> "Aff":
        if self.base and other.base:
            raise LowerError(
                f"address combines two pointers ({self.base} and {other.base}); "
                "Mini_Dice addresses must be one base plus an affine function "
                "of the thread id")
        return Aff(self.base or other.base,
                   self.tid + other.tid, self.const + other.const)

    def scaled(self, factor: int) -> "Aff":
        if self.base:
            raise LowerError(
                f"pointer {self.base} is multiplied in an address; only the "
                "index may be scaled")
        return Aff(None, self.tid * factor, self.const * factor)

    def to_words(self, width: int) -> "Aff":
        if self.tid % width or self.const % width:
            raise LowerError(
                f"address offset tid*{self.tid}+{self.const} bytes is not a "
                f"multiple of the {width}-byte access, so it has no 16-bit "
                "word address")
        return Aff(self.base, self.tid // width, self.const // width)


@dataclass
class MemOp:
    kind: str            # "load" or "store"
    base: str
    stride: int          # words added per thread id
    offset: int          # constant word offset
    name: str            # result name for a load, stored value for a store
    node: int


@dataclass
class AluOp:
    op: str              # "add", "sub", "mul"
    inputs: list         # ("val", name) | ("param", name) | ("const", int) | ("tid",)
    name: str
    node: int


@dataclass
class Lowered:
    ops: list
    bases: list          # (param name, element stride in words), CSR order
    scalars: list        # scalar kernel params read by the compute, CSR order
    constants: list      # literal values that need a CSR, ascending
    approximations: list
    block: int
    grid: int
    kernel: str


def _pointee_bytes(type_str: str) -> int:
    base = type_str[:-1] if type_str.endswith("*") else type_str
    if base not in _POINTEE_BYTES:
        raise LowerError(f"no element size for pointer type {type_str}")
    return _POINTEE_BYTES[base]


def _const(token: str) -> int:
    return int(token.split(":", 1)[1], 0)


def lower(dfg: dict) -> Lowered:
    nodes = {n["id"]: n for n in dfg["nodes"]}
    launch = dfg["meta"]["launch"]
    approximations: list[str] = []

    dropped = _control_nodes(nodes, approximations)
    real = [n for n in dfg["nodes"] if n["id"] not in dropped]
    _reject_unsupported(real)

    produced: dict[int, str] = {}
    ops: list = []

    def value_of(token):
        """The operand a compute op reads: a param, a constant, the thread id,
        or a value an earlier op produced."""
        if token == "thread_id":
            return ("tid",)
        if isinstance(token, str) and token.startswith("const:"):
            return ("const", _const(token))
        node = nodes[token]
        op = node["JSON_opt"]
        if op == "OPT_LD_PARAM":
            if str(node["type"]).endswith("*"):
                raise LowerError(
                    f"compute reads pointer parameter {node['param_name']} "
                    "directly; pointers are only meaningful as addresses")
            return ("param", node["param_name"])
        if op == "OPT_EXT":
            return value_of(node["inputs"][0][0])
        if token not in produced:
            raise LowerError(
                f"node {token} ({op}) feeds the compute but is not a value "
                "the fabric can carry")
        return ("val", produced[token])

    for node in real:
        op = node["JSON_opt"]
        if str(node["type"]).endswith("*") or op in ("OPT_LD_PARAM", "OPT_EXT"):
            continue
        if op == "OPT_LD":
            aff = _address(node["inputs"][0][0], nodes).to_words(_width(node))
            name = f"v{node['id']}"
            produced[node["id"]] = name
            ops.append(MemOp("load", aff.base, aff.tid, aff.const, name, node["id"]))
        elif op == "OPT_STR":
            aff = _address(node["inputs"][1][0], nodes).to_words(_width(node))
            ops.append(MemOp("store", aff.base, aff.tid, aff.const,
                             value_of(node["inputs"][0][0])[1], node["id"]))
        elif op == "OPT_SQRT":
            produced[node["id"]] = value_of(node["inputs"][0][0])[1]
        elif op == "OPT_FDIV":
            numer = value_of(node["inputs"][0][0])
            denom = value_of(node["inputs"][1][0])
            if denom[0] not in ("param", "const"):
                raise LowerError(
                    f"node {node['id']} divides by a per-thread value, and "
                    "Mini_Dice has no divider and no reciprocal instruction")
            name = f"v{node['id']}"
            produced[node["id"]] = name
            ops.append(AluOp("mul", [numer, ("recip", denom)], name, node["id"]))
            approximations.append(
                f"node {node['id']}: division by {denom[1]} lowered to a "
                "multiply by its reciprocal, supplied by the host in a CSR")
        elif op in _ALU or op == "OPT_ADD_CONST":
            opcode = "add" if op == "OPT_ADD_CONST" else _ALU[op]
            name = f"v{node['id']}"
            produced[node["id"]] = name
            ops.append(AluOp(opcode, [value_of(g[0]) for g in node["inputs"]],
                             name, node["id"]))
        else:
            raise LowerError(f"node {node['id']} has unsupported opcode {op}")

    if any(op in _FLOAT_OPS for n in real if (op := n["JSON_opt"])):
        approximations.insert(0,
            "float arithmetic lowered to INT16 on host-quantized inputs; "
            "results wrap modulo 65536")

    bases, scalars, constants = _operands(ops)
    if len(bases) + len(scalars) + len(constants) > CSR_COUNT:
        raise LowerError(
            f"kernel needs {len(bases) + len(scalars) + len(constants)} CSRs "
            f"and the fabric broadcasts {CSR_COUNT}")

    return Lowered(ops, bases, scalars, constants, approximations,
                   launch["block"], launch["grid"], dfg["meta"]["kernel"])


def _width(node: dict) -> int:
    access = node.get("access") or {}
    width = access.get("width_bytes")
    if width not in (2, 4, 8):
        raise LowerError(
            f"node {node['id']} accesses {width} bytes; only 2, 4, or 8 byte "
            "accesses lower onto the 16-bit memory")
    return width


def _address(token, nodes: dict) -> Aff:
    """Evaluate an address expression to base + tid * scale + offset, in bytes."""
    if token == "thread_id":
        return Aff(tid=1)
    if isinstance(token, str) and token.startswith("const:"):
        return Aff(const=_const(token))
    node = nodes[token]
    op = node["JSON_opt"]
    if op == "OPT_LD_PARAM":
        # Kernel pointer arguments are typed "ptr"; derived addresses carry the
        # pointee type and end in "*".
        if node["type"] != "ptr" and not str(node["type"]).endswith("*"):
            raise LowerError(
                f"parameter {node['param_name']} is used as an address but is "
                f"a {node['type']}")
        return Aff(base=node["param_name"])
    if op == "OPT_EXT":
        return _address(node["inputs"][0][0], nodes)
    if op in ("OPT_ADD", "OPT_ADD_CONST") and str(node["type"]).endswith("*"):
        base_tok, index_tok = _base_and_index(node, nodes)
        return _address(base_tok, nodes) + _address(index_tok, nodes).scaled(
            _pointee_bytes(node["type"]))
    if op == "OPT_MUL":
        left = _address(node["inputs"][0][0], nodes)
        right = _address(node["inputs"][1][0], nodes)
        if left.base is None and left.tid == 0:
            return right.scaled(left.const)
        if right.base is None and right.tid == 0:
            return left.scaled(right.const)
        raise LowerError(
            f"node {node['id']} multiplies two non-constant address terms")
    if op in ("OPT_LLS_CONST", "OPT_SHL"):
        shift = _const(node["inputs"][1][0])
        return _address(node["inputs"][0][0], nodes).scaled(1 << shift)
    raise LowerError(
        f"node {node['id']} ({op}) makes the address data-dependent, which "
        "the word-addressed memory cannot express as one base plus a stride")


def _base_and_index(node: dict, nodes: dict):
    """The pointer add's base operand and its index operand.

    LLVM's getelementptr always takes the base first. For anything else the
    base is whichever side is itself a pointer.
    """
    groups = node["inputs"]
    if node.get("org_opt") == "getelementptr":
        return groups[0][0], groups[1][0]

    def carries_base(token) -> bool:
        if isinstance(token, str):
            return False
        kind = nodes[token]["JSON_opt"]
        return kind == "OPT_LD_PARAM" or str(nodes[token]["type"]).endswith("*")

    if carries_base(groups[0][0]):
        return groups[0][0], groups[1][0]
    return groups[1][0], groups[0][0]


def _control_nodes(nodes: dict, approximations: list) -> set:
    """Branch, compare, and return nodes that the fabric run replaces.

    The one control shape Mini_Dice compilation supports is a single bounds
    check that guards the whole kernel: the launch covers exactly the live
    threads, so the compare and its branch are deleted and the body always
    runs. Anything else -- a loop, a second branch, a compare used as data --
    is reported rather than miscompiled.
    """
    branches = [n for n in nodes.values()
                if n["JSON_opt"] == "OPT_BRH" and n.get("branch_targets")]
    conditional = [n for n in branches
                   if n["branch_targets"]["kind"] == "conditional"]
    if len(conditional) > 1:
        raise LowerError(
            f"kernel has {len(conditional)} conditional branches; Mini_Dice "
            "compilation supports one bounds check and an otherwise straight "
            "body. Unroll the rest into separate launches.")
    for node in nodes.values():
        if node.get("is_phi"):
            raise LowerError(
                "kernel has a loop-carried value; unroll the loop before "
                "lowering, the fabric branches forward only")

    dropped = {n["id"] for n in branches}
    dropped |= {n["id"] for n in nodes.values() if n["JSON_opt"] == "OPT_RET"}

    if conditional:
        branch = conditional[0]
        targets = branch["branch_targets"]
        pred = branch["inputs"][targets["predicate_input_index"]][0]
        users = _users_of(pred, nodes) - dropped
        if users:
            raise LowerError(
                f"the comparison at node {pred} feeds {sorted(users)} as well "
                "as its branch; Mini_Dice has no signed compare, only a "
                "nonzero test, so a comparison used as data cannot be lowered")
        pred_node = nodes[pred]
        if pred_node["JSON_opt"] not in ("OPT_LT", "OPT_GT", "OPT_LE", "OPT_GE",
                                         "OPT_EQ", "OPT_NE"):
            raise LowerError(
                f"branch predicate node {pred} is a {pred_node['JSON_opt']}, "
                "not a comparison")
        dropped.add(pred)
        approximations.append(
            f"bounds check at node {branch['id']} (true block "
            f"{targets['true_bb']}, exit block {targets['false_bb']}) removed; "
            "the launch must cover exactly the threads the check would have "
            "kept")
    return dropped


def _users_of(node_id: int, nodes: dict) -> set:
    users = set()
    for node in nodes.values():
        for group in node["inputs"]:
            if node_id in group:
                users.add(node["id"])
    return users


def _reject_unsupported(real: list) -> None:
    handled = set(_ALU) | {"OPT_ADD_CONST", "OPT_SQRT", "OPT_FDIV", "OPT_LD",
                           "OPT_STR", "OPT_LD_PARAM", "OPT_EXT", "OPT_MUL",
                           "OPT_LLS_CONST", "OPT_SHL"}
    for node in real:
        op = node["JSON_opt"]
        if str(node["type"]).endswith("*"):
            continue
        if op not in handled:
            raise LowerError(
                f"node {node['id']} ({node.get('org_opt')}, {op}) has no "
                "Mini_Dice lowering. The fabric ALU is add, sub, and mul on "
                "16-bit integers; shifts, signed compares, and variable "
                "divides are not available.")


def _operands(ops: list):
    """The CSRs the lowered program reads, in assignment order.

    Pointer bases come first so the per-CTA overrides, which rewrite only the
    bases, stay grouped. A base's stride is the tid coefficient of every
    access through it, and two accesses to one base at different strides
    cannot share a single per-CTA base adjustment.
    """
    strides: dict[str, int] = {}
    scalars: list[str] = []
    constants: set[int] = set()

    def add_scalar(name: str) -> None:
        if name not in scalars:
            scalars.append(name)

    for op in ops:
        if isinstance(op, MemOp):
            if op.base is None:
                raise LowerError(
                    f"node {op.node} addresses memory without a pointer base")
            seen = strides.get(op.base)
            if seen is not None and seen != op.stride:
                raise LowerError(
                    f"pointer {op.base} is indexed at stride {seen} and stride "
                    f"{op.stride}; one CSR base can absorb only one stride "
                    "when it is adjusted per CTA")
            strides[op.base] = op.stride
            if op.stride not in (0, 1):
                constants.add(op.stride)
            if op.offset:
                constants.add(op.offset)
            if op.kind == "store" and isinstance(op.name, tuple):
                pass
        if isinstance(op, AluOp):
            for inp in op.inputs:
                if inp[0] == "param":
                    add_scalar(inp[1])
                elif inp[0] == "const":
                    constants.add(inp[1])
                elif inp[0] == "recip":
                    # The reciprocal is a host-supplied scalar keyed by the
                    # divisor it inverts, carried as its own parameter name.
                    add_scalar(f"recip_{inp[1][1]}")
    bases = [(name, stride) for name, stride in strides.items()]
    return bases, scalars, sorted(constants)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P-graphs. Every output of one p-graph is captured on the same cycle, because
# the writeback latency is a single number, so a graph is built to have one
# output depth: a compute graph drives exactly one register, and a memory
# graph only routes registers straight to memory ports. Those routes each
# cross one switch box, so they all arrive together and several ports can
# fire in the same graph.
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    """A memory operation whose address is already a routed value."""
    kind: str
    addr: tuple
    data: str
    node: int


@dataclass
class PGraph:
    name: str
    dot: str
    reads: list          # GPR indices read from the register file
    writes: list         # GPR indices written through the fabric outputs
    load_dests: list     # (rtl port, GPR index) for each load
    num_stores: int
    notes: list


@dataclass
class Program:
    lowered: Lowered
    graphs: list
    csr_index: dict      # ("param", name) | ("const", n) -> CSR number
    gpr_of: dict         # value name -> GPR index


def _assign_csrs(lowered: Lowered) -> dict:
    index = {}
    number = 0
    for name, _stride in lowered.bases:
        index[("param", name)] = number
        number += 1
    for name in lowered.scalars:
        index[("param", name)] = number
        number += 1
    for value in lowered.constants:
        index[("const", value)] = number
        number += 1
    return index


def _addr_ops(op: MemOp) -> tuple[list, tuple]:
    """The arithmetic that produces one memory address, and the operand that
    holds the result. The result is a fresh value so the memory graph can
    route it instead of recomputing it."""
    ops = []
    node = op.node
    term = None
    if op.stride not in (0, 1):
        name = f"a{node}t"
        ops.append(AluOp("mul", [("tid",), ("const", op.stride)], name, node))
        term = ("val", name)
    elif op.stride == 1:
        term = ("tid",)

    base = ("param", op.base) if op.base is not None else None
    acc = _combine(ops, base, term, f"a{node}b", node)
    if op.offset:
        acc = _combine(ops, acc, ("const", op.offset), f"a{node}", node)
    elif acc is None:
        raise LowerError(f"node {node} address has no base and no index")
    elif not ops or ops[-1].name != _operand_name(acc):
        # The address is a bare pin. Give it a name by adding zero so every
        # address the memory graph routes is a value with one producer.
        acc = _combine(ops, acc, ("const", 0), f"a{node}", node)
    return ops, acc


def _operand_name(operand) -> str | None:
    return operand[1] if operand and operand[0] == "val" else None


def _combine(ops: list, left, right, name: str, node: int):
    """Add two address operands, ignoring a side that isn't there."""
    if left is None:
        return right
    if right is None:
        return left
    ops.append(AluOp("add", [left, right], name, node))
    return ("val", name)


def _sequence(lowered: Lowered) -> list:
    """Arithmetic and memory ops in execution order, with each memory address
    computed by the ops placed just before the operation that uses it."""
    seq = []
    for op in lowered.ops:
        if isinstance(op, AluOp):
            seq.append(op)
            continue
        addr_ops, addr = _addr_ops(op)
        seq.extend(addr_ops)
        seq.append(Issue(op.kind, addr, op.name, op.node))
    return seq


def _seq_vals(item) -> tuple[list, str | None]:
    """Values an item reads and the value it produces."""
    if isinstance(item, AluOp):
        return [inp[1] for inp in item.inputs if inp[0] == "val"], item.name
    reads = []
    if item.addr[0] == "val":
        reads.append(item.addr[1])
    if item.kind == "store":
        reads.append(item.data)
        return reads, None
    return reads, item.data


def _schedule(seq: list) -> list:
    """Group the sequence into compute graphs with one output and memory
    graphs that only route.

    Memory ops that don't depend on each other's results are one frontier.
    Their addresses are computed first, each into its own register, and then
    all of them are issued in a single graph so the ports fire together. The
    next frontier -- the store that consumes the loaded values -- follows.
    """
    consumers: dict[str, list] = {}
    producer: dict[str, object] = {}
    for item in seq:
        reads, produced = _seq_vals(item)
        for name in reads:
            consumers.setdefault(name, []).append(item)
        if produced:
            producer[produced] = item

    remaining = list(seq)
    groups: list[list] = []

    def output_of(item):
        return _seq_vals(item)[1]

    def feeds_issue(item, issue_ids: set, seen: set) -> bool:
        """Whether an arithmetic op's result reaches one of the frontier
        memory ops without passing through another memory op."""
        if id(item) in seen:
            return False
        seen.add(id(item))
        produced = output_of(item)
        for user in consumers.get(produced, []):
            if id(user) in issue_ids:
                return True
            if isinstance(user, AluOp) and user in remaining and \
                    feeds_issue(user, issue_ids, seen):
                return True
        return False

    def is_frontier(issue, seen: set) -> bool:
        """A memory op is in the frontier when none of its inputs is the
        result of a memory op that hasn't been issued yet."""
        for name in _seq_vals(issue)[0]:
            source = producer.get(name)
            if source is None or source not in remaining:
                continue
            if isinstance(source, Issue):
                return False
            if not is_frontier_value(source, seen):
                return False
        return True

    def is_frontier_value(item, seen: set) -> bool:
        if id(item) in seen:
            return True
        seen.add(id(item))
        for name in _seq_vals(item)[0]:
            source = producer.get(name)
            if source is None or source not in remaining:
                continue
            if isinstance(source, Issue) or not is_frontier_value(source, seen):
                return False
        return True

    while remaining:
        issues = [item for item in remaining if isinstance(item, Issue)]
        frontier = [item for item in issues if is_frontier(item, set())]
        if not frontier:
            raise LowerError(
                "memory operations depend on each other in a cycle, which a "
                "straight-line p-graph sequence cannot order")

        # Addresses for the whole frontier, one register each, before any of
        # the frontier's memory ops is issued.
        involved = {id(item) for item in frontier}
        address_ops = [item for item in remaining if isinstance(item, AluOp)
                       and feeds_issue(item, involved, set())]
        groups.extend(_single_output_groups(address_ops, consumers))
        for item in address_ops:
            remaining.remove(item)

        for start in range(0, len(frontier), MEM_BUDGET):
            groups.append(frontier[start:start + MEM_BUDGET])
        for item in frontier:
            remaining.remove(item)
    return groups


def _single_output_groups(ops: list, consumers: dict) -> list:
    """Pack arithmetic into graphs that each drive one register.

    An op joins the open graph only while that keeps the graph to a single
    value consumed outside it. The two address chains of one record don't
    fold together, because each is consumed by its own memory port.
    """
    groups = []
    current: list = []

    def outside(group: list) -> int:
        inside = set(id(item) for item in group)
        count = 0
        for item in group:
            produced = item.name
            if any(id(user) not in inside for user in consumers.get(produced, [])):
                count += 1
        return count

    for op in ops:
        if current and outside(current + [op]) > 1:
            groups.append(current)
            current = []
        current.append(op)
    if current:
        groups.append(current)
    return groups


def build(lowered: Lowered) -> Program:
    csr_index = _assign_csrs(lowered)
    groups = _schedule(_sequence(lowered))
    gpr_of: dict[str, int] = {}

    def assign(name: str) -> int:
        if name not in gpr_of:
            if len(gpr_of) >= GPR_COUNT:
                raise LowerError(
                    f"more than {GPR_COUNT} values cross between p-graphs; "
                    "Mini_Dice has eight general registers and no spill slot")
            gpr_of[name] = len(gpr_of)
        return gpr_of[name]

    graphs = []
    for index, group in enumerate(groups):
        graphs.append(_emit(index, group, csr_index, gpr_of, assign,
                            index == len(groups) - 1))
    return Program(lowered, graphs, csr_index, gpr_of)


def _emit(index: int, group: list, csr_index: dict, gpr_of: dict, assign,
          last: bool) -> PGraph:
    node_lines, edge_lines = [], []
    declared: set[str] = set()

    def declare(node: str, opcode: str) -> None:
        if node not in declared:
            declared.add(node)
            node_lines.append(f"    {node:<16} [opcode={opcode}, bitwidth=16];")

    def pin(operand) -> str:
        kind = operand[0]
        if kind == "tid":
            return "REGS0"
        if kind == "param":
            return f"CSRX{csr_index[('param', operand[1])]}"
        if kind == "const":
            return f"CSRX{csr_index[('const', operand[1])]}"
        return operand[1]

    def edge(src: str, dst: str, operand: str | None) -> None:
        extra = f"operand={operand}, " if operand else ""
        edge_lines.append(
            f"    {src} -> {dst} [{extra}bitwidth=16, dist=0, kind=\"dataflow\", "
            "predicate=false, is_loop_back=false];")

    produced_here = {name for item in group
                     if isinstance(item, AluOp) and (name := item.name)}
    reads: set[int] = set()
    writes: list[int] = []
    load_dests: list[tuple[int, int]] = []
    notes: list[str] = []

    def read_pin(operand) -> str:
        src = pin(operand)
        if operand[0] == "val" and operand[1] not in produced_here:
            src = f"R{gpr_of[operand[1]]}"
            reads.add(gpr_of[operand[1]])
        declare(src, "input")
        return src

    if isinstance(group[0], Issue):
        stores = [item for item in group if item.kind == "store"]
        loads = [item for item in group if item.kind == "load"]
        for slot, item in enumerate(stores):
            declare(f"MEM_ADDR{slot}", "output")
            declare(f"MEM_DATA{slot}", "output")
            edge(read_pin(item.addr), f"MEM_ADDR{slot}", None)
            edge(read_pin(("val", item.data)), f"MEM_DATA{slot}", None)
            notes.append(f"store {item.data} on mem port {slot}")
        for slot, item in enumerate(loads):
            port = len(stores) + slot
            dest = assign(item.data)
            declare(f"MEM_ADDR{port}", "output")
            edge(read_pin(item.addr), f"MEM_ADDR{port}", None)
            load_dests.append((port, dest))
            notes.append(f"load -> R{dest} on mem port {port}")
        num_stores = len(stores)
    else:
        for item in group:
            declare(item.name, item.op)
            for operand, label in zip(item.inputs, ("LHS", "RHS")):
                edge(read_pin(operand), item.name, label)
        output = _group_output(group)
        gpr = assign(output)
        declare(f"OUT{gpr}", "output")
        edge(output, f"OUT{gpr}", None)
        writes.append(gpr)
        notes.append(f"{output} -> R{gpr}")
        num_stores = 0

    for edge_line in edge_lines:
        declare(edge_line.split("->")[0].strip(), "input")

    header = f"p-graph {index}" + (" (returns)" if last else "")
    body = "digraph G {\n" + "\n".join(node_lines + edge_lines) + "\n}\n"
    return PGraph(f"pg{index}", f"// {header}\n{body}", sorted(reads), writes,
                  load_dests, num_stores, notes)


def _group_output(group: list) -> str:
    """The one value a compute graph produces for a later graph."""
    produced = [item.name for item in group]
    consumed = {inp[1] for item in group for inp in item.inputs if inp[0] == "val"}
    live = [name for name in produced if name not in consumed]
    if len(live) != 1:
        raise LowerError(
            f"a compute p-graph must drive exactly one register, it drives "
            f"{live or 'nothing'}")
    return live[0]


def metadata(program: Program, latencies: list[int]) -> list:
    """The pgraph_metadata list the mem packer consumes.

    ld_dest_regs is stored reversed: the packer pushes the JSON array
    MSB-first and the RTL's packed array keeps port 0 in its low bits, so
    JSON slot (NUM_MEM_PORTS - 1 - port) holds that port's destination.
    """
    entries = []
    for i, graph in enumerate(program.graphs):
        ld = [UNUSED_DEST] * NUM_MEM_PORTS
        for port, dest in graph.load_dests:
            ld[NUM_MEM_PORTS - 1 - port] = dest
        entries.append({
            "pc": 0x1000 + i * 0x100,
            "meta": {
                "barrier": 0,
                "bitstream_addr": i * 0x200,
                "bitstream_length": 1074,
                "branch_meta": {
                    "branch_ena": 0, "branch_uni": 0, "branch_pred_reg": 0,
                    "branch_neg_pred": 0, "is_return": int(i == len(program.graphs) - 1),
                    "branch_jump_target_offset": 0, "branch_reconv_offset": 0,
                },
                "in_regs_bitmap": sum(1 << r for r in graph.reads),
                "out_regs_bitmap": sum(1 << r for r in graph.writes),
                "lat": latencies[i],
                "ld_dest_regs": ld,
                "num_stores": graph.num_stores,
                "parameter_load": 0,
                "unrolling_factor": 0,
            },
        })
    return entries

