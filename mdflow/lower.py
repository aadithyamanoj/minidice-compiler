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
word addressed, and a CTA is 16 threads, so an address is rewritten as

    word = csr_base + local_thread * stride + offset

The CSR holds the array's word base plus this CTA's share of the index. A
pointer used with two different index formulas gets two base CSRs; the fabric
broadcasts eight. The fabric only ever sees the local thread id.
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
THREADS_PER_CTA = 16
PGRAPH_LIMIT = 32


class LowerError(Exception):
    """The DFG uses something Mini_Dice cannot run, named so it can be fixed."""


def _e_add(left, right):
    if left == ("int", 0):
        return right
    if right == ("int", 0):
        return left
    if left[0] == "int" and right[0] == "int":
        return ("int", left[1] + right[1])
    return ("add", left, right)


def _e_mul(left, right):
    if left == ("int", 0) or right == ("int", 0):
        return ("int", 0)
    if left == ("int", 1):
        return right
    if right == ("int", 1):
        return left
    if left[0] == "int" and right[0] == "int":
        return ("int", left[1] * right[1])
    return ("mul", left, right)


def _e_div(expr, width: int):
    """Divide a byte expression by an access width. Integers must divide exactly."""
    if expr == ("int", 0):
        return expr
    if expr[0] == "int":
        if expr[1] % width:
            raise LowerError(
                f"address offset {expr[1]} bytes is not a multiple of the "
                f"{width}-byte access")
        return ("int", expr[1] // width)
    if expr[0] == "add":
        return _e_add(_e_div(expr[1], width), _e_div(expr[2], width))
    if expr[0] == "mul":
        for factor, other in ((expr[1], expr[2]), (expr[2], expr[1])):
            if factor[0] == "int" and factor[1] % width == 0:
                return _e_mul(("int", factor[1] // width), other)
        for factor, other in ((expr[1], expr[2]), (expr[2], expr[1])):
            try:
                return _e_mul(_e_div(factor, width), other)
            except LowerError:
                continue
    raise LowerError(
        f"address offset is not a multiple of the {width}-byte access")


def _split_host(expr) -> tuple:
    """The part the host bakes into the base, and the constant the fabric adds."""
    pending = [expr]
    const = 0
    varied = ("int", 0)
    while pending:
        item = pending.pop()
        if item == ("int", 0):
            continue
        if item[0] == "add":
            pending.extend((item[1], item[2]))
        elif item[0] == "int":
            const += item[1]
        else:
            varied = _e_add(varied, item)
    return varied, const


@dataclass
class Aff:
    """Byte address as base + host(cta, params) + local_tid * tid.

    `host` is what the host adds to one base CSR for this CTA. It may mention
    the CTA index and kernel parameters. `tid` is the coefficient of the local
    thread id, which is the only index term the fabric computes. A global
    thread id is local_tid plus cta * block, so its host term is `cta * block`.
    """
    base: str | None = None
    tid: int = 0
    host: tuple = ("int", 0)

    def __add__(self, other: "Aff") -> "Aff":
        if self.base and other.base:
            raise LowerError(
                f"address combines two pointers ({self.base} and {other.base}); "
                "Mini_Dice addresses must be one base plus an affine function "
                "of the thread id")
        return Aff(self.base or other.base, self.tid + other.tid,
                   _e_add(self.host, other.host))

    def scaled(self, factor: int) -> "Aff":
        if self.base:
            raise LowerError(
                f"pointer {self.base} is multiplied in an address; only the "
                "index may be scaled")
        return Aff(None, self.tid * factor, _e_mul(self.host, ("int", factor)))

    def to_words(self, width: int) -> "Aff":
        if self.tid % width:
            raise LowerError(
                f"address stride of {self.tid} bytes is not a multiple of the "
                f"{width}-byte access")
        return Aff(self.base, self.tid // width, _e_div(self.host, width))


@dataclass
class MemOp:
    kind: str            # "load" or "store"
    base: str            # pointer parameter the bytes belong to
    stride: int          # words added per local thread id
    host: tuple          # words the host adds to this formula's base CSR
    offset: int          # constant word offset the fabric adds
    name: str            # result name for a load, stored value for a store
    node: int
    slot: str = ""       # CSR key; a second formula of the same pointer differs


@dataclass
class AluOp:
    op: str              # "add", "sub", "mul"
    inputs: list         # ("val", name) | ("param", name) | ("const", int) | ("tid",)
    name: str
    node: int


@dataclass
class Lowered:
    ops: list
    bases: list          # (csr slot, pointer, local stride, host term), CSR order
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


def _fit_launch(grid: int, block: int, approximations: list) -> tuple[int, int]:
    """The launch the chip can run: CTAs of 16 threads, same lanes as asked.

    A block of 64 covering four rows of a 16-wide tile becomes four CTAs of
    one row each. The grid grows by the same factor, so lane N is still lane N.
    """
    if block == THREADS_PER_CTA:
        return grid, block
    if block < THREADS_PER_CTA or block % THREADS_PER_CTA:
        raise LowerError(
            f"block {block} does not split into CTAs of {THREADS_PER_CTA}")
    ctas_per_block = block // THREADS_PER_CTA
    grid *= ctas_per_block
    approximations.append(
        f"launch block {block} split into {ctas_per_block} CTAs of "
        f"{THREADS_PER_CTA}; grid is {grid}")
    return grid, THREADS_PER_CTA


def _lane_column(node) -> bool:
    """`thread_id & 15`, the column of a 16-wide tile. It is the local thread."""
    if not isinstance(node, dict) or node.get("JSON_opt") != "OPT_AND_CONST":
        return False
    groups = node["inputs"]
    if len(groups) != 2 or groups[0][0] != "thread_id":
        return False
    return _const(groups[1][0]) == THREADS_PER_CTA - 1


def _lane_row(node) -> bool:
    """`thread_id >> 4`, the row of a 16-wide tile. It is the CTA index."""
    if not isinstance(node, dict) or node.get("JSON_opt") != "OPT_ASR_CONST":
        return False
    groups = node["inputs"]
    shift = THREADS_PER_CTA.bit_length() - 1
    if len(groups) != 2 or groups[0][0] != "thread_id":
        return False
    return _const(groups[1][0]) == shift


def lower(dfg: dict) -> Lowered:
    nodes = {n["id"]: n for n in dfg["nodes"]}
    launch = dfg["meta"]["launch"]
    approximations: list[str] = []
    grid, block = _fit_launch(launch["grid"], launch["block"], approximations)

    dropped = _control_nodes(nodes, approximations)
    real = [n for n in dfg["nodes"] if n["id"] not in dropped]
    # Shifts and masks that only build an address are not ALU ops. The address
    # walker folds them into a base CSR and a local stride.
    address_only = _address_only(nodes)
    _reject_unsupported(real, block, address_only)

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
        if op in ("OPT_EXT", "OPT_TRUNC"):
            # A trunc to the 16-bit memory result is the low half the ALU
            # already produces. Sign-extends are free on the way into that ALU.
            return value_of(node["inputs"][0][0])
        if block == THREADS_PER_CTA and _lane_column(node):
            return ("tid",)
        if block == THREADS_PER_CTA and _lane_row(node):
            return ("cta",)
        if token not in produced:
            raise LowerError(
                f"node {token} ({op}) feeds the compute but is not a value "
                "the fabric can carry")
        return ("val", produced[token])

    for node in real:
        op = node["JSON_opt"]
        if node["id"] in address_only:
            continue
        if str(node["type"]).endswith("*") or op in (
                "OPT_LD_PARAM", "OPT_EXT", "OPT_TRUNC"):
            continue
        if block == THREADS_PER_CTA and (_lane_column(node) or _lane_row(node)):
            continue
        if op == "OPT_LD":
            aff = _address(node["inputs"][0][0], nodes, block).to_words(_width(node))
            host, offset = _split_host(aff.host)
            name = f"v{node['id']}"
            produced[node["id"]] = name
            ops.append(MemOp("load", aff.base, aff.tid, host, offset,
                             name, node["id"]))
        elif op == "OPT_STR":
            aff = _address(node["inputs"][1][0], nodes, block).to_words(_width(node))
            host, offset = _split_host(aff.host)
            ops.append(MemOp("store", aff.base, aff.tid, host, offset,
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
        elif op in _ALU or op in ("OPT_ADD_CONST", "OPT_MUL_CONST"):
            opcode = {"OPT_ADD_CONST": "add", "OPT_MUL_CONST": "mul"}.get(
                op, _ALU[op])
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
    if any(n["JSON_opt"] == "OPT_TRUNC" for n in real):
        approximations.append(
            "products wider than 16 bits are kept as their low half, which is "
            "the INT16 result")
    if any(n["JSON_opt"] in ("OPT_AND", "OPT_AND_CONST", "OPT_OR", "OPT_ASR",
                             "OPT_ASR_CONST", "OPT_LLS_CONST", "OPT_SHL")
           and n["id"] in address_only for n in real):
        approximations.append(
            "shifts and masks of the thread id in an address were folded into "
            "the base CSR and the local stride")
    if any(_lane_column(n) for n in real):
        approximations.append(
            "lane & 15 lowered to the local thread id on regS")
    if any(_lane_row(n) for n in real):
        approximations.append(
            "lane >> 4 lowered to the CTA index, added into the pointer base")

    bases, scalars, constants = _operands(ops)
    if len(bases) + len(scalars) + len(constants) > CSR_COUNT:
        raise LowerError(
            f"kernel needs {len(bases) + len(scalars) + len(constants)} CSRs "
            f"and the fabric broadcasts {CSR_COUNT}")

    return Lowered(ops, bases, scalars, constants, approximations,
                   block, grid, dfg["meta"]["kernel"])


def _width(node: dict) -> int:
    access = node.get("access") or {}
    width = access.get("width_bytes")
    if width not in (2, 4, 8):
        raise LowerError(
            f"node {node['id']} accesses {width} bytes; only 2, 4, or 8 byte "
            "accesses lower onto the 16-bit memory")
    return width


def _address(token, nodes: dict, block: int) -> Aff:
    """Evaluate an address to base + host(cta, params) + local_tid * tid, in bytes."""
    if token == "thread_id":
        # The DFG's thread id is the global lane. The fabric adds the local
        # part; the host puts this CTA's threads into the pointer base.
        return Aff(tid=1, host=_e_mul(("cta",), ("int", block)))
    if isinstance(token, str) and token.startswith("const:"):
        return Aff(host=("int", _const(token)))
    node = nodes[token]
    op = node["JSON_opt"]
    if op == "OPT_LD_PARAM":
        # Kernel pointer arguments are typed "ptr"; derived addresses carry the
        # pointee type and end in "*". A scalar parameter is an index term
        # the host knows.
        if node["type"] == "ptr" or str(node["type"]).endswith("*"):
            return Aff(base=node["param_name"])
        return Aff(host=("param", node["param_name"]))
    if op == "OPT_EXT":
        return _address(node["inputs"][0][0], nodes, block)
    if block == THREADS_PER_CTA and _lane_column(node):
        return Aff(tid=1)
    if block == THREADS_PER_CTA and _lane_row(node):
        return Aff(host=("cta",))
    if op in ("OPT_ADD", "OPT_ADD_CONST") and str(node["type"]).endswith("*"):
        base_tok, index_tok = _base_and_index(node, nodes)
        return _address(base_tok, nodes, block) + _address(
            index_tok, nodes, block).scaled(_pointee_bytes(node["type"]))
    if op in ("OPT_ADD", "OPT_ADD_CONST"):
        acc = None
        for group in node["inputs"]:
            part = _address(group[0], nodes, block)
            acc = part if acc is None else acc + part
        return acc
    if op == "OPT_MUL":
        left = _address(node["inputs"][0][0], nodes, block)
        right = _address(node["inputs"][1][0], nodes, block)

        def scale_of(aff: Aff):
            if aff.base is not None or aff.tid:
                return None
            return aff.host

        left_scale, right_scale = scale_of(left), scale_of(right)
        if left_scale is not None and left_scale[0] == "int":
            return right.scaled(left_scale[1])
        if right_scale is not None and right_scale[0] == "int":
            return left.scaled(right_scale[1])
        if left_scale is not None and right.tid == 0 and right.base is None:
            return Aff(host=_e_mul(left_scale, right.host))
        if right_scale is not None and left.tid == 0 and left.base is None:
            return Aff(host=_e_mul(left.host, right_scale))
        raise LowerError(
            f"node {node['id']} multiplies the thread id by a value that is "
            "not a constant, so the address stride is not fixed")
    if op in ("OPT_LLS_CONST", "OPT_SHL"):
        shift = _const(node["inputs"][1][0])
        return _address(node["inputs"][0][0], nodes, block).scaled(1 << shift)
    if op in ("OPT_AND", "OPT_AND_CONST", "OPT_OR", "OPT_OR_CONST",
              "OPT_ASR", "OPT_ASR_CONST", "OPT_LSR_CONST", "OPT_XOR"):
        # Clang turns (cta << k) + column into a shift, a mask, and an or.
        # Those are affine in the local thread and the CTA, so they fold.
        return _fit_index(token, nodes, block)
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


def _address_only(nodes: dict) -> set:
    """Nodes that exist only to compute a load or store address."""
    in_address = set()

    def walk(token) -> None:
        if not isinstance(token, int) or token in in_address:
            return
        in_address.add(token)
        for group in nodes[token]["inputs"]:
            walk(group[0])

    for node in nodes.values():
        op = node["JSON_opt"]
        if op == "OPT_LD" and node["inputs"]:
            walk(node["inputs"][0][0])
        elif op == "OPT_STR" and len(node["inputs"]) > 1:
            walk(node["inputs"][1][0])

    value_uses = set()
    for node in nodes.values():
        if node["id"] in in_address:
            continue
        start = 1 if node["JSON_opt"] == "OPT_STR" else 0
        # A store's address was walked above; its data input is a real value.
        # Every other input of a non-address node is a real value too.
        for group in node["inputs"][start:]:
            token = group[0]
            if isinstance(token, int):
                value_uses.add(token)
    return in_address - value_uses


class _NotAffine(Exception):
    pass


def _eval_bits(token, nodes: dict, block: int, local: int, cta: int) -> int:
    """Concrete value of an index expression at one local thread and CTA."""
    if token == "thread_id":
        return cta * block + local
    if isinstance(token, str) and token.startswith("const:"):
        return _const(token)
    node = nodes[token]
    op = node["JSON_opt"]
    args = [_eval_bits(group[0], nodes, block, local, cta)
            for group in node["inputs"]]
    if op in ("OPT_EXT", "OPT_TRUNC"):
        return args[0]
    if op in ("OPT_ADD", "OPT_ADD_CONST"):
        return sum(args)
    if op == "OPT_MUL":
        return args[0] * args[1]
    if op in ("OPT_LLS_CONST", "OPT_SHL"):
        return args[0] << args[1]
    if op in ("OPT_ASR_CONST", "OPT_LSR_CONST"):
        return args[0] >> args[1]
    if op in ("OPT_AND", "OPT_AND_CONST"):
        return args[0] & args[1]
    if op in ("OPT_OR", "OPT_OR_CONST"):
        return args[0] | args[1]
    if op == "OPT_XOR":
        return args[0] ^ args[1]
    raise _NotAffine(op)


def _fit_index(token, nodes: dict, block: int) -> Aff:
    """Fold an index into local_tid * stride + cta * cta_stride + offset.

    The local id only ranges over one CTA, so a shift-and-mask of the global
    thread id is checked at every local id and a few CTA ids. If it is affine,
    the host can absorb the CTA term and the fabric adds the local term.
    """
    def at(local: int, cta: int) -> int:
        return _eval_bits(token, nodes, block, local, cta)

    try:
        origin = at(0, 0)
        stride = at(1, 0) - origin
        cta_k = at(0, 1) - origin
        for cta in range(4):
            for local in range(block):
                if at(local, cta) != stride * local + cta_k * cta + origin:
                    raise _NotAffine("not affine")
    except _NotAffine as error:
        raise LowerError(
            f"node {token} cannot be folded into a base plus a local stride "
            f"({error})") from None
    host = _e_add(_e_mul(("cta",), ("int", cta_k)), ("int", origin))
    return Aff(tid=stride, host=host)


def _reject_unsupported(real: list, block: int, address_only: set) -> None:
    handled = set(_ALU) | {"OPT_ADD_CONST", "OPT_MUL_CONST", "OPT_SQRT", "OPT_FDIV",
                           "OPT_LD", "OPT_STR", "OPT_LD_PARAM", "OPT_EXT",
                           "OPT_TRUNC", "OPT_MUL", "OPT_LLS_CONST", "OPT_SHL"}
    for node in real:
        op = node["JSON_opt"]
        if node["id"] in address_only or str(node["type"]).endswith("*"):
            continue
        if block == THREADS_PER_CTA and (_lane_column(node) or _lane_row(node)):
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
    bases, stay grouped. Each distinct index formula of a pointer — its local
    stride together with the term the host adds per CTA — gets its own base
    CSR. A second formula of the same pointer is named `pointer#1`.
    """
    formulas: dict[tuple, str] = {}
    taken: dict[str, int] = {}
    bases: list = []
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
            key = (op.base, op.stride, op.host)
            slot = formulas.get(key)
            if slot is None:
                n = taken.get(op.base, 0)
                slot = op.base if n == 0 else f"{op.base}#{n}"
                taken[op.base] = n + 1
                formulas[key] = slot
                bases.append((slot, op.base, op.stride, op.host))
            op.slot = slot
            if op.stride not in (0, 1):
                constants.add(op.stride)
            if op.offset:
                constants.add(op.offset)
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
                elif inp[0] == "cta":
                    add_scalar("cta_id")
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
    for slot, _pointer, _stride, _host in lowered.bases:
        index[("param", slot)] = number
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

    base = ("param", op.slot) if op.base is not None else None
    acc = _combine(ops, base, term, f"a{node}b", node)
    if op.offset:
        acc = _combine(ops, acc, ("const", op.offset), f"a{node}", node)
    elif acc is None:
        raise LowerError(f"node {node} address has no base and no index")
    elif not ops or ops[-1].name != _operand_name(acc):
        # The address is already a pin (a base CSR, or the local thread id).
        # The memory graph routes that pin; adding zero would spend a CSR.
        return ops, acc
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

    A sum is scheduled as load, multiply, add, then the next term. Issuing
    every independent load first would hold each term in a register until the
    end, and eight registers cannot hold a long dot product. Loads that feed
    a later multiply wait until the multiply already in hand has been added.
    """
    consumers: dict[str, list] = {}
    producer: dict[str, object] = {}
    order = {id(item): index for index, item in enumerate(seq)}
    for item in seq:
        reads, produced = _seq_vals(item)
        for name in reads:
            consumers.setdefault(name, []).append(item)
        if produced:
            producer[produced] = item

    remaining = list(seq)
    live = set(id(item) for item in remaining)
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
            if isinstance(user, AluOp) and id(user) in live and \
                    feeds_issue(user, issue_ids, seen):
                return True
        return False

    def inputs_ready(item) -> bool:
        for name in _seq_vals(item)[0]:
            source = producer.get(name)
            if source is not None and id(source) in live:
                return False
        return True

    def is_frontier(issue, seen: set) -> bool:
        """A memory op is in the frontier when none of its inputs is the
        result of a memory op that hasn't been issued yet."""
        for name in _seq_vals(issue)[0]:
            source = producer.get(name)
            if source is None or id(source) not in live:
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
            if source is None or id(source) not in live:
                continue
            if isinstance(source, Issue) or not is_frontier_value(source, seen):
                return False
        return True

    def address_only(item, seen: set) -> bool:
        """An add or mul that only builds a load or store address."""
        if id(item) in seen:
            return False
        seen.add(id(item))
        users = consumers.get(item.name, [])
        if not users:
            return False
        for user in users:
            if isinstance(user, Issue):
                used_as_addr = user.addr[0] == "val" and user.addr[1] == item.name
                used_as_data = user.kind == "store" and user.data == item.name
                if not used_as_addr or used_as_data:
                    return False
            elif isinstance(user, AluOp):
                if not address_only(user, seen):
                    return False
            else:
                return False
        return True

    def first_user(item) -> int:
        produced = output_of(item)
        users = consumers.get(produced, []) if produced else []
        if not users:
            return order[id(item)]
        return min(order[id(user)] for user in users)

    def drop(items) -> None:
        gone = set(id(item) for item in items)
        live.difference_update(gone)
        remaining[:] = [item for item in remaining if id(item) not in gone]

    while remaining:
        compute = [item for item in remaining
                   if isinstance(item, AluOp) and inputs_ready(item)
                   and not address_only(item, set())]
        if compute:
            # A Horner chain stays in one p-graph: each product is consumed by
            # the next add, so only the final value leaves the configuration.
            # Nine is as long as the chain can be. A pin can be walked through
            # nine switchbox registers, and the tenth operation is ready one
            # cycle after that.
            chain = []
            chain_ids = set()
            started = False
            ready_ids = {id(item) for item in compute}
            for item in remaining:
                if isinstance(item, Issue) and started:
                    break
                if not isinstance(item, AluOp) or address_only(item, set()):
                    continue
                pending = False
                for name in _seq_vals(item)[0]:
                    source = producer.get(name)
                    if source is not None and id(source) in live and id(source) not in chain_ids:
                        pending = True
                        break
                if pending:
                    if started:
                        break
                    continue
                if not started and id(item) not in ready_ids:
                    continue
                started = True
                chain.append(item)
                chain_ids.add(id(item))
                if len(chain) >= 9:
                    break
            groups.extend(_single_output_groups(chain, consumers))
            drop(chain)
            continue

        frontier = [item for item in remaining
                    if isinstance(item, Issue) and is_frontier(item, set())]
        if not frontier:
            raise LowerError(
                "memory operations depend on each other in a cycle, which a "
                "straight-line p-graph sequence cannot order")
        frontier.sort(key=first_user)
        earliest = first_user(frontier[0])
        wave = [item for item in frontier if first_user(item) == earliest]
        wave = wave[:MEM_BUDGET]

        involved = {id(item) for item in wave}
        address_ops = [item for item in remaining if isinstance(item, AluOp)
                       and feeds_issue(item, involved, set())]
        groups.extend(_single_output_groups(address_ops, consumers))
        drop(address_ops)
        for start in range(0, len(wave), MEM_BUDGET):
            groups.append(wave[start:start + MEM_BUDGET])
        drop(wave)
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

    # A square reused later in the same chain looks like two live values until
    # that later use is included. The whole chain still has one result.
    if ops and outside(ops) <= 1:
        return [ops]

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
    if len(groups) > PGRAPH_LIMIT:
        raise LowerError(
            f"kernel needs {len(groups)} p-graphs and the chip holds "
            f"{PGRAPH_LIMIT}")
    gpr_of: dict[str, int] = {}
    # A register is free once the last p-graph that reads its value has run.
    # Addresses die at the memory graph that consumes them, so a later address
    # can reuse the register instead of demanding a ninth one.
    last: dict[str, int] = {}
    for index, group in enumerate(groups):
        for item in group:
            if isinstance(item, AluOp):
                for inp in item.inputs:
                    if inp[0] == "val":
                        last[inp[1]] = index
                last.setdefault(item.name, index)
            else:
                if item.addr[0] == "val":
                    last[item.addr[1]] = index
                if item.kind == "store":
                    last[item.data] = index
                else:
                    last.setdefault(item.data, index)

    free: list[int] = []
    live = 0
    allocated: dict[str, int] = {}

    def assign(name: str) -> int:
        nonlocal live
        if name in gpr_of:
            return gpr_of[name]
        if free:
            reg = free.pop()
        else:
            if live >= GPR_COUNT:
                raise LowerError(
                    f"more than {GPR_COUNT} values cross between p-graphs; "
                    "Mini_Dice has eight general registers and no spill slot")
            reg = live
            live += 1
        gpr_of[name] = reg
        allocated[name] = reg
        return reg

    graphs = []
    for index, group in enumerate(groups):
        graphs.append(_emit(index, group, csr_index, gpr_of, assign,
                            index == len(groups) - 1))
        for name, end in last.items():
            if end == index and name in gpr_of:
                free.append(gpr_of[name])
                del gpr_of[name]
    return Program(lowered, graphs, csr_index, allocated)


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
        if kind == "cta":
            return f"CSRX{csr_index[('param', 'cta_id')]}"
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

