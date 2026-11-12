"""Host-side inputs, the data memory image, and the expected chip writes.

The interpreter reruns the lowered p-graph program in software, with 16-bit
two's-complement wrap at every step, which is what the fabric ALU does. It
walks the global thread id and the global array bases; on the chip the host
folds the per-CTA part of that base into the CSR and the fabric only adds
the local thread id, and the two produce the same address.
"""
from __future__ import annotations

from .lower import AluOp, MemOp


def _words(n: int, salt: int) -> list:
    """Small signed samples so products of a few of them stay inside int16."""
    return [((i * salt) % 21) - 10 for i in range(n)]


def saxpy_inputs(nthreads: int) -> dict:
    return {"x": _words(nthreads, 3), "y": _words(nthreads, 5), "a": 3}


def hadamard_inputs(nthreads: int) -> dict:
    return {"a": _words(nthreads, 3), "b": _words(nthreads, 5), "c": [0] * nthreads}


def fir3_inputs(nthreads: int) -> dict:
    # Thread i reads x[i], x[i+1], and x[i+2], so the input is two longer.
    return {"x": _words(nthreads + 2, 3), "y": [0] * nthreads,
            "w0": 1, "w1": 2, "w2": 1}


def gemm_k2_inputs(nthreads: int) -> dict:
    """One CTA is one row. A is rows*K, B is K*16, C is rows*16, K=2."""
    rows = nthreads // 16
    return {"A": _words(rows * 2, 3), "B": _words(32, 5), "C": [0] * nthreads}


def sine_inputs(nthreads: int) -> dict:
    """Arguments are radians in -3..3, the domain of the sine polynomial.

    c5, c3 and c1 are the integer coefficients of 3x^5 - 81x^3 + 509x, whose
    value is within 3 counts of round(512*sin(x)) on that domain.
    """
    return {
        "x": [(index % 7) - 3 for index in range(nthreads)],
        "y": [0] * nthreads,
        "c1": 509,
        "c3": -81,
        "c5": 3,
    }


def horner_inputs(nthreads: int) -> dict:
    """Degree-5 Horner. x is one word per thread. c0..c5 are broadcast CSRs."""
    return {
        "x": [(index % 5) - 2 for index in range(nthreads)],
        "y": [0] * nthreads,
        "c0": 1,
        "c1": -1,
        "c2": 2,
        "c3": 0,
        "c4": 1,
        "c5": -2,
    }


def attn_score_inputs(nthreads: int) -> dict:
    """Key dimension 2. q0 and q1 are the query, broadcast through CSRs."""
    return {"K": _words(nthreads * 2, 3), "score": [0] * nthreads, "q0": 2, "q1": -1}


def attn_mix_inputs(nthreads: int) -> dict:
    """Two tokens, 16 channels per CTA. w0 and w1 stand in for host softmax."""
    return {"V": _words(nthreads * 2, 3), "out": [0] * nthreads, "w0": 1, "w1": 1}


def attn_apply_inputs(nthreads: int) -> dict:
    """Four keys, one channel per thread. V is laid out [channel][key]."""
    keys = 4
    return {
        "weight": _words(keys, 3),
        "V": _words(nthreads * keys, 5),
        "out": [0] * nthreads,
    }


def nn_inputs(nthreads: int) -> dict:
    """Quantized inputs for the euclid kernel.

    Coordinates stay inside -10..10 so every difference squared, and the sum
    of the two squares, fits in a signed 16-bit value. The chip result is then
    the exact squared distance, which is what dropping the square root
    preserves the nearest-neighbor ranking of.
    """
    locations = []
    for record in range(nthreads):
        locations.append((record * 3) % 21 - 10)
        locations.append((record * 5) % 21 - 10)
    return {
        "d_locations": locations,
        "d_distances": [0] * nthreads,
        "lat": 2,
        "lng": -3,
    }


def _value(operand, env: dict, inputs: dict, thread: int, block: int) -> int:
    kind = operand[0]
    if kind == "val":
        return env[operand[1]]
    if kind == "param":
        return inputs[operand[1]] & 0xFFFF
    if kind == "const":
        return operand[1] & 0xFFFF
    if kind == "tid":
        return thread % block & 0xFFFF
    if kind == "cta":
        return thread // block & 0xFFFF
    if kind == "recip":
        return inputs[f"recip_{operand[1][1]}"] & 0xFFFF
    raise KeyError(kind)


def _host(expr, inputs: dict, cta: int) -> int:
    """Words the host adds to a pointer base for one CTA."""
    if expr[0] == "int":
        return expr[1]
    if expr[0] == "cta":
        return cta
    if expr[0] == "param":
        return int(inputs[expr[1]])
    if expr[0] == "add":
        return _host(expr[1], inputs, cta) + _host(expr[2], inputs, cta)
    if expr[0] == "mul":
        return _host(expr[1], inputs, cta) * _host(expr[2], inputs, cta)
    raise KeyError(expr[0])


def assemble(program, inputs: dict) -> dict:
    """Lay the input arrays out in word memory and predict every store.

    Returns the data image, the expected writes, the launch CSR values, and
    the per-CTA overrides for the pointer bases. Several base CSRs may point
    into the same array; the array is placed once.
    """
    lowered = program.lowered
    nthreads = lowered.grid * lowered.block

    base_of = {}
    cursor = 0
    for _slot, name, _stride, _host_expr in lowered.bases:
        if name in base_of:
            continue
        words = inputs[name]
        if not isinstance(words, list):
            raise ValueError(f"pointer parameter {name} needs an array of words")
        base_of[name] = cursor
        cursor += len(words)

    def paint(image: dict) -> None:
        seen = set()
        for _slot, name, _stride, _host_expr in lowered.bases:
            if name in seen:
                continue
            seen.add(name)
            for offset, word in enumerate(inputs[name]):
                image[base_of[name] + offset] = word & 0xFFFF

    image = {}
    paint(image)

    writes = []
    for thread in range(nthreads):
        env = {}
        local = thread % lowered.block
        cta = thread // lowered.block
        for op in lowered.ops:
            if isinstance(op, MemOp):
                addr = (base_of[op.base] + _host(op.host, inputs, cta)
                        + local * op.stride + op.offset)
                if op.kind == "load":
                    env[op.name] = image.get(addr, 0)
                else:
                    value = env[op.name] & 0xFFFF
                    image[addr] = value
                    writes.append({"addr": addr, "data": value, "strb": 3})
            elif isinstance(op, AluOp):
                left = _value(op.inputs[0], env, inputs, thread, lowered.block)
                right = _value(op.inputs[1], env, inputs, thread, lowered.block)
                env[op.name] = {
                    "add": left + right,
                    "sub": left - right,
                    "mul": left * right,
                }[op.op] & 0xFFFF

    # The image the chip reads is the memory before any store. Rebuild it from
    # the inputs alone; the stores are checked through expected_writes.
    initial = {}
    paint(initial)

    csr_values = {f"csrX{i}": 0 for i in range(8)}
    per_cta = [{} for _ in range(lowered.grid)]
    layout = {slot: (pointer, host)
              for slot, pointer, _stride, host in lowered.bases}
    for key, index in program.csr_index.items():
        kind, what = key
        if kind == "param" and what in layout:
            pointer, host = layout[what]
            for cta in range(lowered.grid):
                per_cta[cta][index] = base_of[pointer] + _host(host, inputs, cta)
            csr_values[f"csrX{index}"] = per_cta[0][index]
        elif kind == "param" and what == "cta_id":
            for cta in range(lowered.grid):
                per_cta[cta][index] = cta
            csr_values[f"csrX{index}"] = 0
        elif kind == "param":
            csr_values[f"csrX{index}"] = inputs[what] & 0xFFFF
        else:
            csr_values[f"csrX{index}"] = what & 0xFFFF

    return {
        "image": initial,
        "expected_writes": writes,
        "csr_values": csr_values,
        "per_cta": per_cta,
        "base_of": base_of,
    }

