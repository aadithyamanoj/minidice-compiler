"""Host-side inputs, the data memory image, and the expected chip writes.

The interpreter reruns the lowered p-graph program in software, with 16-bit
two's-complement wrap at every step, which is what the fabric ALU does. It
walks the global thread id and the global array bases; on the chip the host
folds the per-CTA part of that base into the CSR and the fabric only adds
the local thread id, and the two produce the same address.
"""
from __future__ import annotations

from .lower import AluOp, MemOp


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


def _value(operand, env: dict, inputs: dict, thread: int) -> int:
    kind = operand[0]
    if kind == "val":
        return env[operand[1]]
    if kind == "param":
        return inputs[operand[1]] & 0xFFFF
    if kind == "const":
        return operand[1] & 0xFFFF
    if kind == "tid":
        return thread & 0xFFFF
    if kind == "recip":
        return inputs[f"recip_{operand[1][1]}"] & 0xFFFF
    raise KeyError(kind)


def assemble(program, inputs: dict) -> dict:
    """Lay the input arrays out in word memory and predict every store.

    Returns the data image, the expected writes, the launch CSR values, and
    the per-CTA overrides for the pointer bases.
    """
    lowered = program.lowered
    nthreads = lowered.grid * lowered.block

    base_of = {}
    cursor = 0
    for name, _stride in lowered.bases:
        words = inputs[name]
        if not isinstance(words, list):
            raise ValueError(f"pointer parameter {name} needs an array of words")
        base_of[name] = cursor
        cursor += len(words)

    image = {}
    for name, _stride in lowered.bases:
        for offset, word in enumerate(inputs[name]):
            image[base_of[name] + offset] = word & 0xFFFF

    writes = []
    for thread in range(nthreads):
        env = {}
        for op in lowered.ops:
            if isinstance(op, MemOp):
                addr = base_of[op.base] + thread * op.stride + op.offset
                if op.kind == "load":
                    env[op.name] = image.get(addr, 0)
                else:
                    value = env[op.name] & 0xFFFF
                    image[addr] = value
                    writes.append({"addr": addr, "data": value, "strb": 3})
            elif isinstance(op, AluOp):
                left = _value(op.inputs[0], env, inputs, thread)
                right = _value(op.inputs[1], env, inputs, thread)
                env[op.name] = {
                    "add": left + right,
                    "sub": left - right,
                    "mul": left * right,
                }[op.op] & 0xFFFF

    # The image the chip reads is the memory before any store. Rebuild it from
    # the inputs alone; the stores are checked through expected_writes.
    initial = {}
    for name, _stride in lowered.bases:
        for offset, word in enumerate(inputs[name]):
            initial[base_of[name] + offset] = word & 0xFFFF

    csr_values = {f"csrX{i}": 0 for i in range(8)}
    per_cta = [{} for _ in range(lowered.grid)]
    for key, index in program.csr_index.items():
        kind, what = key
        if kind == "param" and what in base_of:
            stride = dict(lowered.bases)[what]
            for cta in range(lowered.grid):
                per_cta[cta][index] = base_of[what] + cta * lowered.block * stride
            csr_values[f"csrX{index}"] = per_cta[0][index]
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

