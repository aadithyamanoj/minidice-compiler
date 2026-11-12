# Mini_Dice compiler

Compiles a CUDA kernel into the memory images the Mini_Dice chip loads.
This repository does not simulate the chip. Submodules are the public
repositories the flow uses as a whole. Place and route is Boxman's DICE
mapper, given a Mini_Dice architecture built from `cgra-solve-local`. That
directory is a copy of the private CGRA-Solve tree's routing graph and
feature catalog, not a mapper. Scripts and the config header are copies too.

```
git submodule update --init
source env.sh
python3 minidice_flow.py \
    --cuda path/to/kernel.cu --kernel _Z6euclidP7latLongPfiff \
    --grid 4 --block 16 --workload nn --stem nn --out build/nn
```

`--grid` times `--block` is the number of lanes. The chip runs 16 threads
at a time. A wider block is split into CTAs of 16 and the grid grows so the
same lanes still run; the host changes the pointer-base CSRs between those
CTAs. One pointer can use several of the eight base CSRs when it is indexed
more than one way. In a 16-wide tile the column is the local thread id and
the row is the CTA index. For a stem `nn` the output directory holds:

- `nn_cta_desc.mem`, `nn_meta.mem`, `nn_bitstream.mem`
- `nn_runtime.json` (CSR values, per-CTA pointer bases, expected stores)
- `nn_data.mem` (initial data memory)

## Submodules

| Path | Repository | What this flow uses |
| --- | --- | --- |
| `third_party/boxman` | `pncel/boxman` `2729a56` | CUDA frontend, and the DICE place-and-route mapper (`map_program`). |
| `third_party/dora` | `pncel/dora` `f455a179` | bitstream generation and the taped-out fabric workspace |

`cgra-solve-local` is not the mapper. It is the Mini_Dice architecture that
gets passed into Boxman's mapper: the routing graph and the DORA feature
catalog from `DiceCGRA-Solve` `dice-mapper` `5a1a797`. The mapper itself is
unchanged VEGAS code. It places each p-graph onto that fabric.

`tools/json_to_meta_mem.py`, `tools/bins_to_bitstream_mem.py`, and
`tools/fasm_to_bin.py` are the packers from Mini-Dice-testkernels `26b7b24`.
`collateral/dice_config.vh` is the chip header those packers read, from
Mini_Dice `1d28051`. The rest of the chip, the testbench, and BaseJump STL
are not used.

The LLVM plugin and the VEGAS conda toolchain are build products. `env.sh`
uses copies already built on this machine when the submodule trees have not
been built, and it says so. Set `VEGAS_TOOLS_ROOT` or `BOXMAN_BUILD` to point
somewhere else.

## What the lowering does

Mini_Dice's fabric is a 4x4 grid of 16-bit processing elements with add, sub,
and mul, no per-element immediate, and eight broadcast CSRs. The Boxman CUDA
frontend produces the dataflow graph; this flow retargets it:

- Float arithmetic runs as INT16 on values the host has already quantized, and
  wraps modulo 65536. The `nn` workload keeps every coordinate inside -10..10,
  so the squared distance fits exactly and dropping the square root still
  ranks the nearest neighbor.
- A kernel parameter or a literal arrives in a CSR. Pointer bases occupy the
  low CSRs and are the ones rewritten per CTA; the thread id the fabric sees
  is the local one, and the base absorbs the block offset.
- A bounds check that merely keeps the thread id inside the launch is removed.
  The launch has to cover exactly the threads that check would have kept.
- Each p-graph drives one result. An arithmetic graph writes a single register,
  and a memory graph only routes registers to memory ports, so every output is
  captured on the same cycle. The writeback latency is the switch-box depth the
  mapper measures. Up to four memory operations share one graph; the two loads
  of one record do.
- An operation the fabric cannot do — a variable divide, a shift, a signed
  compare used as data, a loop that survived unrolling — stops the compile and
  names the operation.

## Layout

- `minidice_flow.py` runs the stages: frontend, lower, place and route, bitgen, pack.
- `mdflow/lower.py` turns the dataflow graph into p-graphs.
- `mdflow/workload.py` lays out the data memory and predicts the stores.
- `mdflow/vegas_map.py` turns the Mini_Dice routing graph into the architecture
  document Boxman's mapper accepts, then places each p-graph and writes FASM.
- `mdflow/frontend.py` calls the Boxman CUDA frontend.
- `tools/` packs metadata and bitstreams. Bitgen runs against the DORA submodule.
