# Mini_Dice compiler

Compiles a CUDA kernel and runs it on the Mini_Dice chip in RTL simulation.
This repository is the compiler. Submodules are only the repositories the
flow uses as a whole. Scripts that were a few files inside a larger repo are
copied under `tools/`.

```
git submodule update --init --recursive
source env.sh
python3 minidice_flow.py \
    --cuda path/to/kernel.cu --kernel _Z6euclidP7latLongPfiff \
    --grid 4 --block 16 --workload nn --stem nn --out build/nn
```

`--grid` times `--block` is the number of threads. The chip runs one block of 16
at a time and the flow reprograms the pointer-base CSRs between blocks, so a
larger grid becomes several serial CTAs. The testbench passes when every store
the chip performs matches the host's prediction.

The Rodinia euclid kernel is the worked example. From
`nn_cuda.cu` the command above lowers it, maps it, and the chip simulation
reports `PASS` for all four CTAs.

## Submodules

| Path | Repository | What this flow uses |
| --- | --- | --- |
| `third_party/boxman` | `pncel/boxman` `2729a56` | CUDA frontend. The rest of VEGAS is not used. |
| `third_party/cgra-solve` | `aadithyamanoj/DiceCGRA-Solve` `dice-mapper` `5a1a797` | place and route, the Mini_Dice architecture, and the latency measurement |
| `third_party/mini_dice` | `aadithyamanoj/Mini_Dice` `1d28051` | chip RTL and testbench. Its `dora` submodule is the bitstream generator and the taped-out fabric. |
| `third_party/basejump_stl` | `bespoke-silicon-group/basejump_stl` `uw_ee477_wi23` | libraries the chip filelist compiles |

`git submodule update --init --recursive` checks out DORA inside the chip.
Bitgen uses that same checkout, so the fabric workspace matches the RTL.

`tools/json_to_meta_mem.py`, `tools/bins_to_bitstream_mem.py`, and
`tools/fasm_to_bin.py` are the packers from Mini-Dice-testkernels `26b7b24`.
The rest of that repo is unused.

The LLVM plugin, the VEGAS conda toolchain, and the CGRA-Solve virtualenv are
build products. `env.sh` uses copies already built on this machine when the
submodule trees have not been built, and it says so. Set `VEGAS_TOOLS_ROOT`,
`BOXMAN_BUILD`, or `MINIDICE_MAPPER_PYTHON` to point somewhere else.

Compiled kernels read a data image. The published chip testbench does not.
`patches/mini_dice-data-memory.patch` adds that, and it also skips the FSDB
dump so VCS does not need Verdi. Apply it inside the chip submodule before
simulating:

```
git -C third_party/mini_dice apply ../../patches/mini_dice-data-memory.patch
```

## What the lowering does

Mini_Dice's fabric is a 4x4 grid of 16-bit processing elements with add, sub,
and mul, no per-element immediate, and eight broadcast CSRs. The VEGAS CUDA
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

- `minidice_flow.py` runs the stages: frontend, lower, place and route, bitgen,
  pack, simulate.
- `mdflow/lower.py` turns the dataflow graph into p-graphs.
- `mdflow/workload.py` lays out the data memory and predicts the stores.
- `mdflow/map.py` calls CGRA-Solve. `source env.sh` selects its interpreter,
  and the import path is this repo's submodule.
- `mdflow/frontend.py` calls the Boxman CUDA frontend.
- `tools/` packs metadata and bitstreams. Bitgen runs against the chip's DORA submodule.

The simulation reads a `{stem}_data.mem` image next to the runtime JSON. Kernels
built by this flow ship one; the hand-written vectors in the chip's
`tb/test_vectors` do not, and those still read `addr & 0xFFFF` as before.

The chip testbench is compiled with VCS under the system compiler, because the
compiler in the VEGAS environment rejects VCS's generated C. The build caches
`build/simv` and reuses it.
