#include "minidice.cuh"

// The weighted sum, on the fabric. `weight` is the softmax result from the
// host or FPGA. Each thread is one channel and adds the four key terms into
// one accumulator. Four is as many terms as the eight registers and the
// 32 p-graph cap can hold in one kernel.
extern "C" __global__ void attn_apply(const i16 *weight, const i16 *V, i16 *out) {
    int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    i16 acc = (i16)(weight[0] * V[d * 4]);
    acc = (i16)(acc + weight[1] * V[d * 4 + 1]);
    acc = (i16)(acc + weight[2] * V[d * 4 + 2]);
    acc = (i16)(acc + weight[3] * V[d * 4 + 3]);
    out[d] = acc;
}
