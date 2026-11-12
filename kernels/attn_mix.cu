#include "minidice.cuh"

// Attention output for two tokens and 16 channels per CTA. w0 and w1 are the
// softmax weights from the host or FPGA, broadcast to every channel. V is
// packed [cta][token][channel]. Two tokens fit in registers.
extern "C" __global__ void attn_mix(const i16 *V, i16 *out, i16 w0, i16 w1) {
    int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int d = lane & 15;
    int cta = lane >> 4;
    int base = cta * 32 + d;
    out[cta * 16 + d] = (i16)(w0 * V[base] + w1 * V[base + 16]);
}
