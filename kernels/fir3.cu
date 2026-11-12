#include "minidice.cuh"

// 3-tap convolution. Thread i writes y[i] from x[i .. i+2]. The launch is the
// outputs only, so there is no edge test; x holds two extra samples. The taps
// are CSRs.
extern "C" __global__ void fir3(const i16 *x, i16 *y, i16 w0, i16 w1, i16 w2) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    y[i] = (i16)(w0 * x[i] + w1 * x[i + 1] + w2 * x[i + 2]);
}
