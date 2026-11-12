#include "minidice.cuh"

// c[i] = a[i] * b[i].
extern "C" __global__ void hadamard(const i16 *a, const i16 *b, i16 *c) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    c[i] = (i16)(a[i] * b[i]);
}
