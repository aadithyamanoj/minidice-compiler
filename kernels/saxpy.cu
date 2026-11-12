#include "minidice.cuh"

// y[i] = a * x[i] + y[i]. `a` is one CSR.
extern "C" __global__ void saxpy(const i16 *x, i16 *y, i16 a) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    y[i] = (i16)(a * x[i] + y[i]);
}
