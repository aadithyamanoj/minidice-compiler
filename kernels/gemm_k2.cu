#include "minidice.cuh"

// One CTA is one row of a 16-column matrix. The inner product is unrolled to
// K=2 because a loop cannot be mapped and eight registers cannot hold a
// longer product. The column is the local thread and the row is the CTA.
extern "C" __global__ void gemm_k2(const i16 *A, const i16 *B, i16 *C) {
    int lane = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int col = lane & 15;
    int row = lane >> 4;
    i16 acc = (i16)(A[row * 2] * B[col]);
    acc = (i16)(acc + A[row * 2 + 1] * B[16 + col]);
    C[row * 16 + col] = acc;
}
