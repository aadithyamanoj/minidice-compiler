#include "minidice.cuh"

// Attention scores for head dimension 2. Each thread is one key. The query is
// two CSRs, the same value for every key. Softmax runs on the host or FPGA,
// which reads `score` and writes the weights this chip is later given.
extern "C" __global__ void attn_score(const i16 *K, i16 *score, i16 q0, i16 q1) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const i16 *key = K + i * 2;
    score[i] = (i16)(key[0] * q0 + key[1] * q1);
}
