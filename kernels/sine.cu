#include "minidice.cuh"

// Sine for an integer argument in -3..3 radians. The output is in units of
// 1/512, so y is round(512*sin(x)). Sine is odd, so the even powers are
// zero and the Horner chain is
//   ((((c5*x)*x + c3)*x)*x + c1)*x
// with c5=3, c3=-81, c1=509. Every product fits in int16. On that domain the
// value differs from round(512*sin(x)) by at most 3.
extern "C" __global__ void sine(const i16 *x, i16 *y, i16 c1, i16 c3, i16 c5) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    i16 v = x[i];
    i16 acc = (i16)(c5 * v);
    acc = (i16)(acc * v);
    acc = (i16)(acc + c3);
    acc = (i16)(acc * v);
    acc = (i16)(acc * v);
    acc = (i16)(acc + c1);
    y[i] = (i16)(acc * v);
}
