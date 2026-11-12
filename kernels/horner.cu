#include "minidice.cuh"

// Horner evaluation:
// y = ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
// The coefficients are CSRs. The chain through the last multiply is one
// p-graph. The final add is the next one: a pin can be delayed nine cycles,
// and that add's other input is ready on cycle ten.
extern "C" __global__ void horner(const i16 *x, i16 *y,
                                  i16 c0, i16 c1, i16 c2, i16 c3, i16 c4, i16 c5) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    i16 v = x[i];
    i16 acc = (i16)(c5 * v + c4);
    acc = (i16)(acc * v + c3);
    acc = (i16)(acc * v + c2);
    acc = (i16)(acc * v + c1);
    acc = (i16)(acc * v + c0);
    y[i] = acc;
}
