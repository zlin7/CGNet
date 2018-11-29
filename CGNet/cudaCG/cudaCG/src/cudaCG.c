#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <THC.h>
#include <THCGeneral.h>

#include "cudaCG_gpu.h"
//#include "numpy/arrayobject.h"

extern THCState *state;

int cudaCG_forward(
    THCudaTensor* F_tensor,
    THCudaTensor* output_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor){

    cudaCG_gpu_forward_kernel(
        state,
        F_tensor,
        output_tensor,
        L,
        B,
        taus_tensor
    );
    return 1;
}

int cudaCG_backward(
    THCudaTensor* F_tensor,
    THCudaTensor* grad_in,
    THCudaTensor* grad_out,
    THCudaTensor* CG_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor){

    cudaCG_gpu_backward_kernel(
        state,
        F_tensor,
        grad_in,
        grad_out,
        CG_tensor,
        L,
        B,
        taus_tensor
    );
    return 1;
}
