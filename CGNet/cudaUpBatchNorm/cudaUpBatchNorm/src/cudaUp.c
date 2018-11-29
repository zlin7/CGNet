#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <THC.h>
#include <THCGeneral.h>

#include <time.h>
#include "cudaUp_gpu.h"
//#include "numpy/arrayobject.h"

extern THCState *state;

int cudaUpBatchNorm_forward(
    THCudaTensor* F_tensor,
    THCudaTensor* middle_tensor,
    THCudaTensor* output_tensor,
    THCudaTensor* weight_tensor,
    int L,
    int B,
    THIntTensor* tauIn_tensor,
    THIntTensor* tauOut_tensor,
    THCudaTensor* moving_std_tensor,
    float cnt,
    float eps,
    int update_std){
    //printf("%s update_std %d\n", __func__, update_std);
    cudaUpBatchNorm_gpu_forward_kernel(
        state,
        F_tensor,
        middle_tensor,
        output_tensor,
        weight_tensor,
        L,
        B,
        tauIn_tensor,
        tauOut_tensor,
        moving_std_tensor,
        cnt,
        eps,
        update_std
    );
    return 1;
}

int cudaUpBatchNorm_backward(
    THCudaTensor* weight_tensor,
    THCudaTensor* input_tensor,
    THCudaTensor* grad_in,
    THCudaTensor* grad_weight,
    THCudaTensor* grad_middle,
    THCudaTensor* grad_out,
    THCudaTensor* CG,
    int L,
    int B,
    THIntTensor* tauIn_tensor,
    THIntTensor* tauOut_tensor,
    THCudaTensor* moving_std_tensor,
    float cnt,
    float eps){

    struct timeval t1, t2;
    double elapsedTime;

    gettimeofday(&t1, NULL);
    cudaUpBatchNorm_gpu_backward_kernel(
        state,
        weight_tensor,
        input_tensor,
        grad_in,
        grad_weight,
        grad_middle,
        grad_out,
        CG,
        L,
        B,
        tauIn_tensor,
        tauOut_tensor,
        moving_std_tensor,
        eps
    );
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Backward took %lf mili-seconds\n", elapsedTime);
    return 1;
}
