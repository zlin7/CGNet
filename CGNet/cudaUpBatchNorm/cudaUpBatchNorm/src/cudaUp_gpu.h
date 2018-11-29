

#ifdef __cplusplus
    extern "C" {
#endif

void cudaUpBatchNorm_gpu_forward_kernel(
    THCState* state,
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
    int update_std
);

void cudaUpBatchNorm_gpu_backward_kernel(
    THCState* state,
    THCudaTensor* weight_tensor,
    THCudaTensor* input_tensor,
    THCudaTensor* grad_in_tensor,
    THCudaTensor* grad_weight_tensor,
    THCudaTensor* grad_middle_tensor,
    THCudaTensor* grad_out_tensor,
    THCudaTensor* CG_tensor,
    int L,
    int B,
    THIntTensor* tauIn_tensor,
    THIntTensor* tauOut_tensor,
    THCudaTensor* moving_std_tensor,
    float eps
);

#ifdef __cplusplus
    }
#endif
