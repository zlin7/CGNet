

#ifdef __cplusplus
    extern "C" {
#endif

void cudaCG_gpu_forward_kernel(
    THCState* state,
    THCudaTensor* F_tensor,
    THCudaTensor* output_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor
);

void cudaCG_gpu_backward_kernel(
    THCState* state,
    THCudaTensor* F_tensor,
    THCudaTensor* grad_in_tensor,
    THCudaTensor* grad_out_tensor,
    THCudaTensor* CG_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor
);

#ifdef __cplusplus
    }
#endif
