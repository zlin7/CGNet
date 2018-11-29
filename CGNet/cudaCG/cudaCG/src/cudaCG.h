
int cudaCG_forward(
    THCudaTensor* F_tensor,
    THCudaTensor* output_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor);

int cudaCG_backward(
    THCudaTensor* F_tensor,
    THCudaTensor* grad_in,
    THCudaTensor* grad_out,
    THCudaTensor* CG_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor);