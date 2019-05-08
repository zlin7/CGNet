#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void UpBatchNorm_forward_cuda(
    torch::Tensor input_tensor,
    torch::Tensor middle_tensor,
    torch::Tensor output_tensor,
    torch::Tensor weight_tensor,
    int L,
    int B,
    torch::Tensor tauIn_tensor,
    torch::Tensor tauOut_tensor,
    torch::Tensor moving_std_tensor,
    float cnt,
    float eps,
    int update_std);

void UpBatchNorm_backward_cuda(
    torch::Tensor weight_tensor,
    torch::Tensor input_tensor,
    torch::Tensor grad_in_tensor,
    torch::Tensor grad_weight_tensor,
    torch::Tensor grad_middle_tensor,
    torch::Tensor grad_out_tensor,
    torch::Tensor CG_tensor,
    int L,
    int B,
    torch::Tensor tauIn_tensor,
    torch::Tensor tauOut_tensor,
    torch::Tensor moving_std_tensor,
    float eps);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int UpBatchNorm_forward(
    torch::Tensor input_tensor,
    torch::Tensor middle_tensor,
    torch::Tensor output_tensor,
    torch::Tensor weight_tensor,
    int L,
    int B,
    torch::Tensor tauIn_tensor,
    torch::Tensor tauOut_tensor,
    torch::Tensor moving_std_tensor,
    float cnt,
    float eps,
    int update_std) {
    CHECK_INPUT(input_tensor);
    CHECK_INPUT(middle_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(weight_tensor);
    CHECK_CONTIGUOUS(tauIn_tensor);
    CHECK_CONTIGUOUS(tauOut_tensor);
    CHECK_INPUT(moving_std_tensor);
    UpBatchNorm_forward_cuda(input_tensor, middle_tensor, output_tensor, weight_tensor,
                              L, B, tauIn_tensor, tauOut_tensor,
                              moving_std_tensor, cnt, eps, update_std);
    return 0;
}

int UpBatchNorm_backward(
    torch::Tensor weight_tensor,
    torch::Tensor input_tensor,
    torch::Tensor grad_in_tensor,
    torch::Tensor grad_weight_tensor,
    torch::Tensor grad_middle_tensor,
    torch::Tensor grad_out_tensor,
    torch::Tensor CG_tensor,
    int L,
    int B,
    torch::Tensor tauIn_tensor,
    torch::Tensor tauOut_tensor,
    torch::Tensor moving_std_tensor,
    float eps) {
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT(input_tensor);
    CHECK_INPUT(grad_in_tensor);
    CHECK_INPUT(grad_weight_tensor);
    CHECK_INPUT(grad_middle_tensor);
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(CG_tensor);
    CHECK_CONTIGUOUS(tauIn_tensor);
    CHECK_CONTIGUOUS(tauOut_tensor);
    CHECK_INPUT(moving_std_tensor);

    UpBatchNorm_backward_cuda(weight_tensor, input_tensor, grad_in_tensor,
                              grad_weight_tensor, grad_middle_tensor, grad_out_tensor,
                              CG_tensor, L, B, tauIn_tensor, tauOut_tensor,
                              moving_std_tensor, eps);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &UpBatchNorm_forward, "CG + Batch Normalization forward (CUDA)");
  m.def("backward", &UpBatchNorm_backward, "CG + Batch Normalization backward (CUDA)");
}
