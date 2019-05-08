#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void CG_cuda_forward(
    torch::Tensor input,
    torch::Tensor output,
    int L,
    int B,
    torch::Tensor taus_tensor);

void CG_cuda_backward(
    torch::Tensor input,
    torch::Tensor grad_in,
    torch::Tensor grad_out,
    torch::Tensor CG_tensor,
    int L,
    int B,
    torch::Tensor taus_tensor);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int CG_forward(
    torch::Tensor input,
    torch::Tensor output,
    int L,
    int B,
    torch::Tensor taus) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_CONTIGUOUS(taus);

    CG_cuda_forward(input, output, L, B, taus);
    return 0;
}

int CG_backward(
    torch::Tensor input,
    torch::Tensor grad_in,
    torch::Tensor grad_out,
    torch::Tensor CG_tensor,
    int L,
    int B,
    torch::Tensor taus) {
    CHECK_INPUT(input);
    CHECK_INPUT(grad_in);
    CHECK_INPUT(grad_out);
    CHECK_INPUT(CG_tensor);
    CHECK_CONTIGUOUS(taus);

    //int L = taus.size(0);
    //int B = input.size(0);
    CG_cuda_backward(input, grad_in, grad_out, CG_tensor, L, B, taus);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &CG_forward, "CG forward (CUDA)");
  m.def("backward", &CG_backward, "CG backward (CUDA)");
}
