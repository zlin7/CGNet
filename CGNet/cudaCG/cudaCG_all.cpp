#include <torch/extension.h>

#include <vector>
//========================================= CUDA forward declarations
#define MAX_LMAX 512-1

void CG_sparse_cuda_forward(
        torch::Tensor input,
        torch::Tensor output,
        int L,
        int B,
        //torch::Tensor taus_tensor,
        int* d_t_F, int* d_cumu_tm_F,
        int* d_t_FF, int* d_cumu_tm_FF,
        int* llls, int* d_ll1_to_lllidx_offsets, int nthreads,
        int* t_offsets,
        float* CG, int* CG_offsets);

void CG_sparse_cuda_backward(
        torch::Tensor input,
        torch::Tensor grad_in,
        torch::Tensor grad_out,
        int L,
        int B,
        int* d_t_F, int* d_cumu_tm_F,
        int* d_t_FF, int* d_cumu_tm_FF,
        int* llls, int nllls, int size,
        int* t_offsets,
        float* CG, int* CG_offsets);

//========================================= C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int CG_sparse_forward(
        torch::Tensor input,
        torch::Tensor output,
        int L,
        int B,
        torch::Tensor t_F, torch::Tensor cumu_tm_F,
        torch::Tensor t_FF, torch::Tensor cumu_tm_FF,
        torch::Tensor llls,
        torch::Tensor ll1_to_lllidx,
        torch::Tensor t_offsets,
        torch::Tensor CG,
        torch::Tensor CG_offsets) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(t_F);
    CHECK_INPUT(cumu_tm_F);
    CHECK_INPUT(t_FF);
    CHECK_INPUT(cumu_tm_FF);
    CHECK_INPUT(llls);
    CHECK_INPUT(ll1_to_lllidx);
    CHECK_INPUT(t_offsets);
    CHECK_INPUT(CG);
    CHECK_INPUT(CG_offsets);

    CG_sparse_cuda_forward(input, output, L, B,
                           t_F.data_ptr<int>(), cumu_tm_F.data_ptr<int>(),
                           t_FF.data_ptr<int>(), cumu_tm_FF.data_ptr<int>(),
                           //llls.data_ptr<int>(), ll1_to_lllidx.data_ptr<int>(), llls.size(0),
                           llls.data_ptr<int>(), ll1_to_lllidx.data_ptr<int>(), output.size(1),
                           t_offsets.data_ptr<int>(),
                           CG.data_ptr<float>(), CG_offsets.data_ptr<int>());
    return 0;
}

int CG_sparse_backward(
        torch::Tensor input,
        torch::Tensor grad_in,
        torch::Tensor grad_out,
        //torch::Tensor CG_tensor,
        int L,
        int B,
        torch::Tensor t_F, torch::Tensor cumu_tm_F,
        torch::Tensor t_FF, torch::Tensor cumu_tm_FF,
        torch::Tensor llls,
        torch::Tensor t_offsets,
        torch::Tensor CG,
        torch::Tensor CG_offsets) {
    CHECK_INPUT(input);
    CHECK_INPUT(grad_in);
    CHECK_INPUT(grad_out);
    //CHECK_INPUT(CG_tensor);
    CHECK_INPUT(t_F);
    CHECK_INPUT(cumu_tm_F);
    CHECK_INPUT(t_FF);
    CHECK_INPUT(cumu_tm_FF);
    CHECK_INPUT(llls);
    CHECK_INPUT(t_offsets);
    CHECK_INPUT(CG);
    CHECK_INPUT(CG_offsets);

    //int L = taus.size(0);
    //int B = input.size(0);
    CG_sparse_cuda_backward(input, grad_in, grad_out, L, B,
                            t_F.data_ptr<int>(), cumu_tm_F.data_ptr<int>(),
                            t_FF.data_ptr<int>(), cumu_tm_FF.data_ptr<int>(),
                            llls.data_ptr<int>(), llls.size(0), grad_in.size(1),
                            t_offsets.data_ptr<int>(),
                            CG.data_ptr<float>(), CG_offsets.data_ptr<int>());
    return 0;
}

void CG_sparse_precompute_cuda(float* CGspace, int L, int* llls, int nllls, int* CG_offsets);

int CG_sparse_precompute(torch::Tensor CG_tensor, int L, torch::Tensor llls, torch::Tensor CG_offsets) {
    CHECK_INPUT(CG_tensor);
    CHECK_CONTIGUOUS(llls);
    CHECK_CONTIGUOUS(CG_offsets);
    CG_sparse_precompute_cuda(CG_tensor.data_ptr<float>(), L,
                              llls.data_ptr<int>(), llls.size(0),
                              CG_offsets.data_ptr<int>()
                              );
    return 0;
}


int sparse_maxL() {
    return MAX_LMAX;
}

//===============================================Non-CG stuff
void FN_WMM_forward_cuda(torch::Tensor FF_tensor,
                 torch::Tensor O_tensor,
                 torch::Tensor W_tensor,
                 int L, int B, int size_FN, int size_WMM,
                 int* d_t_FF, int* d_cumu_tm_FF, int* d_cumu_tm_O, int* d_cumu_tt_W,
                 float* FN_stds, float FN_cnt, float FN_eps, int FN_flags
                 );

int FN_WMM_forward(torch::Tensor FF_tensor,
           torch::Tensor output_tensor,
           torch::Tensor W_tensor,
           int L, int B, int size_FN, int size_WMM,
           torch::Tensor t_FF, torch::Tensor cumu_tm_FF,
           torch::Tensor cumu_tm_O, torch::Tensor cumu_tt_W,
           //FN related things
           torch::Tensor FN_stds_tensor, float FN_cnt, float FN_eps, int FN_flags){
    CHECK_INPUT(FF_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(W_tensor);
    CHECK_INPUT(t_FF);
    CHECK_INPUT(cumu_tm_FF);
    CHECK_INPUT(cumu_tm_O);
    CHECK_INPUT(cumu_tt_W);

    float* FN_stds = NULL;
    if (FN_flags & 0x1){
        CHECK_INPUT(FN_stds_tensor);
        FN_stds = FN_stds_tensor.data_ptr<float>();
    }

    FN_WMM_forward_cuda(FF_tensor, output_tensor, W_tensor,
                L, B, size_FN, size_WMM,
                t_FF.data_ptr<int>(), cumu_tm_FF.data_ptr<int>(),
                cumu_tm_O.data_ptr<int>(), cumu_tt_W.data_ptr<int>(),
                FN_stds, FN_cnt, FN_eps, FN_flags);
    return 0;
}
void FN_WMM_backward_cuda(torch::Tensor grad_out_tensor,
                          torch::Tensor grad_FF_tensor, torch::Tensor grad_W_tensor,
                          torch::Tensor FF_tensor, torch::Tensor W_tensor,
                          int L, int B, int size_FF, int size_W,
                          int* d_t_FF, int* d_cumu_tm_FF, int* d_t_O, int* d_cumu_tm_O, int* d_cumu_tt_W,
                          float* FN_stds, float FN_cnt, float FN_eps, int FN_flags);
int FN_WMM_backward(torch::Tensor grad_out_tensor,
                    torch::Tensor grad_FF_tensor, torch::Tensor grad_W_tensor,
                    torch::Tensor FF_tensor, torch::Tensor W_tensor,
                    int L, int B, int size_FF, int size_W,
                    torch::Tensor t_FF, torch::Tensor cumu_tm_FF,
                    torch::Tensor t_O, torch::Tensor cumu_tm_O, torch::Tensor cumu_tt_W,
                    //FN related things
                    torch::Tensor FN_stds_tensor, float FN_cnt, float FN_eps, int FN_flags){
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(grad_FF_tensor);
    CHECK_INPUT(grad_W_tensor);
    CHECK_INPUT(FF_tensor);
    CHECK_INPUT(W_tensor);
    CHECK_INPUT(t_FF);
    CHECK_INPUT(cumu_tm_FF);
    CHECK_INPUT(t_O);
    CHECK_INPUT(cumu_tm_O);
    CHECK_INPUT(cumu_tt_W);

    float* FN_stds = NULL;
    if (FN_flags & 0x1){
        CHECK_INPUT(FN_stds_tensor);
        FN_stds = FN_stds_tensor.data_ptr<float>();
    }

    FN_WMM_backward_cuda(grad_out_tensor,
                         grad_FF_tensor, grad_W_tensor,
                         FF_tensor, W_tensor,
                         L, B, size_FF, size_W,
                         t_FF.data_ptr<int>(), cumu_tm_FF.data_ptr<int>(),
                         t_O.data_ptr<int>(), cumu_tm_O.data_ptr<int>(), cumu_tt_W.data_ptr<int>(),
                                 FN_stds, FN_cnt, FN_eps, FN_flags);
    return 0;
}
//===================================================================Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_precomputeCG", &CG_sparse_precompute, "precompute CG");
  m.def("sparse_forward", &CG_sparse_forward, "CG forward (CUDA) sparse");
  m.def("sparse_backward", &CG_sparse_backward, "CG backward (CUDA) sparse");
  m.def("sparse_maxL", &sparse_maxL, "max maxL supported by CG backward (CUDA) sparse");

  m.def("FN_WMM_forward", &FN_WMM_forward, "fragment-normalization forward");
  m.def("FN_WMM_backward", &FN_WMM_backward, "fragment-normalization backward");
}
