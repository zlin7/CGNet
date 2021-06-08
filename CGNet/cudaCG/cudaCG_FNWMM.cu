#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cudaCG_all.h"

//#define NUM_THREADS 512
#define BLOCK 512
#define CEIL_DIV(num, denum) (num+denum-1)/denum
#define IDX(b,l,t,m,i,cum,L) (i+2*(m+t*(2*l+1)+cum[l]+b*cum[L+1])) //cum[l] remembers the start of the middle channel for l (in (?, tm, 2))
#define WIDX(l,tOut,tMid,i,cum,tauMids) (i+2*(tMid+tauMids[l]*tOut+cum[l]))
#define PLUSMINUS(k) ((k%2==1) ? -1 : 1)
#define MAX_LMAX 512-1
#define MAX_(a,b) ((a<b)?b:a)
#define MIN_(a,b) ((a>b)?b:a)

__constant__ int RESERVED_T[MAX_LMAX+1]; //
__constant__ int RESERVED_CUMU_TM[MAX_LMAX+2]; //

namespace {
//==================================================================================================================
    __global__ void cuda_FN_forward_job(  //from cudaBatchNorm_forward_job
            const float* FF,
            float* moving_std,
            const int* t_FF,
            const int* cumu_t_FF,
            float cnt,
            float eps,
            int Lmax,
            int Batch_size,
            int update_std,
            int nthreads){
        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_threadId < nthreads){
            int t_offset = 0, l=0;
            while(l<=Lmax){
                t_offset+=t_FF[l];
                if (t_offset <= global_threadId){
                    l++;
                } else {
                    t_offset -= t_FF[l];
                    break;
                }
            }

            int tmid = global_threadId - t_offset;

            if (update_std){
                //calculate mean

                double N = (double) Batch_size * (2*l+1);
                double mean = 0., mean_sq = 0., norm = 0.;
                for (int b = 0; b < Batch_size; b++){
                    for (int m = 0; m < 2*l+1; m++){
                        float realm = FF[IDX(b,l,tmid,m,0,cumu_t_FF,Lmax)];
                        float imagm = FF[IDX(b,l,tmid,m,1,cumu_t_FF,Lmax)];
                        norm = realm*realm+imagm*imagm;
                        mean_sq += norm;
                        mean += sqrt(norm);
                    }
                }
                double std = mean_sq/N - (mean/N * mean/N);
                if (std <= 0){ //numerical stability
                    std = 0.;
                } else{
                    std = sqrt(std);
                }
                moving_std[t_offset + tmid] *= cnt / (cnt + 1);
                moving_std[t_offset + tmid] += std / (cnt + 1);
            }
        }
    }

    __global__ void cudaWeightTransform_forward_job(
            const float* FF,
            const float* W,
            float* out,
            const int* t_FF,
            const int* cumu_tm_FF,
            const int* cumu_tm_O,
            const int* cumu_tt_W,
            const float* FN_stds, float eps,
            int Lmax) {
        int b = blockIdx.z;
        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_threadId < cumu_tm_O[Lmax+1]){
            //first, loop to get l
            int ltm = global_threadId;
            int l=0;
            while (cumu_tm_O[l] <= ltm){l++;}
            l--;
            int tmid_offset = 0;
            for (int templ = 0; templ< l; templ++){
                tmid_offset += t_FF[templ];
            }
            int tout = (ltm - cumu_tm_O[l]) / (2*l+1);
            int m = (ltm - cumu_tm_O[l]) % (2*l+1);

            float real=0.0, imag=0.0, divisor = 1.0;
            for (int tmid = 0; tmid < t_FF[l]; tmid++){
                float realw = W[WIDX(l,tout,tmid,0,cumu_tt_W,t_FF)];
                float imagw = W[WIDX(l,tout,tmid,1,cumu_tt_W,t_FF)];
                float realm = FF[IDX(b,l,tmid,m,0,cumu_tm_FF,Lmax)];
                float imagm = FF[IDX(b,l,tmid,m,1,cumu_tm_FF,Lmax)];
                if (FN_stds){
                    divisor = MAX_(eps, FN_stds[tmid_offset+tmid]);
                }
                real += (realw * realm - imagw * imagm) / divisor;
                imag += (realw * imagm + imagw * realm) / divisor;
            }

            out[IDX(b,l,tout,m,0,cumu_tm_O,Lmax)] = real;
            out[IDX(b,l,tout,m,1,cumu_tm_O,Lmax)] = imag;
        }

    }


    __global__ void cudaWeightGrad1_backward_job(
            const float* FF,
            float* __restrict__ grad_W,
            const float* grad_out,
            const int* t_FF,
            const int* cumu_tm_FF,
            const int* cumu_tm_O,
            const int* cumu_tt_W,
            const float* FN_stds, float eps,
            int Lmax,
            int size_W){

        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.z;
        if (global_threadId < size_W){
            int l=0;
            while (cumu_tt_W[l] <= global_threadId){l++;}
            l--;
            int tmid_offset = 0;
            for (int templ = 0; templ< l; templ++){
                tmid_offset += t_FF[templ];
            }
            int tout = (global_threadId - cumu_tt_W[l]) / t_FF[l];
            int tmid = (global_threadId - cumu_tt_W[l]) % t_FF[l];

            float real=0.0, imag=0.0, divisor = 1.0;
            if (FN_stds){
                divisor = MAX_(eps, FN_stds[tmid_offset+tmid]);
            }
            for (int m = 0; m < 2*l+1; m++){
                float realo = grad_out[IDX(b,l,tout,m,0,cumu_tm_O,Lmax)];
                float imago = grad_out[IDX(b,l,tout,m,1,cumu_tm_O,Lmax)];
                float realm = FF[IDX(b,l,tmid,m,0,cumu_tm_FF,Lmax)];
                float imagm = FF[IDX(b,l,tmid,m,1,cumu_tm_FF,Lmax)];
                real += (realm * realo + imagm * imago)/divisor;
                imag += (realm * imago - realo * imagm)/divisor;
            }
            atomicAdd(&(grad_W[WIDX(l,tout,tmid,0,cumu_tt_W,t_FF)]), real);
            atomicAdd(&(grad_W[WIDX(l,tout,tmid,1,cumu_tt_W,t_FF)]), imag);
        }

    }

    __global__ void cudaMiddleGrad_backward_job(
            float* grad_FF,
            const float* W,
            const float* grad_out,
            const float* moving_std,
            const int* t_FF,
            const int* cumu_tm_FF,
            const int* t_O,
            const int* cumu_tm_O,
            const int* cumu_tt_W,
            const float* FN_stds, float eps,
            int Lmax){

        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.z;
        if (global_threadId < cumu_tm_FF[Lmax+1]){
            int l=0;
            while (cumu_tm_FF[l] <= global_threadId){l++;}
            l--;
            int tmid_offset = 0;
            for (int templ = 0; templ< l; templ++){
                tmid_offset += t_FF[templ];
            }

            int tm = global_threadId - cumu_tm_FF[l];
            int tmid = tm / (2*l+1), m = tm % (2*l+1);
            float real=0.0, imag=0.0, divisor = 1.0;
            if (FN_stds){
                divisor = MAX_(eps, FN_stds[tmid_offset+tmid]);
            }
            for (int tout = 0; tout < t_O[l]; tout++){
                float realo = grad_out[IDX(b,l,tout,m,0,cumu_tm_O,Lmax)];
                float imago = grad_out[IDX(b,l,tout,m,1,cumu_tm_O,Lmax)];
                float realw = W[WIDX(l,tout,tmid,0,cumu_tt_W,t_FF)];
                float imagw = W[WIDX(l,tout,tmid,1,cumu_tt_W,t_FF)];
                real += realw * realo + imagw * imago;
                imag += realw * imago - realo * imagw;
            }
            grad_FF[IDX(b,l,tmid,m,0,cumu_tm_FF,Lmax)] = real / divisor;
            grad_FF[IDX(b,l,tmid,m,1,cumu_tm_FF,Lmax)] = imag / divisor;
            //}
        }
    }
} // namespace


void FN_WMM_forward_cuda(torch::Tensor FF_tensor,
                 torch::Tensor O_tensor,
                 torch::Tensor W_tensor,
                 int L, int B, int size_FN, int size_WMM,
                 int* d_t_FF, int* d_cumu_tm_FF, int* d_cumu_tm_O, int* d_cumu_tt_W,
                 float* FN_stds, float FN_cnt, float FN_eps, int FN_flags){
    float* FF = FF_tensor.data<float>();
    float* O = O_tensor.data<float>();
    float* W = W_tensor.data<float>();

    dim3 DimBlock(BLOCK, 1, 1);
    int update_std = (FN_flags & 0x2) ? 1 : 0;

    if (FN_stds && update_std){
        dim3 DimGrid2(CEIL_DIV(size_FN, BLOCK), 1, 1);
        cuda_FN_forward_job<<<DimGrid2, DimBlock>>>(FF, FN_stds,
                                                    d_t_FF, d_cumu_tm_FF,
                                                    FN_cnt, FN_eps,
                                                    L, B, update_std, size_FN);

        cudaDeviceSynchronize();
    }


    dim3 DimGrid(CEIL_DIV(size_WMM, BLOCK), 1, B);
    cudaWeightTransform_forward_job<<<DimGrid, DimBlock>>>(FF, W, O,
                                                           d_t_FF, d_cumu_tm_FF,
                                                           d_cumu_tm_O, d_cumu_tt_W,
                                                           FN_stds, FN_eps, L);
    cudaDeviceSynchronize();
}




void FN_WMM_backward_cuda(torch::Tensor grad_out_tensor,
                          torch::Tensor grad_FF_tensor, torch::Tensor grad_W_tensor,
                          //inputs
                          torch::Tensor FF_tensor, torch::Tensor W_tensor,
                          int L, int B, int size_FF, int size_W,
                          int* d_t_FF, int* d_cumu_tm_FF, int* d_t_O, int* d_cumu_tm_O, int* d_cumu_tt_W,
                          float* FN_stds, float FN_cnt, float FN_eps, int FN_flags){
    float* FF = FF_tensor.data<float>();
    float* W = W_tensor.data<float>();
    float* grad_FF = grad_FF_tensor.data<float>();
    float* grad_out = grad_out_tensor.data<float>();
    float* grad_W = grad_W_tensor.data<float>();

    dim3 DimBlock(BLOCK, 1, 1);

    dim3 DimGrid(CEIL_DIV(size_W, BLOCK), 1, B);
    cudaWeightGrad1_backward_job<<<DimGrid, DimBlock>>>(
            FF, grad_W, grad_out,
            d_t_FF, d_cumu_tm_FF,
            d_cumu_tm_O, d_cumu_tt_W,
            FN_stds, FN_eps, L, size_W);
    cudaDeviceSynchronize();


    dim3 DimGrid0(CEIL_DIV(size_FF, BLOCK), 1, B);
    cudaMiddleGrad_backward_job<<<DimGrid0, DimBlock>>>(grad_FF, W, grad_out, FN_stds,
                                                        d_t_FF, d_cumu_tm_FF,
                                                        d_t_O, d_cumu_tm_O, d_cumu_tt_W,
                                                        FN_stds, FN_eps,
                                                        L);
    cudaThreadSynchronize();

}
