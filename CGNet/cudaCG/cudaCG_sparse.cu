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
#define PLUSMINUS(k) ((k%2==1) ? -1 : 1)
#define LLL2L(lll, L) (lll / (L+1) / (L+1))
#define LLL2L1(lll, L) (lll / (L+1) % (L+1))
#define LLL2L2(lll, L) (lll % (L+1))
#define MAX_LMAX 512-1
#define MAX_LOGFACT_FROM_L(L) (5*(L+1) + 20)
#define MAX_(a,b) ((a<b)?b:a)
#define MIN_(a,b) ((a>b)?b:a)


__constant__ double LogFact_CONST[MAX_LOGFACT_FROM_L(MAX_LMAX)]; //

#define LOGFACT(n,mem) ((n < 2) ? 0. : LogFact_CONST[n])

namespace {
    __device__ __forceinline__ float _naiveCG(
            int l1, int l2, int l, int m1, int m2, int m,
            const double* mem){
        int m3=-m;
        int t1=l2-m1-l;
        int t2=l1+m2-l;
        int t3=l1+l2-l;
        int t4=l1-m1;
        int t5=l2+m2;

        int tmin=max(0,max(t1,t2));
        int tmax=min(t3,min(t4,t5));

        double wigner=0;

        double logA=(log((double)2*l+1)+LOGFACT(l+l1-l2,mem)+LOGFACT(l-l1+l2,mem)+LOGFACT(l1+l2-l,mem)-LOGFACT(l1+l2+l+1,mem))/2;

        logA+=(LOGFACT(l-m3,mem)+LOGFACT(l+m3,mem)+LOGFACT(l1-m1,mem)+LOGFACT(l1+m1,mem)+LOGFACT(l2-m2,mem)+LOGFACT(l2+m2,mem))/2;
        for(int t=tmin; t<=tmax; t++){
            double logB = LOGFACT(t,mem)+LOGFACT(t3-t,mem)+LOGFACT(t4-t,mem)+LOGFACT(t5-t,mem)+LOGFACT(-t1+t,mem)+LOGFACT(-t2+t,mem);
            wigner += PLUSMINUS(t)*exp(logA-logB);
        }

        return (float) PLUSMINUS(l1-l2-m3)*PLUSMINUS(l1-l2+m)*wigner;
    }

    __device__ __forceinline__ float naiveCG_cal_m(
            int l1, int l2, int l, int m1, int m2){
        return _naiveCG(l1, l2, l, m1, m2, m1+m2, NULL);
    }

    __device__ float naiveCG_cal_m1(
            int l1, int l2, int l, int m, int m2){
        return _naiveCG(l1, l2, l, m - m2, m2, m, NULL);
    }
    __global__ void new_precomputeCG_kernel(float* __restrict__ CG, int Lmax,
                                            int* __restrict__ llls, int nllls, int* CG_offsets) {

        const int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        int lllidx = blockIdx.x * blockDim.x + threadIdx.x;
        int l, l1, l2, lll;
        if (lllidx < nllls){
            //compute the l, l1, l2 indices
            lll = llls[lllidx];
            l = LLL2L(lll, Lmax);
            l1 = LLL2L1(lll, Lmax);
            l2 = LLL2L2(lll, Lmax);

            int start = CG_offsets[lllidx];
            for (int m = 0; m < 2 * l +1 ; m++){
                for (int m2 = 0; m2 < 2 * l2 + 1; m2++){
                    int m1 = (m-l) - (m2-l2);
                    if (-l1 <= m1 && m1 <= l1){
                        CG[start + m * (2*l2+1) + m2] = naiveCG_cal_m1(l1,l2,l,m-l,m2-l2);
                    }
                    //start += 1;
                }
            }
        }
    }

//==================================================================================================================

    __global__ void cudaCG_forward_kernel(
            const float*  F,
            float*  FF,
            const int*  t_F,
            const int*  cumu_tm_F,
            const int*  cumu_tm_FF,
            int Lmax,
            const int*  llls,
            const int*  ll1_to_lllidx_offsets, //new
            int nthreads,
            const int*  t_offsets,
            const float*  CG,
            const int*  CG_offsets) {
        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.z;
        int lllidx;
        int l1, l2, t1, t2, CG_offset;//need to be computed from threadid
        int l, t, m;
        int t_offset;
        if (global_threadId < nthreads){
            //compute l, t, m
            l = 0;
            while (cumu_tm_FF[l] <= global_threadId){
                l++;
            }
            l--; //cumu_tm_FF[l] <= global_threadId < cumu_tm_FF[l]
            t = (global_threadId - cumu_tm_FF[l]) / (2 * l + 1);
            m = (global_threadId - cumu_tm_FF[l]) % (2 * l + 1) - l; //[-l, l]

            //compute lllidx
#define LL_to_LLLIDX(l_, l1_) ll1_to_lllidx_offsets[l_ * (Lmax+2) + l1_]
            t_offset = 0;
            l1 = 0;
            while (l1 <= Lmax){
                t_offset = t_offsets[LL_to_LLLIDX(l, l1)];
                if (t_offset > t){
                    l1 -= 1;
                    break;
                } else if (l1 == Lmax) {
                    break;
                }
                l1++;
            }
            //now, t_offsets[LL_to_LLLIDX(l, l1)] <= t < t_offsets[LL_to_LLLIDX(l, l1+1)])
            lllidx = LL_to_LLLIDX(l, l1);
            while (lllidx < LL_to_LLLIDX(l, l1+1)){
                t_offset = t_offsets[lllidx];
                if (t_offset > t){
                    lllidx -= 1;
                    break;
                } else if (lllidx == LL_to_LLLIDX(l, l1+1) - 1){
                    break;
                }
                lllidx ++;
            }
            l2 = LLL2L2(llls[lllidx], Lmax);
#undef LL_to_LLLIDX
            int t_offset = t_offsets[lllidx];
            //now, t_offsets[lllidx] <= t < t_offsets[lllidx+1], so lllidx maps to the l1&l2 that generates t
            t2 = (t - t_offset) % t_F[l2];
            t1 = (t - t_offset) / t_F[l2];

            //compute CG_offset as well
            int CG_offset = CG_offsets[lllidx];

            float real0 = 0., imag0 = 0.;
            for (int m2 = -l2; m2 <= l2; m2++){
                int m1 = m - m2;
                if (-l1 <= m1 && m1 <= l1){
                    float CGcoef = CG[CG_offset + (m+l) * (2*l2+1) + (m2+l2)]; //This is cached (read from global ram)
                    //float CGcoef = naiveCG_cal_m1(l1,l2,l,m,m2);
                    float real1 = F[IDX(b,l1,t1,m1+l1,0,cumu_tm_F,Lmax)];
                    float imag1 = F[IDX(b,l1,t1,m1+l1,1,cumu_tm_F,Lmax)];
                    float real2 = F[IDX(b,l2,t2,m2+l2,0,cumu_tm_F,Lmax)];
                    float imag2 = F[IDX(b,l2,t2,m2+l2,1,cumu_tm_F,Lmax)];
                    real0 += (real1 * real2 - imag1 * imag2) * CGcoef;
                    imag0 += (real1 * imag2 + real2 * imag1) * CGcoef;
                }
            }
            //FF[IDX(b,l,t,m+l,0,cumu_tm_FF,Lmax)] = 1000 * b + 100 * l1 + 10 * t1 + 1 * (m + l);
            //FF[IDX(b,l,t,m+l,1,cumu_tm_FF,Lmax)] = 1000 * b + 100 * l2 + 10 * t2 + 1 * (m + l);
            //FF[IDX(b,l,t,m+l,0,cumu_tm_FF,Lmax)] = 1000 * b + 100 * l + 10 * t + 1 * (m + l); //good
            //FF[IDX(b,l,t,m+l,1,cumu_tm_FF,Lmax)] = l2 + l1 * (Lmax + 1) + l * (Lmax + 1) * (Lmax + 1); //good
            FF[IDX(b,l,t,m+l,0,cumu_tm_FF,Lmax)] = real0;
            FF[IDX(b,l,t,m+l,1,cumu_tm_FF,Lmax)] = imag0;
        }


    }

    __global__ void cudaCG_backward_kernel(
            const float*  F,
            float*  grad_F,
            const float*  grad_FF,
            const int*  t_F,
            const int*  cumu_tm_F,
            const int*  cumu_tm_FF,
            int Lmax,
            const int*  llls,
            int nllls,
            const int*  t_offsets,
            const float*  CG,
            const int*  CG_offsets) {

        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.z;
        int l1, l2, l, m1, m2, m, t1, t2, t;
        int lll;
        int thread_l, thread_t, thread_m;
        float real1, imag1, real2, imag2, real, imag;
        //Here we paralellized over l1, t1 and m1.
        if (global_threadId < cumu_tm_F[Lmax+1]){
            //Compute the l1, m1, t1 for this thread
            thread_l = 0;
            while (cumu_tm_F[thread_l]<=global_threadId) {
                thread_l++;
            }
            thread_l -= 1;
            thread_t = (global_threadId - cumu_tm_F[thread_l]) / (2*thread_l+1);
            thread_m = (global_threadId - cumu_tm_F[thread_l]) % (2*thread_l+1);


            //init the gradients to 0
            real1=imag1=0;
            int old_lllidx1 = -1, old_lllidx2 = -1, lllidx1 = -1, lllidx2 = -1;
            while (old_lllidx1 < nllls || old_lllidx2 < nllls){
                for (int lllidx = MIN_(old_lllidx1, old_lllidx2) + 1; lllidx < nllls; lllidx++){
                    lll = llls[lllidx];
                    if (lllidx1 <= old_lllidx1 && lllidx > old_lllidx1 && LLL2L1(lll, Lmax) == thread_l){
                        lllidx1 = lllidx;
                    }
                    if (lllidx2 <= old_lllidx2 && lllidx > old_lllidx2 && LLL2L2(lll, Lmax) == thread_l){
                        lllidx2 = lllidx;
                    }
                }
                if (lllidx1 > old_lllidx1){
                    lll = llls[lllidx1];
                    l2 = LLL2L2(lll, Lmax);
                    l = LLL2L(lll, Lmax);
                    l1 = thread_l;
                    int CG_offset = CG_offsets[lllidx1];
                    //This is the first case
                    //iterate over l2 and l to compute the gradient
                    int t_offset = t_offsets[lllidx1];
                    for (m2 = 0; m2 < 2*l2+1; m2++){
                        m = thread_m-thread_l + m2-l2 + l;
                        if (0 <= m && m <= 2*l){
                            float CGcoef = CG[CG_offset + (m) * (2*l2+1) + (m2)];
                            //float CGcoef = naiveCG_cal_m1(l1,l2,l,m-l,m2-l2);
                            for (t2 = 0; t2 < t_F[l2]; t2++){
                                t = t_F[l2] * thread_t + t2 + t_offset;
                                real = grad_FF[IDX(b,l,t,m,0,cumu_tm_FF,Lmax)];
                                imag = grad_FF[IDX(b,l,t,m,1,cumu_tm_FF,Lmax)];
                                real2 = F[IDX(b,l2,t2,m2,0,cumu_tm_F,Lmax)];
                                imag2 = F[IDX(b,l2,t2,m2,1,cumu_tm_F,Lmax)];
                                real1 += (real * real2 + imag * imag2) * CGcoef;
                                imag1 += (real2 * imag - real * imag2) * CGcoef;
                                //grad_FF[IDX(b,l,t,m,0,cumu_tm_FF,Lmax)] = 1000 * lllidx1 + 000 * b + 100 * thread_l + 10 * thread_t + (thread_m);
                            }
                        }
                    }
                    old_lllidx1 = lllidx1;
                } else { //stop
                    old_lllidx1 = nllls;
                    lllidx1 = nllls;
                }

                if (lllidx2 > old_lllidx2){
                    //l2 == thread_l
                    lll = llls[lllidx2];
                    l1 = LLL2L1(lll, Lmax);
                    l = LLL2L(lll, Lmax);
                    l2 = thread_l;
                    int CG_offset = CG_offsets[lllidx2];
                    int t_offset = t_offsets[lllidx2];
                    for (m1 = 0; m1 < 2*l1+1; m1++){
                        m = m1-l1 + thread_m-thread_l + l;
                        if (0 <= m && m <= 2 * l){
                            float CGcoef = CG[CG_offset + (m) * (2*thread_l+1) + (thread_m)];
                            //float CGcoef = naiveCG_cal_m1(l1,l2,l,m-l,thread_m-thread_l);
                            for (t1 = 0; t1 < t_F[l1]; t1++){
                                t = t_F[thread_l] * t1 + thread_t + t_offset;
                                real = grad_FF[IDX(b,l,t,m,0,cumu_tm_FF,Lmax)];
                                imag = grad_FF[IDX(b,l,t,m,1,cumu_tm_FF,Lmax)];
                                //This time we need to access l1 t1 and m1
                                real2 = F[IDX(b,l1,t1,m1,0,cumu_tm_F,Lmax)];
                                imag2 = F[IDX(b,l1,t1,m1,1,cumu_tm_F,Lmax)];
                                real1 += (real * real2 + imag * imag2) * CGcoef;
                                imag1 += (real2 * imag - real * imag2) * CGcoef;
                                //grad_FF[IDX(b,l,t,m,1,cumu_tm_FF,Lmax)] = 1000 * lllidx2 + 000 * b + 100 * thread_l + 10 * thread_t + (thread_m);
                            }
                        }
                    }
                    old_lllidx2 = lllidx2;
                } else { //stop
                    lllidx2 = nllls;
                    old_lllidx2 = nllls;
                }
            }
            grad_F[(global_threadId + cumu_tm_F[Lmax+1] * b)*2] = real1;
            grad_F[(global_threadId + cumu_tm_F[Lmax+1] * b)*2+1] = imag1;
        }
    }



} // namespace

void CG_sparse_precompute_cuda(float* CGspace, int L, int* llls, int nllls, int* CG_offsets){
    double *logfact;
    //device ptrs
    int logfact_size = MAX_LOGFACT_FROM_L(L);
    logfact_size = init_logfactorials_new(&logfact, logfact_size);
    cudaMemcpyToSymbol(LogFact_CONST, logfact, logfact_size * sizeof(double ));
    //len(CG_offsets) == nllls+1

    dim3 DimBlock(BLOCK, 1, 1);
    dim3 DimGrid0(CEIL_DIV(nllls, BLOCK), 1, 1);
    new_precomputeCG_kernel<<<DimGrid0, DimBlock>>>(CGspace, L, llls, nllls, CG_offsets);
    cudaDeviceSynchronize();
    free(logfact);
}



void CG_sparse_cuda_forward(
        torch::Tensor input,
        torch::Tensor output,
        int L,
        int B,
        int* d_t_F, int* d_cumu_tm_F,
        int* d_t_FF, int* d_cumu_tm_FF,
        int* llls, int* d_ll1_to_lllidx_offsets, int nthreads,
        int* t_offsets,
        float* CG, int* CG_offsets){

    //auto output = torch::zeros_like(old_cell);
    float* F = input.data<float>();
    float* FF = output.data<float>();

    double *logfact;
    int logfact_size = MAX_LOGFACT_FROM_L(L);
    logfact_size = init_logfactorials_new(&logfact, logfact_size);
    cudaMemcpyToSymbol(LogFact_CONST, logfact, logfact_size * sizeof(double ));

    dim3 DimBlock(BLOCK, 1, 1);
    dim3 DimGrid(CEIL_DIV(nthreads, BLOCK), 1, B);
    cudaCG_forward_kernel<<<DimGrid, DimBlock>>>(F, FF,
                                                  d_t_F, d_cumu_tm_F, d_cumu_tm_FF,
                                                  L, llls, d_ll1_to_lllidx_offsets, nthreads, t_offsets,
                                                  CG, CG_offsets);
    cudaDeviceSynchronize();
    free(logfact);
}

void CG_sparse_cuda_backward(
        torch::Tensor input,
        torch::Tensor grad_in,
        torch::Tensor grad_out,
        //torch::Tensor CG_tensor,
        int L,
        int B,
        int* d_t_F, int* d_cumu_tm_F,
        int* d_t_FF, int* d_cumu_tm_FF,
        int* llls, int nllls, int size,
        int* t_offsets,
        float* CG, int* CG_offsets){

    float* F = input.data<float>();
    float* grad_F = grad_in.data<float>();
    float* grad_O = grad_out.data<float>();

    double *logfact;
    int logfact_size = MAX_LOGFACT_FROM_L(L);
    logfact_size = init_logfactorials_new(&logfact, logfact_size);
    cudaMemcpyToSymbol(LogFact_CONST, logfact, logfact_size * sizeof(double ));
    dim3 DimBlock(BLOCK, 1, 1);
    dim3 DimGrid(CEIL_DIV(size, BLOCK), 1, B);
    cudaCG_backward_kernel<<<DimGrid, DimBlock>>>(F, grad_F, grad_O,
                                                  d_t_F, d_cumu_tm_F, d_cumu_tm_FF,
                                                  L, llls, nllls, t_offsets,
                                                  CG, CG_offsets);
    cudaDeviceSynchronize();
    free(logfact);
}


