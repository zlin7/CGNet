//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <THC.h>
#include <THCGeneral.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cudaCG_gpu.h"
//#include "math_functions.h"

#ifdef __cplusplus
    extern "C" {
#endif

#define NUM_THREADS 512
#define BLOCK 512
#define IDX(b,l,t,m,i,cum,L) (i+2*(m+t*(2*l+1)+cum[l]+b*cum[L+1]))
#define PLUSMINUS(k) ((k%2==1) ? -1 : 1)
#define LOGFACT(n,mem) ((n < 2) ? 0. : mem[n])

int rounded_division(int number1, int number2) {
    if (number1 % number2 == 0) {
        return number1 / number2;
    }
    return number1 / number2 + 1;
}

dim3 cuda_gridsize(int n){
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if (x > 65535){
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    return d;
}


//VERIFIED
/*
__device__ float stupidCG_(
    int l1, int l2, int l, int m1, int m2,
    const double* logfact){
    int m = m1 + m2; //requires the caller to be careful!
    
    int m3=-m;
    int t1=l2-m1-l;
    int t2=l1+m2-l;
    int t3=l1+l2-l;
    int t4=l1-m1;
    int t5=l2+m2;
  
    int tmin=max(0,max(t1,t2));
    int tmax=min(t3,min(t4,t5));

    double wigner=0;

    // for(int t=tmin; t<=tmax; t++)
    //   wigner += plusminus(t)/(fact(t)*fact(t-t1)*fact(t-t2)*fact(t3-t)*fact(t4-t)*fact(t5-t));
    // wigner*=plusminus(l1-l2-m3)*sqrt((double)(fact(l1+l2-l)*fact(l1-l2+l)*fact(-l1+l2+l))/fact(l1+l2+l+1));
    // wigner*=sqrt((double)(fact(l1+m1)*fact(l1-m1)*fact(l2+m2)*fact(l2-m2)*fact(l+m3)*fact(l-m3)));

    //add a log(2*l+1) to logA
    double logA=(log((double)2*l+1)+logfact[l+l1-l2]+logfact[l-l1+l2]+logfact[l1+l2-l]-logfact[l1+l2+l+1])/2;
    logA+=(logfact[l-m3]+logfact[l+m3]+logfact[l1-m1]+logfact[l1+m1]+logfact[l2-m2]+logfact[l2+m2])/2;
    for(int t=tmin; t<=tmax; t++){
        // cout<<t<<endl;
        //double logB=logfact(t)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t)+logfact(t-t1)+logfact(t-t2);
        double logB = logfact[t]+logfact[t3-t]+logfact[t4-t]+logfact[t5-t]+logfact[-t1+t]+logfact[-t2+t];
        wigner += PLUSMINUS(t)*exp(logA-logB);
    }

    return (float) PLUSMINUS(l1-l2-m3)*PLUSMINUS(l1-l2+m)*wigner; 
}
*/

__device__ float stupidCG(
    int l1, int l2, int l, int m1, int m2,
    const double* mem){
    int m = m1 + m2; //requires the caller to be careful!
    
    int m3=-m;
    int t1=l2-m1-l;
    int t2=l1+m2-l;
    int t3=l1+l2-l;
    int t4=l1-m1;
    int t5=l2+m2;
  
    int tmin=max(0,max(t1,t2));
    int tmax=min(t3,min(t4,t5));

    double wigner=0;

    //double logA=(log((double)2*l+1)+logfact[l+l1-l2]    +logfact[l-l1+l2]    +logfact[l1+l2-l]    -logfact[l1+l2+l+1])/2;
    double logA=(log((double)2*l+1)+LOGFACT(l+l1-l2,mem)+LOGFACT(l-l1+l2,mem)+LOGFACT(l1+l2-l,mem)-LOGFACT(l1+l2+l+1,mem))/2;

    logA+=(LOGFACT(l-m3,mem)+LOGFACT(l+m3,mem)+LOGFACT(l1-m1,mem)+LOGFACT(l1+m1,mem)+LOGFACT(l2-m2,mem)+LOGFACT(l2+m2,mem))/2;
    //logA+=(logfact[l-m3]    +logfact[l+m3]    +logfact[l1-m1]    +logfact[l1+m1]    +logfact[l2-m2]    +logfact[l2+m2])/2;
    for(int t=tmin; t<=tmax; t++){
        double logB = LOGFACT(t,mem)+LOGFACT(t3-t,mem)+LOGFACT(t4-t,mem)+LOGFACT(t5-t,mem)+LOGFACT(-t1+t,mem)+LOGFACT(-t2+t,mem);
        //double logB = logfact[t]    +logfact[t3-t]    +logfact[t4-t]    +logfact[t5-t]    +logfact[-t1+t]    +logfact[-t2+t];
        wigner += PLUSMINUS(t)*exp(logA-logB);
    }

    return (float) PLUSMINUS(l1-l2-m3)*PLUSMINUS(l1-l2+m)*wigner; 
}

__device__ float stupidCGcal(
    int l1, int l2, int l, int m, int m2,
    const double* mem){
    //int m = m1+m2;
    int m1 = m - m2; //requires the caller to be careful!
    
    int m3=-m;
    int t1=l2-m1-l;
    int t2=l1+m2-l;
    int t3=l1+l2-l;
    int t4=l1-m1;
    int t5=l2+m2;
  
    int tmin=max(0,max(t1,t2));
    int tmax=min(t3,min(t4,t5));

    double wigner=0;

    //double logA=(log((double)2*l+1)+logfact[l+l1-l2]    +logfact[l-l1+l2]    +logfact[l1+l2-l]    -logfact[l1+l2+l+1])/2;
    double logA=(log((double)2*l+1)+LOGFACT(l+l1-l2,mem)+LOGFACT(l-l1+l2,mem)+LOGFACT(l1+l2-l,mem)-LOGFACT(l1+l2+l+1,mem))/2;

    logA+=(LOGFACT(l-m3,mem)+LOGFACT(l+m3,mem)+LOGFACT(l1-m1,mem)+LOGFACT(l1+m1,mem)+LOGFACT(l2-m2,mem)+LOGFACT(l2+m2,mem))/2;
    //logA+=(logfact[l-m3]    +logfact[l+m3]    +logfact[l1-m1]    +logfact[l1+m1]    +logfact[l2-m2]    +logfact[l2+m2])/2;
    for(int t=tmin; t<=tmax; t++){
        double logB = LOGFACT(t,mem)+LOGFACT(t3-t,mem)+LOGFACT(t4-t,mem)+LOGFACT(t5-t,mem)+LOGFACT(-t1+t,mem)+LOGFACT(-t2+t,mem);
        //double logB = logfact[t]    +logfact[t3-t]    +logfact[t4-t]    +logfact[t5-t]    +logfact[-t1+t]    +logfact[-t2+t];
        wigner += PLUSMINUS(t)*exp(logA-logB);
    }

    return (float) PLUSMINUS(l1-l2-m3)*PLUSMINUS(l1-l2+m)*wigner; 
}

__global__ void cudaprecomputeCG_job(
        float* CG, 
        const double* logfact,
        int Lmax,
        int Batch_size) {

    int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int L1 = (Lmax + 1);
    int L2 = L1*L1, L3=L1*L1*L1;
    if (global_threadId < L3*(2*Lmax+1)){
        int m2 = global_threadId % (2*Lmax+1);
        int l_remainder = global_threadId / (2*Lmax+1);
        int l1 = l_remainder / L2;
        int l2 = (l_remainder / L1) % L1;
        int l = l_remainder % L1;
        if (l2 <= l1 && l1-l2 <= l && l <= l1+l2 && m2 < 2*l2+1){
            int start = 0;
            for (int templ1=0; templ1 <= l1; templ1++){
                for (int templ2=0; (templ2<l2 && templ1==l1) || (templ2<=templ1 && templ1<l1);templ2++){
                    int low = templ1-templ2, high=(templ2+templ1 > Lmax) ? Lmax : templ2+templ1;
                    for (int templ=low; templ<=high ; templ++){
                        start += (2*templ2+1)*(2*templ+1);
                    }
                }
            }
            for (int templ = l1-l2; templ<l; templ++){
                start += (2*l2+1)*(templ*2+1);
            }
            //offset m2
            start += m2*(2*l+1);
            for (int m = 0; m < 2*l+1;m++){
                int m1 = (m-l) - (m2-l2);
                if (-l1 <= m1 && m1 <= l1){
                    CG[start + m] = stupidCGcal(l1,l2,l,m-l,m2-l2,logfact);
                    //CG[start + m] = 100*l1 + 10*l2 + l + 0.1*(m1+l1) + 0.01*m2 + 0.001*m;
                }
            }
        }
    } 
}

__global__ void cudaCG_forward_job(
        float* tensor, 
        float* out_tensor, 
        const int* taus, 
        const int* cum_tauIn_m,
        const int* cum_tauMiddle_m,
        const double* logfact,
        int Lmax,
        int Batch_size) {
    
    int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int L1 = (Lmax+1);
    int Entry_size = L1 * (Lmax+2) * (Lmax +1) / 2;
    if (global_threadId < Batch_size * Entry_size){
        int b = global_threadId / Entry_size;
        int l = global_threadId % L1;
        int remainder_for_l = (global_threadId % Entry_size) / L1;
        int l1 = 0, l2 = remainder_for_l * 2;
        while (l1*(l1+1) <= l2){l1++;}
        l1 -= 1;
        l2 = (l2 - l1*(l1+1))/2;
        
        if (l2 <= l1 && l1 - l2 <= l && l <= l1 + l2){
            int t_offset = 0;
            for (int templ1 = 0; templ1<l1; templ1++){
                for (int templ2 = 0; templ2<=templ1; templ2++){
                    if (l <= templ2 + templ1 && l >= templ1- templ2){
                        t_offset += taus[templ1]*taus[templ2];  
                    }
                }
            }
            for (int templ2 = 0; templ2<=l2; templ2++){
                if (l <= templ2 + l1 && l >= l1- templ2){
                    t_offset += taus[l1]*taus[templ2];  
                }
            }
            t_offset -= taus[l1]*taus[l2];

            for (int m1 = -l1; m1 <= l1; m1++){
                for (int m2 = -l2; m2 <= l2; m2++){
                    int m = m1 + m2;
                    if (-l <= m && m <= l){
                        float CGcoef = stupidCG(l1,l2,l,m1,m2,logfact);
                        for (int t1 = 0; t1 < taus[l1]; t1++){
                            for (int t2 = 0; t2 < taus[l2]; t2++){
                                int t = t1 * taus[l2] + t2 + t_offset;
                                float real1 = tensor[IDX(b,l1,t1,m1+l1,0,cum_tauIn_m,Lmax)];
                                float imag1 = tensor[IDX(b,l1,t1,m1+l1,1,cum_tauIn_m,Lmax)];
                                float real2 = tensor[IDX(b,l2,t2,m2+l2,0,cum_tauIn_m,Lmax)];
                                float imag2 = tensor[IDX(b,l2,t2,m2+l2,1,cum_tauIn_m,Lmax)];
                                out_tensor[IDX(b,l,t,m+l,0,cum_tauMiddle_m,Lmax)] += (real1 * real2 - imag1 * imag2) * CGcoef;
                                out_tensor[IDX(b,l,t,m+l,1,cum_tauMiddle_m,Lmax)] += (real1 * imag2 + real2 * imag1) * CGcoef;
                                //out_tensor[IDX(b,l,t,m+l,0,cum_tauMiddle_m,Lmax)] = t + 0.01 * t_offset;
                                //out_tensor[IDX(b,l,t,m+l,1,cum_tauMiddle_m,Lmax)] = m+l+0.1 * l1 + 0.01 * l2 + 0.001*l;
                                //return;
                                
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ void cudaCG_backward_job(
        float* tensor, 
        float* g_in, 
        float* g_out, 
        const int* taus, 
        const int* cum_taus,
        const int* cum_new_taus,
        const float* CG,
        int Lmax,
        int Batch_size) {

    int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_threadId < Batch_size * cum_taus[Lmax+1]){
        int b = global_threadId / cum_taus[Lmax + 1];
        int ltm1 = global_threadId % cum_taus[Lmax + 1];
        int l1 = 0;
        while (cum_taus[l1]<=ltm1) {
            l1++;
        }
        l1 -= 1;
        int tm1 = ltm1 - cum_taus[l1];
        int t1 = tm1 / (2*l1+1);
        int m1 = tm1 % (2*l1+1);

        //m1 -= l1;
        int l2 = 0, m2 = 0, t2 = 0;

        float real1=0, imag1=0;
        for (l2 = 0; l2 <= l1; l2++){
            for (int l = l1 - l2; l <= Lmax && l <= l1 + l2; l++){
                int CG_offset=0, t_offset=0; 
                for (int templ1=0; templ1 <= l1; templ1++){
                    for (int templ2=0; (templ2<l2 && templ1==l1) || (templ2<=templ1 && templ1<l1);templ2++){
                        int low = templ1-templ2, high=(templ2+templ1 > Lmax) ? Lmax : templ2+templ1;
                        for (int templ=low; templ<=high ; templ++){
                            CG_offset += (2*templ2+1)*(2*templ+1);
                        }
                        if (l <= templ1 + templ2 && l >= templ1 - templ2){
                            t_offset += taus[templ1]*taus[templ2]; 
                        }
                    }
                }
                for (int templ = l1-l2; templ<l; templ++){
                    CG_offset += (2*l2+1)*(templ*2+1);
                }
                
                for (m2 = 0; m2 < 2*l2+1; m2++){
                    for (int m = 0; m < 2*l+1; m++){
                        if (m1-l1 + m2-l2 == m-l){
                            float CGcoef = CG[CG_offset+(2*l+1)*m2+m];
                            for (t2 = 0; t2 < taus[l2]; t2++){
                                int t = taus[l2] * t1 + t2 + t_offset;
                                float real = g_out[IDX(b,l,t,m,0,cum_new_taus,Lmax)];
                                float imag = g_out[IDX(b,l,t,m,1,cum_new_taus,Lmax)];
                                float real2 = tensor[IDX(b,l2,t2,m2,0,cum_taus,Lmax)];
                                float imag2 = tensor[IDX(b,l2,t2,m2,1,cum_taus,Lmax)];
                                real1 += (real * real2 + imag * imag2) * CGcoef;
                                imag1 += (real2 * imag - real * imag2) * CGcoef;
                            }
                        }
                    }
                }
            }
        }

        //Now switching to treat l1 as a "l2"
        l2 = l1;
        t2 = t1;
        m2 = m1;

        for (l1 = l2; l1 <= Lmax; l1++){
            for (int l = l1 - l2; l <= Lmax && l <= l1 + l2; l++){
                int CG_offset=0, t_offset=0; 
                for (int templ1=0; templ1 <= l1; templ1++){
                    for (int templ2=0; (templ2<l2 && templ1==l1) || (templ2<=templ1 && templ1<l1);templ2++){
                        int low = templ1-templ2, high=(templ2+templ1 > Lmax) ? Lmax : templ2+templ1;
                        for (int templ=low; templ<=high ; templ++){
                            CG_offset += (2*templ2+1)*(2*templ+1);
                        }
                        if (l <= templ1 + templ2 && l >= templ1 - templ2){
                            t_offset += taus[templ1]*taus[templ2]; 
                        }
                    }
                }
                for (int templ = l1-l2; templ<l; templ++){
                    CG_offset += (2*l2+1)*(templ*2+1);
                }

                for (m1 = 0; m1 < 2*l1+1; m1++){
                    for (int m = 0; m < 2*l+1; m++){
                        if (m1-l1 + m2-l2 == m-l){
                            float CGcoef = CG[CG_offset+(2*l+1)*m2+m];
                            for (t1 = 0; t1 < taus[l1]; t1++){
                                int t = taus[l2] * t1 + t2 + t_offset;
                                float real = g_out[IDX(b,l,t,m,0,cum_new_taus,Lmax)];
                                float imag = g_out[IDX(b,l,t,m,1,cum_new_taus,Lmax)];
                                //This time we need to access l1 t1 and m1
                                float real2 = tensor[IDX(b,l1,t1,m1,0,cum_taus,Lmax)];
                                float imag2 = tensor[IDX(b,l1,t1,m1,1,cum_taus,Lmax)];
                                real1 += (real * real2 + imag * imag2) * CGcoef;
                                imag1 += (real2 * imag - real * imag2) * CGcoef;
                            }
                        }
                    }
                }
            }
        }
        g_in[global_threadId*2] = real1;
        g_in[global_threadId*2+1] = imag1;
    }
}

    

void print_arr(int* v, int l){
    printf("vector: (");
    for (int i = 0; i < l; i++){
        printf("%d, ", v[i]);
    }
    printf(")\n");
    return;
}

int* _get_cum_tau(int* taus, int L){
    int* cum_tau = (int*) malloc((L+2)*sizeof(int));
    cum_tau[0] = 0;
    for (int l = 0; l <= L; l++){
        cum_tau[l+1] = cum_tau[l] + (2 * l + 1) * taus[l];
    }
    return cum_tau;
}

void cudaCG_gpu_forward_kernel(
    THCState* state,
    THCudaTensor* F_tensor,
    THCudaTensor* output_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor
){
    cudaStream_t stream = THCState_getCurrentStream(state);

    float* F = THCudaTensor_data(state, F_tensor);
    float* out = THCudaTensor_data(state, output_tensor);

    int* taus = THIntTensor_data(taus_tensor);
    int* new_taus = (int*) calloc(L+1, sizeof(int));
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2; l <=L && l <= l1 + l2; l++){
                new_taus[l] += taus[l1] * taus[l2];
            }
        }
    }
    int* cum_tauIn_m = _get_cum_tau(taus, L);
    int* cum_tauMiddle_m = _get_cum_tau(new_taus, L);
    //Compute taus such that taus[l] is the starting position of the tensor

    //int size = B * (L+1) * (L+1) * (L+1);
    int size = B * (L+1) * (L+2) * (L+1) /2;
    //printf("Need %d threads\n", size);

    //Prep for CG
    int LOGFACT_SIZE=5*L+20;
    double* logfact = (double*) calloc(LOGFACT_SIZE, sizeof(double));
    for (int i = 2; i < LOGFACT_SIZE; i++){
        logfact[i] = logfact[i-1] + log((double) i);
    }
    double* cuda_logfact;
    cudaMalloc((void**) &cuda_logfact, LOGFACT_SIZE*sizeof(double)); 
    cudaMemcpy(cuda_logfact, logfact, LOGFACT_SIZE*sizeof(double), cudaMemcpyHostToDevice);

    int *cuda_tauIn, *cuda_cum_tauIn_m, *cuda_cum_tauMiddle_m;

    cudaMalloc((void**) &cuda_tauIn, (L+1)*sizeof(int)); 
    cudaMemcpy(cuda_tauIn, taus, (L+1)*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &cuda_cum_tauIn_m, (L+2)*sizeof(int)); 
    cudaMemcpy(cuda_cum_tauIn_m, cum_tauIn_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &cuda_cum_tauMiddle_m, (L+2)*sizeof(int)); 
    cudaMemcpy(cuda_cum_tauMiddle_m, cum_tauMiddle_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);

    cudaCG_forward_job<<<cuda_gridsize(size), BLOCK, 0, stream>>>(
            F, out,
            cuda_tauIn, cuda_cum_tauIn_m, cuda_cum_tauMiddle_m,
            cuda_logfact, L, B);
}


void cudaCG_gpu_backward_kernel(
    THCState* state,
    THCudaTensor* F_tensor,
    THCudaTensor* grad_in_tensor,
    THCudaTensor* grad_out_tensor,
    THCudaTensor* CG_tensor,
    int L,
    int B,
    THIntTensor* taus_tensor
){
    cudaStream_t stream = THCState_getCurrentStream(state);

    float* F = THCudaTensor_data(state, F_tensor);
    float* g_in = THCudaTensor_data(state, grad_in_tensor);
    float* g_out = THCudaTensor_data(state, grad_out_tensor);
    int* taus = THIntTensor_data(taus_tensor);    
    int* new_taus = (int*) calloc(L+1, sizeof(int));
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2; l <=L && l <= l1 + l2; l++){
                new_taus[l] += taus[l1] * taus[l2];
            }
        }
    }
    int* cum_tauIn_m = _get_cum_tau(taus, L);
    int* cum_tauMiddle_m = _get_cum_tau(new_taus, L);
    int *cuda_tauIn, *cuda_cum_tauIn_m, *cuda_cum_tauMiddle_m;
    cudaMalloc((void**) &cuda_tauIn, (L+1)*sizeof(int)); 
    cudaMalloc((void**) &cuda_cum_tauIn_m, (L+2)*sizeof(int)); 
    cudaMalloc((void**) &cuda_cum_tauMiddle_m, (L+2)*sizeof(int)); 
    cudaMemcpy(cuda_tauIn, taus, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauIn_m, cum_tauIn_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauMiddle_m, cum_tauMiddle_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    //Compute taus such that taus[l] is the starting position of the tensor

    int size = B * cum_tauIn_m[L+1];
    //printf("Need %d threads\n", size);

    //Prep for CG
    int LOGFACT_SIZE=5*L+20;
    double* logfact = (double*) calloc(LOGFACT_SIZE, sizeof(double));
    for (int i = 2; i < LOGFACT_SIZE; i++){
        logfact[i] = logfact[i-1] + log((double) i);
    }
    double* cuda_logfact;
    cudaMalloc((void**) &cuda_logfact, LOGFACT_SIZE*sizeof(double)); 
    cudaMemcpy(cuda_logfact, logfact, LOGFACT_SIZE*sizeof(double), cudaMemcpyHostToDevice);

    float* CG = THCudaTensor_data(state, CG_tensor);
    int size0 = (L+1)*(L+1)*(L+1)*(2*L+1);
    cudaprecomputeCG_job<<<cuda_gridsize(size0), BLOCK, 0, stream>>>(
            CG,
            cuda_logfact, L, B);
    cudaThreadSynchronize();


    cudaCG_backward_job<<<cuda_gridsize(size), BLOCK, 0, stream>>>(
            F, g_in, g_out, 
            cuda_tauIn, cuda_cum_tauIn_m, cuda_cum_tauMiddle_m, 
            CG, L, B);
    cudaThreadSynchronize();



    cudaFree(cuda_logfact);
    cudaFree(cuda_tauIn);
    cudaFree(cuda_cum_tauIn_m);
    cudaFree(cuda_cum_tauMiddle_m);

    free(logfact);
    free(cum_tauIn_m);
    free(cum_tauMiddle_m);
}

#ifdef __cplusplus
    }
#endif
