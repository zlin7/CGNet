#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define NUM_THREADS 512
#define BLOCK 512
#define IDX(b,l,t,m,i,cum,L) (i+2*(m+t*(2*l+1)+cum[l]+b*cum[L+1]))
#define PLUSMINUS(k) ((k%2==1) ? -1 : 1)
#define LOGFACT(n,mem) ((n < 2) ? 0. : mem[n])
#define WIDX(l,tOut,tMid,i,cum,tauMids) (i+2*(tMid+tauMids[l]*tOut+cum[l]))

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
    int l1, int l2, int l, int m1, int m2,
    const double* mem){
    return _naiveCG(l1, l2, l, m1, m2, m1+m2, mem);
}

__device__ float naiveCG_cal_m1(
    int l1, int l2, int l, int m, int m2,
    const double* mem){
    return _naiveCG(l1, l2, l, m - m2, m2, m, mem);
}

__global__ void cudaprecomputeCG_job(
        float* __restrict__ CG,
        const double* __restrict__ logfact,
        int Lmax,
        int Batch_size) {

    const int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int L1 = (Lmax + 1);
    const int L2 = L1*L1, L3=L1*L1*L1;
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
                    CG[start + m] = naiveCG_cal_m1(l1,l2,l,m-l,m2-l2,logfact);
                    //CG[start + m] = 100*l1 + 10*l2 + l + 0.1*(m1+l1) + 0.01*m2 + 0.001*m;
                }
            }
        }
    }
}

//==================================================================================================================
__global__ void cudaCG_forward_kernel(
        const float* tensor,
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
                        float CGcoef = naiveCG_cal_m(l1,l2,l,m1,m2,logfact);
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

__global__ void cudaBatchNorm_forward_job(
        float* mid_tensor,
        float* moving_std,
        const int* tauMiddle,
        const int* cum_tauMiddle_m,
        float cnt,
        float eps,
        int Lmax,
        int Batch_size,
        int update_std){
    int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int t_offset = 0, l=0;
    while(l<=Lmax){
        t_offset+=tauMiddle[l];
        if (t_offset <= global_threadId){
            l++;
        } else {
            t_offset -= tauMiddle[l];
            break;
        }
    }
    if (l <= Lmax){
        int tmid = global_threadId - t_offset;

        if (update_std){
            //calculate mean
            double N = (double) Batch_size * (2*l+1);
            double mean = 0.;
            for (int b = 0; b < Batch_size; b++){
                for (int m = 0; m < 2*l+1; m++){
                    float realm = mid_tensor[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)];
                    float imagm = mid_tensor[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)];
                    double norm = sqrt(realm*realm+imagm*imagm);
                    mean += norm / N;
                    //moving_std[t_offset + tmid] = norm;
                    //return;
                }
            }
            //calculate std
            double std = 0.;
            for (int b = 0; b < Batch_size; b++){
                for (int m = 0; m < 2*l+1; m++){
                    float realm = mid_tensor[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)];
                    float imagm = mid_tensor[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)];
                    double norm = sqrt(realm*realm+imagm*imagm);
                    std += (norm - mean) * (norm - mean) / N;
                }
            }

            std = sqrt(std);
            //update std
            //moving_std[t_offset + tmid] = std;
            moving_std[t_offset + tmid] *=  cnt / (cnt + 1);
            moving_std[t_offset + tmid] += std / (cnt + 1);
        }

        //actually performing batch norm. Note eval mode only has this, not update in as in the previous code
        double divisor = (eps > moving_std[t_offset + tmid]) ? eps : moving_std[t_offset + tmid];
        for (int b = 0; b < Batch_size; b++){
            for (int m = 0; m < 2*l+1; m++){
                mid_tensor[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)] /= divisor;
                mid_tensor[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)] /= divisor;
            }
        }
    }

}

__global__ void cudaWeightTransform_forward_job(
        const float* mid_tensor,
        const float* weight_tensor,
        float* out_tensor,
        const int* tauMiddle,
        const int* cum_tauMiddle_m,
        const int* cum_tauOut_m,
        const int* cumW_tauOut_m,
        int Lmax,
        int Batch_size) {

    int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_threadId < Batch_size*cum_tauOut_m[Lmax+1]){
        //first, loop to get l
        int b = global_threadId / cum_tauOut_m[Lmax+1];
        int ltm = global_threadId % cum_tauOut_m[Lmax+1], l=0;
        while (cum_tauOut_m[l] <= ltm){l++;}
        l--;
        int tout = (ltm - cum_tauOut_m[l]) / (2*l+1);
        int m = (ltm - cum_tauOut_m[l]) % (2*l+1);

        float real=0.0, imag=0.0;
        for (int tmid = 0; tmid < tauMiddle[l]; tmid++){
            float realw = weight_tensor[WIDX(l,tout,tmid,0,cumW_tauOut_m,tauMiddle)];
            float imagw = weight_tensor[WIDX(l,tout,tmid,1,cumW_tauOut_m,tauMiddle)];
            float realm = mid_tensor[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)];
            float imagm = mid_tensor[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)];
            real += realw * realm - imagw * imagm;
            imag += realw * imagm + imagw * realm;
        }
        //weight_tensor[WIDX(l,tout,tmid,0,cumW_tauOut_m,tauMiddle)] = global_threadId;
        //weight_tensor[WIDX(l,tout,tmid,1,cumW_tauOut_m,tauMiddle)] = 300 + l + 0.1 * tout + 0.01 * tmid;

        out_tensor[IDX(b,l,tout,m,0,cum_tauOut_m,Lmax)] = real;
        out_tensor[IDX(b,l,tout,m,1,cum_tauOut_m,Lmax)] = imag;
    }

}


//==========================================================backward


__global__ void cudaWeightGrad1_backward_job(
        const float* middle,
        float* grad_weight,
        const float* grad_out,
        const int* tauMiddle,
        const int* cum_tauMiddle_m,
        const int* cum_tauOut_m,
        const int* cumW_tauOut_m,
        int Lmax,
        int Batch_size){

        int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_threadId < cumW_tauOut_m[Lmax+1]){
            int l=0;
            while (cumW_tauOut_m[l] <= global_threadId){l++;}
            l--;
            int tout = (global_threadId - cumW_tauOut_m[l]) / tauMiddle[l];
            int tmid = (global_threadId - cumW_tauOut_m[l]) % tauMiddle[l];

            float real=0.0, imag=0.0;
            for (int b = 0; b < Batch_size; b++){
                for (int m = 0; m < 2*l+1; m++){
                    float realo = grad_out[IDX(b,l,tout,m,0,cum_tauOut_m,Lmax)];
                    float imago = grad_out[IDX(b,l,tout,m,1,cum_tauOut_m,Lmax)];
                    float realm = middle[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)];
                    float imagm = middle[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)];
                    real += realm * realo + imagm * imago;
                    imag += realm * imago - realo * imagm;
                }
            }
            grad_weight[WIDX(l,tout,tmid,0,cumW_tauOut_m,tauMiddle)] = real;
            grad_weight[WIDX(l,tout,tmid,1,cumW_tauOut_m,tauMiddle)] = imag;
        }

}

__global__ void cudaMiddleGrad_backward_job(
        float* grad_middle,
        const float* weight,
        const float* grad_out,
        const float* moving_std,
        const int* tauMiddle,
        const int* cum_tauMiddle_m,
        const int* tauOut,
        const int* cum_tauOut_m,
        const int* cumW_tauOut_m,
        int Lmax,
        int Batch_size,
        float eps){

    int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_threadId < cum_tauMiddle_m[Lmax+1]){
        int l=0;
        while (cum_tauMiddle_m[l] <= global_threadId){l++;}
        l--;

        int t_offset = 0;
        for (int templ = 0; templ <= l; templ++){
            t_offset += tauMiddle[templ];
        }
        t_offset -= tauMiddle[l];

        int tm = global_threadId - cum_tauMiddle_m[l];
        int tmid = tm / (2*l+1), m = tm % (2*l+1);
        float divisor = (eps > moving_std[t_offset+tmid]) ? eps : moving_std[t_offset+tmid];
        //divisor = divisor * divisor;
        for (int b = 0; b < Batch_size; b++){
            float real=0.0, imag=0.0;
            for (int tout = 0; tout < tauOut[l]; tout++){
                float realo = grad_out[IDX(b,l,tout,m,0,cum_tauOut_m,Lmax)];
                float imago = grad_out[IDX(b,l,tout,m,1,cum_tauOut_m,Lmax)];
                float realw = weight[WIDX(l,tout,tmid,0,cumW_tauOut_m,tauMiddle)];
                float imagw = weight[WIDX(l,tout,tmid,1,cumW_tauOut_m,tauMiddle)];
                real += realw * realo + imagw * imago;
                imag += realw * imago - realo * imagw;
            }
            //grad_middle[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)] = real;
            //grad_middle[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)] = imag;
            grad_middle[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)] = real / divisor;
            grad_middle[IDX(b,l,tmid,m,1,cum_tauMiddle_m,Lmax)] = imag / divisor;
            //grad_middle[IDX(b,l,tmid,m,0,cum_tauMiddle_m,Lmax)] = b + 0.1*l + 0.01*tmid+0.001*m;
        }
        //moving_std[t_offset+tmid] = t_offset + 0.1*tmid;
    }
}

__global__ void cudaCG_backward_kernel(
        const float* tensor,
        float* g_in,
        const float* g_out,
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


} // namespace

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


//=====================================================================================

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
    int update_std){

    float* F = input_tensor.data<float>();
    float* middle = middle_tensor.data<float>();

    int* tauIn = tauIn_tensor.data<int>();

    //printf("len(taus) = %d\n", taus_tensor.size(0));

    int* tauMiddle = (int*) calloc(L+1, sizeof(int));
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2; l <=L && l <= l1 + l2; l++){
                tauMiddle[l] += tauIn[l1] * tauIn[l2];
            }
        }
    }
    int* cum_tauIn_m = _get_cum_tau(tauIn, L);
    int* cum_tauMiddle_m = _get_cum_tau(tauMiddle, L);

    int* tauOut = tauOut_tensor.data<int>();
    int* cum_tauOut_m = _get_cum_tau(tauOut, L);
    int* cumW_tauOut_m = (int*) malloc((L+2)*sizeof(int));
    cumW_tauOut_m[0] = 0;
    for (int l = 0; l <= L; l++){
        cumW_tauOut_m[l+1] = cumW_tauOut_m[l] + tauOut[l] * tauMiddle[l];
    }

    int *cuda_tauIn, *cuda_cum_tauIn_m;
    int *cuda_tauMiddle, *cuda_cum_tauMiddle_m;
    int *cuda_tauOut, *cuda_cum_tauOut_m, *cuda_cumW_tauOut_m;
    cudaMalloc((void**) &cuda_tauIn, (L+1)*sizeof(int));
    cudaMalloc((void**) &cuda_cum_tauIn_m, (L+2)*sizeof(int));
    cudaMalloc((void**) &cuda_tauMiddle, (L+1)*sizeof(int));
    cudaMalloc((void**) &cuda_cum_tauMiddle_m, (L+2)*sizeof(int));
    cudaMalloc((void**) &cuda_tauOut, (L+1)*sizeof(int));
    cudaMalloc((void**) &cuda_cum_tauOut_m, (L+2)*sizeof(int));
    cudaMalloc((void**) &cuda_cumW_tauOut_m, (L+2)*sizeof(int));

    cudaMemcpy(cuda_tauIn, tauIn, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauIn_m, cum_tauIn_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_tauMiddle, tauMiddle, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauMiddle_m, cum_tauMiddle_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_tauOut, tauOut, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauOut_m, cum_tauOut_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cumW_tauOut_m, cumW_tauOut_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);

    int size1 = B * (L+1) * (L+2) * (L+1) /2;

    int LOGFACT_SIZE=5*L+20;
    double* logfact = (double*) calloc(LOGFACT_SIZE, sizeof(double));
    for (int i = 2; i < LOGFACT_SIZE; i++){
        logfact[i] = logfact[i-1] + log((double) i);
    }
    double* cuda_logfact;
    cudaMalloc((void**) &cuda_logfact, LOGFACT_SIZE*sizeof(double));
    cudaMemcpy(cuda_logfact, logfact, LOGFACT_SIZE*sizeof(double), cudaMemcpyHostToDevice);

    cudaCG_forward_kernel<<<cuda_gridsize(size1), NUM_THREADS, 0>>>(
            F, middle,
            cuda_tauIn, cuda_cum_tauIn_m, cuda_cum_tauMiddle_m,
            cuda_logfact, L, B);

    cudaThreadSynchronize();
    //return;
    //Step 2, batch normalization
    int size2 = 0;
    for (int templ = 0; templ <= L; templ++){
        size2 += tauMiddle[templ];
    }
    //printf("Step 2 Need %d threads \n", size2);
    //printf("This is the %f-th update %d\n", cnt, update_std);
    float* moving_std = moving_std_tensor.data<float>();
    cudaBatchNorm_forward_job<<<cuda_gridsize(size2), NUM_THREADS, 0>>>(
            middle, moving_std,
            cuda_tauMiddle, cuda_cum_tauMiddle_m,
            cnt, eps,
            L, B, update_std);
    cudaThreadSynchronize();

    //Step 3 is weight transform
    float* out = output_tensor.data<float>();
    float* weights = weight_tensor.data<float>();

    int size3 = B * cum_tauOut_m[L+1];
    //printf("Step 3 Need %d threads \n", size3);
    cudaWeightTransform_forward_job<<<cuda_gridsize(size3), NUM_THREADS, 0>>>(
            middle, weights,out,
            cuda_tauMiddle, cuda_cum_tauMiddle_m,
            cuda_cum_tauOut_m, cuda_cumW_tauOut_m,
            L, B);
    cudaThreadSynchronize();

    cudaFree(cuda_logfact);
    cudaFree(cuda_tauIn);
    cudaFree(cuda_cum_tauIn_m);
    cudaFree(cuda_tauMiddle);
    cudaFree(cuda_cum_tauMiddle_m);
    cudaFree(cuda_tauOut);
    cudaFree(cuda_cumW_tauOut_m);
    cudaFree(cuda_cum_tauOut_m);

    free(cum_tauIn_m);
    free(tauMiddle);
    free(cum_tauMiddle_m);
    free(cum_tauOut_m);
    free(cumW_tauOut_m);
    free(logfact);

}

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
    float eps){

    float* Fin = input_tensor.data<float>();

    int* tauIn = tauIn_tensor.data<int>();
    int* cum_tauIn_m = _get_cum_tau(tauIn, L);
    int* tauMiddle = (int*) calloc(L+1, sizeof(int));
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2; l <=L && l <= l1 + l2; l++){
                tauMiddle[l] += tauIn[l1] * tauIn[l2];
            }
        }
    }
    int* cum_tauMiddle_m = _get_cum_tau(tauMiddle, L);
    int* tauOut = tauOut_tensor.data<int>();
    int* cum_tauOut_m = _get_cum_tau(tauOut, L);
    int* cumW_tauOut_m = (int*) malloc((L+2)*sizeof(int));
    cumW_tauOut_m[0] = 0;
    for (int l = 0; l <= L; l++){
        cumW_tauOut_m[l+1] = cumW_tauOut_m[l] + tauOut[l] * tauMiddle[l];
    }

    int *cuda_tauIn, *cuda_cum_tauIn_m;
    int *cuda_tauMiddle, *cuda_cum_tauMiddle_m;
    int *cuda_tauOut, *cuda_cum_tauOut_m, *cuda_cumW_tauOut_m;
    cudaMalloc((void**) &cuda_tauIn, (L+1)*sizeof(int));
    cudaMalloc((void**) &cuda_cum_tauIn_m, (L+2)*sizeof(int));
    cudaMalloc((void**) &cuda_tauMiddle, (L+1)*sizeof(int));
    cudaMalloc((void**) &cuda_cum_tauMiddle_m, (L+2)*sizeof(int));
    cudaMalloc((void**) &cuda_tauOut, (L+1)*sizeof(int));
    cudaMalloc((void**) &cuda_cum_tauOut_m, (L+2)*sizeof(int));
    cudaMalloc((void**) &cuda_cumW_tauOut_m, (L+2)*sizeof(int));

    cudaMemcpy(cuda_tauIn, tauIn, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauIn_m, cum_tauIn_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_tauMiddle, tauMiddle, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauMiddle_m, cum_tauMiddle_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_tauOut, tauOut, (L+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cum_tauOut_m, cum_tauOut_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_cumW_tauOut_m, cumW_tauOut_m, (L+2)*sizeof(int), cudaMemcpyHostToDevice);

    float* grad_out = grad_out_tensor.data<float>();
    float* grad_weight = grad_weight_tensor.data<float>();

    //Prep for CG
    int LOGFACT_SIZE=5*L+20;
    double* logfact = (double*) calloc(LOGFACT_SIZE, sizeof(double));
    for (int i = 2; i < LOGFACT_SIZE; i++){
        logfact[i] = logfact[i-1] + log((double) i);
    }
    double* cuda_logfact;
    cudaMalloc((void**) &cuda_logfact, LOGFACT_SIZE*sizeof(double));
    cudaMemcpy(cuda_logfact, logfact, LOGFACT_SIZE*sizeof(double), cudaMemcpyHostToDevice);


    struct timeval t1, t2;
    double elapsedTime;


    float* CG = CG_tensor.data<float>();
    gettimeofday(&t1, NULL);

    int size0 = (L+1)*(L+1)*(L+1)*(2*L+1);
    cudaprecomputeCG_job<<<cuda_gridsize(size0), NUM_THREADS, 0>>>(
            CG,
            cuda_logfact, L, B);
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Step 0 Need %d threads, took %lf mili-seconds\n", size0, elapsedTime);



    float* grad_middle = grad_middle_tensor.data<float>();

    int size1a = B * (L+1) * (L+2) * (L+1) /2;

    //printf("Step 1.a Need %d threads\n", size1a);
    gettimeofday(&t1, NULL);
    cudaCG_forward_kernel<<<cuda_gridsize(size1a), NUM_THREADS, 0>>>(
            Fin, grad_middle,
            cuda_tauIn, cuda_cum_tauIn_m, cuda_cum_tauMiddle_m,
            cuda_logfact, L, B);
    cudaThreadSynchronize();

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Step 1.a Need %d threads, took %lf mili-seconds\n", size1a, elapsedTime);

    //Step 1.b, batch normalization
    //int size1b = cum_tauMiddle_m[L+1];
    int size1b = 0;
    for (int templ = 0; templ <= L; templ++){
        size1b += tauMiddle[templ];
    }
    //printf("Step 1.b Need %d threads \n", size1b);
    float* moving_std = moving_std_tensor.data<float>();
    gettimeofday(&t1, NULL);
    cudaBatchNorm_forward_job<<<cuda_gridsize(size1b), NUM_THREADS, 0>>>(
            grad_middle, moving_std,
            cuda_tauMiddle, cuda_cum_tauMiddle_m,
            1, eps,
            L, B, 0);
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Step 1.b Need %d threads, took %lf mili-seconds\n", size1b, elapsedTime);

    //Step 1.c calculate grad on weights
    int size1c = cumW_tauOut_m[L+1];
    //printf("Step 1.c Need %d threads \n", size1c);

    gettimeofday(&t1, NULL);
    cudaWeightGrad1_backward_job<<<cuda_gridsize(size1c), NUM_THREADS, 0>>>(
            grad_middle, grad_weight, grad_out,
            cuda_tauMiddle, cuda_cum_tauMiddle_m,
            cuda_cum_tauOut_m, cuda_cumW_tauOut_m,
            L, B);
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Step 1.c Need %d threads, took %lf mili-seconds\n", size1c, elapsedTime);


    //Second update the middle grad
    //SHOULD CLEAR grad_middle FIRST!
    float* weights = weight_tensor.data<float>();
    int size2 = cum_tauMiddle_m[L+1];//This is the most balanced
    gettimeofday(&t1, NULL);
    cudaMiddleGrad_backward_job<<<cuda_gridsize(size2), NUM_THREADS, 0>>>(
            grad_middle, weights, grad_out, moving_std,
            cuda_tauMiddle, cuda_cum_tauMiddle_m,
            cuda_tauOut, cuda_cum_tauOut_m, cuda_cumW_tauOut_m,
            L, B, eps);
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Step 2 Need %d threads, took %lf mili-seconds\n", size2, elapsedTime);


    //Last, update the grad on input
    float* grad_in = grad_in_tensor.data<float>();
    int size3 = B * cum_tauIn_m[L+1];
    //printf("Step 3 Need %d threads\n", size3);

    gettimeofday(&t1, NULL);
    cudaCG_backward_kernel<<<cuda_gridsize(size3), NUM_THREADS, 0>>>(
            Fin, grad_in, grad_middle,
            cuda_tauIn, cuda_cum_tauIn_m, cuda_cum_tauMiddle_m,
            //cuda_logfact, CG,
            CG,
            L, B);
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    //printf("Step 3 Need %d threads, took %lf mili-seconds\n", size3, elapsedTime);


    gettimeofday(&t1, NULL);

    cudaFree(cuda_logfact);
    cudaFree(cuda_tauIn);
    cudaFree(cuda_cum_tauIn_m);
    cudaFree(cuda_tauMiddle);
    cudaFree(cuda_cum_tauMiddle_m);
    cudaFree(cuda_tauOut);
    cudaFree(cuda_cumW_tauOut_m);
    cudaFree(cuda_cum_tauOut_m);

    free(logfact);
    free(cum_tauIn_m);
    free(tauMiddle);
    free(cum_tauMiddle_m);
    free(cum_tauOut_m);
    free(cumW_tauOut_m);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms


}