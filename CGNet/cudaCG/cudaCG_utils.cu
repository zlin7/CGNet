#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cudaCG_all.h"

#define BLOCK 512
#define CEIL_DIV(num, denum) (num+denum-1)/denum
#define IDX(b,l,t,m,i,cum,L) (i+2*(m+t*(2*l+1)+cum[l]+b*cum[L+1]))
#define PLUSMINUS(k) ((k%2==1) ? -1 : 1)
#define LOGFACT(n,mem) ((n < 2) ? 0. : mem[n])


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

void print_arr(int* v, int l){
    printf("vector: (");
    for (int i = 0; i < l; i++){
        printf("%d, ", v[i]);
    }
    printf(")\n");
    return;
}


int* _get_cum_tau(const int* taus, int L){
    int* cum_tau = (int*) malloc((L+2)*sizeof(int));
    cum_tau[0] = 0;
    for (int l = 0; l <= L; l++){
        cum_tau[l+1] = cum_tau[l] + (2 * l + 1) * taus[l];
    }
    return cum_tau;
}

void init_taum_ptrs_FG(const int* taus_F, const int* taus_G, int L,
                    int** taus_FG, int** cumu_tm_F, int** cumu_tm_G, int** cumu_tm_FG){
    //Compute the taus for all the l1*l2 components
    int *temp_taus_FG = (int*) calloc(L+1, sizeof(int));
    if (!taus_G){
        taus_G = taus_F;
    }
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2; l <=L && l <= l1 + l2; l++){
                temp_taus_FG[l] += taus_F[l1] * taus_G[l2]; //Note here l2 <= l1
            }
        }
    }
    if (cumu_tm_F){
        *cumu_tm_F = _get_cum_tau(taus_F, L);
    }
    if (cumu_tm_G){
        *cumu_tm_G = _get_cum_tau(taus_G, L);
    }
    if (cumu_tm_FG){
        *cumu_tm_FG = _get_cum_tau(temp_taus_FG, L);
    }
    *taus_FG = temp_taus_FG;
}

void init_taum_ptrs_F(const int* taus_F, int L, int** taus_FF, int** cumu_tm_F, int** cumu_tm_FF){
    return init_taum_ptrs_FG(taus_F, NULL, L, taus_FF, cumu_tm_F, NULL, cumu_tm_FF);
}


int init_logfactorials_new(double** logfact, int max_size){
    double* v = (double*) calloc(max_size, sizeof(double));
    for (int i = 2; i < max_size; i++){
        v[i] = v[i-1] + log((double) i);
    }
    *logfact = v;
    return max_size;
}

int init_logfactorials(int L, double** logfact){
    int LOGFACT_SIZE=5*L+20;
    return init_logfactorials_new(logfact, LOGFACT_SIZE);
}

void* cuda_init_and_memcpy(void* host_ptr, size_t size){
    void* device_ptr;
    cudaMalloc((void**) &device_ptr, size);
    cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    return device_ptr;
}



