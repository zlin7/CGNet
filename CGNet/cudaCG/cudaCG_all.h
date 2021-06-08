//
// Created by zhen7 on 5/2/2021.
//
#include <cuda.h>

#ifndef CUDACG_ALL_CUDACG_ALL_H
#define CUDACG_ALL_CUDACG_ALL_H

void print_arr(int* v, int l);
int* _get_cum_tau(const int* taus, int L);
void init_taum_ptrs_FG(const int* taus_F, const int* taus_G, int L,
                       int** taus_FG, int** cumu_tm_F, int** cumu_tm_G, int** cumu_tm_FG);
void init_taum_ptrs_F(const int* taus_F, int L, int** taus_FF, int** cumu_tm_F, int** cumu_tm_FF);
int init_logfactorials_new(double** logfact, int max_size);
int init_logfactorials(int L, double** logfact);
void* cuda_init_and_memcpy(void* host_ptr, size_t size);
dim3 cuda_gridsize(int n);
#endif //CUDACG_ALL_CUDACG_ALL_H
