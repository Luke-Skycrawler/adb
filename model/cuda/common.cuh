#pragma once
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define func __device__ __host__
#define kernel __global__

#include "scalar_types.h"

#define CUDA_SOURCE


// Generic template for atomicMin
template <typename T>
inline __device__ T atomic_min(T* address, T val);

template <>
inline __device__ float atomic_min<float>(float* address, float val) {
    int *address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (val < __int_as_float(old)) 
	{
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }

    return __int_as_float(old);
}

// Specialization for double using CAS-based implementation
template <>
inline __device__ double atomic_min<double>(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    while (val < __longlong_as_double(old)) {
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    }
    return __longlong_as_double(old);    
}
