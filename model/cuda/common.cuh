#pragma once
#include <cuda_runtime.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include "helper_math.h"

#include "scalar_types.h"

#define CUDA_SOURCE

#define TID(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return
#define TID_NO_RETURN(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x

 
#define CUDA_CALL(function, totalThreads)  \
	if (totalThreads == 0) return; \
	uint function ## _numBlocks, function ## _numThreads; \
	ComputeGridSize(totalThreads, function ## _numBlocks, function ## _numThreads); \
	function <<<function ## _numBlocks, function ## _numThreads >>>
#define CUDA_CALL_S(function, totalThreads, stream)  \
	if (totalThreads == 0) return; \
	uint function ## _numBlocks, function ## _numThreads; \
	ComputeGridSize(totalThreads, function ## _numBlocks, function ## _numThreads); \
	function <<<function ## _numBlocks, function ## _numThreads, 0,stream>>>
#define CUDA_CALL_V(function, ...) \
	function <<<__VA_ARGS__>>>

namespace ShayCUDA
{
	typedef unsigned int uint;

	const uint BLOCK_SIZE = 256;

	// __device__ inline float length2(glm::vec3 vec)
	// {
	// 	return glm::dot(vec, vec);
	// }

	inline void ComputeGridSize(const uint& n, uint& numBlocks, uint& numThreads)
	{
		if (n == 0)
		{
			//fmt::print("Error(Solver): numParticles is 0\n");
			numBlocks = 0;
			numThreads = 0;
			return;
		}
		numThreads = min(n, BLOCK_SIZE);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}

	template<class T>
	inline T* VtAllocBuffer(size_t elementCount)
	{
		T* devPtr = nullptr;
		#ifdef __CUDARUN__
		checkCudaErrors(cudaMalloc((void **)&devPtr, elementCount * sizeof(T)));
		#else
		// checkCudaErrors(cudaMallocManaged((void**)&devPtr, elementCount * sizeof(T)));
		devPtr = (T*)malloc(elementCount * sizeof(T));
		#endif
		cudaDeviceSynchronize(); // this is necessary, otherwise realloc can cause crash
		return devPtr;
	}

	inline void VtFreeBuffer(void* buffer)
	{
		#ifdef __CUDARUN__
		cudaDeviceSynchronize();
		checkCudaErrors(cudaFree(buffer));
		#else

		free(buffer);
		#endif
	}

}
class ParallelFunction {
public:
	int run_count; 
    
	ParallelFunction(int run_count) : run_count(run_count) {}
	virtual func void run(int tid) = 0;
	
	
	virtual void gpu(){}

	virtual void cpu() {
        #pragma omp parallel for
        for (int i = 0; i < run_count; i++)
            run(i);
	}
};
#ifdef __CUDACC__
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
#endif