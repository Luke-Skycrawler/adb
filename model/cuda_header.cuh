#pragma once 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda/std/array>

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess){ printf("\nCUDA Error: %s (err_num = %d) \n", cudaGetErrorString(a), a); cudaDeviceReset(); assert(0);}}

#define MAX_THREADS_NUM 1024
#define MAX_BLOCKS_NUM 1024
#define THREADS_NUM_768 768
#define THREADS_NUM_576 576
#define THREADS_NUM 512
#define THREADS_NUM_256 256
#define THREADS_NUM_192 192
#define THREADS_NUM_128 128
#define THREADS_NUM_64 64
#define THREADS_NUM_32 32
#define THREADS_NUM_16 16

#define MAX_CELLS_NUM 10000
#define MAX_CELLS_CONTSTRAINT_TRIANGLE_NUM 500
static const int n_cuda_threads_per_block = 256;

#ifdef USE_DOUBLE_PRECISION
#define CUDA_ZERO 5e-13
#define IS_Numerical_ZERO(d) (CUDA_ABS(d) < CUDA_ZERO)
#define CUDA_SQRT(d) sqrt(d)
#define CUDA_ABS(d) fabs(d)
#define CUDA_LOG(d) log(d)
#define IS_CUDA_ZERO(d) (fabs(d) < CUDA_ZERO)
#define Check_CUDA_ZERO(d) (fabs(d) > CUDA_ZERO ? d : 0)
#define CUDA_MAX(a, b) fmax(a, b)
#define CUDA_MIN(a, b) fmin(a, b)
#define CUDA_MAX3(a, b, c) fmax(fmax(a, b), c)
#define CUDA_MIN3(a, b, c) fmin(fmin(a, b), c)
#else
#define CUDA_ZERO 1e-6
#define IS_Numerical_ZERO(d) (CUDA_ABS(d) < CUDA_ZERO)
#define CUDA_SQRT(d) sqrtf(d)
#define CUDA_ABS(d) fabsf(d)
#define CUDA_LOG(d) logf(d)
#define IS_CUDA_ZERO(d) (fabsf(d) < CUDA_ZERO)
#define Check_CUDA_ZERO(d) (fabsf(d) > CUDA_ZERO ? d : 0)
#define CUDA_MAX(a, b) fmaxf(a, b)
#define CUDA_MIN(a, b) fminf(a, b)
#define CUDA_MAX3(a, b, c) fmaxf(fmaxf(a, b), c)
#define CUDA_MIN3(a, b, c) fminf(fminf(a, b), c)

using luf = cuda::std::array<float, 6>;
using vec3f = cuda::std::array<float, 3>;
using Facef = cuda::std::array<vec3f, 3>;

inline __host__ __device__ vec3f operator+(const vec3f& a, const vec3f& b)
{
    return { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}
inline __host__ __device__ vec3f operator-(const vec3f& a, const vec3f& b)
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}
inline __host__ __device__ vec3f operator/(const vec3f& a, float k)
{
    return { a[0] / k, a[1] / k, a[2] / k };
}
inline __host__ __device__ vec3f operator*(const vec3f& a, float k)
{
    return { a[0] * k, a[1] * k, a[2] * k };
}

inline __host__ __device__ float dot(const vec3f& a, const vec3f& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline __host__ __device__ float norm(const vec3f& a)
{
    return CUDA_SQRT(dot(a, a));
}
inline __host__ __device__ vec3f normalize(const vec3f& a)
{
    return a / norm(a);
}

inline __host__ __device__ bool intersects(const luf& a, const luf& b)
{
    return a[0] <= b[3] && a[3] >= b[0] && a[1] <= b[4] && a[4] >= b[1] && a[2] <= b[5] && a[5] >= b[2];
}

inline __host__ __device__ vec3f cross(const vec3f& a, const vec3f& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}
inline __host__ __device__ vec3f unit_normal(const Facef& f)
{
    return normalize(cross(f[1] - f[0], f[2] - f[0]));
}
inline __host__ __device__ float area(const vec3f& a, const vec3f& b, const vec3f& c)
{
    return norm(cross(c - a, b - a)) / 2.0f;
}

inline __host__ __device__ float ab(const vec3f& a, const vec3f& b)
{
    return norm(b - a);
}

inline __host__ __device__ float h(const vec3f& e0, const vec3f& e1, const vec3f& p)
{
    return area(e0, e1, p) / ab(e0, e1);
}
inline __host__ __device__ float ab_sqr(const vec3f& a, const vec3f& b)
{
    return dot(b - a, b - a);
}

inline __host__ __device__ bool is_obtuse_triangle(const vec3f& e0, const vec3f& e1, const vec3f& p)
{
    auto ab = ab_sqr(e0, e1);
    auto bc = ab_sqr(e1, p);
    auto ca = ab_sqr(e0, p);
    return CUDA_ABS(ca - bc) > ab;
}

#endif

