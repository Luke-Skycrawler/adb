#pragma once 
#ifdef TESTING
#define __host__ 
#define __device__ 
#define __forceinline__ inline 
#define __global__
#define __constant__ 
struct float3 {
    float x, y, z;
};
inline float3 make_float3(float x, float y, float z) {
    return { x, y, z };
}
#define USE_DOUBLE_PRECISION
#include <cmath>
#include <cstdio>
#include <array>
using i2 = std::array<int, 2>;
using i4 = std::array<int, 4>;

#else 
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

using i2 = cuda::std::array<int, 2>;
using i4 = cuda::std::array<int, 4>;
template <typename T>
std::vector<T> from_thrust(thrust::device_vector<T> &a)
{
    thrust::host_vector<T> b = a;
    std::vector<T> ret;
    ret.resize(b.size());
    for (int i = 0; i < b.size(); i++) {
        ret[i] = b[i];
    }
    return ret;
}

template <typename T>
std::vector<T> from_thrust(thrust::host_vector<T>& b)
{
    std::vector<T> ret;
    ret.resize(b.size());
    for (int i = 0; i < b.size(); i++) {
        ret[i] = b[i];
    }
    return ret;
}

template <typename T>
std::vector<T> from_dev_ptr(T * dev_ptr, int n) {
    thrust::device_vector<T> a(dev_ptr, dev_ptr + n);
    return from_thrust(a);
}

#endif

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

#endif

// using luf = cuda::std::array<float, 6>;

// using vec3f = cuda::std::array<float, 3>;
// using Facef = cuda::std::array<vec3f, 3>;
using vec3f = float3;

struct luf {
    vec3f l, u;
};
struct Facef {
    vec3f t0, t1, t2;
};
struct Edgef {
    vec3f e0, e1;
};

struct cudaAffineBody {
    float3 q[4], q0[4], dqdt[4], q_update[4];
    float mass, Ic;
    int n_vertices, n_faces, n_edges;
    int *faces, *edges;
    float3* vertices, *projected, *updated;
    __device__ __host__ void q_minus_qtiled(float3 dq[4]);
    inline __device__ __host__ Facef triangle(int i)
    {
        return Facef{
            projected[faces[i * 3]],
            projected[faces[i * 3 + 1]],
            projected[faces[i * 3 + 2]]
        };
    }
    inline __device__ __host__ Edgef edge(int i)
    {
        return Edgef{
            projected[edges[i * 2]],
            projected[edges[i * 2 + 1]]
        };
    }
    inline __device__ __host__ Facef triangle_at_rest(int i) {
        return Facef{
            vertices[faces[i * 3]],
            vertices[faces[i * 3 + 1]],
            vertices[faces[i * 3 + 2]]
        };
    }
    inline __device__ __host__ Edgef edge_at_rest(int i) {
        return Edgef{
            vertices[edges[i * 2]],
            vertices[edges[i * 2 + 1]]
        };
    }
    inline __device__ __host__ Facef triangle_updated(int i) {
        return Facef{
            updated[faces[i * 3]],
            updated[faces[i * 3 + 1]],
            updated[faces[i * 3 + 2]]
        };
    }
    inline __device__ __host__ Edgef edge_updated(int i) {
        return Edgef{
            updated[edges[i * 2]],
            updated[edges[i * 2 + 1]]
        };
    }
};

__forceinline__ __host__ __device__ vec3f operator+(vec3f a, vec3f b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ vec3f operator-(vec3f a, vec3f b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ vec3f operator/(vec3f a, float k)
{
    return k == 0.0 ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(a.x / k, a.y / k, a.z / k);
}
__forceinline__ __host__ __device__ vec3f operator*(vec3f a, float k)
{
    return make_float3(a.x * k, a.y * k, a.z * k);
}
__forceinline__ __host__ __device__ vec3f operator+(vec3f a, float k)
{
    return make_float3(a.x + k, a.y + k, a.z + k);
}
__forceinline__ __host__ __device__ vec3f operator-(vec3f a, float k)
{
    return make_float3(a.x - k, a.y - k, a.z - k);
}

__forceinline__ __host__ __device__ float dot(vec3f a, vec3f b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __host__ __device__ float norm(vec3f a)
{
    return CUDA_SQRT(dot(a, a));
}
__forceinline__ __host__ __device__ vec3f normalize(vec3f a)
{
    return a / norm(a);
}

__forceinline__ __host__ __device__ bool intersects(luf a, luf b)
{
    // return a[0].x < b[1].x && a[1].x > b[0].x && a[0].y < b[1].y && a[1].y > b[0].y && a[0].z < b[1].z && a[1].z > b[0].z;
    return a.l.x < b.u.x && a.u.x > b.l.x && a.l.y < b.u.y && a.u.y > b.l.y && a.l.z < b.u.z && a.u.z > b.l.z;
}

__forceinline__ __host__ __device__ vec3f cross(vec3f a, vec3f b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}
__forceinline__ __host__ __device__ vec3f unit_normal(Facef f)
{
    // return normalize(cross(f[1] - f[0], f[2] - f[0]));
    return normalize(cross(f.t1 - f.t0, f.t2 - f.t0));
}
__forceinline__ __host__ __device__ float area_x2(vec3f a, vec3f b, vec3f c)
{
    return norm(cross(c - a, b - a));
}

__forceinline__ __host__ __device__ float ab(vec3f a, vec3f b)
{
    return norm(b - a);
}

__forceinline__ __host__ __device__ float h(vec3f e0, vec3f e1, vec3f p)
{
    return area_x2(e0, e1, p) / ab(e0, e1);
}
__forceinline__ __host__ __device__ float ab_sqr(vec3f a, vec3f b)
{
    return dot(b - a, b - a);
}

__forceinline__ __host__ __device__ bool is_obtuse_triangle(vec3f e0, vec3f e1, vec3f p)
{
    auto ab = ab_sqr(e0, e1);
    auto bc = ab_sqr(e1, p);
    auto ca = ab_sqr(e0, p);
    return CUDA_ABS(ca - bc) > ab;
}


__forceinline__ __device__ float3 matmul(float3 _q[4], float3 x)
{
    float3* q = _q + 1;
    float3 ret = make_float3(
        x.x * q[0].x + x.y * q[1].x + x.z * q[2].x,
        x.x * q[0].y + x.y * q[1].y + x.z * q[2].y,
        x.x * q[0].z + x.y * q[1].z + x.z * q[2].z);
    return ret + _q[0];
}


__forceinline__ __device__ __host__ float kronecker(int i, int j)
{
    return i == j ? 1.0f : 0.0f;
}
namespace dev {
__device__ __constant__ static const float kappa = 1e-4f, d_hat = 1e-4f, d_hat_sqr = 1e-2f, eps = 1e-3f;

__host__ __device__ float barrier_function(float d);
__host__ __device__ float barrier_derivative_d(float x);
__host__ __device__ float barrier_second_derivative(float d);

__host__ __device__ float point_triangle_distance(vec3f p, vec3f t0, vec3f t1, vec3f t2, int type = 7);
__host__ __device__ void point_triangle_distance_gradient(vec3f p, vec3f t0, vec3f t1, vec3f t2, float* pt_grad, int type = 7, float *local_grad = nullptr);
__host__ __device__ void point_triangle_distance_hessian(vec3f p, vec3f t0, vec3f t1, vec3f t2, float* pt_hess, int type = 7, float *local_hess = nullptr);
__host__ __device__ void point_point_distance_gradient(vec3f p, vec3f t0, float* pt_hess);
__host__ __device__ void point_point_distance_hessian(vec3f p, vec3f t0, float* pt_hess);
__host__ __device__ float edge_edge_distance(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, int type = 9);
__host__ __device__ void edge_edge_distance_gradient(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float* ee_grad, int type = 9, float *local_grad = nullptr);
__host__ __device__ void edge_edge_distance_hessian(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float* ee_hess, int type = 9, float *local_hess = nullptr);

__host__ __device__ int point_triangle_distance_type(vec3f p, vec3f t0, vec3f t1, vec3f t2);
__host__ __device__ int edge_edge_distance_type(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1);
__host__ __device__ float edge_edge_mollifier(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float eps_x);
__host__ __device__ void edge_edge_mollifier_gradient(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float eps_x, float* grad);
__host__ __device__ void edge_edge_mollifier_hessian(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float eps_x, float *grad_input, float* hess);
__host__ __device__ float edge_edge_mollifier(float x, float eps_x);
__host__ __device__ float edge_edge_mollifier_gradient(float x, float eps_x);
__host__ __device__ float edge_edge_mollifier_hessian(float x, float eps_x);
__host__ __device__ float edge_edge_cross_squarednorm(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1);
}
__host__ __device__ float inertia(cudaAffineBody &c, float dt);
__host__ __device__ void inertia_grad(cudaAffineBody& c, float dt, float ret[12]);
__host__ __device__ void inertia_hess(cudaAffineBody& c, float ret[144]);
__device__ __host__ float vf_distance(vec3f _v, Facef f, int& pt_type);



static const int n_cuda_threads_per_block = 256;
#define PTR(x) thrust::raw_pointer_cast((x).data())