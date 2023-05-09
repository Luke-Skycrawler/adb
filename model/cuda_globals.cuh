#pragma once
#include <cuda/std/array>
#include <thrust/device_vector.h>
#include "cuda_header.cuh"
#include "cuda_glue.h"
#include <vector>
using i2 = cuda::std::array<int, 2>;
using i4 = cuda::std::array<int, 4>;
struct CollisionSets {
    int pt_cnt[7], ee_cnt[9];
    i2* pt_set[7];
    i2* pt_set_body_index[7];
    /* corresponds to enum class PointTriangleDistanceType {
        P_T0, 
        P_T1, 
        P_T2, 
        P_E0, 
        P_E1, 
        P_E2, 
        P_T
    };*/
    i2* ee_set[9];
    i2* ee_set_body_index[9];
    /* corresponds to enum class EdgeEdgeDistanceType {
        EA0_EB0, 
        EA0_EB1, 
        EA1_EB0, 
        EA1_EB1, 
        EA_EB0,
        EA_EB1,
        EA0_EB,
        EA1_EB,
        EA_EB
    }; */
    
};

struct CsrSparseMatrix
{
	int rows = 0;
	int cols = 0;
	int nnz = 0;
	thrust::device_vector<int> outer_start;
	// thrust::host_vector<int> outer_start;
	thrust::device_vector<int> inner;
	// thrust::host_vector<int> inner;
	thrust::device_vector<float> values;
	// thrust::host_vector<float> values;
};


struct FrictionInfo {
    float lambda;
    float basis[24];
};

struct CollisionSet {
    i2* b; // body index
    i2* p; // primitive index
};

static const int max_pairs_per_thread = 512, max_aabb_list_size = 512;
static const int max_n_vertices = 1024 * 1024;
static const int n_blocks = 64;
static const int n_aabb_list_per_block = 16, max_pairs_per_block = 1024 * 2;
static const int max_prmts_per_block = 1024 * 16;

struct CudaGlobals {
    thrust::device_vector<FrictionInfo> friction_info;
    cudaAffineBody* cubes;
    std::vector<cudaAffineBody> host_cubes;
    luf* aabbs;
    thrust::device_vector<i2> prim_idx, body_idx;
    thrust::device_vector<i2> prim_idx_update, body_idx_update;
    thrust::device_vector<int> pt_types, pt_types_update;
    int npt, nee;
    int n_cubes, lut_size;
    CollisionSets collision_sets;
    CsrSparseMatrix hess;
    // FIXME: __device__ refer lan's imple
    // thrust::device_vector<float> b;

    // permenant designated buffers
    float *b, *hess_diag, *dq, *float_buffer;
    float3 *float3_buffer;
    float dt;
    thrust::device_vector<i2> lut;
    void* buffer_chunk;

    int device_id, per_stream_buffer_size;
    int* cnt_ret;

    // temporary buffers
    const int st_size = 128, blk_size = max_pairs_per_block * (sizeof(i2) * 2 + sizeof(int) * 2) * 8;
    // only fit for small data;
    char **small_temporary_buffer, **bulk_buffer;
    char **small_temporary_buffer_back, **bulk_buffer_back;
    char *leader_thread_buffer_back, *leader_thread_buffer;
    vec3f *vertices_at_rest, *projected_vertices;
    int *edges, *faces;
    CollisionSet pt, ee;

    cudaStream_t* streams;
    float3 gravity;
    CudaGlobals(int n_cubes = 0);
    ~CudaGlobals();
    void allocate_buffers();
    void free_buffers();
    __device__ __host__ CudaGlobals(CudaGlobals& CudaGlobals);
};

// __constant__ CudaGlobals *cuda_globals;
extern CudaGlobals host_cuda_globals;
namespace dev {
__device__ __constant__ static const float kappa = 1e-1f, d_hat = 1e-4f, d_hat_sqr = 1e-2f;

__host__ __device__ float barrier_function(float d);
__host__ __device__ float barrier_derivative_d(float x);
__host__ __device__ float barrier_second_derivative(float d);

__device__ void point_triangle_distance_hessian(vec3f p, vec3f t0, vec3f t1, vec3f t2, float* pt_hess);
__device__ float point_triangle_distance(vec3f p, vec3f t0, vec3f t1, vec3f t2);
__device__ void point_triangle_distance_gradient(vec3f p, vec3f t0, vec3f t1, vec3f t2, float *pt_grad);

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

extern void make_lut(int lut_size, i2* lut);

__forceinline__ __device__ __host__ float kronecker(int i, int j)
{
    return i == j ? 1.0f : 0.0f;
}

