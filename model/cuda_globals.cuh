#pragma once
#include <cuda/std/array>
#include <thrust/device_vector.h>
#include "cuda_header.cuh"

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
	thrust::device_vector<int> inner;
	thrust::device_vector<float> values;
};


struct cudaAffineBody {
        float3 q[4], q0[4], dqdt[4], q_update[4];
        float mass, Ic;
        int n_vertices, n_faces, n_edges;
        int *faces, *edges;
        float3* vertices;
        int global_vertices_offset;
        __device__ __host__ void q_minus_qtiled(float3 dq[4]);
};
struct FrictionInfo {
    float lambda;
    float basis[24];
};

struct CudaGlobals {
    thrust::device_vector<FrictionInfo> friction_info;
    cudaAffineBody *cubes;
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
    float *b, *hess_diag, *dq, * float_buffer;
    float dt;
    float3 *projected_vertices, *float3_buffer;
    thrust::device_vector<i2> lut;
    // i2 * lut;
    void *buffer_chunk;
    int device_id, per_stream_buffer_size;
    cudaStream_t* streams;
    float3 gravity;
    CudaGlobals(int n_cubes = 0);
    ~CudaGlobals();
    __device__ __host__ CudaGlobals(CudaGlobals &CudaGlobals);
};

static const int max_pairs_per_thread = 512, max_aabb_list_size = 512;
static const int max_n_vertices = 1024 * 1024;

// __constant__ CudaGlobals *cuda_globals;
extern CudaGlobals host_cuda_globals;
namespace dev {
__device__ __constant__ static const float kappa = 1e-1f, d_hat = 1e-4f, d_hat_sqr = 1e-2f;

__host__ __device__ float barrier_function(float d);
__host__ __device__ float barrier_derivative_d(float x);
__host__ __device__ float barrier_second_derivative(float d);


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

void make_lut(int lut_size, i2* lut);
