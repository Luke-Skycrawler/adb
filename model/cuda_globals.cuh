#pragma once
#include "cuda_header.cuh"
#include <vector>
#include <string>
#include <map>

// struct CollisionSets {
//     int pt_cnt[7], ee_cnt[9];
//     i2* pt_set[7];
//     i2* pt_set_body_index[7];
//     /* corresponds to enum class PointTriangleDistanceType {
//         P_T0, 
//         P_T1, 
//         P_T2, 
//         P_E0, 
//         P_E1, 
//         P_E2, 
//         P_T
//     };*/
//     i2* ee_set[9];
//     i2* ee_set_body_index[9];
//     /* corresponds to enum class EdgeEdgeDistanceType {
//         EA0_EB0, 
//         EA0_EB1, 
//         EA1_EB0, 
//         EA1_EB1, 
//         EA_EB0,
//         EA_EB1,
//         EA0_EB,
//         EA1_EB,
//         EA_EB
//     }; */
    
// };

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
    int npt, nee;
    int n_cubes, lut_size;
    CsrSparseMatrix hess;
    // permenant designated buffers
    float *b, *hess_diag, *dq; //, *float_buffer;
    // float3 *float3_buffer;
    float dt;
    thrust::device_vector<i2> lut;
    void* buffer_chunk;

    int device_id, per_stream_buffer_size;

    // temporary buffers
    const int st_size = 1024, blk_size = max_pairs_per_block * (sizeof(i2) * 2 + sizeof(int) * 2) * 8;
    // only fit for small data;
    char **small_temporary_buffer, **bulk_buffer;
    char **small_temporary_buffer_back, **bulk_buffer_back;
    char *leader_thread_buffer_back, *leader_thread_buffer;
    vec3f *vertices_at_rest, *projected_vertices, *updated_vertices;
    int n_vertices, n_edges, n_faces;
    int *edges, *faces;
    CollisionSet pt, ee, pt_line_search, ee_line_search;
    int npt_line_search, nee_line_search;
    cudaStream_t* streams;
    float3 gravity;
    CudaGlobals(int n_cubes = 0);
    ~CudaGlobals();
    void allocate_buffers();
    void free_buffers();
    __device__ __host__ CudaGlobals(CudaGlobals& CudaGlobals);

    std::map<std::string, int> params;
};

extern CudaGlobals host_cuda_globals;

// seen in all cuda files & glued files-----------------------------------------
// high-level functions, can be directly called outside of cuda environment

// full-fledged cuda ipc with cpu verification
void cuda_ipc();

// full-fledged cuda iaabb
float iaabb_brute_force_cuda_pt_only(
    int n_cubes,
    cudaAffineBody* cubes,
    luf* aabbs,
    int vtn,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx);

// build csr matrix from look up table
void build_csr(int n_cubes, const thrust::device_vector<i2>& lut, CsrSparseMatrix& sparse_matrix);

// solve Ax = b, A is the hess matrix
void gpuCholSolver(CsrSparseMatrix& hess, float* x, float *b);

// defined in constraint_set.cu
void project_glue(int vtn);

float barrier_plus_inert_glue(float dt);

