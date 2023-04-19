#include "cuda_header.cuh"
#include "cuda_globals.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>
// #include "autogen/autogen.cuh"
using namespace std;
CudaGlobals cuda_globals;


__device__ float dev_barrier_derivative_d(float x){ return 0.0f; }
__device__ float dev_barrier_second_derivative(float x) {return 0.0f; } 

__device__ float point_triangle_distance(vec3f p, vec3f t0, vec3f t1, vec3f t2) {return 0.0f;}
__device__ void point_triangle_distance_gradient(vec3f p, vec3f t0, vec3f t1, vec3f t2, float *pt_grad) {
    // autogen::point_plane_distance_gradient(
    //     p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0],
    //     t2[1], t2[2], pt_grad);
}

__device__ void point_triangle_distance_hessian(vec3f p, vec3f t0, vec3f t1, vec3f t2, float *pt_hess){
    // autogen::point_plane_distance_hessian(
    //     p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0],
    //     t2[1], t2[2], pt_hess);
}
__device__ void dev_project_to_psd(int dim, float* A){

}

__global__ void fill_inner_outers_kernel(int n_cubes, int lut_size, const i2 * lut, int *inners, int *outers) {
    auto tid = threadIdx.x;
    int nnz = lut_size * 12 * 12;
    auto n_task_per_thread = (nnz + blockDim.x - 1)/ blockDim.x;  
    // precondition: lut contains all the symmetric pairs (i, j) and (j, i), sorted in ascending order

    for (int _i = 0; _i < n_task_per_thread; _i ++) {
        // detect stairs, fill outers
        auto I = _i + n_task_per_thread * tid;
        if (I < nnz) {
            int i = lut[I][0];
            int last = I == 0 ? -1: lut[I - 1][0];
            
            if (last != i) {
                inners[i] = I;
                // temporary storage
            }
        }
    }
    __syncthreads();
    int n_cols_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_cols_per_thread; _i ++) {
        auto I = _i + n_cols_per_thread * tid;
        if (I < n_cubes) {
            auto next = I == n_cubes - 1 ? lut_size: inners[I + 1];
            auto stride = next - inners[I];
            auto start = inners[I] * 12 * 12;

            for (int i =0; i < 12; i ++) {
                outers[I * 12 + i] = start + 12 * (i * stride);
            }
        }
    }
    __syncthreads(); // wait until finish, then break inners array


    for (int _i = 0; _i < n_task_per_thread; _i ++) {
        // fiil inners
        auto I = _i + n_task_per_thread * tid;
        if (I < nnz) {
            int j = lut[I][1];
            for (int r = 0; r <12; r ++) {
                inners[r + 12 * I] = j * 12 + r;
            }
        }
    }

}
void build_csr(int n_cubes, const thrust::device_vector<i2> &lut, CsrSparseMatrix & sparse_matrix) {
    int lut_size = lut.size();
    int nnz = lut_size * 12 * 12;
    sparse_matrix.rows = n_cubes * 12;
    sparse_matrix.cols = n_cubes * 12;
    sparse_matrix.nnz = nnz;
    sparse_matrix.outer_start.resize(n_cubes);
    sparse_matrix.values.resize(nnz);
    thrust::fill(sparse_matrix.values.begin(), sparse_matrix.values.end(), 0.0f);
    
    auto lut_ptr = thrust::raw_pointer_cast(lut.data());
    auto inner_ptr = thrust::raw_pointer_cast(sparse_matrix.inner.data());
    auto outer_ptr = thrust::raw_pointer_cast(sparse_matrix.outer_start.data());

    fill_inner_outers_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, lut_size, lut_ptr, inner_ptr, outer_ptr);
}


void make_placeholder_sparse_matrix(int n_cubes, CsrSparseMatrix &sparse_matrix) {
    auto& lut{ cuda_globals.lut}; 
    {
        thrust::host_vector<i2> diagonals(n_cubes);
        thrust::device_vector<i2> dev_diagonals(n_cubes);
        for (int i = 0; i < n_cubes; i++) diagonals[i] = { i, i };
        dev_diagonals = diagonals;
        lut.insert(lut.end(), dev_diagonals.begin(), dev_diagonals.end());
    }
 
    
    thrust::sort(lut.begin(), lut.end());
    auto new_end = thrust::unique(lut.begin(), lut.end());

    auto lut_size = new_end - lut.begin();
    lut.resize(lut_size);
    lut.shrink_to_fit();
    build_csr(n_cubes, lut, sparse_matrix);
}


__device__ void pt_grad_hess12x12(vec3f p, Facef t, float *pt_grad, float * pt_hess, bool psd = true) {

    auto dist = point_triangle_distance(p, t.t0, t.t1, t.t2);
    point_triangle_distance_gradient(p, t.t0, t.t1, t.t2, pt_grad);
    point_triangle_distance_hessian(p, t.t0, t.t1, t.t2, pt_hess);

    auto B_ = dev_barrier_derivative_d(dist);
    auto B__ = dev_barrier_second_derivative(dist);

    //pt_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;
    for (int i = 0; i < 12;  i ++ ) for (int j = 0; j < 12; j ++) {
        pt_hess[j * 12  + i] = B_ * pt_hess[j * 12  + i] + pt_grad[i] * pt_grad[j] * B__;
        // column major
    }
    for (int i =0; i < 12; i ++)
        pt_grad[i] *= B_;

    if (psd)
        dev_project_to_psd(12, pt_hess);
}

static const int ncthreads = 64;
__global__ void ipc_pt_kernel(int npt, vec3f* p, Facef* t, i2 *ij, cuda::std::array<bool, 2> *is_static, float *values, int *outers, float *pt_grad_hess) {
    // input: p, t data
    // output: basis Tk (2x12), uk (relative displacement, 2x1), gradient g (12x1), hessian H (12x12)
    // __shared__ float pt_grad_hess[13 * 12 * n_cuda_threads_per_block];
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto hess_start = pt_grad_hess + 12 * blockDim.x;
    // FIXME: align with cacheline size 
    int n_tasks_per_thread = (npt + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < npt) { 
            pt_grad_hess12x12(p[I], t[I], pt_grad_hess + tid * 12, hess_start + 144 * tid);
            
        }
    }

}

#include <cusolver_common.h>
#include <cusolverSp.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

cusolverDnHandle_t dnHandle;
cusolverSpHandle_t cusolverSpH;
cusparseHandle_t cusparseH;
cudaStream_t stream;
cusparseMatDescr_t spdescrA;
csrcholInfo_t sp_chol_info;
// cublasFillMode_t dnUplo;
// cublasHandle_t blasHandle;

// void setCublasAndCuSparse()
// {
// 	cusolverDnCreate(&dnHandle);
// 	// dnUplo = CUBLAS_FILL_MODE_LOWER;
// 	// cublasCreate(&blasHandle);
// 	cusolverSpCreate(&cusolverSpH);
// 	cudaStreamCreate(&stream);
// 	cusolverSpSetStream(cusolverSpH, stream);
// 	cusparseCreate(&cusparseH);
// 	cusparseSetStream(cusparseH, stream);
// 	cusparseCreateMatDescr(&spdescrA);
// 	cusolverSpCreateCsrcholInfo(&sp_chol_info);
// 	cusparseSetMatType(spdescrA, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(spdescrA, CUSPARSE_INDEX_BASE_ZERO);
// }

// void freeCublasAndCusparse()
// {
// 	cusolverDnDestroy(dnHandle);
// 	// cublasDestroy(blasHandle);
// 	cusolverSpDestroy(cusolverSpH);
// 	cudaStreamDestroy(stream);
// 	cusparseDestroy(cusparseH);
// 	cusolverSpDestroyCsrcholInfo(sp_chol_info);
// }

// void gpuCholSolver(CsrSparseMatrix& hess, float* x)
// {
//     // hess must be filled by all nonzero value.
//     float tol = 1.e-12f;
//     const int reorder = 0; // symrcm
//     int singularity = 0;


//     auto values = thrust::raw_pointer_cast(hess.values.data());
//     auto outer_start = thrust::raw_pointer_cast(hess.outer_start.data());
//     auto inner = thrust::raw_pointer_cast(hess.inner.data());
//     auto rhs = thrust::raw_pointer_cast(cuda_globals.b.data());
//     cusolverStatus_t t = cusolverSpScsrlsvchol(
//         cusolverSpH, hess.rows, hess.nnz,
//         spdescrA, values, outer_start, inner,
//         rhs, tol, reorder, x, &singularity);
//     cudaDeviceSynchronize();
//     if (0 <= singularity)
//     {
//         printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
//     }

//     //checkNumericalPrecisionHost(m_activeDims, x);
// }



// __global__ void find_stairs_kernel(int * start sorted_and_unique, int *offset)

// build map i, j -> i 
// with sorted array whose ith element is (i,j)

// why not directly binary search? on the array?


