#include "cuda_header.cuh"
#include "cuda_globals.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>
// #include "autogen/autogen.cuh"
using namespace std;

namespace dev {
//__device__ __constant__ float kappa = 1e-1f, d_hat = 1e-4f, d_hat_sqr = 1e-2f;

__host__ __device__ float barrier_derivative_d(float x)
{
    if (x >= d_hat)
        return 0.0f;
    return -(x - d_hat) * kappa * (2 * log(x / d_hat) + (x - d_hat) / x) / (d_hat * d_hat);
}
__host__ __device__ float barrier_second_derivative(float d)
{
    if (d >= d_hat)
        return 0.0f;
    return -kappa * (2 * log(d / d_hat) + (d - d_hat) / d + (d - d_hat) * (2 / d + d_hat / d / d)) / (d_hat * d_hat);
}

}

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

__device__ i2 offset_and_stride(int I, const i2* lut, int* outers)
{
    // TEST COVERED
    int i = lut[I][0];
    int j = lut[I][1];
    int col_start = outers[i * 12] / (12 * 12);
    int sub_mat_offset = I - col_start;
    int sub_mat_start = outers[i * 12] + sub_mat_offset * 12;
    int stride = outers[i * 12 + 1] - outers[i * 12];

    return { sub_mat_start, stride };
}

__global__ void fill_inner_outers_kernel(int n_cubes, int lut_size, const i2 * lut, int *inners, int *outers) {
    // TESTED
    auto tid = threadIdx.x;
    auto n_task_per_thread = (lut_size+ blockDim.x - 1)/ blockDim.x;  
    // precondition: lut contains all the symmetric pairs (i, j) and (j, i), sorted in ascending order

    for (int _i = 0; _i < n_task_per_thread; _i ++) {
        // detect stairs, fill outers
        auto I = _i + n_task_per_thread * tid;
        if (I < lut_size) {
            int i = lut[I][0];
            int last = I == 0 ? -1: lut[I - 1][0];
            
            if (last != i) {
                inners[i] = I;
                // temporary storage, outers should be filled according to this later
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

    int n_block_per_thread = (lut_size + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_block_per_thread; _i ++) {
        // fiil inners
        auto I = _i + n_block_per_thread * tid;
        if (I < lut_size) {

            i2 os = offset_and_stride(I, lut, outers);
            // int i = lut[I][0];
            // int j = lut[I][1];
            // int col_start = outers[i * 12] / (12 * 12);
            // int sub_mat_offset = I - col_start;
            // int sub_mat_start = outers[i * 12] + sub_mat_offset * 12;
            // int stride = outers[i * 12 + 1] - outers[i * 12];
            int sub_mat_start = os[0];
            int stride = os[1];
            int j = lut[I][1];
            for (int c = 0; c < 12; c++) {
                for (int r = 0; r < 12; r++) {
                    inners[sub_mat_start + stride * c + r] = j * 12 + r;
                }
            }
        }
    }

}
void build_csr(int n_cubes, const thrust::device_vector<i2> &lut, CsrSparseMatrix & sparse_matrix) {
    // TESTED
    int lut_size = lut.size();
    int nnz = lut_size * 12 * 12;
    sparse_matrix.rows = n_cubes * 12;
    sparse_matrix.cols = n_cubes * 12;
    sparse_matrix.nnz = nnz;

    sparse_matrix.outer_start.resize(n_cubes * 12);
    sparse_matrix.inner.resize(nnz);
    sparse_matrix.values.resize(nnz);
    
    thrust::device_vector<int> dev_inner(nnz), dev_outer(n_cubes * 12);
    
    thrust::fill(sparse_matrix.values.begin(), sparse_matrix.values.end(), 0.0f);
    
    auto lut_ptr = thrust::raw_pointer_cast(lut.data());
    auto inner_ptr = thrust::raw_pointer_cast(dev_inner.data());
    auto outer_ptr = thrust::raw_pointer_cast(dev_outer.data());
    
    fill_inner_outers_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, lut_size, lut_ptr, inner_ptr, outer_ptr);
    
    sparse_matrix.inner = dev_inner;
    sparse_matrix.outer_start = dev_outer;
    CUDA_CALL(cudaGetLastError());
}


void make_placeholder_sparse_matrix(int n_cubes, CsrSparseMatrix &sparse_matrix) {
    auto& lut{ host_cuda_globals.lut}; 
    build_csr(n_cubes, lut, sparse_matrix);
}

__device__ void pt_grad_hess12x12(vec3f* pt, float* pt_grad, float* pt_hess, bool psd = true)
{

    // auto dist = point_triangle_distance(p, t.t0, t.t1, t.t2);
    // point_triangle_distance_gradient(p, t.t0, t.t1, t.t2, pt_grad);
    // point_triangle_distance_hessian(p, t.t0, t.t1, t.t2, pt_hess);
    auto dist = point_triangle_distance(pt[0], pt[1], pt[2], pt[3]);
    point_triangle_distance_gradient(pt[0], pt[1], pt[2], pt[3], pt_grad);
    point_triangle_distance_hessian(pt[0], pt[1], pt[2], pt[3], pt_hess);

    auto B_ = dev::barrier_derivative_d(dist);
    auto B__ = dev::barrier_second_derivative(dist);

    // pt_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++) {
            pt_hess[j * 12 + i] = B_ * pt_hess[j * 12 + i] + pt_grad[i] * pt_grad[j] * B__;
            // column major
        }
    for (int i = 0; i < 12; i++)
        pt_grad[i] *= B_;

    if (psd)
        dev_project_to_psd(12, pt_hess);
}

__device__ int binary_search(int lut_size, i2* lut, i2 value)
{
    int l = 0, u = lut_size;
    while (l < u) {
        int mid = (l + u) / 2;
        if (lut[mid] < value)
            l = mid + 1;
        else
            u = mid;
    }
    return l;
}

__forceinline__ int __device__ rc_to_1d(int r, int c)
{
    return c * 12 + r;
}

// void ipc_term_pt(int npt, i2 *ij, i2 *body_index, int lut_size, i2 * lut)
// __global__ void ipc_pt_batch_kernel(int npt, vec3f *p, Facef * t) {
// }

void ipc_pt_kernel(
    int npt, vec3f* pt, i2* ij, bool* is_static,
    int lut_size, i2* lut,
    CsrSparseMatrix& sparse_hess,

    float* buffer,
    float* lambdas, float* Tk)
{
}

__device__ void JTJ(vec3f a, float* ipc_hess)
{
}

__device__ void plain_matrix_product(int ar, int ac, int bc, float* a, float* b, float* c)
{
    // assume all in column major
    for (int i = 0; i < ar; i++) {
        for (int j = 0; j < bc; j++) {
            float cij = 0.0f;

            for (int k = 0; k < ac; k++) {
                // cij += aik * bkj
                cij += a[i + k * ar] * b[k + j * ac];
            }
            c[i + j * ar] = cij;
        }
    }
}
__global__ void ipc_pt(
    int npt, vec3f* pt, i2* ij, bool* is_static,
    int lut_size, i2* lut,
    float* values, int* inners, int* outers,
    // CsrSparseMatrix& sparse_hess,
    float* b, // rhs
    float* buffer,
    float* lambdas, float* Tk

)
{

    // input: pt data, body index, is static
    // output: basis Tk (2x12), lambda, gradient g (12x1), hessian H (12x12)
    // __shared__ float pt_grad_hess[13 * 12 * n_cuda_threads_per_block];
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto hess_start = buffer + 12 * blockDim.x;
    // FIXME: align with cacheline size

    // auto values = sparse_hess.values;
    // auto inners = sparse_hess.inner;
    // auto outers = sparse_hess.outer_start;

    int n_tasks_per_thread = (npt + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;

    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < npt) {
            float* ipc_hess = hess_start + 144 * tid;
            float* pt_grad = buffer + tid * 12;

            pt_grad_hess12x12(pt + I * 4, pt_grad, ipc_hess);
            int k = binary_search(lut_size, lut, ij[I]);

            vec3f p_tile, t0_tile, t1_tile, t2_tile; // TODO: forward declare, fill in the args later

            const auto to_3x12 = [] __device__(vec3f x, float* J) {
                // cudaMemset(J, 0, 36 * sizeof(float));
                for (int i = 0; i < 4; i++) {
                    auto e = i == 0 ? 1.0f : i == 1 ? x.x
                        : i == 2                    ? x.y
                                                    : x.z;
                    for (int d = 0; d < 3; d++) {
                        // Jdd = x_{d-1}
                        J[d + d * 3] = e;
                    }
                }
            };

            float Jp[36], Jt[36 * 3];
            to_3x12(p_tile, Jp);
            to_3x12(t0_tile, Jt);
            to_3x12(t1_tile, Jt + 36);
            to_3x12(t2_tile, Jt + 72);

            // float kerp[4]{ 1.0, p_tile.x, p_tile.y, p_tile.z },
            //     kert[][4]{
            //         { 1.0, t0_tile.x, t0_tile.y, t0_tile.z },
            //         { 1.0, t1_tile.x, t1_tile.y, t1_tile.z },
            //         { 1.0, t2_tile.x, t2_tile.y, t2_tile.z }
            //     };

            float *hess_p, *hess_t, *off_diag; // TODO: forward declare, fill in the args later
            // for (int i = 0; i < 4; i++)
            //         for (int j = 0; j < 4; j++) {
            //             for (int c = 0; c < 3; c++)
            //                 for (int r = 0; r < 3; r++) {
            //                     hess_p[rc_to_1d(i * 3 + r, j * 3 + c)] = ipc_hess[rc_to_1d(r, c)] * kerp[i] * kerp[j];
            //                 }
            //             // FIXME: make sure ipc autogen is row-major
            //             for (int k = 0; k < 3; k++)
            //                 for (int l = 0; l < 3; l++) {
            //                     for (int c = 0; c < 3; c++)
            //                         for (int r = 0; r < 3; r++) {
            //                             hess_t[rc_to_1d(i * 3 + r, j * 3 + c)] += ipc_hess[rc_to_1d((k + 1) * 3 + r, (l + 1) * 3 + c)] * kert[k][i] * kert[l][j];
            //                         }
            //                 }
            //             for (int c = 0; c < 3; c++)
            //         }
            // }

            // auto outer_ptr = thrust::raw_pointer_cast(outers.data());
            // auto value_ptr = thrust::raw_pointer_cast(values.data());
            auto os = offset_and_stride(k, lut, outers);
            int offset = os[0], stride = os[1];

            for (int c = 0; c < 12; c++)
                for (int r = 0; r < 12; r++) {
                    // FIXME: add lock, static body
                    atomicAdd(values + offset + c * stride + r, hess_start[144 * tid + c * 12 + r]);
                }
            float dgp[12], dgt[12];
            for (int j = 0; j < 12; j++) {
                float dgpj = 0.0f, dgtj = 0.0f;
                for (int k = 0; k < 3; k++) {
                    // dgp = JpT * pt_grad_3
                    // dgp i = Jp ki * pt_grad_k
                    dgpj += pt_grad[k] * Jp[k + j * 3];
                }
                dgp[j] = dgpj;

                for (int k = 3; k < 12; k++) {
                    dgtj += pt_grad[k] * Jt[k - 3 + j * 3];
                }
                dgt[j] = dgtj;
            }
            for (int i = 0; i < 12; i++) {
                atomicAdd(b + i + ij[I][0] * 12, dgp[i]);
                atomicAdd(b + i + ij[I][1] * 12, dgt[i]);
            }
        }
        // for (int r = 0; r < 12; r++) {
        //     atomicAdd()
        //             pt_grad_hess
        //         +
        // }
    }
}


void make_lut(int lut_size, i2* _lut) {
    int n_cubes = host_cuda_globals.n_cubes;
    
    auto& lut{ host_cuda_globals.lut}; 
    {
        thrust::host_vector<i2> diagonals(n_cubes);
        thrust::device_vector<i2> dev_diagonals(n_cubes);
        for (int i = 0; i < n_cubes; i++) diagonals[i] = { i, i };
        dev_diagonals = diagonals;
        lut.insert(lut.end(), dev_diagonals.begin(), dev_diagonals.end());
    }
 
    
    thrust::sort(lut.begin(), lut.end());
    auto new_end = thrust::unique(lut.begin(), lut.end());

    lut_size = new_end - lut.begin();
    lut.resize(lut_size);
    lut.shrink_to_fit();

}

__global__ void ipc_pt_kernel(
    int n_cubes, int npt,
    i2* prims, i2* body,
    CsrSparseMatrix& hess,
    int lut_size, i2* lut){}