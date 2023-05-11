#include "cuda_header.cuh"
#include "cuda_globals.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "autogen/autogen.cuh"
using namespace std;

__device__ __host__ float vf_distance(vec3f _v, Facef f, int& _pt_type);
__host__ __device__ void dev_project_to_psd(int dim, float* A){

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

__host__ __device__ void pt_grad_hess12x12(vec3f pt[4], float pt_grad[12], float pt_hess[144], bool psd = true)
{

    int type;
    auto dist = vf_distance(pt[0], Facef{pt[1], pt[2], pt[3]}, type);
    dev::point_triangle_distance_gradient(pt[0], pt[1], pt[2], pt[3], pt_grad, type);
    dev::point_triangle_distance_hessian(pt[0], pt[1], pt[2], pt[3], pt_hess, type);

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

__device__ void put(float* values, i2 offset_stride, float* mat12x12)
{
    int offset = offset_stride[0], stride = offset_stride[1];
    for (int c = 0; c < 12; c++)
        for (int r = 0; r < 12; r++) {
            atomicAdd(values + offset + c * stride + r, mat12x12[c * 12 + r]);
        }
}


__device__ void put_T(float* values, i2 offset_stride, float* mat12x12)
{
    int offset = offset_stride[0], stride = offset_stride[1];
    for (int c = 0; c < 12; c++)
        for (int r = 0; r < 12; r++) {
            atomicAdd(values + offset + c * stride + r, mat12x12[r * 12 + c]);
        }
}

__device__ i2 to_os(i2 ij, int lut_size, i2* lut, int* outers)
{
    int k = binary_search(lut_size, lut, ij);
    return offset_and_stride(k, lut, outers);
}

__global__ void ipc_pt_kernel(
    int npt,
    i2* pt, i2* ij,
    cudaAffineBody* cubes,

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

    auto grad_start = buffer;
    auto hess_start = buffer + 12 * blockDim.x;
    auto hess_p_start = hess_start + 144 * blockDim.x;
    auto hess_t_start = hess_p_start + 144 * blockDim.x;
    auto off_diag_start = hess_t_start + 144 * blockDim.x;

    // FIXME: align with cacheline size
    // FIXME: make sure ipc autogen is row-major


    int n_tasks_per_thread = (npt + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;

    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < npt) {
            float* ipc_hess = hess_start + 144 * tid;
            float* pt_grad = grad_start + tid * 12;

            int ii = ij[I][0], jj = ij[I][1];
            auto &ci{ cubes[ii] }, &cj{ cubes[jj] };
            auto fp {cj.triangle(pt[I][1])};
            vec3f projected[4] {
                ci.projected[pt[I][0]],
                fp.t0,
                fp.t1,
                fp.t2
            };

            pt_grad_hess12x12(projected, pt_grad, ipc_hess);

            vec3f p_tile, t0_tile, t1_tile, t2_tile;
            p_tile = ci.vertices[pt[I][0]];
            Facef f = cj.triangle_at_rest(pt[I][1]);
            t0_tile = f.t0;
            t1_tile = f.t1;
            t2_tile = f.t2;

            float kerp[4]{ 1.0, p_tile.x, p_tile.y, p_tile.z },
                kert[3][4]{
                    { 1.0, t0_tile.x, t0_tile.y, t0_tile.z },
                    { 1.0, t1_tile.x, t1_tile.y, t1_tile.z },
                    { 1.0, t2_tile.x, t2_tile.y, t2_tile.z }
                };

            float *hess_p, *hess_t, *off_diag; 
            float dgp[12], dgt[12];

            hess_p = hess_p_start + 144 * tid;
            hess_t = hess_t_start + 144 * tid;
            off_diag = off_diag_start + 144 * tid;

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    // set hess_p
                    for (int c = 0; c < 3; c++)
                        for (int r = 0; r < 3; r++) {
                            hess_p[rc_to_1d(i * 3 + r, j * 3 + c)] = ipc_hess[rc_to_1d(r, c)] * kerp[i] * kerp[j];
                        }


                    // set hess_t
                    for (int c=  0; c < 3; c++)
                        for (int r= 0; r< 3; r++) {
                            hess_t[rc_to_1d(i * 3 + r, j * 3 + c)] = 0.0f;
                        }
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++) {
                            for (int c = 0; c < 3; c++)
                                for (int r = 0; r < 3; r++) {
                                    hess_t[rc_to_1d(i * 3 + r, j * 3 + c)] += ipc_hess[rc_to_1d((k + 1) * 3 + r, (l + 1) * 3 + c)] * kert[k][i] * kert[l][j];
                                }
                        }

                    // set off_diag
                    for (int c=  0; c < 3; c++)
                        for (int r= 0; r< 3; r++) {
                            off_diag[rc_to_1d(i * 3 + r, j * 3 + c)] = 0.0f;
                        }
                    for (int l = 0; l < 3; l++) {
                        for (int c = 0; c < 3; c++) {
                            for (int r = 0; r < 3; r++) {
                                off_diag[rc_to_1d(i * 3 + r, j * 3 + c)] += ipc_hess[rc_to_1d(r, (l + 1) * 3 + c)] * (kerp[i] * kert[l][j]);
                            }
                        }
                    }
                }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    dgp[i * 3 + j] = pt_grad[j] * kerp[i];
                    dgt[i * 3 + j] = pt_grad[j + 3] * kert[0][i]
                        + pt_grad[j + 6] * kert[1][i]
                        + pt_grad[j + 9] * kert[2][i];
                }
            }

            auto osii = to_os({ ii, ii }, lut_size, lut, outers);
            auto osij = to_os({ ii, jj }, lut_size, lut, outers);
            auto osjj = to_os({ jj, jj }, lut_size, lut, outers);
            auto osji = to_os({ jj, ii }, lut_size, lut, outers);

            if (ci.mass > 0.0f) {
                if (cj.mass > 0.0f)
                    put_T(values, osji, off_diag);
                put(values, osii, hess_p);
                for (int i = 0; i < 12; i++) {
                    atomicAdd(b + i + ii * 12, dgp[i]);
                }
            }
            if (cj.mass > 0.0f) {
                if (ci.mass > 0.0f)
                    put(values, osij, off_diag);
                put(values, osjj, hess_t);
                for (int i = 0; i < 12; i++) {
                    atomicAdd(b + i + jj * 12, dgt[i]);
                }
            }
        }
    }
}

