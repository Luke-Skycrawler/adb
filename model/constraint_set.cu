#include "cuda_header.cuh"
#include "cuda_globals.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "autogen/autogen.cuh"
using namespace std;

__device__ __host__ float vf_distance(vec3f _v, Facef f, int& _pt_type);
__host__ __device__ void dev_project_to_psd(int dim, float* A){

}

__host__ __device__ i2 offset_and_stride(int I, const i2* lut, int* outers)
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


__host__ __device__ void pt_grad_hess12x12(vec3f *pt, 
    float *pt_grad, float *pt_hess, bool psd, 
    float *buf  // for local grad and hess return
)
{

    int type;
    auto dist = vf_distance(pt[0], Facef{pt[1], pt[2], pt[3]}, type);
    dev::point_triangle_distance_gradient(pt[0], pt[1], pt[2], pt[3], pt_grad, type, buf);
    dev::point_triangle_distance_hessian(pt[0], pt[1], pt[2], pt[3], pt_hess, type, buf);

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

__host__ __device__ int binary_search(int lut_size, i2* lut, i2 value)
{
    int l = 0, u = lut_size - 1;
    while (l < u) {
        int mid = (l + u) / 2;
        if (lut[mid] < value)
            l = mid + 1;
        else
            u = mid;
    }
    // u is the immediate value >= x
    return u;
}

__forceinline__ int __device__ __host__ rc_to_1d(int r, int c)
{
    return c * 12 + r;
}

// void ipc_term_pt(int npt, i2 *ij, i2 *body_index, int lut_size, i2 * lut)
// __global__ void ipc_pt_batch_kernel(int npt, vec3f *p, Facef * t) {
// }

// __device__ void JTJ(vec3f a, float* ipc_hess)
// {
// }

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
#define CPU_REF
#ifdef CPU_REF
void _put(float* values, i2 offset_stride, float* mat12x12)
{
    int offset = offset_stride[0], stride = offset_stride[1];
    for (int c = 0; c < 12; c++)
        for (int r = 0; r < 12; r++) {
            values[offset + c * stride + r]+= mat12x12[c * 12 + r];
        }
}


void _put_T(float* values, i2 offset_stride, float* mat12x12)
{
    int offset = offset_stride[0], stride = offset_stride[1];
    for (int c = 0; c < 12; c++)
        for (int r = 0; r < 12; r++) {
            values [offset + c * stride + r]+= mat12x12[r * 12 + c];
        }
}
#endif
__host__ __device__ i2 to_os(i2 ij, int lut_size, i2* lut, int* outers)
{
    int k = binary_search(lut_size, lut, ij);
    return offset_and_stride(k, lut, outers);
}

void get_submat_glue(
    int ii, int jj, 
    float *submat12x12
) 
{
    auto &g{host_cuda_globals};
    auto &h {g.hess};
    auto outers = from_thrust(h.outer_start);
    auto values = from_thrust(h.values);
    auto lut = from_thrust(g.lut);
    auto inners = from_thrust(h.inner);
    auto os = to_os(i2{ii, jj}, g.lut_size, lut.data(), outers.data());
    int offset = os[0], stride = os[1];
    for (int c = 0; c < 12; c++) for (int r = 0; r < 12; r ++){
        submat12x12[c * 12 + r] = values[offset + c * stride + r];
        if (inners[offset + c * stride + r] != jj * 12 + r) {
            printf("index error, should be (%d %d) but inner index = %d\n", jj * 12 + r, ii * 12 + c, inners[offset + c * stride + r]);
        }
    }
}

__global__ void ipc_pt_kernel(
    int npt,
    i2* pt, i2* ij,
    cudaAffineBody* cubes,

    int lut_size, i2* lut,

    float* values, int* outers,
    // CsrSparseMatrix& sparse_hess,
    float* b, // rhs
    float* buffer,
    float* lambdas, float* Tk

)
{

    // input: pt data, body index, is static
    // output: basis Tk (2x12), lambda, gradient g (12x1), hessian H (12x12)

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grad_start = buffer;
    auto hess_start = buffer + 12 * blockDim.x;
    auto hess_p_start = hess_start + 144 * blockDim.x;
    auto hess_t_start = hess_p_start + 144 * blockDim.x;
    auto off_diag_start = hess_t_start + 144 * blockDim.x;

    // FIXME: align with cacheline size


    int n_tasks_per_thread = (npt + blockDim.x - 1) / blockDim.x;

    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < npt) {

            // printf("tid = %d  created\n", tid);
            int ii = ij[I][0], jj = ij[I][1];
            auto &ci{ cubes[ii] }, &cj{ cubes[jj] };
            // printf("tid = %d  ci, cj access, ij = {%d, %d}\n pt[I] = {%d, %d}", tid, ii, jj, pt[I][0], pt[I][1]);

            auto fp {cj.triangle(pt[I][1])};
            vec3f projected[4] {
                ci.projected[pt[I][0]],
                fp.t0,
                fp.t1,
                fp.t2
            };
            // printf("tid = %d  projected vertices\n", tid);
            
            float* ipc_hess = hess_start + 144 * tid;
            float* pt_grad = grad_start + tid * 12;

            float *hess_p, *hess_t, *off_diag; 
            float *dgp = ipc_hess, *dgt = ipc_hess + 12;
            // reuse ipc_hess buffer, when dgp computation ipc_hess should be used up

            hess_p = hess_p_start + 144 * tid;
            hess_t = hess_t_start + 144 * tid;
            off_diag = off_diag_start + 144 * tid;

            pt_grad_hess12x12(projected, pt_grad, ipc_hess, true, hess_p);
            const auto all_zero = [](float* a) -> int {
                for (int i = 0; i < 144; i++) {
                    if (a[i] != 0.0f) return 0;
                }
                return 1;
            };

            // printf("tid = %d  pt grad hess checkpoint passed\n", tid);
            // printf("pt grad hess zero status: %d\n", all_zero(ipc_hess));

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
                
            // printf("tid = %d  grad_p, grad_t checkpoint passed\n", tid);
            // printf(" zero status: p: %d t: %d, off_diag: %d\n", all_zero(hess_p), all_zero(hess_t), all_zero(off_diag));
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    dgp[i * 3 + j] = pt_grad[j] * kerp[i];
                    dgt[i * 3 + j] = pt_grad[j + 3] * kert[0][i]
                        + pt_grad[j + 6] * kert[1][i]
                        + pt_grad[j + 9] * kert[2][i];
                }
            }
            // printf("tid = %d  dgp checkpoint passed\n", tid);

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
            // printf("tid = %d  output checkpoint passed, success\n", tid);

        }
    }
}

__host__ __device__ void ee_grad_hess12x12(vec3f *ee, float *ee_grad, float *ipc_hess, float * buf_start) {
    int type;
    
    float *buf = buf_start;
    float *mollifier_grad = buf;
    buf += 12;
    float* mollifier_hess = buf;
    buf += 144;
    
    type = dev::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
    float dist = dev::edge_edge_distance(ee[0], ee[1], ee[2], ee[3], type);
    auto ei = ee[1] - ee[0], ej = ee[3] - ee[2];
    float eps_x = dev::eps * dot(ei, ei) * dot(ej, ej);

    float p = dev::edge_edge_mollifier(ee[0], ee[1], ee[2], ee[3], eps_x);
    dev::edge_edge_mollifier_gradient(ee[0], ee[1], ee[2], ee[3], eps_x, mollifier_grad);
    dev::edge_edge_mollifier_hessian(ee[0], ee[1], ee[2], ee[3], eps_x, mollifier_grad, mollifier_hess);
    
    
    
    dev::edge_edge_distance_gradient(ee[0], ee[1], ee[2], ee[3], ee_grad, type, buf);
    dev::edge_edge_distance_hessian(ee[0], ee[1], ee[2], ee[3], ipc_hess, type, buf);

    float B = dev::barrier_function(dist);
    float B_ = dev::barrier_derivative_d(dist);
    float B__ = dev::barrier_second_derivative(dist);

    for (int I = 0; I < 144; I++) {
        int i = I % 12, j = I / 12; // column major
        ipc_hess[i] = mollifier_hess[i] * B + B_ * (mollifier_grad[i] * ee_grad[j] + mollifier_grad[j] * ee_grad[i]) + p * (B__ * ee_grad[i] * ee_grad[j] + B_ * ipc_hess[i]);
    }
    for (int i = 0; i < 12; i++) {
        ee_grad[i] = ee_grad[i] * B_ + mollifier_grad[i] * B;
    }
    dev_project_to_psd(12, ipc_hess);
}

__global__ void put_inertia_kernel(
    int n_cubes, 
    cudaAffineBody *cubes,
    int lut_size, i2 *lut,
    float *values, int *outers,
    float *b,
    float *diag
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i =0; _i < n_task_per_thread; _i ++) {
        int I = _i + tid * n_tasks_per_thread;
        if (I < n_cubes){
            auto osii = to_os({ I, I }, lut_size, lut, outers);
            int offset = osii[0], stride = osii[1];

            for (int c = 0; c < 12; c++)
                for (int r =0; r < 12; r ++) {
                    values[offset + c * stride + r] += diag[c * 12 + r];
                }
            // grad is already added to globals.b
        }
    }
}
__global__ void ipc_ee_kernel(
    int nee, 
    i2 *ee, i2 *ij, 
    cudaAffineBody *cubes,
    int lut_size, i2 *lut,
    float *values, int *outers,
    float *b,
    float *buffer,
    float *lambdas, float *Tk
) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grad_start = buffer;
    auto hess_start = buffer + 12 * blockDim.x;
    auto hess_0_start = hess_start + 144 * blockDim.x;
    auto hess_1_start = hess_0_start + 144 * blockDim.x;
    auto off_diag_start = hess_1_start + 144 * blockDim.x;

    int n_tasks_per_thread = (nee + blockDim.x - 1) / blockDim.x;

    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < nee) {
            int ii = ij[I][0], jj = ij[I][1];
            auto &ci{ cubes[ii] }, &cj{ cubes[jj] };

            auto ei {ci.edge(ee[I][0])}, ej {cj.edge(ee[I][1])};
            vec3f projected[4] {
                ei.e0, ei.e1, ej.e0, ej.e1
            };
            float* ipc_hess = hess_start + 144 * tid;
            float* ee_grad = grad_start + tid * 12;

            float *hess_0, *hess_1, *off_diag; 
            float *dg0 = ipc_hess, *dg1 = ipc_hess + 12;
            hess_0 = hess_0_start + 144 * tid;
            hess_1 = hess_1_start + 144 * tid;
            off_diag = off_diag_start + 144 * tid;

            ee_grad_hess12x12(projected, ee_grad, ipc_hess, hess_0);

            vec3f ei0_tile, ei1_tile, ej0_tile, ej1_tile;
            auto eir {ci.edge_at_rest(ee[I][0])}, ejr {cj.edge_at_rest(ee[I][1])};
            ei0_tile = eir.e0; ei1_tile = eir.e1;
            ej0_tile = ejr.e0; ej1_tile = ejr.e1;

            float ker0[][4] {
                {1.0f, ei0_tile.x, ei0_tile.y, ei0_tile.z},
                {1.0f, ei1_tile.x, ei1_tile.y, ei1_tile.z},
            }, ker1[][4]{
                {1.0f, ej0_tile.x, ej0_tile.y, ej0_tile.z},
                {1.0f, ej1_tile.x, ej1_tile.y, ej1_tile.z}
            };

            // fill hess_0, hess_2, off_diag
            for(int i = 0;  i < 4; i ++)
                for (int j = 0; j < 4; j ++) {
                    for (int c = 0; c < 3; c++)
                        for (int  r = 0; r < 3; r++) {
                            hess_1[rc_to_1d(i * 3+ r, j * 3 + c)] = 0.0f;
                            hess_0[rc_to_1d(i * 3+ r, j * 3 + c)] = 0.0f;
                            off_diag[rc_to_1d(i * 3+ r, j * 3 + c)] = 0.0f;
                        }
                    for (int k = 0; k < 2; k++) 
                        for(int l = 0; l < 2; l ++){
                            
                            for (int c = 0; c < 3; c++)
                                for (int r = 0; r < 3; r++) {
                                    hess_0[rc_to_1d(i *  3 + r, j * 3 + c)] += ipc_hess[rc_to_1d(k * 3 + r, l * 3 + c)] * ker0[k][i] * ker0[l][j];

                                    hess_1[rc_to_1d(i  * 3+ r, j * 3 + j)] += ipc_hess[rc_to_1d((k + 2) * 3 + r, (l + 2) * 3 + c)] * ker1[k][i] * ker1[l][j];

                                    off_diag[rc_to_1d(i * 3 + r, j * 3 + c)] += ipc_hess[rc_to_1d(k * 3 + r, (l + 2) * 3 + c)] * ker0[k][i] * ker1[l][j];
                                }
                        }

                }
            
            // fill dg0, dg1
            for (int i = 0; i < 12; i++) {
                dg0[i] = ker0[0][i / 3] * ee_grad[i % 3] + ker0[1][i / 3] * ee_grad[i % 3 + 3];
                dg1[i] = ker1[0][i / 3] * ee_grad[i % 3 + 6] + ker1[1][i / 3] * ee_grad[i % 3 + 9];
            }
            
            auto osii = to_os({ ii, ii }, lut_size, lut, outers);
            auto osij = to_os({ ii, jj }, lut_size, lut, outers);
            auto osjj = to_os({ jj, jj }, lut_size, lut, outers);
            auto osji = to_os({ jj, ii }, lut_size, lut, outers);

            if (ci.mass > 0.0f) {
                if (cj.mass > 0.0f)
                    put_T(values, osji, off_diag);
                put(values,osii, hess_0);
                for (int i = 0; i < 12; i ++) {
                    atomicAdd(b + i + ii * 12, dg0[i]);
                }
            }
            if (cj.mass > 0.0f) {
                if (ci.mass > 0.0f)
                    put(values, osij, off_diag);
                put(values, osjj, hess_1);
                for (int i = 0; i < 12; i++) {
                    atomicAdd(b + i + jj * 12, dg1[i]);
                }
            }

        }
    }
}
#define CPU_REF
#ifdef CPU_REF
void ipc_pt_cpu(
    int npt,
    i2* pt, i2* ij,
    cudaAffineBody* cubes,

    int lut_size, i2* lut,

    float* values, int* outers,
    // CsrSparseMatrix& sparse_hess,
    float* b, // rhs
    float* buffer,
    float* lambdas, float* Tk

)
{

    // input: pt data, body index, is static
    // output: basis Tk (2x12), lambda, gradient g (12x1), hessian H (12x12)

    // FIXME: align with cacheline size
    // FIXME: make sure ipc autogen is row-major

    for (int I = 0; I < npt; I++) {

        int ii = ij[I][0], jj = ij[I][1];
        auto &ci{ cubes[ii] }, &cj{ cubes[jj] };

        auto fp{ cj.triangle(pt[I][1]) };
        vec3f projected[4]{
            ci.projected[pt[I][0]],
            fp.t0,
            fp.t1,
            fp.t2
        };

        float ipc_hess[144];
        float pt_grad[12];

        float hess_p[144], hess_t[144], off_diag[144];
        float *dgp = ipc_hess, *dgt = ipc_hess + 12;
        // reuse ipc_hess buffer, when dgp computation ipc_hess should be used up

        pt_grad_hess12x12(projected, pt_grad, ipc_hess, true, hess_p);

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

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                // set hess_p
                for (int c = 0; c < 3; c++)
                    for (int r = 0; r < 3; r++) {
                        hess_p[rc_to_1d(i * 3 + r, j * 3 + c)] = ipc_hess[rc_to_1d(r, c)] * kerp[i] * kerp[j];
                    }

                // set hess_t
                for (int c = 0; c < 3; c++)
                    for (int r = 0; r < 3; r++) {
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
                for (int c = 0; c < 3; c++)
                    for (int r = 0; r < 3; r++) {
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
                _put_T(values, osji, off_diag);
            _put(values, osii, hess_p);
            for (int i = 0; i < 12; i++) {
                b [i + ii * 12] += dgp[i];
            }
        }
        if (cj.mass > 0.0f) {
            if (ci.mass > 0.0f)
                _put(values, osij, off_diag);
            _put(values, osjj, hess_t);
            for (int i = 0; i < 12; i++) {
                b[i + jj * 12] += dgt[i];
            }
        }
    }
}

#endif
void project_glue(int vtn);
void cuda_ipc_glue()
{
    auto& g{ host_cuda_globals };
    auto& lt{ g.leader_thread_buffer_back };
    auto lt_stashed = lt;
    
    auto ps = from_thrust(thrust::device_vector<i2>(g.pt.p, g.pt.p + g.npt));
    auto bs = from_thrust(thrust::device_vector<i2>(g.pt.b, g.pt.b + g.npt));
    

    project_glue(1);
    if (g.params["ipc_cpu_debug"]) {
        float b[12], buf[144];
        auto host_cubes = host_cuda_globals.host_cubes;
        vec3f* host_projected = new vec3f[host_cuda_globals.n_vertices], *host_vertices = new vec3f[host_cuda_globals.n_vertices];
        int* host_edges = new int[host_cuda_globals.n_edges * 2];
        int* host_faces = new int[host_cuda_globals.n_faces * 3];

        int start = 0;
        cudaMemcpy(host_projected, host_cuda_globals.projected_vertices, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_edges, host_cuda_globals.edges, sizeof(int) * host_cuda_globals.n_edges * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_faces, host_cuda_globals.faces, sizeof(int) * host_cuda_globals.n_faces * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_vertices, host_cuda_globals.vertices_at_rest, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
        for (int i = 0; i < host_cubes.size(); i++) {
            host_cubes[i].projected = host_projected + start;
            host_cubes[i].vertices  = host_vertices + start;
            start += host_cubes[i].n_vertices;
            host_cubes[i].edges = host_cubes[i].edges - host_cuda_globals.edges + host_edges;
            host_cubes[i].faces = host_cubes[i].faces - host_cuda_globals.faces + host_faces;
        }

        thrust::host_vector<float> ret_values = g.hess.values;
        if (g.params["pt_enable"])
        ipc_pt_cpu(g.npt, ps.data(), bs.data(),
            host_cubes.data(),
            g.lut_size, from_thrust(g.lut).data(),
            ret_values.data(), from_thrust(g.hess.outer_start).data(),
            b, buf,
            nullptr, nullptr);
        // if (g.params["ee_enable"])
        // ipc_ee_cpu(g.nee, from_thrust)

        g.hess.values = ret_values;

        delete []host_projected;
        delete []host_edges;
        delete []host_faces;
        delete []host_vertices;

    }
    else {
        if (g.params["pt_enable"])
        ipc_pt_kernel<<<1, 1>>>(g.npt, g.pt.p, g.pt.b,
            g.cubes,
            g.lut_size, PTR(g.lut),
            PTR(g.hess.values), PTR(g.hess.outer_start),
            g.b, (float*)lt,
            nullptr, nullptr);
        if (g.params["ee_enable"]) 
        ipc_ee_kernel<<<1, 1>>>(g.nee, g.ee.p, g.ee.b,
            g.cubes,
            g.lut_size, PTR(g.lut),
            PTR(g.hess.values), PTR(g.hess.outer_start),
            g.b, (float*)lt,
            nullptr, nullptr);
        put_inertia_kernel<<<1, 1>>>(
            g.n_cubes,
            g.cubes,
            g.lut_size, PTR(g.lut),
            PTR(g.hess.values), PTR(g.hess.outer_start),
            g.b, g.hess_diag
        );
    }
    
    CUDA_CALL(cudaDeviceSynchronize());
    const auto all_zero = [](vector<float> values, int n) {
        for (int i = 0; i < n; i++) {
            if (values[i] != 0.0f) return false;
        }
        return true;
    };
    if (g.npt && all_zero(from_thrust(g.hess.values), g.hess.nnz)) {
    printf("\nerror: all zero in sparse matrix\n");
    }
    lt = lt_stashed;
}


inline __host__ __device__ vec3f linerp(vec3f pt1, vec3f pt0, float t){
    return t * pt1 + (1.0f - t) * pt0;
}
inline __host__ __device__ Facef linerp(Facef tt1, Facef tt0, float t) {
    return Facef {
        linerp(tt1.t0, tt0.t0, t),
        linerp(tt1.t1, tt0.t1, t),
        linerp(tt1.t2, tt0.t2, t),
    };
}
inline __host__ __device__ Edgef linerp(Edgef et1, Edgef et0, float t) {
    return Edgef {
        linerp(et1.e0, et0.e0, t),
        linerp(et1.e1, et0.e1, t),
    };
}

__host__ __device__ void cubic_binomial (float a[3], float b[3], float ret_polynomial[4]){
    polynomial[0] += b[0] * b[1] * b[2];
    polynomial[1] += a[0] * b[1] * b[2] + b[0] * b[1] * a[2] + b[0] * a[1] * b[2];
    polynomial[2] += a[0] * a[1] * b[2] + b[0] * a[1] * a[2] + a[0] * b[1] * a[2];
    polynomial[3] += a[0] * a[1] * a[2];
}

__forceinline__ int rc_3(int i, int j) {
    return 3 * j + i;
}
__host__ __device__ void det_polynomial(
    // const mat3& a, const mat3& b
    float *a, float *b, float *ret_polynomial
)
{
    float pos_polynomial[4]{ 0.0 }, neg_polynomial[4]{ 0.0 };
    float c11c22c33[2][3]{
        { a[rc_3(0, 0)], a[rc_3(1, 1)], a[rc_3(2, 2)] },
        { b[rc_3(0, 0)], b[rc_3(1, 1)], b[rc_3(2, 2)] }
    },
        c12c23c31[2][3]{
            { a[rc_3(0, 1)], a[rc_3(1, 2)], a[rc_3(2, 0)] },
            { b[rc_3(0, 1)], b[rc_3(1, 2)], b[rc_3(2, 0)] }
        },
        c13c21c32[2][3]{
            { a[rc_3(0, 2)], a[rc_3(1, 0)], a[rc_3(2, 1)] },
            { b[rc_3(0, 2)], b[rc_3(1, 0)], b[rc_3(2, 1)] }
        };
    float c11c23c32[2][3]{
        { a[rc_3(0, 0)], a[rc_3(1, 2)], a[rc_3(2, 1)] }, { b[rc_3(0, 0)], b[rc_3(1, 2)], b[rc_3(2, 1)] }
    },
        c12c21c33[2][3]{
            { a[rc_3(0, 1)], a[rc_3(1, 0)], a[rc_3(2, 2)] }, { b[rc_3(0, 1)], b[rc_3(1, 0)], b[rc_3(2, 2)] }
        },
        c13c22c31[2][3]{
            { a[rc_3(0, 2)], a[rc_3(1, 1)], a[rc_3(2, 0)] }, { b[rc_3(0, 2)], b[rc_3(1, 1)], b[rc_3(2, 0)] }
        };
    cubic_binomial(
        c11c22c33[0],
        c11c22c33[1],
        pos_polynomial);
    cubic_binomial(
        c12c23c31[0],
        c12c23c31[1],
        pos_polynomial);
    cubic_binomial(
        c13c21c32[0],
        c13c21c32[1],
        pos_polynomial);
    cubic_binomial(
        c11c23c32[0],
        c11c23c32[1],
        neg_polynomial);
    cubic_binomial(
        c12c21c33[0],
        c12c21c33[1],
        neg_polynomial);
    cubic_binomial(
        c13c22c31[0],
        c13c22c31[1],
        neg_polynomial);
    for (int i = 0; i < 4; i++) ret_polynomial[i] = pos_polynomial[i] - neg_polynomial[i];
}



int build_and_solve_4_points_coplanar(
    const vec3f& p0_t0,
    const vec3f& p1_t0,
    const vec3f& p2_t0,
    const vec3f& p3_t0,

    const vec3f& p0_t1,
    const vec3f& p1_t1,
    const vec3f& p2_t1,
    const vec3f& p3_t1,

    float roots[3])
{
    mat3 a1, a2, a3, a4;
    mat3 b1, b2, b3, b4;

    b1 << p1_t0, p2_t0, p3_t0;
    b2 << p0_t0, p2_t0, p3_t0;
    b3 << p0_t0, p1_t0, p3_t0;
    b4 << p0_t0, p1_t0, p2_t0;

    a1 << p1_t1, p2_t1, p3_t1;
    a2 << p0_t1, p2_t1, p3_t1;
    a3 << p0_t1, p1_t1, p3_t1;
    a4 << p0_t1, p1_t1, p2_t1;

    a1 -= b1;
    a2 -= b2;
    a3 -= b3;
    a4 -= b4;

    float a1[9] {
        pt_t1.x - p1_t0.x, pt_t1.y - p1_t0.y, pt_t1.z - p1_t0.z,
        pt_t2.x - p2_t0.x, pt_t2.y - p2_t0.y, pt_t2.z - p2_t0.z,
        pt_t3.x - p3_t0.x, pt_t3.y - p3_t0.y, pt_t3.z - p3_t0.z
    },
    a2[9] {
        p0_t1.x - p0_t0.x, p0_t1.y - p0_t0.y, p0_t1.z - p0_t0.z,
        p2_t1.x - p2_t0.x, p2_t1.y - p2_t0.y, p2_t1.z - p2_t0.z,
        p3_t1.x - p3_t0.x, p3_t1.y - p3_t0.y, p3_t1.z - p3_t0.z
    }, 
    a3[9] {
        p0_t1.x - p0_t0.x, p0_t1.y - p0_t0.y, p0_t1.z - p0_t0.z,
        p1_t1.x - p1_t0.x, p1_t1.y - p1_t0.y, p1_t1.z - p1_t0.z,
        p3_t1.x - p3_t0.x, p3_t1.y - p3_t0.y, p3_t1.z - p3_t0.z
    },
    a4[9] {
        p0_t1.x - p0_t0.x, p0_t1.y - p0_t0.y, p0_t1.z - p0_t0.z,
        p1_t1.x - p1_t0.x, p1_t1.y - p1_t0.y, p1_t1.z - p1_t0.z,
        p2_t1.x - p2_t0.x, p2_t1.y - p2_t0.y, p2_t1.z - p2_t0.z
    };
    float b1[9] {
        p1_t0.x, p1_t0.y, p1_t0.z,
        p2_t0.x, p2_t0.y, p2_t0.z,
        p3_t0.x, p3_t0.y, p3_t0.z
    }, 
    b2[9] {
        p0_t0.x, p0_t0.y, p0_t0.z,
        p2_t0.x, p2_t0.y, p2_t0.z,
        p3_t0.x, p3_t0.y, p3_t0.z
    },
    b3[9] {
        p0_t0.x, p0_t0.y, p0_t0.z,
        p1_t0.x, p1_t0.y, p1_t0.z,
        p3_t0.x, p3_t0.y, p3_t0.z
    },
    b4[9] {
        p0_t0.x, p0_t0.y, p0_t0.z,
        p1_t0.x, p1_t0.y, p1_t0.z,
        p2_t0.x, p2_t0.y, p2_t0.z
    }; 
    
    float ret_polynomial[4] {0.0f};
    float tmp_polynomial[4] {0.0f};

    det_polynomial(a1, b1, ret_polynomial);
    det_polynomial(a2, b2, tmp_polynomial);
    for (int i = 0 ; i < 4; i ++) ret_polynomial[i] -= tmp_polynomial[i];
    det_polynomial(a3, b3, tmp_polynomial);
    for (int i = 0 ; i < 4; i ++) ret_polynomial[i] += tmp_polynomial[i];
    det_polynomial(a4, b4, tmp_polynomial);
    for (int i = 0 ; i < 4; i ++) ret_polynomial[i] -= tmp_polynomial[i];

    double root = 1.0;
    int found = cubic_roots(roots, ret_polynomial, 0.0, 1.0);
    return found;
}


__device__ __host__ bool _cross(const Edgef &ei, const Edgef &ej){
    auto vei = ei.e1 - ei.e0;
    auto vej0 = ej.e0 - ei.e0;
    auto vej1 = ej.e1 - ei.e0;
    return (cross(vei, vej0), cross(vei, vej1)) < 0.0f;
}

__device__ __host__ bool verify_root_ee(
    const Edgef &ei, 
    const Edgef &ej
) {
    return _cross(ei, ej) && _cross(ej, ei);
}

__forceinline__ __device__ __host__ bool inside(const Facef &f, const vec3f &p) {
    auto f01 = cross(t0 - p, t1- p);
    auto f12 = cross(t1 - p, t2- p);
    auto f20 = cross(t2 - p, t0- p);
    return dot(f01, f12) >= 0.0f && dot(f12, f20) >= 0.0f;
}

__device__ __host__ verify_root_pt(
    const vec3f &p, const Facef &f
 ) {
    auto n = f.unit_normal();
    double d = dot(n, p - f.t0);
    auto v = p - d * n;
    return inside(f, v);
 }
__device__ __host__ float pt_collision_time(
    const vec3f &p0,
    const Facef &t0, 
    const vec3f &p1,
    const Facef &t1
){
    float roots[3];
    int found = build_and_solve_4_points_coplanar(p0, t0.t0, t0.t1, t0.t2, p1, t1.t0, t1.t1, t1.t2, roots);
    bool true_root = false;
    for (int i = 0; i < found && !true_root; i ++) {
        root = roots[i];
        true_root = verify_root_pt(linerp(p1, p0, root), linerp(t1, t0, root));
    }
    return found && true_root? root: 1.0f;
}

__device__ __host__ float ee_collision_time(
    const Edgef &ei0, 
    const Edgef &ej0,
    const Edgef &ei1,
    const Edgef &ej1
){
    float roots[3];
    int found = build_and_solve_4_points_coplanar(
        ei0.e0, ei0.e1, ej0.e0, ej0.e1,
        ei1.e0, ei1.e1, ej1.e0, ej1.e1,
        roots
    );
    float root = 1.0f;
    bool true_root = false;
    for (int i = 0; i < found && !true_root; i ++) {
        root = roots[i];
        true_root = verify_root_ee(linerp(ei1, ei0, root), lerp(ej1, ej0, root));
    }
    return found && true_root? root: 1.0f;
}