#include "cuda_globals.cuh"
#include <omp.h>

using namespace std;

static __constant__ float kappa = 1e9f, dt = 1e-2f;
CudaGlobals host_cuda_globals;
//__constant__ CudaGlobals *cuda_globals;
//__constant__ float3 gravity;
__host__ __device__ void orthogonal_grad(float3 q[4], float dt, float ret[12]);
__host__ __device__ void orthogonal_hess(float3 q[4], float dt, float ret[144]);
__host__ __device__ void inertia_grad(cudaAffineBody& c, float dt, float ret[12]);
__host__ __device__ void inertia_hess(cudaAffineBody& c, float ret[144]);

void gpuCholSolver(CsrSparseMatrix& hess, float* x, float *b);
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

);

__global__ void norm_kernel(int dim, float *x, float *ret_norm) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (dim + blockDim.x - 1) / blockDim.x;
    __shared__ float norm;
    norm = 0.0f;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < dim) {
            norm += x[i] * x[i];
        }
    }
    __syncthreads();
    *ret_norm = norm;
}

__global__ void set_q_kernel(int n_cubes, cudaAffineBody* cubes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            for (int j = 0; j < 4; j ++)
                c.q_update[j] = c.q[j] = c.q0[j];
        }
    }
}


void iaabb_brute_force_cuda(int n_cubes, luf* aabbs, int ti, int& lut_size, i2* lut, float* dq,
    i2* prim_ret,
    i2* body_ret,
    int * pt_types_ret)
{
}

float upper_bound_cuda(int n_cubes, luf *aabbs, int lut_size, i2 *lut){
    return 1.0f;   
}

__global__ void update_line_search_kernel(int n_cubes, cudaAffineBody *cubes, float* dq, float alpha)
{
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            for (int j = 0; j < 4; j++)
                c.q_update[j] = c.q[j] + make_float3(dq[i * 12 + j * 3 + 0], dq[i * 12 + j * 3 + 1], dq[i * 12 + j * 3 + 2]) * alpha;
        }
    }
}

__global__ void barrier_kernel(float& E, int n_col, float* distance)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float e[n_cuda_threads_per_block];
    auto n_tasks_per_thread = (n_col + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        auto i = tid * n_tasks_per_thread + _i;
        e[tid] = 0.0f;
        if (i < n_col) {
             e[tid] += dev::barrier_function(distance[i]);
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < n_cuda_threads_per_block; i++) e[0] += e[i];
        E += e[0];
    }
}

__global__ void inertia_kernel(float& E, int n_cubes, cudaAffineBody* cubes)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    __shared__ float e[n_cuda_threads_per_block];
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        e[tid] = 0.0f;
        auto i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            float3 dq[4];
            auto& c{ cubes[i] };
            c.q_minus_qtiled(dq);
            for (int j = 0; j < 4; j++) {
                auto w = (j == 0 ? c.mass : c.Ic) * 0.5f;
                e[tid] += w * dot(dq[j], dq[j]);
                for (int j = 1; j < 4; j++)
                    for (int k = 1; k < 4; k++) {
                        auto t = dot(c.q[j], c.q[k]) - kronecker(j, k);
                        t = t * t * kappa;
                        e[tid] += t;
                    }
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < n_cuda_threads_per_block; i++) e[0] += e[i];
        E += e[0];
    }
}
__global__ void friction_kernel(float &E){}

// float line_search_cuda(int n_cubes, float* dq, float toi, i2* prims_ret, i2* body_ret, int * pt_types_ret)
// {
//     float E0 = 0.0f, E1 = 0.0f;
//     float alpha = toi;
//     auto& bulk{ host_cuda_globals.leader_thread_buffer_back };
//     auto bulk_stashed = bulk;
//     float* buf = (float*)buf;
//     barrier_kernel<<<1, n_cuda_threads_per_block>>>(E0, host_cuda_globals.npt + host_cuda_globals.nee, host_cuda_globals.float_buffer);

//     inertia_kernel<<<1, n_cuda_threads_per_block>>>(E0, host_cuda_globals.n_cubes, host_cuda_globals.cubes);
//     friction_kernel<<<1, n_cuda_threads_per_block>>>(E0);
//     do {
//         update_line_search_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, dq, alpha);
//         int npt_new, nee_new;

//         iaabb_brute_force_cuda(
//             n_cubes, host_cuda_globals.aabbs, 2,
//             host_cuda_globals.lut_size, PTR(host_cuda_globals.lut),
//             dq,
//             prims_ret, body_ret, pt_types_ret);
//         E1 = 0.0f;
//         barrier_kernel<<<1, n_cuda_threads_per_block>>>(
//             E1, npt_new + nee_new, buf);
//         inertia_kernel<<<1, n_cuda_threads_per_block>>>(
//             E1,
//             host_cuda_globals.n_cubes, host_cuda_globals.cubes);
//         friction_kernel<<<1, n_cuda_threads_per_block>>>(E1);

//         bool wolfe = E1 <= E0;
//         alpha /= 2.0f;
//         if (wolfe) break;
//     } while (true);
//     return alpha;
// }


__global__ void project_vt1_kernel(int n_cubes, cudaAffineBody *cubes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            for (int j = 0; j < c.n_vertices; j++) {
                c.projected[j] = matmul(c.q, c.vertices[j]);
            }
        }
    }
}
__global__ void project_vt2_kernel(int n_cubes, cudaAffineBody *cubes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            for (int j = 0; j < c.n_vertices; j++) {
                c.updated[j] = c.projected[j] = matmul(c.q_update, c.vertices[j]);
            }
        }
    }
}



__global__ void inertia_grad_hess_kernel(int n_cubes, cudaAffineBody *cubes, float dt, float* b, float* diag)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            if (c.mass < 0) {
                for (int k = 0; k < 144; k++) {
                    diag[k + 144 * i] = k % 13 == 0 ? 1.0f : 0.0f;
                }
                for (int k = 0; k < 12; k++) {
                    b[k + 12 * i] = 0.0f;
                }
            }
            else {
                for (int j = 0; j < 4; j++) {
                    c.q_update[j] = c.q[j];
                }
                float g12[12]{ 0.0 };
                orthogonal_grad(c.q, dt, g12);
                inertia_grad(c, dt, g12);
                for (int j = 0; j < 12; j++) b[j + 12 * i] = g12[j];

                orthogonal_hess(c.q, dt, diag + 144 * i);
                inertia_hess(c, diag + 144 * i);
            }
        }
    }
}

__global__ void update_timestep_kernel(int n_cubes, cudaAffineBody* cubes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_task_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int I = tid * n_task_per_thread + _i;
        if (I < n_cubes) {
            auto& c{ cubes[I] };
            for (int i = 0; i < 4; i++) {
                c.dqdt[i] = (c.q[i] - c.q0[i]) / dt;
                c.q0[i] = c.q[i];
            }
        }
    }
}

__global__ void update_newton_kernel(int n_cubes, cudaAffineBody* cubes, float* dq, float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_task_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int I = tid * n_task_per_thread + _i;
        if (I < n_cubes) {
            auto& c{ cubes[I] };
            for (int i = 0; i < 4; i++) {
                c.q[i] = c.q[i] + make_float3(dq[I * 12 + i * 3 + 0], dq[I * 12 + i * 3 + 1], dq[I * 12 + i * 3 + 2]);
            }
        }
    }
}


// void implicit_euler_cuda()
// {
//     auto &g {host_cuda_globals};
//     auto &bulk {g.bulk_buffer_back[0]};
//     auto &st {g.small_temporary_buffer_back[0]};
//     auto st_stashed = st;
//     auto bulk_stashed = bulk;

//     int n_cubes = g.n_cubes;
//     float dt = g.dt, tol = 1e-2;
//     set_q_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes);

//     // iaabb_brute_force_cuda(n_cubes, host_cuda_globals.aabbs, 1, host_cuda_globals.lut_size, PTR(host_cuda_globals.lut), nullptr, PTR(host_cuda_globals.prim_idx), PTR(host_cuda_globals.body_idx), PTR(host_cuda_globals.pt_types));
//     vector<array<int, 4>> foo;
//     iaabb_brute_force_cuda_pt_only(
//         n_cubes, g.cubes, g.aabbs, 1, foo);

//     do {
//         // make_lut(host_cuda_globals.lut_size, PTR(host_cuda_globals.lut));
//         // make_placeholder_sparse_matrix(g.lut_size, PTR(g.lut), g.hess);

//         project_vt1_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, g.projected_vertices);
//         inertia_grad_hess_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, dt, g.b, g.hess_diag);

//         auto &hess {g.hess}; 
//         ipc_pt_kernel<<<1, n_cuda_threads_per_block>>> (
//             g.npt, g.pt.p, g.pt.b, g.cubes, g.lut_size, PTR(g.lut),
//             PTR(hess.values), PTR(hess.inner), PTR(hess.outer_start),
//             g.b, 
//             nullptr, // buffer undecided
//             nullptr, 
//             nullptr     // lambda and Tk undecided
//         );
//         // TODO: merge the two kernels

//         // ipc_ee_kernel<<<1, n_cuda_threads_per_block, 0, host_cuda_globals.streams[0]>>>(
//         //     n_cubes, host_cuda_globals.nee, PTR(host_cuda_globals.prim_idx), PTR(cuda_globlas.body_idx),
//         //     host_cuda_globals.hess,
//         //     host_cuda_globals.lut_size, host_cuda_globals.lut
//         // );

//         // ipc vg kernel

//         gpuCholSolver(host_cuda_globals.hess, host_cuda_globals.dq, g.b);
//         float* ret_norm = (float*)st;
//         bulk += sizeof(float);


//         norm_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes * 12, g.dq, ret_norm);
//         float sup_dq;
//         cudaMemcpy(&sup_dq, ret_norm, sizeof(float), cudaMemcpyDeviceToHost);

//         float toi = upper_bound_cuda(n_cubes, host_cuda_globals.aabbs, host_cuda_globals.lut_size, PTR(host_cuda_globals.lut));
//         toi = toi == 1.0f ? 1.0f : 0.8f * toi;
//         float alpha = line_search_cuda(n_cubes, host_cuda_globals.dq, toi, PTR(host_cuda_globals.prim_idx_update), PTR(host_cuda_globals.body_idx_update), PTR(host_cuda_globals.pt_types_update));
//         update_newton_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, host_cuda_globals.dq, alpha);
//         bool term_cond = sup_dq < tol;
//         // cudaDeviceSynchronize();
//         if (term_cond) break;
//     } while (true);
//     update_timestep_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes);
// }

void cuda_inert_hess_glue(int n_cubes, float dt, float *grads, float * hess) {
    auto &cubes{ host_cuda_globals.cubes };

    //project_vt1_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, host_cuda_globals.projected_vertices);
    inertia_grad_hess_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, dt, host_cuda_globals.b, host_cuda_globals.hess_diag);
    CUDA_CALL(cudaDeviceSynchronize());
    cudaMemcpy(grads, host_cuda_globals.b, n_cubes* 12 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, host_cuda_globals.hess_diag, n_cubes* 144 * sizeof(float), cudaMemcpyDeviceToHost);
}
void freeCublasAndCusparse();
void setCublasAndCuSparse();

CudaGlobals::CudaGlobals(int n_cubes)
{
    cudaGetDevice(&device_id);
    int n_proc = omp_get_num_procs();
    per_stream_buffer_size = n_cuda_threads_per_block * max_pairs_per_thread * sizeof(i2) * 8;
    streams = new cudaStream_t[n_proc];
    for (int i = 0; i < n_proc; i++) {
        cudaStreamCreate(&streams[i]);
    }

    gravity = make_float3(0.0f, -9.8f, 0.0f);
    // cudaMemcpyToSymbol("cuda_globals", this, sizeof(CudaGlobals), size_t(0), cudaMemcpyHostToDevice);
}

void CudaGlobals::allocate_buffers()
{
    cudaMallocManaged(&pt.p, max_pairs_per_block * n_blocks * sizeof(i2));
    cudaMallocManaged(&pt.b, max_pairs_per_block * n_blocks * sizeof(i2));
    cudaMallocManaged(&ee.p, max_pairs_per_block * n_blocks * sizeof(i2));
    cudaMallocManaged(&ee.b, max_pairs_per_block * n_blocks * sizeof(i2));

    cudaMallocManaged(&pt_line_search.p, max_pairs_per_block * n_blocks * sizeof(i2));
    cudaMallocManaged(&pt_line_search.b, max_pairs_per_block * n_blocks * sizeof(i2));
    cudaMallocManaged(&ee_line_search.p, max_pairs_per_block * n_blocks * sizeof(i2));
    cudaMallocManaged(&ee_line_search.b, max_pairs_per_block * n_blocks * sizeof(i2));

    cudaMallocManaged(&cubes, n_cubes * sizeof(cudaAffineBody));
    cudaMallocManaged(&b, 12 * n_cubes * sizeof(float));
    cudaMallocManaged(&dq, 12 * n_cubes * sizeof(float));
    int n_proc = omp_get_num_procs();
    cudaMallocManaged(&buffer_chunk, per_stream_buffer_size * n_proc);
    // cudaMallocManaged(&float3_buffer, sizeof(float3) * n_cuda_threads_per_block * max_pairs_per_thread * 4);
    cudaMallocManaged(&hess_diag, 144 * n_cubes * sizeof(float));
    small_temporary_buffer = new char*[n_proc];
    small_temporary_buffer_back = new char*[n_proc];
    bulk_buffer = new char*[n_proc];
    bulk_buffer_back = new char*[n_proc];
    cudaMallocManaged(&small_temporary_buffer[0], st_size * n_proc);
    for (int i = 0; i < n_proc; i++) {
        small_temporary_buffer[i] = small_temporary_buffer[0] + i * st_size;
        small_temporary_buffer_back[i] = small_temporary_buffer[i];
    }
    cudaMallocManaged(&bulk_buffer[0], blk_size * n_proc);
    for (int i = 0; i < n_proc; i++) {
        bulk_buffer[i] = bulk_buffer[0] + i * blk_size;
        bulk_buffer_back[i] = bulk_buffer[i];
    }

    cudaMallocManaged(&leader_thread_buffer, 
        sizeof(int) * max_prmts_per_block * 64 + // iaabb body-body intersection 
         + sizeof(float) * 144 * 4 * n_cuda_threads_per_block // ipc, 4 12x12 matrices (ipc_hess, hess_p, hess_t, off_diag)
         + sizeof(float) * n_cuda_threads_per_block * 12 * 4 // ipc, 12x1 vectors (ipc_grad, grad_p, grad_t)
        );
    leader_thread_buffer_back = leader_thread_buffer;
    
    setCublasAndCuSparse();
}
void CudaGlobals::free_buffers()
{

    freeCublasAndCusparse();
    cudaFree(vertices_at_rest);
    cudaFree(projected_vertices);
    cudaFree(updated_vertices);
    cudaFree(edges);
    cudaFree(faces);
    cudaFree(aabbs);
    cudaFree(bulk_buffer[0]);
    cudaFree(small_temporary_buffer[0]);
    cudaFree(leader_thread_buffer);
    // cudaFree(buffer_chunk);
    // cudaFree(float3_buffer);
    cudaFree(dq);
    cudaFree(hess_diag);
    cudaFree(b);
    cudaFree(cubes);
    cudaFree(pt.p);
    cudaFree(pt.b);
    cudaFree(ee.p);
    cudaFree(ee.b);
    cudaFree(pt_line_search.p);
    cudaFree(pt_line_search.b);
    cudaFree(ee_line_search.p);
    cudaFree(ee_line_search.b);
    delete[] small_temporary_buffer;
    delete[] small_temporary_buffer_back;
    delete[] bulk_buffer;
    delete[] bulk_buffer_back;
}
CudaGlobals::~CudaGlobals()
{
    free_buffers();
    for (int i = 0; i < omp_get_num_procs(); i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

__host__ __device__ CudaGlobals::CudaGlobals(CudaGlobals & a) {
    cubes = a.cubes;
    aabbs = a.aabbs;
    npt = a.npt; nee = a.nee;
    n_cubes = a.n_cubes; lut_size = a.lut_size;
    hess = a.hess; 
    b = a.b; dq = a.dq; hess_diag = a.hess_diag; // float_buffer = a.float_buffer;
    dt = a.dt; gravity = a.gravity;
    projected_vertices = a.projected_vertices;
    // float3_buffer = a.float3_buffer;
    // buffer_chunk = a.buffer_chunk;
    device_id = a.device_id;
    per_stream_buffer_size = a.per_stream_buffer_size;
    streams = a.streams;
    // lut = a.lut;
}

void project_glue(int vtn)
{
    auto &g {host_cuda_globals};
    switch (vtn) {
    case 1:
        project_vt1_kernel<<<1, n_cuda_threads_per_block>>>(g.n_cubes, g.cubes);
        break;
    case 2:
        project_vt2_kernel<<<1, n_cuda_threads_per_block>>>(g.n_cubes, g.cubes);
        break;
    case 3:
        project_vt2_kernel<<<1, n_cuda_threads_per_block>>>(g.n_cubes, g.cubes);
        project_vt1_kernel<<<1, n_cuda_threads_per_block>>>(g.n_cubes, g.cubes);
        // updated will have vt2 value and projected will have vt1 value
        break;
    }
}