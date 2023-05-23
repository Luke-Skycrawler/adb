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

void freeCublasAndCusparse();
void setCublasAndCuSparse();
void gpuCholSolver(CsrSparseMatrix& hess, float* x, float *b);
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

);
__global__ void ipc_ee_kernel(
    int nee,
    i2* ee, i2* ij,
    cudaAffineBody* cubes,
    int lut_size, i2* lut,
    float* values, int* outers,
    float* b,
    float* buffer,
    float* lambdas, float* Tk);
__global__ void put_inertia_kernel(
    int n_cubes,
    cudaAffineBody* cubes,
    int lut_size, i2* lut,
    float* values, int* outers,
    float* b,
    float* diag);

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

__global__ void set_q_to_q0_kernel(int n_cubes, cudaAffineBody* cubes)
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


__global__ void update_line_search_kernel(int n_cubes, cudaAffineBody *cubes, float* dq, float alpha)
{
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            for (int j = 0; j < 4; j++)
                c.q_update[j] = c.q[j] - make_float3(dq[i * 12 + j * 3 + 0], dq[i * 12 + j * 3 + 1], dq[i * 12 + j * 3 + 2]) * alpha;
                // newton dq, negative sign
        }
    }
}

__global__ void friction_kernel(float &E){}

float line_search_cuda(int n_cubes, float* dq, float toi, float dt = 1e-2f)
{
    auto& g{ host_cuda_globals };
    float E0 = 0.0f, E1 = 0.0f, E1f = 0.0f;
    float alpha = toi;
    auto& bulk{ g.leader_thread_buffer_back };
    auto bulk_stashed = bulk;
    update_line_search_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, dq, 0.0f);
    project_glue(3);
    // vtn = 3 to prepare for friction
    E0 = barrier_plus_inert_glue(dt);
    // TODO: add friction energy here
    vector<array<int, 4>> foo, bar;
    do {
        update_line_search_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, dq, alpha);
        project_glue(3);
        
        int npt_new, nee_new;
        CUDA_CALL(cudaDeviceSynchronize());
        E1 = iaabb_brute_force_cuda_pt_only(
            n_cubes, g.cubes, g.aabbs, 2, foo, bar);
        // directly returns the barrier + inert energy when vtn = 2

        // friction_kernel<<<1, n_cuda_threads_per_block>>>(E1);

        bool wolfe = E1 <= E0;
        if (wolfe) break;
        alpha /= 2.0f;
    } while (true);
    {
        g.npt = g.npt_line_search;
        g.nee = g.nee_line_search;
        // swap pt and pt_line_search pointers
        auto tmp_pt = g.pt, tmp_ee = g.ee;
        g.pt = g.pt_line_search;
        g.ee = g.ee_line_search;
        g.pt_line_search = tmp_pt;
        g.ee_line_search = tmp_ee;
    }
    return alpha;
}

__global__ void project_vt1_kernel(int n_cubes, cudaAffineBody* cubes)
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
__global__ void project_vt2_kernel(int n_cubes, cudaAffineBody* cubes)
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

__global__ void inertia_grad_hess_kernel(int n_cubes, cudaAffineBody* cubes, float dt, float* b, float* diag)
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

__global__ void update_timestep_kernel(int n_cubes, cudaAffineBody* cubes, float dt)
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
                c.q[i] = c.q[i] - make_float3(dq[I * 12 + i * 3 + 0], dq[I * 12 + i * 3 + 1], dq[I * 12 + i * 3 + 2]);
            }
        }
    }
}
void implicit_euler_cuda(float dt)
{
    auto& g{ host_cuda_globals };
    auto& hess{ g.hess };
    auto& bulk0{ g.bulk_buffer_back[0] };
    auto& st0{ g.small_temporary_buffer_back[0] };
    auto st0_stashed = st0;
    auto bulk0_stashed = bulk0;

    int n_cubes = g.n_cubes;
    float tol = 1e-2;
    set_q_to_q0_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes);
    CUDA_CALL(cudaDeviceSynchronize());
    vector<array<int, 4>> foo, bar;
    iaabb_brute_force_cuda_pt_only(
        n_cubes, g.cubes, g.aabbs, 1, foo, bar);
    // contains make_lut step
    auto& lt{ g.leader_thread_buffer_back };
    auto lt_stashed = lt;
    do {
        build_csr(g.n_cubes, g.lut, g.hess);
        project_glue(1);
        inertia_grad_hess_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, dt, g.b, g.hess_diag);

        ipc_pt_kernel<<<1, 1>>>(g.npt, g.pt.p, g.pt.b,
            g.cubes,
            g.lut_size, PTR(g.lut),
            PTR(g.hess.values), PTR(g.hess.outer_start),
            g.b, (float*)lt,
            nullptr, nullptr);
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
            g.b, g.hess_diag);

        CUDA_CALL(cudaDeviceSynchronize());
        gpuCholSolver(g.hess, g.dq, g.b);
        float* ret_norm = (float*)lt;
        // lt += sizeof(float);
        norm_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes * 12, g.dq, ret_norm);
        float sup_dq;
        cudaDeviceSynchronize();
        cudaMemcpy(&sup_dq, ret_norm, sizeof(float), cudaMemcpyDeviceToHost);

        update_line_search_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, g.dq, 1.0f);
        CUDA_CALL(cudaDeviceSynchronize());
        // q_update = q + dq
        project_glue(3);
        
        float toi = iaabb_brute_force_cuda_pt_only(
            n_cubes, g.cubes, g.aabbs, 3, foo, bar);

        toi = toi == 1.0f ? 1.0f : 0.8f * toi;
        float alpha = line_search_cuda(n_cubes, g.dq, toi, dt);
        update_newton_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, g.dq, alpha);
        cudaDeviceSynchronize();
        bool term_cond = sup_dq < tol;
        // cudaDeviceSynchronize();
        if (term_cond) break;
    } while (true);
    update_timestep_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, g.cubes, dt);
    cudaDeviceSynchronize();
}

void cuda_inert_hess_glue(int n_cubes, float dt, float *grads, float * hess) {
    auto &cubes{ host_cuda_globals.cubes };

    //project_vt1_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, host_cuda_globals.projected_vertices);
    inertia_grad_hess_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, dt, host_cuda_globals.b, host_cuda_globals.hess_diag);
    CUDA_CALL(cudaDeviceSynchronize());
    cudaMemcpy(grads, host_cuda_globals.b, n_cubes* 12 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, host_cuda_globals.hess_diag, n_cubes* 144 * sizeof(float), cudaMemcpyDeviceToHost);
}

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
    cudaDeviceSynchronize();
}