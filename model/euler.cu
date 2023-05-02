#include "cuda_header.cuh"
#include "cuda_globals.cuh"


__constant__ float kappa = 1e9f, dt = 1e-2f;

CudaGlobals host_cuda_globals;
//__constant__ CudaGlobals *cuda_globals;
//__constant__ float3 gravity;

void gpuCholSolver(CsrSparseMatrix& hess, float* x);
__global__ void ipc_pt_kernel(
    int n_cubes, int npt,
    i2* prims, i2* body,
    CsrSparseMatrix& hess,
    int lut_size, i2* lut);

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

__forceinline__ __device__ __host__ float kronecker(int i, int j)
{
    return i == j ? 1.0f : 0.0f;
}
void initialize_precompute_aabbs()
{
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

__host__ __device__ float dev::barrier_function(float d)
{
    if (d >= dev::d_hat) return 0.0;
    return dev::kappa * -(d - dev::d_hat) * (d - dev::d_hat) * log(d / dev::d_hat) / (dev::d_hat * dev::d_hat);
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

//__device__ void cudaAffineBody::q_minus_qtiled(float3 dq[4])
//{
//    __constant__ float  h =  cuda_globals->dt;
//    __constant__ float h2 = h * h;
//    for (int i = 0; i < 4; i++) {
//        dq[i] = q_update[i] - (q[i] + dqdt[i] * h + cuda_globals->gravity * h2);
//    }
//}
__host__ __device__ void cudaAffineBody::q_minus_qtiled(float3 dq[4])
{
    float h = dt;
    float h2 = h * h;
    for (int i = 0; i < 4; i++) {
        dq[i] = q_update[i] - (q[i] + dqdt[i] * h + make_float3(0.0f, -9.8f, 0.0f) * h2);
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

float line_search_cuda(int n_cubes, float* dq, float toi, i2* prims_ret, i2* body_ret, int * pt_types_ret)
{
    float E0 = 0.0f, E1 = 0.0f;
    float alpha = toi;
    barrier_kernel<<<1, n_cuda_threads_per_block>>>(E0, host_cuda_globals.npt + host_cuda_globals.nee, host_cuda_globals.float_buffer);

    inertia_kernel<<<1, n_cuda_threads_per_block>>>(E0, host_cuda_globals.n_cubes, host_cuda_globals.cubes);
    friction_kernel<<<1, n_cuda_threads_per_block>>>(E0);
    do {
        update_line_search_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, dq, alpha);
        // collision_set_kernel<<<1, n_cuda_threads_per_block>>>(alpha, dq, host_cuda_globals.cubes);
        int npt_new, nee_new;

        iaabb_brute_force_cuda(
            n_cubes, host_cuda_globals.aabbs, 2,
            host_cuda_globals.lut_size, PTR(host_cuda_globals.lut),
            dq,
            prims_ret, body_ret, pt_types_ret);
        // TODO: should make it cull kernel, precise kernel, and output options
        E1 = 0.0f;
        barrier_kernel<<<1, n_cuda_threads_per_block>>>(
            E1, npt_new + nee_new, host_cuda_globals.float_buffer);
        inertia_kernel<<<1, n_cuda_threads_per_block>>>(
            E1,
            host_cuda_globals.n_cubes, host_cuda_globals.cubes);
        friction_kernel<<<1, n_cuda_threads_per_block>>>(E1);

        bool wolfe = E1 <= E0;
        alpha /= 2.0f;
        if (wolfe) break;
    } while (true);
    return alpha;
}

void make_placeholder_sparse_matrix(int lut_size, i2* lut, CsrSparseMatrix& hess)
{
}

__global__ void project_vt1_kernel(int n_cubes, cudaAffineBody *cubes, float3* buffer)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (n_cubes + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int i = tid * n_tasks_per_thread + _i;
        if (i < n_cubes) {
            auto& c{ cubes[i] };
            auto offset = c.global_vertices_offset;
            for (int j = 0; j < c.n_vertices; j++) {
                buffer[offset + j] = matmul(c.q, c.vertices[j]);
            }
        }
    }
}


__host__ __device__ void orthogonal_grad(float3 q[4], float dt, float ret[12])
{
    auto h2 = dt * dt;

    for (int i = 1; i < 4; i++) {
        float3 g = make_float3(0.0f, 0.0f, 0.0f);
        for (int j = 1; j < 4; j++) {
            g = g + q[j] * (dot(q[i], q[j]) - kronecker(i, j));
        }

        g = g * (4 * kappa * h2);
        ret[i * 3 + 0] += g.x;
        ret[i * 3 + 1] += g.y;
        ret[i * 3 + 2] += g.z;
    }
}

__host__ __device__ void orthogonal_hess(float3 q[4], float dt, float ret[144])
{
    auto h2 = dt * dt;
    for (int i = 1; i < 4; i++)
        for (int j = 1; j < 4; j++) {
            float h[9]{ 0.0f };
            if (i == j) {

                for (int k = 1; k < 4; k++) {
                    float w = k == i ? 2.0f : 1.0f;
                    float _q[3]{ q[k].x, q[k].y, q[k].z };
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++) {
                            auto qii = _q[ii], qjj = _q[jj];
                            h[ii + jj * 3] = w * qii * qjj;
                        }
                    for (int ii = 0; ii < 3; ii++) {
                        h[ii * 4] += dot(q[i], q[i]) - 1.0f;
                    }
                }
            }
            else {
                float d = dot(q[i], q[j]);
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++) {
                        auto qjii = ii == 0 ? q[j].x : (ii == 1 ? q[j].y : q[j].z);
                        auto qijj = jj == 0 ? q[i].x : (jj == 1 ? q[i].y : q[i].z);
                        h[ii + jj * 3] = qjii * qijj + kronecker(ii, jj) * d;
                    }
            }
            auto scale = 4 * kappa * h2;
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++) {
                    ret[(i * 3 + ii) + (j * 3 + jj) * 12] = scale * h[ii + jj * 3];
                }
        }
}
__host__ __device__ void inertia_grad(cudaAffineBody& c, float dt, float ret[12])
{
    float3 g[4];
    c.q_minus_qtiled(g);
    for (int i = 0; i < 4; i++) {
        float w = i == 0 ? c.mass : c.Ic;
        ret[i * 3 + 0] += w * g[i].x;
        ret[i * 3 + 1] += w * g[i].y;
        ret[i * 3 + 2] += w * g[i].z;
    }
}

__host__ __device__ void inertia_hess(cudaAffineBody& c, float ret[144])
{
    for (int i = 0; i < 3; i++) {
        ret[i * 13] += c.mass;
    }
    for (int i = 3; i < 12; i++) {
        ret[i * 13] += c.Ic;
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
            float g12[12]{ 0.0 };
            orthogonal_grad(c.q, dt, g12);
            inertia_grad(c, dt, g12);

            orthogonal_hess(c.q, dt, diag + 144 * i);
            inertia_hess(c, diag + 144 * i);
        }
    }
}

__global__ void prepare_pt_kernel(int n_cubes, cudaAffineBody *cubes, int npt, i2* prims, i2* body, vec3f* buffer, vec3f *projected_vertices)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks_per_thread = (npt + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < npt) {
            auto &ci{ cubes[body[I][0]] },
                &cj{ cubes[body[I][1]] };
            int i0, i1, i2;
            i0 = cj.faces[prims[I][1] * 3 + 0];
            i1 = cj.faces[prims[I][1] * 3 + 1];
            i2 = cj.faces[prims[I][1] * 3 + 2];
            vec3f p(projected_vertices[ci.global_vertices_offset + prims[I][0]]),
                t0{ projected_vertices[cj.global_vertices_offset + i0] },
                t1{ projected_vertices[cj.global_vertices_offset + i1] },
                t2{ projected_vertices[cj.global_vertices_offset + i2] };

            buffer[I * 4 + 0] = p;
            buffer[I * 4 + 1] = t0;
            buffer[I * 4 + 2] = t1;
            buffer[I * 4 + 3] = t2;
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

void implicit_euler_cuda()
{
    int n_cubes = host_cuda_globals.n_cubes;
    float dt = host_cuda_globals.dt, tol = 1e-2;
    set_q_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes);

    iaabb_brute_force_cuda(n_cubes, host_cuda_globals.aabbs, 1, host_cuda_globals.lut_size, PTR(host_cuda_globals.lut), nullptr, PTR(host_cuda_globals.prim_idx), PTR(host_cuda_globals.body_idx), PTR(host_cuda_globals.pt_types));

    do {
        make_lut(host_cuda_globals.lut_size, PTR(host_cuda_globals.lut));
        make_placeholder_sparse_matrix(host_cuda_globals.lut_size, PTR(host_cuda_globals.lut), host_cuda_globals.hess);

        project_vt1_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, host_cuda_globals.projected_vertices);
        inertia_grad_hess_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, dt, host_cuda_globals.b, host_cuda_globals.hess_diag);

        prepare_pt_kernel<<<1, n_cuda_threads_per_block>>>(
            n_cubes, host_cuda_globals.cubes,host_cuda_globals.npt,
            PTR(host_cuda_globals.prim_idx), PTR(host_cuda_globals.body_idx),
            host_cuda_globals.float3_buffer, host_cuda_globals.projected_vertices);

        ipc_pt_kernel<<<1, n_cuda_threads_per_block, 0, host_cuda_globals.streams[0]>>>(
            n_cubes, host_cuda_globals.npt, PTR(host_cuda_globals.prim_idx), PTR(host_cuda_globals.body_idx),
            host_cuda_globals.hess,
            host_cuda_globals.lut_size, PTR(host_cuda_globals.lut));
        // TODO: merge the two kernels

        // ipc_ee_kernel<<<1, n_cuda_threads_per_block, 0, host_cuda_globals.streams[0]>>>(
        //     n_cubes, host_cuda_globals.nee, PTR(host_cuda_globals.prim_idx), PTR(cuda_globlas.body_idx),
        //     host_cuda_globals.hess,
        //     host_cuda_globals.lut_size, host_cuda_globals.lut
        // );

        // ipc vg kernel
        gpuCholSolver(host_cuda_globals.hess, host_cuda_globals.dq);
        float sup_dq = thrust::reduce(host_cuda_globals.dq, host_cuda_globals.dq + 12 * n_cubes, 0.0f);
        // FIXME: squared sum
        float toi = upper_bound_cuda(n_cubes, host_cuda_globals.aabbs, host_cuda_globals.lut_size, PTR(host_cuda_globals.lut));
        toi = toi == 1.0f ? 1.0f : 0.8f * toi;
        float alpha = line_search_cuda(n_cubes, host_cuda_globals.dq, toi, PTR(host_cuda_globals.prim_idx_update), PTR(host_cuda_globals.body_idx_update), PTR(host_cuda_globals.pt_types_update));
        update_newton_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes, host_cuda_globals.dq, alpha);
        bool term_cond = sup_dq < tol;
        // cudaDeviceSynchronize();
        if (term_cond) break;
    } while (true);
    update_timestep_kernel<<<1, n_cuda_threads_per_block>>>(n_cubes, host_cuda_globals.cubes);
}

#include <omp.h>

CudaGlobals::CudaGlobals(int n_cubes)
{
    cudaGetDevice(&device_id);
    int n_proc = omp_get_num_procs();
    per_stream_buffer_size = n_cuda_threads_per_block * max_pairs_per_thread * sizeof(i2) * 8;
    streams = new cudaStream_t[n_proc];
    for (int i = 0; i < n_proc; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaMallocManaged(&b, 12 * n_cubes * sizeof(float));
    cudaMallocManaged(&dq, 12 * n_cubes * sizeof(float));
    cudaMallocManaged(&buffer_chunk, per_stream_buffer_size * n_proc);
    cudaMallocManaged(&projected_vertices, sizeof(float3) * max_n_vertices);
    cudaMallocManaged(&float3_buffer, sizeof(float3) * n_cuda_threads_per_block * max_pairs_per_thread * 4);
    cudaMallocManaged(&hess_diag, 144 * n_cubes * sizeof(float));

    gravity = make_float3(0.0f, -9.8f, 0.0f);
    // cudaMemcpyToSymbol("cuda_globals", this, sizeof(CudaGlobals), size_t(0), cudaMemcpyHostToDevice);
}

CudaGlobals::~CudaGlobals()
{
    cudaFree(buffer_chunk);
    cudaFree(projected_vertices);
    cudaFree(float3_buffer);
    cudaFree(dq);
    cudaFree(hess_diag);
    cudaFree(b);
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
    b = a.b; dq = a.dq; hess_diag = a.hess_diag; float_buffer = a.float_buffer;
    dt = a.dt; gravity = a.gravity;
    projected_vertices = a.projected_vertices;
    float3_buffer = a.float3_buffer;
    buffer_chunk = a.buffer_chunk;
    device_id = a.device_id;
    per_stream_buffer_size = a.per_stream_buffer_size;
    streams = a.streams;
    // lut = a.lut;
}