#include "cuda_globals.cuh"
#include <spdlog/spdlog.h>
__global__ void pt_barrier_kernel(
    int npt,
    i2* b,
    i2* p,
    cudaAffineBody* cubes,
    float* E)
{
    auto tid = threadIdx.x;
    auto n_tasks = npt;
    auto n_tasks_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    __shared__ float energy[n_cuda_threads_per_block];
    energy[tid] = 0;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        auto I = _i + tid * n_tasks_per_thread;
        if (I < n_tasks) {
            auto ij{ b[I] };
            int i = ij[0];
            int j = ij[1];

            int vi = p[I][0];
            int fj = p[I][1];

            auto &ci{ cubes[i] }, &cj{ cubes[j] };

            auto p{ ci.updated[vi] };
            auto t{ cj.triangle_updated(fj) };
            int pt_type;
            auto d = vf_distance(p, t, pt_type);
            energy[tid] += dev::barrier_function(d);
        }
    }
    __syncthreads();
    if (tid == 0) {
        float e = 0;
        for (int i = 0; i < blockDim.x; i++) {
            e += energy[i];
        }
        *E = e;
    }
}

__global__ void ee_barrier_kernel(
    int nee,
    i2* b,
    i2* p,
    cudaAffineBody* cubes,
    float* E)
{
    auto tid = threadIdx.x;
    auto n_tasks = nee;
    auto n_tasks_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    __shared__ float energy[n_cuda_threads_per_block];
    energy[tid] = 0;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        auto I = _i + tid * n_tasks_per_thread;
        if (I < n_tasks) {
            auto ij{ b[I] };
            int i = ij[0];
            int j = ij[1];

            int ei = p[I][0];
            int ej = p[I][1];

            auto &ci{ cubes[i] }, &cj{ cubes[j] };

            auto edgei{ ci.edge_updated(ei) };
            auto edgej{ cj.edge_updated(ej) };
            int ee_type = dev::edge_edge_distance_type(edgei.e0, edgei.e1, edgej.e0, edgej.e1);
            auto d = dev::edge_edge_distance(edgei.e0, edgei.e1, edgej.e0, edgej.e1, ee_type);
            energy[tid] += dev::barrier_function(d);
        }
    }
    __syncthreads();
    if (tid == 0) {
        float e = 0;
        for (int i = 0; i < blockDim.x; i++) {
            e += energy[i];
        }
        *E = e;
    }
}

__global__ void compute_inertia_energy_kernel(
    int n_cubes,
    cudaAffineBody* cubes,
    float* ret_energy,
    float dt = 1e-2f);



float barrier_plus_inert_glue(
    float dt)
{
    auto& g{ host_cuda_globals };
    auto& lt{ g.leader_thread_buffer_back };
    lt += 3 * sizeof(float);
    auto lt_stashed = lt;
    float* ebpt = (float*)lt;
    float* ebee = ebpt + 1;
    float* ei = ebee + 1;

    pt_barrier_kernel<<<1, n_cuda_threads_per_block>>>(
        g.npt,
        g.pt.b,
        g.pt.p,
        g.cubes,
        ebpt);
    ee_barrier_kernel<<<1, n_cuda_threads_per_block>>>(
        g.nee,
        g.ee.b,
        g.ee.p,
        g.cubes,
        ebee);
    compute_inertia_energy_kernel<<<1, n_cuda_threads_per_block>>>(
        g.n_cubes,
        g.cubes,
        ei,
        dt);

    float host_energy[3];
    cudaDeviceSynchronize();
    cudaMemcpy(host_energy, ebpt, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    lt = lt_stashed;
    #define CPU_REF
    #ifdef CPU_REF
    auto cubes = g.host_cubes;

    cudaMemcpy(cubes.data(), g.cubes, g.n_cubes * sizeof(cudaAffineBody), cudaMemcpyDeviceToHost);
    float host_inertia = 0;
    for (int  i = 0; i < g.n_cubes; i ++) {
        host_inertia += inertia(cubes[i], 1e-2f);
    }
    if (abs(host_inertia - host_energy[2]) > 1e-4f) 
        spdlog::error("inertia energy mismatch: ref {} vs {}", host_inertia, host_energy[2]);
    #endif
    return host_energy[0] + host_energy[1] + host_energy[2];
}
