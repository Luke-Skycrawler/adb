
// #include "iaabb.h"
#include "cuda_header.cuh"
#include "cuda_globals.cuh"
#include "timer.h"
#include <omp.h>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <spdlog/spdlog.h>
#include <tuple>

#include <assert.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <thrust/set_operations.h>

using namespace std;
using namespace ipc;
// using namespace cuda::std;
// using namespace Eigen;
// FIXME: tid probably not in 1 block



__device__ __host__ float vf_distance(vec3f _v, Facef f, int& pt_type);
__device__ luf intersection(const luf &a, const luf &b);
__device__ luf affine(luf aabb, cudaAffineBody &c, int vtn);
__device__ luf compute_aabb(const Facef& f, float d_hat_sqrt);
__device__ luf compute_aabb(const Edgef& e, float d_hat_sqrt);

tuple<float, PointTriangleDistanceType> vf_distance(vec3f vf, Facef ff);
void per_intersection_core(int n_overlaps, luf* culls, i2* overlaps);

/* NOTE: kernel argument convention:
    buf_: device buffer
    ret_: return buffer
    io_: input and output from the same buffer, i.e. in-place
    : (none) inputs
*/
__global__ void aabb_intersection_test_kernel(luf* dev_aabbs, int nvi, int nfj, i2* ret_ij, int* ret_cnt)
{

    __shared__ luf aabbs[max_aabb_list_size];
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_task_per_thread = (nvi * nfj + blockDim.x - 1) / blockDim.x;
    int n_copies_per_thread = (nvi + nfj + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < n_copies_per_thread; i++) {
        int idx = tid * n_copies_per_thread + i;
        if (idx < nvi + nfj && idx < max_aabb_list_size)
            aabbs[idx] = dev_aabbs[idx];
    }
    ret_cnt[tid] = 0;
    __syncthreads();
    // copys the bounding boxes to shared memory

    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int I = tid * n_task_per_thread + _i;
        if (I < nvi * nfj) {
            int i = I / nfj;
            int j = I % nfj;
            if (intersects(aabbs[i], aabbs[nvi + j])) {
                auto put = ret_cnt[tid]++ + tid * max_pairs_per_thread;
                ret_ij[put] = { i, j };
            }
        }
    }
}

__global__ void inclusive_scan_kernel(int* io_cnt)
{
    for (int i = 1; i < n_cuda_threads_per_block; i++) {
        io_cnt[i] = io_cnt[i - 1] + io_cnt[i];
    }
}

__global__ void filter_distance_kernel_atomic(i2* buf_ij, i2* io_tmp,
    int* cnt, 
    vec3f* vis, Facef* fjs,

    int* vilist, int* fjlist,
    int* buf_pt_types,
    int* io_pt_types,
    float dhat = 1e-4, int* meta_vifj = nullptr)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks = cnt[n_cuda_threads_per_block - 1];
    int n_task_per_thread = (n_tasks + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;
    __shared__ int n;
    n = 0;
    auto nvi = meta_vifj? meta_vifj[0]: 0;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int idx = tid * n_task_per_thread + _i;
        if (idx < n_tasks) {
            auto ij = io_tmp[idx];
            int i = ij[0];
            int j = ij[1];
            // int vi = vilist[i];
            // int fj = fjlist[j];

            // compute the distance and type
            auto v{ vis[i] };
            Facef f{ fjs ? fjs[j] : Facef{ vis[j * 3 + nvi], vis[j * 3 + 1 + nvi], vis[j * 3 + 2 + nvi] } };
            auto put = atomicAdd(&n, 1);

            auto d = vf_distance(v, f, buf_pt_types[put]);
            if (d < dhat) {
                if (vilist && fjlist)
                    buf_ij[put] = { vilist[i], fjlist[j] };
                else
                    buf_ij[put] = { i, j };
            }
        }
    }
    __syncthreads();
    n_tasks =  n;
    n_task_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i  < n_task_per_thread; _i ++) {
        int idx = tid * n_task_per_thread + _i;
        if (idx < n_tasks) {
            io_tmp[idx] = buf_ij[idx];
            io_pt_types[idx] = buf_pt_types[idx];
        }
    }
    cnt[n_cuda_threads_per_block - 1] = n;
}
__global__ void filter_distance_kernel(i2* ret_ij, int* ret_cnt, i2* tmp,
    // int* vilist, int* fjlist,
    vec3f* vis, Facef* fjs,
    int* ret_pt_types,
    int* tmp_pt_types,
    float dhat = 1e-4)
{
    // // squeeze the ret_ij list according to a prefix sum array ret_cnt
    // // FIXME: asserting blockDim.x == n_cuda_threads_per_block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int start = tid == 0 ? 0 : ret_cnt[tid - 1];
    int n_tasks = ret_cnt[blockDim.x - 1];
    // do the vf distance and type computation
    ret_cnt[tid] = 0;
    int n_task_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int idx = tid * n_task_per_thread + _i;
        if (idx < n_tasks) {
            auto _ret_ij = tmp[idx];
            int i = _ret_ij[0];
            int j = _ret_ij[1];
            // int vi = vilist[i];
            // int fj = fjlist[j];

            // compute the distance and type
            auto v{ vis[i] };
            auto f{ fjs[j] };
            auto put = ret_cnt[tid] + tid * max_pairs_per_thread;
            auto d = vf_distance(v, f, ret_pt_types[put]);
            if (d < dhat) {
                ret_ij[put] = { i, j };
                ret_cnt[tid]++;
            }
        }
    }
}
__global__ void squeeze_ij_kernel(
    i2* ij, int* cnt, 
    i2* ret_tmp, 
    int* pt_types, 
    int* ret_tmp_pt_types)
{
    // squeeze again and copy back to ij matrix
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto start = tid == 0 ? 0 : cnt[tid - 1];
    auto copy_size = cnt[tid] - start;
    for (int i = 0; i < copy_size; i++) {
        int dst = i + start;
        int src = tid * max_pairs_per_thread + i;

        ret_tmp[dst] = ij[src];
        ret_tmp_pt_types[dst] = pt_types[src];
        // ret_tmp is now dense
    }
}
void vf_col_set_cuda(
    int nvi, int nfj,
    const thrust::host_vector<luf>& aabbs,
    const thrust::host_vector<vec3f>& vis,
    const thrust::host_vector<Facef>& fjs,
    const std::vector<int>& vilist, const std::vector<int>& fjlist,
    vector<array<int, 4>>& idx,
    int I, int J,
    int tid)
{
    if (nvi && nfj)
        ;
    else
        return;
    auto& stream{ host_cuda_globals.streams[tid] };

// #define THRUST_DEV_VECTOR
#ifdef THRUST_DEV_VECTOR
    thrust::device_vector<luf> dev_aabbs(aabbs.begin(), aabbs.end());
    thrust::device_vector<vec3f> dev_vis(vis.begin(), vis.end());
    thrust::device_vector<Facef> dev_fjs(fjs.begin(), fjs.end());
    thrust::device_vector<int>
        dev_vilist(vilist.begin(), vilist.end()),
        dev_fjlist(fjlist.begin(), fjlist.end());

    // allocate memory on device
    thrust::device_vector<int> dev_cnt(n_cuda_threads_per_block, 0);
    thrust::device_vector<i2>
        ij(n_cuda_threads_per_block * max_pairs_per_thread),
        tmp(n_cuda_threads_per_block * max_pairs_per_thread);

    thrust::device_vector<int> pt_types(n_cuda_threads_per_block * max_pairs_per_thread), pt_types_buffer(n_cuda_threads_per_block * max_pairs_per_thread);

    auto ij_ptr = thrust::raw_pointer_cast(ij.data());
    auto cnt_ptr = thrust::raw_pointer_cast(dev_cnt.data());
    auto tmp_ptr = thrust::raw_pointer_cast(tmp.data());
    auto vilist_ptr = thrust::raw_pointer_cast(dev_vilist.data());
    auto fjlist_ptr = thrust::raw_pointer_cast(dev_fjlist.data());
    auto vis_ptr = thrust::raw_pointer_cast(dev_vis.data());
    auto fjs_ptr = thrust::raw_pointer_cast(dev_fjs.data());
    auto aabbs_ptr = thrust::raw_pointer_cast(dev_aabbs.data());
    auto pt_types_ptr = thrust::raw_pointer_cast(pt_types.data());
    auto tmp_pt_types_ptr = thrust::raw_pointer_cast(pt_types_buffer.data());
#else
    auto chunk_int = (int*)((char*)host_cuda_globals.buffer_chunk + tid * host_cuda_globals.per_stream_buffer_size);
    auto ij_size = n_cuda_threads_per_block * max_pairs_per_thread * 2;
    i2* ij_ptr = (i2*)(chunk_int);
    i2* tmp_ptr = (i2*)(chunk_int + ij_size);
    int* pt_types_ptr = (int*)(chunk_int + ij_size * 2);
    int* tmp_pt_types_ptr = (int*)(pt_types_ptr + ij_size / 2);
    int* cnt_ptr = (int*)(chunk_int + ij_size * 3);
    int* tmp_cnt = cnt_ptr + n_cuda_threads_per_block * 2;
    vec3f* vis_ptr = (vec3f*)(cnt_ptr + n_cuda_threads_per_block * 3);
    Facef* fjs_ptr = (Facef*)(vis_ptr + max_aabb_list_size);
    luf* aabbs_ptr = (luf*)(fjs_ptr + max_aabb_list_size);

    
    
    CUDA_CALL(cudaMemcpyAsync((void*)vis_ptr, vis.data(), vis.size() * sizeof(vec3f), cudaMemcpyHostToDevice), stream);
    CUDA_CALL(cudaMemcpyAsync((void*)fjs_ptr, fjs.data(), fjs.size() * sizeof(Facef), cudaMemcpyHostToDevice), stream);

    CUDA_CALL(cudaMemcpyAsync((void*)aabbs_ptr, aabbs.data(), aabbs.size() * sizeof(luf), cudaMemcpyHostToDevice), stream);

    // CUDA_CALL(cudaMemPrefetchAsync(chunk_int, host_cuda_globals.per_stream_buffer_size, host_cuda_globals.device_id, stream));

#endif
    {
        // cuda kernels
        aabb_intersection_test_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(aabbs_ptr, nvi, nfj, ij_ptr, cnt_ptr);

        CUDA_CALL(cudaGetLastError());

        // thrust::inclusive_scan(thrust::cuda::par_nosync, dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());
        // thrust::inclusive_scan(thrust::cuda::par_nosync, cnt_ptr, cnt_ptr + n_cuda_threads_per_block, cnt_ptr);
        inclusive_scan_kernel<<<1, 1, 0, stream>>>(cnt_ptr);
        CUDA_CALL(cudaGetLastError());

        squeeze_ij_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, cnt_ptr, tmp_ptr, pt_types_ptr, tmp_pt_types_ptr);

        // now culled pairs are gathered in the front of tmp
        // dev_cnt.back() has the length of the culled pairs

        bool run_on_gpu = true, run_thrust = false, run_kernel = !run_thrust;
        if (run_on_gpu) {
            if (run_kernel) {
                bool non_atomic = true;
                if (non_atomic) {
                    // pass
                    filter_distance_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, cnt_ptr, tmp_ptr,
                        // vilist_ptr, fjlist_ptr,
                        vis_ptr, fjs_ptr, pt_types_ptr, tmp_pt_types_ptr);

                    CUDA_CALL(cudaGetLastError());
                    inclusive_scan_kernel<<<1, 1, 0, stream>>>(cnt_ptr);
                    // thrust::inclusive_scan(thrust::cuda::par_nosync, cnt_ptr, cnt_ptr + n_cuda_threads_per_block, cnt_ptr);

                    // thrust::inclusive_scan(thrust::cuda::par_nosync, dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());

                    squeeze_ij_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, cnt_ptr, tmp_ptr, pt_types_ptr, tmp_pt_types_ptr);
                }

                else {

                    filter_distance_kernel_atomic<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, tmp_ptr, cnt_ptr, vis_ptr, fjs_ptr, nullptr, nullptr, pt_types_ptr, tmp_pt_types_ptr);
                }
                CUDA_CALL(cudaStreamSynchronize(stream));
                
            }
            else {
#ifdef THRUST_DEV_VECTOR
                // pass
                int cnt_gpu = dev_cnt.back();
                auto ij_end = thrust::copy_if(thrust::device, tmp.begin(), tmp.begin() + cnt_gpu, ij.begin(), [=] __device__(i2 I) -> bool {
                    auto i = I[0], j = I[1];
                    PointTriangleDistanceType ptt;
                    auto vi{ vis_ptr[i] };
                    auto fj{ fjs_ptr[j] };
                    auto d = vf_distance(vis_ptr[i], fjs_ptr[j], ptt);
                    // __device__ float d = 1e-5f;
                    return (d < 1e-4f);
                });
                thrust::copy(thrust::device, ij.begin(), ij_end, tmp.begin());
                dev_cnt.back() = ij_end - ij.begin();
#endif
            }
        }
        else {
#ifdef THRUST_DEV_VECTOR
            // run_on_gpu = false, pass
            // NOTE: copy_if does not relocate the space. should allocate storage manually
            if (run_thrust) {
                int n_copy = dev_cnt.back();
                thrust::host_vector<i2> host_ij(tmp.begin(), tmp.begin() + n_copy), host_tmp(n_copy);
                int cnt = 0;
                thrust::copy_if(thrust::host, host_ij.begin(), host_ij.begin() + n_copy, host_tmp.begin(), [&](i2 I) -> bool {
                    auto i = I[0], j = I[1];
                    PointTriangleDistanceType ptt;
                    auto vi{ vis[i] };
                    auto fj{ fjs[j] };
                    auto d = vf_distance(vi, fj, ptt);
                    // __device__ float d = 1e-5f;
                    bool ret = d < 1e-4f;
                    if (ret) cnt++;
                    return ret;
                });
                // thrust::inclusive_scan(thrust::host, dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());
                thrust::copy(host_tmp.begin(), host_tmp.begin() + cnt, tmp.begin());
                dev_cnt.back() = cnt;
            }
            else {
                // exact solution on cpu, pass
                thrust::host_vector<i2> host_ij;

                for (int i = 0; i < nvi; i++)
                    for (int j = 0; j < nfj; j++)
                        if (intersects(aabbs[i], aabbs[j + nvi])) {
                            PointTriangleDistanceType ptt;
                            auto d = vf_distance(vis[i], fjs[j], ptt);
                            auto tup = vf_distance(vis[i], fjs[j]);
                            auto d2 = std::get<0>(tup);
                            auto type_ref = std::get<1>(tup);
                            if (d < 1e-4f) {
                                host_ij.push_back({ i, j });
                            }
                            if (fabs(d - d2) > 1e-6f) {
                                spdlog::error("d1 = {}, d2 = {}", d, d2);
                                spdlog::error("type, cuda = {}, ref = {}", static_cast<cuda::std::underlying_type_t<ipc::PointTriangleDistanceType>>(ptt), static_cast<cuda::std::underlying_type_t<ipc::PointTriangleDistanceType>>(type_ref));
                            }
                        }
                thrust::copy(host_ij.begin(), host_ij.end(), tmp.begin());
                dev_cnt.back() = host_ij.size();
            }
#endif
        }
    }
    // now tmp has the exact collsion set with d < dhat,
    // and dev_cnt.back has the information of the length of the set

    {
#ifdef THRUST_DEV_VECTOR
        int n_collision_set = dev_cnt.back();
        thrust::host_vector<i2> host_ij(tmp.begin(), tmp.begin()+ n_collision_set);
        
        #else 
        int n_collision_set;
        cudaMemcpyAsync(&n_collision_set, cnt_ptr + n_cuda_threads_per_block - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
        vector<i2> host_ij;
        host_ij.resize(n_collision_set);
        cudaMemcpyAsync(host_ij.data(), tmp_ptr, sizeof(i2) * n_collision_set, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

    #endif
        for (int i = 0; i < n_collision_set; i++) {  
            auto vi = host_ij[i][0], fj = host_ij[i][1];

            idx.push_back({ I, vilist[vi], J, fjlist[fj] });
        }
    }

}
__device__ __constant__ const int max_overlap_size = 1024;
__global__ void iaabb_culling_kernel_atomic(
    // inputs:
    int n_cubes, cudaAffineBody* cubes,
    luf* aabbs, 
    int vtn,

    // outputs:
    int* ret_n_overlaps,
    i2* ret_overlaps, // n_overlaps
    luf* ret_culls // n_overlaps
)
{
    __shared__ luf affine_aabb[n_cuda_threads_per_block];
    __shared__ int n;
    n = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tasks_per_thread = (n_cubes + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid + _i * n_cuda_threads_per_block;
        if (I < n_cubes) {
            auto& c{ cubes[I] };
            affine_aabb[tid] = affine(aabbs[I], c, vtn);
        }
    }
    __syncthreads();
    int n_tasks = n_cubes * n_cubes;
    n_tasks_per_thread = (n_tasks + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        int i = I / n_cubes;
        int j = I % n_cubes;
        if (I < n_cubes * n_cubes && i < j) {
            if (intersects(affine_aabb[i], affine_aabb[j])) {
                int put = atomicAdd(&n, 1);
                ret_overlaps[put] = { i, j };
            }
        }
    }
    
    __syncthreads();
    n_tasks = n;
    n_tasks_per_thread = (n_tasks + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = tid * n_tasks_per_thread + _i;
        if (I < n_tasks) {
            int i = ret_overlaps[I][0], j = ret_overlaps[I][1];
            ret_culls[I] = intersection(affine_aabb[i], affine_aabb[j]);
        }
    }
    *ret_n_overlaps = n_tasks;
    __syncthreads();
}


inline luf __device__ aabb(cudaAffineBody& c, int idx, int type)
{
    if (type == 0) {
        auto v = c.vertices[idx];
        return luf{ v - dev::d_hat_sqr, v + dev::d_hat_sqr };
    }
    else if (type == 1) {
        return compute_aabb(c.edge(idx), dev::d_hat_sqr);
    }
    else {
        return compute_aabb(c.triangle(idx), dev::d_hat_sqr);
    }
}

__global__ void primitive_intersection_test_kernel(
    int type, // 0: vertex, 1: edge, 2: face
    int i_overlap, luf* culls, i2* body_index,
    cudaAffineBody* cubes,
    int* ret_prmts_meta_sizes,
    int* ret_prmts)
{
    // __shared__ int cnt[n_cuda_threads_per_block * 2];
    __shared__ int n;
    n = 0;

    int tid = threadIdx.x; // each block handles one overlap

    auto& cull{ culls[i_overlap] };
    auto& body_idx{ body_index[i_overlap] };
    int i = body_idx[0], j = body_idx[1];
    int nvi, nv;
    switch (type) {
    case 0:
        nvi = cubes[i].n_vertices;
        nv = cubes[i].n_vertices + cubes[j].n_vertices;
        break;
    case 1:
        nvi = cubes[i].n_edges;
        nv = cubes[i].n_edges + cubes[j].n_edges;
        break;
    case 2:
        nvi = cubes[i].n_faces;
        nv = cubes[i].n_faces + cubes[j].n_faces;
        break;
    }
    int n_tasks_per_thread = (nv + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;

    for (int _j = 0; _j < n_tasks_per_thread; _j++) {
        int vi = tid * n_tasks_per_thread + _j;
        if (vi < nv) {
            int j = vi < nvi ? 0 : 1;
            auto b = vi < nvi ? i : j;
            int idx = vi < nvi ? vi : vi - nvi;

            auto t = aabb(cubes[b], idx, type);

            if (intersects(cull, t)) {
                int put = atomicAdd(&n, 1);
                ret_prmts[put] = b;
                atomicAdd(ret_prmts_meta_sizes + j, 1);
            }
        }
    }

}

double iaabb_brute_force_cuda(
    int n_cubes,
    thrust::device_vector<cudaAffineBody>& cubes,
    thrust::device_vector<luf>& aabbs,
    int vtn,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx)

{
    int n_overlaps;

    auto stashed_lt_back = host_cuda_globals.leader_thread_buffer_back;
    auto &lt_back { host_cuda_globals.leader_thread_buffer_back };
    int * dev_n_overlaps = (int *)lt_back;
    lt_back += sizeof(int);

    i2 * overlaps = (i2 *)lt_back;
    lt_back += sizeof(i2) * max_overlap_size;
    
    luf *culls = (luf *)lt_back;
    lt_back += sizeof(luf) * max_overlap_size;
    
    
    iaabb_culling_kernel_atomic<<<1, n_cuda_threads_per_block>>>(n_cubes, PTR(cubes), PTR(aabbs), vtn, dev_n_overlaps, overlaps, culls);

    cudaMemcpy(&n_overlaps, dev_n_overlaps, sizeof(int), cudaMemcpyDeviceToHost);
    //make_lut(n_overlaps, PTR(host_cuda_globals.lut));


    per_intersection_core(n_overlaps, culls, overlaps);

    lt_back = stashed_lt_back;

    return 1.0;
}

__device__ luf intersection(const luf& a, const luf& b)
{
    vec3f l, u;
    l = make_float3(
        CUDA_MAX(a.l.x, b.l.x),
        CUDA_MAX(a.l.y, b.l.y),
        CUDA_MAX(a.l.z, b.l.z));
    u = make_float3(
        CUDA_MIN(a.u.x, b.u.x),
        CUDA_MIN(a.u.y, b.u.y),
        CUDA_MIN(a.u.z, b.u.z));
    return { l, u };
}
__device__ luf affine(luf aabb, cudaAffineBody& c, int vtn)
{
    vec3f cull[8];
    vec3f l, u;
    if (vtn == 3) {
        auto updated =  affine(aabb, c, 2);
        l.x = CUDA_MIN(l.x, updated.l.x);
        l.y = CUDA_MIN(l.y, updated.l.y);
        l.z = CUDA_MIN(l.z, updated.l.z);
        u.x = CUDA_MAX(u.x, updated.u.x);
        u.y = CUDA_MAX(u.y, updated.u.y);
        u.z = CUDA_MAX(u.z, updated.u.z);
        return {l, u};
    }
    auto q{ vtn == 2 ? c.q_update : c.q };
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                auto I = (i << 2) | (j << 1) | k;
                cull[I] = make_float3(
                    i ? aabb.u.x : aabb.l.x,
                    j ? aabb.u.y : aabb.l.y,
                    k ? aabb.u.z : aabb.l.z);
                cull[I] = matmul(q, cull[I]);
            }
    for (int i = 0; i < 8; i++) {
        if (i == 0) {
            l = u = cull[i];
        }
        else {
            l.x = CUDA_MIN(l.x, cull[i].x);
            l.y = CUDA_MIN(l.y, cull[i].y);
            l.z = CUDA_MIN(l.z, cull[i].z);
            u.x = CUDA_MAX(u.x, cull[i].x);
            u.y = CUDA_MAX(u.y, cull[i].y);
            u.z = CUDA_MAX(u.z, cull[i].z);
        }
    }
    return { l, u };
}

__device__ __host__ float vf_distance(vec3f _v, Facef f, int& _pt_type)
{
    auto n = unit_normal(f);
    auto d = dot(n, _v - f.t0);
    auto a1 = area_x2(f.t1, f.t0, f.t2);
    auto v = _v - n * d;
    d = d * d;
    // float a2 = ((f[0] - v).cross(f[1] - v).norm() + (f[1] - v).cross(f[2] - v).norm() + (f[2] - v).cross(f[0] - v).norm());
    // auto a2 = area_x2(f[0], f[1], v) + area_x2(f[1], f[2], v) + area_x2(f[2], f[0], v);
    auto _a1 = dot(cross(f.t0 - v, f.t1 - v), n);
    auto _a2 = dot(cross(f.t1 - v, f.t2 - v), n);
    auto _a3 = dot(cross(f.t2 - v, f.t0 - v), n);
    bool inside = _a1 * _a2 > 0.0f && _a2 * _a3 > 0.0f;
    PointTriangleDistanceType pt_type;
    // if (a2 > a1 + 1e-8) {
    if (!inside) {
        // projection outside of triangle

        auto d_ab = h(f.t0, f.t1, v);
        auto d_bc = h(f.t1, f.t2, v);
        auto d_ac = h(f.t0, f.t2, v);

        auto d_a = ab(v, f.t0);
        auto d_b = ab(v, f.t1);
        auto d_c = ab(v, f.t2);

        auto dab = is_obtuse_triangle(f.t0, f.t1, v) ? CUDA_MIN(d_a, d_b) : d_ab;
        auto dbc = is_obtuse_triangle(f.t2, f.t1, v) ? CUDA_MIN(d_c, d_b) : d_bc;
        auto dac = is_obtuse_triangle(f.t0, f.t2, v) ? CUDA_MIN(d_a, d_c) : d_ac;

        auto d_projected = CUDA_MIN3(dab, dbc, dac);
        d += d_projected * d_projected;

        if (d_projected == d_ab)
            pt_type = PointTriangleDistanceType::P_E0;
        else if (d_projected == d_bc)
            pt_type = PointTriangleDistanceType::P_E1;
        else if (d_projected == d_ac)
            pt_type = PointTriangleDistanceType::P_E2;
        else if (d_projected == d_a)
            pt_type = PointTriangleDistanceType::P_T0;
        else if (d_projected == d_b)
            pt_type = PointTriangleDistanceType::P_T1;
        else
            pt_type = PointTriangleDistanceType::P_T2;
    }
    else
        pt_type = PointTriangleDistanceType::P_T;
    _pt_type = static_cast<underlying_type_t<PointTriangleDistanceType>>(pt_type);
    return d;
}

__device__ luf compute_aabb(const Edgef& e, float d_hat_sqrt)
{
    vec3f l, u;
    l = make_float3(
        CUDA_MIN(e.e0.x, e.e1.x),
        CUDA_MIN(e.e0.y, e.e1.y),
        CUDA_MIN(e.e0.z, e.e1.z));
    u = make_float3(
        CUDA_MAX(e.e0.x, e.e1.x),
        CUDA_MAX(e.e0.y, e.e1.y),
        CUDA_MAX(e.e0.z, e.e1.z));
    return { l, u };
}

__device__ luf compute_aabb(const Facef& f, float d_hat_sqrt)
{
    vec3f l, u;
    l = make_float3 (
        CUDA_MIN3(f.t0.x, f.t1.x, f.t2.x),
        CUDA_MIN3(f.t0.y, f.t1.y, f.t2.y),
        CUDA_MIN3(f.t0.z, f.t1.z, f.t2.z));
    u = make_float3 (
        CUDA_MAX3(f.t0.x, f.t1.x, f.t2.x),
        CUDA_MAX3(f.t0.y, f.t1.y, f.t2.y),
        CUDA_MAX3(f.t0.z, f.t1.z, f.t2.z));
    return { l, u };
}

__global__ void strided_memset_kernel(
    i2* start, i2 value, int n)
{
    auto tid = threadIdx.x;
    int n_tasks_per_thread = (n + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_tasks_per_thread; _i++) {
        int I = _i + tid * n_tasks_per_thread;
        if (I < n) {
            start[I] = value;
        }
    }
}
__global__ void prepare_aabb_vi_fj_kernel(
    int i_overlap, i2* overlaps,
    cudaAffineBody* cubes,
    int type, // vi fj; vj fi; ei ej (i < j)

    int* vi_meta_sizes, int* fj_meta_sizes, // i.e. nvi, nfj
    int* vilist, int* fjlist,
    vec3f* vifjs, // not necessarily vi fj, just groups of 4 vec3f
    luf* joint_aabbs)
{
    auto tid = threadIdx.x;
    auto i = overlaps[i_overlap][0], j = overlaps[i_overlap][1];
    auto &ci{ cubes[i] }, &cj{ cubes[j] };

    auto T_t = vi_meta_sizes[0];
    int n_tasks_per_thread = T_t + n_cuda_threads_per_block - 1 / n_cuda_threads_per_block;
    for (int _j = 0; _j < n_tasks_per_thread; _j++) {
        auto I = _j + tid * n_tasks_per_thread;
        if (I < T_t) {
            int i = vilist[I];
            switch (type) {
            case 1: i += vi_meta_sizes[0];
            case 0: {
                auto& p{ ci.vertices[i] };
                vifjs[I] = p;
                joint_aabbs[I] = { p - dev::d_hat_sqr, p + dev::d_hat_sqr };
                break;
            }
            case 2: {
                auto& e{ ci.edge(i) };
                vifjs[I * 2] = e.e0;
                vifjs[I * 2 + 1] = e.e1;
                joint_aabbs[I] = compute_aabb(e, dev::d_hat_sqr);
                break;
            }
            }
        }
    }
    T_t = fj_meta_sizes[1];
    n_tasks_per_thread = T_t + n_cuda_threads_per_block - 1 / n_cuda_threads_per_block;
    for (int _j = 0; _j < n_tasks_per_thread; _j++) {
        auto I = _j + tid * n_tasks_per_thread;
        if (I < T_t) {
            auto i = fjlist[I];
            switch (type) {
            case 0: i += fj_meta_sizes[0];
            case 1: {
                Facef f{ cj.triangle(i) };
                vifjs[vi_meta_sizes[0] + I * 3] = f.t0;
                vifjs[vi_meta_sizes[0] + I * 3 + 1] = f.t1;
                vifjs[vi_meta_sizes[0] + I * 3 + 2] = f.t2;

                joint_aabbs[vi_meta_sizes[0] + I] = compute_aabb(f, dev::d_hat_sqr);
                break;
            }
            case 2: {
                auto& e{ ci.edge(i + fj_meta_sizes[0]) };
                vifjs[vi_meta_sizes[0] * 2 + I * 2] = e.e0;
                vifjs[vi_meta_sizes[0] * 2 + I * 2 + 1] = e.e1;
                joint_aabbs[I] = compute_aabb(e, dev::d_hat_sqr);
                break;
            }
            }
        }
    }
}

void per_intersection_core(int n_overlaps, luf* culls, i2* overlaps)
{
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n_overlaps; i++) {
        auto tid = omp_get_thread_num();
        auto& stream{ host_cuda_globals.streams[tid] };
        cudaAffineBody* cubes{ host_cuda_globals.cubes };
        // preparing pointers

        int *vlist, *elist, *flist;
        int *vlist_meta, *elist_meta, *flist_meta;

        char *st_back_stashed{ host_cuda_globals.small_temporary_buffer_back[tid] }, *bulk_back_stashed{ host_cuda_globals.bulk_buffer_back[tid] };

        for (int type = 0; type < 3; type++) {
            // designate buffers
            auto& st{ host_cuda_globals.small_temporary_buffer_back[tid] };

            int* ret_meta_sizes = (int*)st;
            st += 2 * sizeof(int);

            auto& bulk{ host_cuda_globals.bulk_buffer_back[tid] };

            int* ret_prmts = (int*)bulk;
            bulk += sizeof(i2) * max_prmts_per_block;

            primitive_intersection_test_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(
                type, i, culls, overlaps, cubes, ret_meta_sizes, ret_prmts);

            switch (type) {
            case 0:
                vlist = ret_prmts;
                vlist_meta = ret_meta_sizes;
                break;
            case 1:
                elist = ret_prmts;
                elist_meta = ret_meta_sizes;
                break;
            case 2:
                flist = ret_prmts;
                flist_meta = ret_meta_sizes;
                break;
            }
        }


        int sizes[6];
        cudaStreamSynchronize(stream);
        cudaMemcpy(sizes, vlist_meta, sizeof(int) * 6, cudaMemcpyDeviceToHost);
        i2 ij;
        cudaMemcpy(&ij, overlaps + i, sizeof(i2), cudaMemcpyDeviceToHost);
        i2 ji = { ij[1], ij[0] };

        host_cuda_globals.bulk_buffer_back[tid] = bulk_back_stashed;
        host_cuda_globals.small_temporary_buffer_back[tid] = st_back_stashed;

        int nvifj, nvjfi, neiej;
        int *host_cnts[3], cnt[3];
        i2 *vifj_ptr, *vjfi_ptr, *eiej_ptr;
        for (int type = 0; type < 3; type++) {
            // vi fj, vj fi, ei ej (i < j)

            auto& bulk{ host_cuda_globals.bulk_buffer_back[tid] };
            auto& st{ host_cuda_globals.small_temporary_buffer_back[tid] };
            float3* vifjs = (float3*)bulk;
            bulk += sizeof(float3) * 4 * max_prmts_per_block;
            luf* joint_aabbs = (luf*)bulk;

            prepare_aabb_vi_fj_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(
                i, overlaps, cubes, type, vlist_meta, flist_meta, vlist, flist, vifjs, joint_aabbs);

            i2* ret_ij = (i2*)bulk;
            bulk += sizeof(i2) * max_pairs_per_block;
            i2* buf_ij = (i2*)bulk;
            bulk += sizeof(i2) * max_pairs_per_block;
            int* pt_types = (int*)bulk;
            bulk += sizeof(int) * max_pairs_per_block;
            int* tmp_pt_types = (int*)bulk;
            bulk += sizeof(int) * max_pairs_per_block;

            int* ret_cnt = (int*)st;
            st += sizeof(int);

            int nvi, nfj;
            switch (type) {
                case 0: nvi = sizes[0]; nfj = sizes[5]; break;
                case 1: nvi = sizes[1]; nfj = sizes[4]; break;
                case 2: nvi = sizes[2]; nfj = sizes[3]; break;
            }

            aabb_intersection_test_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(joint_aabbs, nvi, nfj, ret_ij, ret_cnt);
            filter_distance_kernel_atomic<<<1, n_cuda_threads_per_block, 0, stream>>>(
                buf_ij, ret_ij, ret_cnt, vifjs, nullptr,
                vlist, flist,
                tmp_pt_types, pt_types, 
                1e-4f, 
                vlist_meta + type);
            // gather_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ret_cnt, ret_ij, buf_ij, vlist, flist);
            host_cnts[type] = ret_cnt;
            switch (type) {
                case 0: vifj_ptr = ret_ij; break;
                case 1: vjfi_ptr = ret_ij; break;
                case 2: eiej_ptr = ret_ij; break;
            }
        }



        cudaStreamSynchronize(stream);
        cudaMemcpy(cnt, host_cnts[0], sizeof (int ) * 3, cudaMemcpyDeviceToHost);
        nvifj = cnt[0]; 
        nvjfi = cnt[1];
        neiej = cnt[2];
        int pt_put, ee_put;
#pragma omp critical
        {
            pt_put = host_cuda_globals.npt;
            host_cuda_globals.npt += (nvifj + nvjfi);
            ee_put = host_cuda_globals.nee;
            host_cuda_globals.nee += neiej;
        }
        cudaMemcpy(host_cuda_globals.pt.p + pt_put, vifj_ptr, nvifj * sizeof(i2), cudaMemcpyDeviceToDevice);
        cudaMemcpy(host_cuda_globals.pt.p + pt_put + nvifj, vjfi_ptr, nvjfi * sizeof(i2), cudaMemcpyDeviceToDevice);
        cudaMemcpy(host_cuda_globals.ee.p + ee_put, eiej_ptr, neiej * sizeof(i2), cudaMemcpyDeviceToDevice);

        // cudaMemset(host_cuda_globals.pt.b + pt_put, (long long)ij, nvifj);
        // cudaMemset(host_cuda_globals.pt.b + pt_put + nvifj, ji, nvjfi);
        // cudaMemset(host_cuda_globals.ee.b + ee_put, ij, sizeof(i2) * neiej);
        strided_memset_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(host_cuda_globals.pt.b + pt_put, ij, nvifj);
        strided_memset_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(host_cuda_globals.pt.b + pt_put + nvifj, ji, nvjfi);
        strided_memset_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(host_cuda_globals.ee.b + ee_put, ij, neiej);
    }
}

void make_lut(int lut_size, thrust::device_vector<i2>& lut) {
    int n_cubes = host_cuda_globals.n_cubes;
    
    {
        thrust::host_vector<i2> diagonals(n_cubes);
        thrust::device_vector<i2> dev_diagonals(n_cubes);
        for (int i = 0; i < n_cubes; i++) diagonals[i] = { i, i };
        dev_diagonals = diagonals;
        lut.insert(lut.begin() + lut_size, dev_diagonals.begin(), dev_diagonals.end());
    }
 
    
    thrust::sort(lut.begin(), lut.end());
    // auto new_end = thrust::unique(lut.begin(), lut.end());

    // lut.resize(lut_size);
    lut.resize(lut_size + n_cubes);
    // lut.shrink_to_fit();

}

void cuda_culling_glue(
    int vtn,
    thrust::device_vector<luf>& aabbs,
    thrust::device_vector<luf>& ret_culls
) {


    int n_cubes = host_cuda_globals.n_cubes;
    auto cubes { host_cuda_globals.cubes };

    int n_overlaps;

    auto stashed_lt_back = host_cuda_globals.leader_thread_buffer_back;
    auto &lt_back { host_cuda_globals.leader_thread_buffer_back };
    int * dev_n_overlaps = (int *)lt_back;
    lt_back += sizeof(int);

    i2 * overlaps = (i2 *)lt_back;
    lt_back += sizeof(i2) * max_overlap_size;
    
    luf *culls = (luf *)lt_back;
    lt_back += sizeof(luf) * max_overlap_size;

    iaabb_culling_kernel_atomic<<<1, n_cuda_threads_per_block>>>(n_cubes, cubes, PTR(aabbs), vtn, dev_n_overlaps, overlaps, culls);
    cudaDeviceSynchronize();
    cudaMemcpy(&n_overlaps, dev_n_overlaps, sizeof(int), cudaMemcpyDeviceToHost);


    host_cuda_globals.lut = thrust::device_vector<i2>{
        overlaps, overlaps + n_overlaps
    };
    ret_culls = thrust::device_vector<luf> {
        culls, culls + n_overlaps
    };
    
    lt_back = stashed_lt_back; 
}
/*
// deprecated
void stencil_classifier(
    thrust::device_vector<i2>& pt_idx,
    thrust::device_vector<i2>& pt_body_idx,
    thrust::device_vector<PointTriangleDistanceType>& pt_types)
{
    static const PointTriangleDistanceType types[] = {
        PointTriangleDistanceType::P_T0, // 0
        PointTriangleDistanceType::P_T1, // 1
        PointTriangleDistanceType::P_T2, // 2
        PointTriangleDistanceType::P_E0, // 3
        PointTriangleDistanceType::P_E1, // 4
        PointTriangleDistanceType::P_E2, // 5
        PointTriangleDistanceType::P_T // 6
    };
    // static const EdgeEdgeDistanceType edge_types[] = {
    // };
    assert(pt_idx.size() == pt_types.size() && pt_idx.size() == pt_body_idx.size());
    for (int i = 0; i < 7; i++) {
        auto &cset{ host_cuda_globals.collision_sets.pt_set[i] }, &bset{ host_cuda_globals.collision_sets.pt_set_body_index[i] };
        // cset.clear();
        // bset.clear();
        thrust::copy_if(
            thrust::make_zip_iterator(thrust::make_tuple(pt_idx.begin(), pt_body_idx.begin(), pt_types.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(pt_idx.end(), pt_body_idx.end(), pt_types.end())),
            thrust::make_zip_iterator(thrust::make_tuple(cset, bset, thrust::make_discard_iterator())),
            [=] __device__(const thrust::tuple<i2, i2, PointTriangleDistanceType>& tup) {
                return static_cast<cuda::std::underlying_type_t<PointTriangleDistanceType>>(thrust::get<2>(tup)) == i;
            });
    }
    // FIXME: make sure i < j before merging into lut? 
    {
    // static thrust::device_vector<i2> merged_lut;
        // // generate look up table for spare hess 
        // merged_lut .resize(0);
        // thrust::set_union(thrust::device, pt_body_idx.begin(), pt_body_idx.end(), host_cuda_globals.lut.begin(), host_cuda_globals.lut.end(), merged_lut.begin());
        // host_cuda_globals.lut = merged_lut;

        // should be sorted, just not worth it

        thrust::copy(pt_body_idx.begin(), pt_body_idx.end(), host_cuda_globals.lut.end());
    }
}
*/
/*
// coded but necessity in doubt
__global__ void copy_kernel(i2* ij, PointTriangleDistanceType* pt_types, int* cnt, i2** dst_prim, i2** dst_body, int I, int J, int* cset_cnt)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int put[7], type_cnt[7 * n_cuda_threads_per_block];

    int n_tasks = cnt[n_cuda_threads_per_block - 1];
    int n_task_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < 7; i++)
        type_cnt[i * n_cuda_threads_per_block + tid] = 0;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        auto i = tid * n_task_per_thread + _i;
        if (i < n_tasks) {
            auto t = static_cast<underlying_type_t<PointTriangleDistanceType>>(pt_types[i]);
            type_cnt[t * n_cuda_threads_per_block + i]++;
        }
    }

    __syncthreads();

    if (tid < 7) {
        auto sum = 0;
        for (int i = 0; i < n_cuda_threads_per_block; i++) {
            sum += type_cnt[tid * n_cuda_threads_per_block + i];
        }
        put[tid] = atomicAdd(cset_cnt + tid, sum);
    }
    __syncthreads();

    for (int _i = 0; _i < n_task_per_thread; _i++) {
        auto i = tid * n_task_per_thread + _i;
        if (i < n_tasks) {
            auto t = static_cast<underlying_type_t<PointTriangleDistanceType>>(pt_types[i]);

            dst_prim[t][i + put[t]] = ij[i];
            dst_body[t][i + put[t]] = { I, J };
        }
    }
}
*/
