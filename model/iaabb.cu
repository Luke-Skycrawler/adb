
#include "cuda_header.cuh"
#include "cuda_globals.cuh"
#include "timer.h"
#include <omp.h>
#include <spdlog/spdlog.h>
#include <tuple>

#include <assert.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/set_operations.h>
#include <algorithm>
using namespace std;
#define CPU_REF
#define PT_ONLY
#ifdef PT_ONLY
#define MAX_TYPES 2
#else
#define MAX_TYPES 3
#endif
// FIXME: tid probably not in 1 block

static const int max_overlap_size = 1024;

__device__ __host__ float vf_distance(vec3f _v, Facef f, int& pt_type);
__device__ luf intersection(const luf &a, const luf &b);
__device__ luf affine(luf aabb, cudaAffineBody &c, int vtn);
__device__ __host__ luf compute_aabb(const Facef& f, float d_hat_sqrt);
__device__ __host__ luf compute_aabb(const Edgef& e, float d_hat_sqrt);
__host__ __device__ luf merge(luf a, luf b);
void make_lut(int n_overlaps, i2* overlaps);

void per_intersection_core(int n_overlaps, luf* culls, i2* overlaps, int vtn, float* toi = nullptr);

/* NOTE: kernel argument convention:
    buf_: device buffer
    ret_: return buffer
    io_: input and output from the same buffer, i.e. in-place
    : (none) inputs
*/

__global__ void aabb_intersection_test_kernel_atomic(luf *dev_aabbs, int nvi, int nfj, i2 * ret_ij, int *ret_cnt){
    __shared__ luf aabbs[max_aabb_list_size];
    __shared__ int n;
    n = 0;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_task_per_thread = (nvi * nfj + blockDim.x - 1) / blockDim.x;
    int n_copies_per_thread = (nvi + nfj + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < n_copies_per_thread; i++) {
        int idx = tid * n_copies_per_thread + i;
        if (idx < nvi + nfj && idx < max_aabb_list_size)
            aabbs[idx] = dev_aabbs[idx];
    }
    __syncthreads();
    // copys the bounding boxes to shared memory

    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int I = tid * n_task_per_thread + _i;
        if (I < nvi * nfj) {
            int i = I / nfj;
            int j = I % nfj;
            if (intersects(aabbs[i], aabbs[nvi + j])) {
                auto put = atomicAdd(&n, 1);
                ret_ij[put] = { i, j };
            }
        }
    }
    __syncthreads();
    *ret_cnt = n;
}
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
__device__ __host__ float pt_collision_time(
    const vec3f& p0,
    const Facef& t0,
    const vec3f& p1,
    const Facef& t1);

__device__ __host__ float ee_collision_time(
    const Edgef& ei0,
    const Edgef& ej0,
    const Edgef& ei1,
    const Edgef& ej1);


__global__ void toi_decision_kernel(
    i2 *ijs, 
    int *cnt,
    vec3f *vifjs,
    int _nvi,
    float* ret_toi
){
    // NOTE: can only launched with n_cuda_threads_per_block threads
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks = *cnt;
    int n_task_per_thread = (n_tasks + n_cuda_threads_per_block - 1) / n_cuda_threads_per_block;
    int nvi = _nvi * 2;
    __shared__ float tois[n_cuda_threads_per_block];
    tois[tid] = 1.0f;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int idx = tid * n_task_per_thread + _i;
        if (idx < n_tasks) {
            auto ij = ijs[idx];
            int i = ij[0];
            int j = ij[1];

            auto vt1{ vifjs[i * 2] }, vt2{vifjs[i * 2 + 1]};
            Facef ft1{ vifjs[j * 6 + nvi], vifjs[j * 6 + 1 + nvi], vifjs[j * 6 + 2 + nvi] },
                ft2 {vifjs[j * 6 + 3 + nvi], vifjs[j * 6 + 4 + nvi], vifjs[j * 6 + 5 + nvi]};
            
            auto toi = pt_collision_time(vt1, ft1, vt2, ft2);
            tois[tid] = CUDA_MIN(tois[tid], toi);
        }
    }
    __syncthreads();
    if (tid == 0) {
        float toi_min = 1.0f;
        for(int i = 0; i < n_cuda_threads_per_block; i ++) {
            toi_min = CUDA_MIN(toi_min, tois[i]);
        }
        *ret_toi = toi_min;
    }
}


__global__ void filter_distance_kernel_atomic(i2* buf_ij, i2* io_tmp,
    int* io_cnt, 
    vec3f* vis, Facef* fjs,

    int* vilist, int* fjlist,
    int* ret_pt_types,
    float dhat = 1e-4f, int nvi = 0)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_tasks = *io_cnt;
    int n_task_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    __shared__ int n;
    n = 0;
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
            int pt_type;
            auto d = vf_distance(v, f, pt_type);
            if (d < dhat) {
                auto put = atomicAdd(&n, 1);
                if (vilist && fjlist)
                    buf_ij[put] = { vilist[i], fjlist[j] };
                else
                    buf_ij[put] = { i, j };
                ret_pt_types[put] = pt_type;
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
        }
    }
    *io_cnt = n;
}

// deprecated
__global__ void filter_distance_kernel(i2* ret_ij, int* ret_cnt, i2* tmp,
    // int* vilist, int* fjlist,
    vec3f* vis, Facef* fjs,
    int* ret_pt_types,
    int* tmp_pt_types,
    float dhat = 1e-4)
{
    // // squeeze the ret_ij list according to a prefix sum array ret_cnt
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

// deprecated
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

// deprecated
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
        bool non_atomic = false;
        if (non_atomic) {
            // cuda kernels
            aabb_intersection_test_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(aabbs_ptr, nvi, nfj, ij_ptr, cnt_ptr);

            CUDA_CALL(cudaGetLastError());

            // thrust::inclusive_scan(thrust::cuda::par_nosync, dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());
            // thrust::inclusive_scan(thrust::cuda::par_nosync, cnt_ptr, cnt_ptr + n_cuda_threads_per_block, cnt_ptr);
            inclusive_scan_kernel<<<1, 1, 0, stream>>>(cnt_ptr);
            CUDA_CALL(cudaGetLastError());

            squeeze_ij_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, cnt_ptr, tmp_ptr, pt_types_ptr, tmp_pt_types_ptr);

        }
        else {
            aabb_intersection_test_kernel_atomic<<< 1, n_cuda_threads_per_block, 0, stream >>>(aabbs_ptr, nvi, nfj, tmp_ptr, cnt_ptr + n_cuda_threads_per_block - 1);
        }
        // now culled pairs are gathered in the front of tmp
        // dev_cnt.back() has the length of the culled pairs

        bool run_on_gpu = true, run_thrust = false, run_kernel = !run_thrust;
        if (run_on_gpu) {
            if (run_kernel) {
                bool non_atomic = false;
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

                    filter_distance_kernel_atomic<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, tmp_ptr, cnt_ptr + n_cuda_threads_per_block - 1, vis_ptr, fjs_ptr, nullptr, nullptr, tmp_pt_types_ptr);
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
__global__ void culling_kernel_atomic (
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
    // NOTE: can only be launched with <<<1, n_cuda_threads_per_block>>>
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


inline __device__ __host__ luf aabb(cudaAffineBody& c, int idx, int type, int  vtn)
{
    if (vtn != 3) {
        if (type == 0) {
            auto v = c.projected[idx];
            return luf{ v - dev::d_hat_sqr, v + dev::d_hat_sqr };
        }
        else if (type == 2) {
            return compute_aabb(c.edge(idx), dev::d_hat_sqr);
        }
        else {
            return compute_aabb(c.triangle(idx), dev::d_hat_sqr);
        }

    }
    else  {
        if (type == 0) {
            Edgef p_u{c.projected[idx], c.updated[idx]};
            return compute_aabb(p_u, 0.0f);
        }
        else if (type == 2) {
            auto et2 = c.edge_updated(idx), et1  = c.edge(idx);
            luf lu2 = compute_aabb(et2, 0.0f), lu1 = compute_aabb(et1, 0.0f);
            return merge(lu1, lu2);
        }
        else {
            auto t2 = c.triangle_updated(idx), t1 = c.triangle(idx);
            luf lu2 = compute_aabb(t2, 0.0f), lu1 = compute_aabb(t1, 0.0f);
            return merge(lu1, lu2);
        }
    }
}

__global__ void primitive_intersection_test_kernel(
    int type, // 0: vertex, 2: edge, 1: face
    int vtn,
    int i_overlap, luf* culls, i2* body_index,
    cudaAffineBody* cubes,
    int* ret_prmts_meta_sizes,
    int* ret_prmts)
{
    __shared__ int n;
    n = 0;

    int tid = threadIdx.x; // each block handles one overlap

    auto cull{ culls[i_overlap] };
    auto body_idx{ body_index[i_overlap] };
    int i = body_idx[0], j = body_idx[1];
    int nvi, nv;
    switch (type) {
    case 0:
        nvi = cubes[i].n_vertices;
        nv = cubes[i].n_vertices + cubes[j].n_vertices;
        break;
    case 2:
        nvi = cubes[i].n_edges;
        nv = cubes[i].n_edges + cubes[j].n_edges;
        break;
    case 1:
        nvi = cubes[i].n_faces;
        nv = cubes[i].n_faces + cubes[j].n_faces;
        break;
    }
    int n_tasks_per_thread = (nv + blockDim.x - 1) / blockDim.x;
    ret_prmts_meta_sizes[0] = ret_prmts_meta_sizes[1] = 0;
    for (int _j = 0; _j < n_tasks_per_thread; _j++) {
        int vi = tid * n_tasks_per_thread + _j;
        if (vi < nv) {
            int J = vi < nvi ? 0 : 1;
            auto b = vi < nvi ? i : j;
            int idx = vi < nvi ? vi : vi - nvi;

            auto t = aabb(cubes[b], idx, type, vtn);

            if (intersects(cull, t)) {
                int put = atomicAdd(&n, 1);
                ret_prmts[put] = idx;
                atomicAdd(ret_prmts_meta_sizes + J, 1);
            }
        }
    }

}


#ifdef CPU_REF
void primtive_intersection_host(
    int type,
    int vtn,
    int i,
    vector<cudaAffineBody> &host_cubes,
    const vector<luf>& host_culls,
    const vector<i2> &host_overlaps,
    vector<int> host_prmts[MAX_TYPES][2]
){
    // CPU_CODE
    auto cull{ host_culls[i] };
    auto idx{ host_overlaps[i] };
    int I{ idx[0] }, J{ idx[1] };
    int nvi, nv;
    switch (type) {
    case 0:
        nvi = host_cubes[I].n_vertices;
        nv = host_cubes[I].n_vertices + host_cubes[J].n_vertices;
        break;
    case 2:
        nvi = host_cubes[I].n_edges;
        nv = host_cubes[I].n_edges + host_cubes[J].n_edges;
        break;
    case 1:
        nvi = host_cubes[I].n_faces;
        nv = host_cubes[I].n_faces + host_cubes[J].n_faces;
        break;
    }
    for (int vi = 0; vi < nv; vi++) {
        auto b = vi < nvi ? I : J;
        int id = vi < nvi ? vi : vi - nvi;
        int iorj = vi < nvi ? 0 : 1;
        auto t = aabb(host_cubes[b], id, type, vtn);
        if (intersects(cull, t)) {
            host_prmts[type][iorj].push_back(id);
        }
    }
}

#endif
float iaabb_brute_force_cuda_pt_only(
    int n_cubes,
    cudaAffineBody* cubes,
    luf* aabbs,
    int vtn,
    std::vector<std::array<int, 4>>& idx
    // std::vector<std::array<int, 4>>& eidx,
    // std::vector<std::array<int, 2>>& vidx
    )

{
    int n_overlaps;

    auto stashed_lt_back = host_cuda_globals.leader_thread_buffer_back;
    auto &lt_back { host_cuda_globals.leader_thread_buffer_back };
    int * dev_n_overlaps = (int *)lt_back;
    lt_back += sizeof(int);

    host_cuda_globals.nee = host_cuda_globals.npt = 0;
    idx.resize(0);

    i2 * overlaps = (i2 *)lt_back;
    lt_back += sizeof(i2) * max_overlap_size;
    
    luf *culls = (luf *)lt_back;
    lt_back += sizeof(luf) * max_overlap_size;
    
    
    culling_kernel_atomic <<<1, n_cuda_threads_per_block>>>(n_cubes, cubes, aabbs, vtn, dev_n_overlaps, overlaps, culls);
    CUDA_CALL(cudaGetLastError());
    cudaMemcpy(&n_overlaps, dev_n_overlaps, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CALL(cudaDeviceSynchronize());
    make_lut(n_overlaps, overlaps);

    float ret_toi = 1.0f;
    per_intersection_core(n_overlaps, culls, overlaps, vtn, &ret_toi);

    lt_back = stashed_lt_back;

    int npt = host_cuda_globals.npt, nee = host_cuda_globals.nee;
    i2 *plist = new i2[npt], *blist = new i2[npt];
    cudaMemcpy(plist, host_cuda_globals.pt.p, sizeof(i2) * npt, cudaMemcpyDeviceToHost);
    cudaMemcpy(blist, host_cuda_globals.pt.b, sizeof(i2) * npt, cudaMemcpyDeviceToHost);

    for (int i = 0; i < npt; i++) {
        auto p = plist[i], b = blist[i];
        idx.push_back({ b[0], p[0], b[1], p[1] });
    }
    delete[] plist;
    delete[] blist;
    return ret_toi;
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
    int vtn,

    int* vi_meta_sizes, int* fj_meta_sizes, // i.e. nvi, nfj
    // int nvi, int nfj,
    int* vilist, int* fjlist,
    vec3f* vifjs, // not necessarily vi fj, just groups of 4 vec3f
    luf* joint_aabbs)
{
    auto tid = threadIdx.x;
    auto i = overlaps[i_overlap][0], j = overlaps[i_overlap][1];
    auto &ci{ cubes[type == 1 ? j : i] }, &cj{ cubes[type == 1 ? i : j] };
    int nvi = vi_meta_sizes[0], nvj = vi_meta_sizes[1], nfj = fj_meta_sizes[1], nfi = fj_meta_sizes[0];

    auto T_t = type == 1 ? nvj : nvi;
    // type = 1 is vj fi intersection test
    int n_tasks_per_thread = T_t + blockDim.x - 1 / blockDim.x;
    for (int _j = 0; _j < n_tasks_per_thread; _j++) {
        auto I = _j + tid * n_tasks_per_thread;
        if (I < T_t) {
            int i = vilist[I];
            switch (type) {
            case 1: i = vilist[I + nvi];
            case 0: {
                auto& p{ ci.projected[i] };


                if (vtn != 3){
                    vifjs[I] = p;
                    joint_aabbs[I] = { p - dev::d_hat_sqr, p + dev::d_hat_sqr };

                }
                else {
                    auto &pt2 = ci.updated[i];
                    vifjs[I * 2] = p;
                    vifjs[I * 2 + 1] = pt2;
                    joint_aabbs[I] = compute_aabb(Edgef{p, pt2}, 0.0f);
                }
                break;
            }
            case 2: {
                auto& e{ ci.edge(i) };
                if (vtn != 3) {

                    vifjs[I * 2] = e.e0;
                    vifjs[I * 2 + 1] = e.e1;
                    joint_aabbs[I] = compute_aabb(e, dev::d_hat_sqr);
                }
                else {
                    auto  &e2 = ci.edge_updated(i);
                    vifjs[I * 4] = e.e0;
                    vifjs[I * 4 + 1] = e.e1;
                    vifjs[I * 4 + 2] = e2.e0;
                    vifjs[I * 4 + 3] = e2.e1;
                    joint_aabbs[I] = merge(compute_aabb(e, 0.0f), compute_aabb(e2, 0.0f));
                }
                break;
            }
            }
        }
    }
    auto T_t2 = type == 1 ? nfi : nfj;
    n_tasks_per_thread = T_t2 + blockDim.x - 1 / blockDim.x;
    for (int _j = 0; _j < n_tasks_per_thread; _j++) {
        auto I = _j + tid * n_tasks_per_thread;
        if (I < T_t2) {
            auto i = fjlist[I + nfi];
            switch (type) {
            case 1: i = fjlist[I];
            case 0: {
                Facef f{ cj.triangle(i) };

                if (vtn != 3) {

                    vifjs[T_t + I * 3] = f.t0;
                    vifjs[T_t + I * 3 + 1] = f.t1;
                    vifjs[T_t + I * 3 + 2] = f.t2;
    
                    joint_aabbs[T_t + I] = compute_aabb(f, dev::d_hat_sqr);
                }
                else {
                    vifjs[T_t * 2 + I * 6] = f.t0;
                    vifjs[T_t * 2 + I * 6 + 1] = f.t1;
                    vifjs[T_t * 2 + I * 6 + 2] = f.t2;
                    auto &f2 = cj.triangle_updated(i);
                    vifjs[T_t * 2 + I * 6 + 3] = f2.t0;
                    vifjs[T_t * 2 + I * 6 + 4] = f2.t1;
                    vifjs[T_t * 2 + I * 6 + 5] = f2.t2;
                    joint_aabbs[T_t + I] = merge(compute_aabb(f, 0.0f), compute_aabb(f2, 0.0f));
                }
                break;
            }
            case 2: {
                auto& e{ cj.edge(i) };
                if (vtn != 3) {
                    vifjs[T_t * 2 + I * 2] = e.e0;
                    vifjs[T_t * 2 + I * 2 + 1] = e.e1;
                    joint_aabbs[T_t + I] = compute_aabb(e, dev::d_hat_sqr);
                }
                else {
                    vifjs[T_t * 4 + I * 4] = e.e0;
                    vifjs[T_t * 4 + I * 4 + 1] = e.e1;
                    auto &e2 = cj.edge_updated(i);
                    vifjs[T_t * 4 + I * 4 + 2] = e2.e0;
                    vifjs[T_t * 4 + I * 4 + 3] = e2.e1;
                }
                break;
            }
            }
        }
    }
}



#ifdef CPU_REF
void prepare_vifj(
    i2 ij, vector<cudaAffineBody>& cubes, int type,
    vector<int>& vis, vector<int>& vjs,
    vector<int>& fis, vector<int>& fjs,
    vector<vec3f>& vifjs, vector<luf>& joint_aabbs)
{
    int nvi = vis.size(), nvj = vjs.size(),
        nfi = fis.size(), nfj = fjs.size();
    int T = type == 1 ? nvj : nvi;
    auto T_2 = type == 1 ? nfi : nfj;

    vifjs.resize(T + T_2 * 3);
    joint_aabbs.resize(T + T_2);
    int ii = type == 1 ? ij[1] : ij[0], jj = type == 1 ? ij[0] : ij[1];
    auto &ci{ cubes[ii] }, &cj{ cubes[jj] };
    for (int I = 0; I < T; I++) {
        int i = type == 1 ? vjs[I] : vis[I];
        switch (type) {
        case 1:
        case 0: {
            auto& p{ ci.projected[i] };
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
    for (int I = 0; I < T_2; I++) {
        int i = type == 1 ? fis[I] : fjs[I];
        switch (type) {
        case 1:
        case 0: {
            Facef f{ cj.triangle(i) };
            vifjs[T + I * 3] = f.t0;
            vifjs[T + I * 3 + 1] = f.t1;
            vifjs[T + I * 3 + 2] = f.t2;

            joint_aabbs[T + I] = compute_aabb(f, dev::d_hat_sqr);
            break;
        }
        case 2: {
            auto& e{ cj.edge(i) };
            vifjs[T * 2 + I * 2] = e.e0;
            vifjs[T * 2 + I * 2 + 1] = e.e1;
            joint_aabbs[T + I] = compute_aabb(e, dev::d_hat_sqr);
            break;
        }
        }
    }
}

std::vector<cudaAffineBody> get_host_cubes_copy(
    vec3f * &projected, vec3f * &updated, vec3f *&vertices,
    int *&edges , int *&faces
){
    // create cpu-accessible copy of cubes, 
    // for debugging purpose only
    auto &g {host_cuda_globals};
    auto &cubes{ g.host_cubes };
    auto host_cubes = cubes;

    vec3f* host_projected = new vec3f[host_cuda_globals.n_vertices],
     * host_updated = new vec3f[host_cuda_globals.n_vertices],
     * host_vertices = new vec3f[host_cuda_globals.n_vertices];
    int* host_edges = new int[host_cuda_globals.n_edges * 2];
    int* host_faces = new int[host_cuda_globals.n_faces * 3];

    cudaMemcpy(host_projected, host_cuda_globals.projected_vertices, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_updated, host_cuda_globals.updated_vertices, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vertices, host_cuda_globals.vertices_at_rest, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_edges, host_cuda_globals.edges, sizeof(int) * host_cuda_globals.n_edges * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_faces, host_cuda_globals.faces, sizeof(int) * host_cuda_globals.n_faces * 3, cudaMemcpyDeviceToHost);
    
    int start = 0;
    for (int i = 0; i < host_cubes.size(); i++) {
        host_cubes[i].projected = host_projected + start;
        host_cubes[i].updated = host_updated + start;
        host_cubes[i].vertices = host_vertices + start;
        start += host_cubes[i].n_vertices;
        host_cubes[i].edges = host_cubes[i].edges - host_cuda_globals.edges + host_edges;
        host_cubes[i].faces = host_cubes[i].faces - host_cuda_globals.faces + host_faces;
    }

    projected = host_projected;
    updated = host_updated;
    vertices = host_vertices;
    edges = host_edges;
    faces = host_faces;

    return host_cubes;
}
#endif




void per_intersection_core(int n_overlaps, luf* culls, i2* overlaps, int vtn, float* toi)
{

    const bool kernel = true;

    static vector<i2> host_overlaps;
    host_overlaps.resize(n_overlaps);
#ifdef CPU_REF
    static vector<luf> host_culls;

    host_culls.resize(n_overlaps);

    static vector<cudaAffineBody> host_cubes_stashed;
    host_cubes_stashed = host_cuda_globals.host_cubes;
    auto& host_cubes{ host_cuda_globals.host_cubes };
    vec3f* host_projected = new vec3f[host_cuda_globals.n_vertices],
     * host_updated = new vec3f[host_cuda_globals.n_vertices];
    int* host_edges = new int[host_cuda_globals.n_edges * 2];
    int* host_faces = new int[host_cuda_globals.n_faces * 3];

    cudaMemcpy(host_projected, host_cuda_globals.projected_vertices, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_updated, host_cuda_globals.updated_vertices, sizeof(vec3f) * host_cuda_globals.n_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_edges, host_cuda_globals.edges, sizeof(int) * host_cuda_globals.n_edges * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_faces, host_cuda_globals.faces, sizeof(int) * host_cuda_globals.n_faces * 3, cudaMemcpyDeviceToHost);
    
    int start = 0;
    for (int i = 0; i < host_cubes.size(); i++) {
        host_cubes[i].projected = host_projected + start;
        host_cubes[i].updated = host_updated + start;
        start += host_cubes[i].n_vertices;
        host_cubes[i].edges = host_cubes[i].edges - host_cuda_globals.edges + host_edges;
        host_cubes[i].faces = host_cubes[i].faces - host_cuda_globals.faces + host_faces;
    }

    {
        cudaMemcpy(host_culls.data(), culls, sizeof(luf) * n_overlaps, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_overlaps.data(), overlaps, sizeof(i2) * n_overlaps, cudaMemcpyDeviceToHost);
    }
#endif

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n_overlaps; i++) {
        auto tid = omp_get_thread_num();
        auto& stream{ host_cuda_globals.streams[tid] };
        cudaAffineBody* cubes{ host_cuda_globals.cubes };
        // preparing pointers

        int *vlist, *elist, *flist;
        int *vlist_meta, *elist_meta, *flist_meta;

        char *st_back_stashed{ host_cuda_globals.small_temporary_buffer_back[tid] }, *bulk_back_stashed{ host_cuda_globals.bulk_buffer_back[tid] };
        auto& st{ host_cuda_globals.small_temporary_buffer_back[tid] };
        auto& bulk{ host_cuda_globals.bulk_buffer_back[tid] };

        vector<int> host_prmts[MAX_TYPES][2];
        for (int type = 0; type < MAX_TYPES; type++) {
            // designate buffers

            int* ret_meta_sizes = (int*)st;
            st += 2 * sizeof(int);

            int* ret_prmts = (int*)bulk;
            bulk += sizeof(i2) * max_prmts_per_block;
            if (kernel) {
                primitive_intersection_test_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(
                    type, vtn, i, culls, overlaps, cubes, ret_meta_sizes, ret_prmts);
            }
#ifdef CPU_REF
            // else
            {
                // CPU_CODE
                auto cull{ host_culls[i] };
                auto idx{ host_overlaps[i] };
                int I{ idx[0] }, J{ idx[1] };
                int nvi, nv;
                switch (type) {
                case 0:
                    nvi = host_cubes[I].n_vertices;
                    nv = host_cubes[I].n_vertices + host_cubes[J].n_vertices;
                    break;
                case 2:
                    nvi = host_cubes[I].n_edges;
                    nv = host_cubes[I].n_edges + host_cubes[J].n_edges;
                    break;
                case 1:
                    nvi = host_cubes[I].n_faces;
                    nv = host_cubes[I].n_faces + host_cubes[J].n_faces;
                    break;
                }
                for (int vi = 0; vi < nv; vi++) {
                    auto b = vi < nvi ? I : J;
                    int id = vi < nvi ? vi : vi - nvi;
                    int iorj = vi < nvi ? 0 : 1;
                    auto t = aabb(host_cubes[b], id, type, vtn);
                    if (intersects(cull, t)) {
                        host_prmts[type][iorj].push_back(id);
                    }
                }
            }
#endif
            CUDA_CALL(cudaStreamSynchronize(stream));

            CUDA_CALL(cudaGetLastError());
            switch (type) {
            case 0:
                vlist = ret_prmts;
                vlist_meta = ret_meta_sizes;
                break;
            case 2:
                elist = ret_prmts;
                elist_meta = ret_meta_sizes;
                break;
                // edges are handled last
            case 1:
                flist = ret_prmts;
                flist_meta = ret_meta_sizes;
                break;
            }
        }   // end of primitive-intersection culling loop

        int sizes[6];
        CUDA_CALL(cudaStreamSynchronize(stream));
        CUDA_CALL(cudaMemcpyAsync(sizes, vlist_meta, sizeof(int) * 6, cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        i2 ij = host_overlaps[i];
        i2 ji = { ij[1], ij[0] };

#ifdef CPU_REF
        // CPU code
        bool all_sizes_match = true;
        for (int type = 0; type < MAX_TYPES; type++) {
            all_sizes_match &= (sizes[type * 2] == host_prmts[type][0].size() && sizes[type * 2 + 1] == host_prmts[type][1].size());

            // perhaps sorting not required

            // auto &a = host_prmts[type][0], &b = host_prmts[type][1];
            // sort(a.begin(), a.end());
            // sort(b.begin(), b.end());
        }
        thrust::device_vector<int> d_vi(vlist, vlist + sizes[0]);
        thrust::device_vector<int> d_vj(vlist + sizes[0], vlist + sizes[0] + sizes[1]);
        thrust::device_vector<int> d_fi(flist, flist + sizes[2]);
        thrust::device_vector<int> d_fj(flist + sizes[2], flist + sizes[2] + sizes[3]);
        auto _vi = from_thrust(d_vi), _vj = from_thrust(d_vj), _fi = from_thrust(d_fi), _fj = from_thrust(d_fj);
        auto vi = _vi, vj = _vj, fi = _fi, fj = _fj;

        sort(_vi.begin(), _vi.end());
        sort(_vj.begin(), _vj.end());
        sort(_fi.begin(), _fi.end());
        sort(_fj.begin(), _fj.end());
        vector<int> garbage;
        set_difference(_vi.begin(), _vi.end(), host_prmts[0][0].begin(), host_prmts[0][0].end(), back_inserter(garbage));
        set_difference(_vj.begin(), _vj.end(), host_prmts[0][1].begin(), host_prmts[0][1].end(), back_inserter(garbage));
        set_difference(_fi.begin(), _fi.end(), host_prmts[1][0].begin(), host_prmts[1][0].end(), back_inserter(garbage));
        set_difference(_fj.begin(), _fj.end(), host_prmts[1][1].begin(), host_prmts[1][1].end(), back_inserter(garbage));

        if (!all_sizes_match) {
            spdlog::error("sizes do not match\n");
            // TODO: check if prmitives match
        }
        if (garbage.size()) {
            spdlog::error("primitives do not match\n");
        }
        // set the host vector to the devices vector
        {
            host_prmts[0][0] = vi;
            host_prmts[0][1] = vj;
            host_prmts[1][0] = fi;
            host_prmts[1][1] = fj;
        }
        // declare of next stage
        vector<vec3f> host_vifjs;
        vector<luf> host_joint_aabbs;

#endif
        int nvifj, nvjfi, neiej;
        int *host_cnts[3], cnt[3];
        i2 *vifj_ptr, *vjfi_ptr, *eiej_ptr;
        float toi_by_type[3];
        for (int type = 0; type < MAX_TYPES; type++) {
            // vi fj, vj fi, ei ej (i < j)

            float3* vifjs = (float3*)bulk;
            bulk += sizeof(float3) * 4 * max_prmts_per_block;
            luf* joint_aabbs = (luf*)bulk;

            i2* ret_ij = (i2*)bulk;
            bulk += sizeof(i2) * max_pairs_per_block;
            i2* buf_ij = (i2*)bulk;
            bulk += sizeof(i2) * max_pairs_per_block;
            int* pt_types = (int*)bulk;
            bulk += sizeof(int) * max_pairs_per_block;

            int* ret_cnt = (int*)st;
            st += sizeof(int);

            int nvi, nfj, f_offset = 0, v_offset = 0;
            switch (type) {
            case 0:
                nvi = sizes[0];
                nfj = sizes[3];
                f_offset = sizes[2];
                break;
            case 1:
                nvi = sizes[1];
                nfj = sizes[2];
                v_offset = sizes[0];
                break;
            case 2:
                nvi = sizes[4];
                nfj = sizes[5];
                f_offset = sizes[4];
                break;
            }
            
            prepare_aabb_vi_fj_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(
                i, overlaps, cubes, type, vtn,
                type == 2 ? elist_meta : vlist_meta,
                type == 2 ? elist_meta : flist_meta,
                // nvi, nfj,
                vlist, flist, vifjs, joint_aabbs);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaStreamSynchronize(stream));

#ifdef CPU_REF
            {
                prepare_vifj(ij, host_cuda_globals.host_cubes, type,
                    host_prmts[0][0], host_prmts[0][1],
                    host_prmts[1][0], host_prmts[1][1],
                    host_vifjs, host_joint_aabbs);
                thrust::device_vector<vec3f> d_vifjs(vifjs, vifjs + nvi + nfj * 3);
                thrust::device_vector<luf> d_joint_aabbs(joint_aabbs, joint_aabbs + nvi + nfj);
                auto _vifjs = from_thrust(d_vifjs);
                auto _joint_aabbs = from_thrust(d_joint_aabbs);
                if (_vifjs.size() != host_vifjs.size()) {
                    spdlog::error("vifjs size mismatch\n");
                }
                if (_joint_aabbs.size() != host_joint_aabbs.size()) {
                    spdlog::error("joint_aabbs size mismatch\n");
                }
                if (_vifjs.size() == host_vifjs.size() && _joint_aabbs.size() == host_joint_aabbs.size()) {
                    for (int i = 0; i < host_vifjs.size(); i++) {
                        if (norm(host_vifjs[i] - _vifjs[i]) > 1e-4f) {
                            spdlog::error("vifjs mismatch\n");
                        }
                    }
                    for (int i = 0; i < host_joint_aabbs.size(); i++) {
                        auto norm2 = norm(host_joint_aabbs[i].l - _joint_aabbs[i].l) + norm(host_joint_aabbs[i].u - _joint_aabbs[i].u);
                        if (norm2 > 1e-4f) {
                            spdlog::error("joint_aabbs mismatch\n");
                        }
                    }
                }
            }
#endif

            aabb_intersection_test_kernel_atomic<<<1, n_cuda_threads_per_block, 0, stream>>>(joint_aabbs, nvi, nfj, ret_ij, ret_cnt);

            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaStreamSynchronize(stream));
#ifdef CPU_REF
            int host_ret_cnt;
            cudaMemcpy(&host_ret_cnt, ret_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            auto host_culled_pairs{ from_thrust(thrust::device_vector<i2>(ret_ij, ret_ij + host_ret_cnt)) };

#endif
            auto _flist = flist + f_offset;
            auto _vlist = vlist + v_offset;

            float* ret_toi = (float*)st;
            st += sizeof(float);

            if (vtn != 3)
                filter_distance_kernel_atomic<<<1, n_cuda_threads_per_block, 0, stream>>>(
                    buf_ij, ret_ij, ret_cnt, vifjs, nullptr,
                    _vlist, _flist,
                    pt_types,
                    1e-4f,
                    nvi);
            else
                toi_decision_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ret_ij, ret_cnt, vifjs, nvi, ret_toi);
            if (vtn == 3 && type == 2) {
                cudaMemcpyAsync(toi_by_type, ret_toi - 2, sizeof(float) * 3, cudaMemcpyDeviceToHost, stream);
            }
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaStreamSynchronize(stream));

#ifdef CPU_REF
            cudaMemcpy(&host_ret_cnt, ret_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            auto host_filtered_pairs{ from_thrust(thrust::device_vector<i2>(ret_ij, ret_ij + host_ret_cnt)) };

#endif
            host_cnts[type] = ret_cnt;
            switch (type) {
            case 0: vifj_ptr = ret_ij; break;
            case 1: vjfi_ptr = ret_ij; break;
            case 2: eiej_ptr = ret_ij; break;
            }
        }   // end of primtive pairwise loop

        if (vtn != 3) {
            CUDA_CALL(cudaStreamSynchronize(stream));
            CUDA_CALL(cudaMemcpyAsync(cnt, host_cnts[0], sizeof(int) * 3, cudaMemcpyDeviceToHost, stream));
            CUDA_CALL(cudaStreamSynchronize(stream));

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
            cudaMemcpyAsync(host_cuda_globals.pt.p + pt_put, vifj_ptr, nvifj * sizeof(i2), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(host_cuda_globals.pt.p + pt_put + nvifj, vjfi_ptr, nvjfi * sizeof(i2), cudaMemcpyDeviceToDevice, stream);
#ifndef PT_ONLY
            cudaMemcpyAsync(host_cuda_globals.ee.p + ee_put, eiej_ptr, neiej * sizeof(i2), cudaMemcpyDeviceToDevice, stream);
#endif
            strided_memset_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(host_cuda_globals.pt.b + pt_put, ij, nvifj);
            strided_memset_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(host_cuda_globals.pt.b + pt_put + nvifj, ji, nvjfi);
#ifndef PT_ONLY
            strided_memset_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(host_cuda_globals.ee.b + ee_put, ij, neiej);
#endif
            host_cuda_globals.bulk_buffer_back[tid] = bulk_back_stashed;
            host_cuda_globals.small_temporary_buffer_back[tid] = st_back_stashed;
        }


        else {
            // vtn == 3, ccd
            *toi = std::min({ toi_by_type[0], toi_by_type[1], toi_by_type[2] });
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());
#ifdef CPU_REF
    {
        delete[] host_edges;
        delete[] host_faces;
        delete[] host_projected;
        delete[] host_updated;
        host_cubes = host_cubes_stashed;
    }
#endif
}

// called by "cuda_intersection" option
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

    culling_kernel_atomic <<<1, n_cuda_threads_per_block>>>(n_cubes, cubes, PTR(aabbs), vtn, dev_n_overlaps, overlaps, culls);
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
