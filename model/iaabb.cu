
// #include "iaabb.h"
#include "cuda_header.cuh"
#include "timer.h"
#include <omp.h>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#include "cuda_globals.cuh"
#include <spdlog/spdlog.h>
#include <tuple>
using namespace std;
using namespace ipc;
// using namespace cuda::std;
// using namespace Eigen;
static const int max_pairs_per_thread = 512, max_aabb_list_size = 512;
// FIXME: tid probably not in 1 block

tuple<float, PointTriangleDistanceType> vf_distance(vec3f vf, Facef ff);

__device__ __host__ float vf_distance(vec3f _v, Facef f, PointTriangleDistanceType& pt_type)
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
    return d;
}

__global__ void aabb_intersection_test_kernel(luf* dev_aabbs, int nvi, int nfj, i2* ij, int* cnt)
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
    cnt[tid] = 0;
    __syncthreads();
    // copys the bounding boxes to shared memory

    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int I = tid * n_task_per_thread + _i;
        if (I < nvi * nfj) {
            int i = I / nfj;
            int j = I % nfj;
            if (intersects(aabbs[i], aabbs[nvi + j])) {
                auto put = cnt[tid]++ + tid * max_pairs_per_thread;
                ij[put] = { i, j };
            }
        }
    }
}

__global__ void inclusive_scan_kernel(int * cnt) {
    for (int i = 1; i < n_cuda_threads_per_block; i ++) {
        cnt[i] = cnt[i - 1] + cnt[i];
    }
}
__global__ void filter_distance_kernel(i2* ij, int* cnt, i2* tmp,
    // int* vilist, int* fjlist,
    vec3f* vis, Facef* fjs,
    PointTriangleDistanceType* pt_types,
    PointTriangleDistanceType* tmp_pt_types,
    float dhat = 1e-4)
{
    // // squeeze the ij list according to a prefix sum array cnt
    // // FIXME: asserting blockDim.x == n_cuda_threads_per_block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int start = tid == 0 ? 0 : cnt[tid - 1];
    int n_tasks = cnt[blockDim.x - 1];
    // do the vf distance and type computation
    cnt[tid] = 0;
    int n_task_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int idx = tid * n_task_per_thread + _i;
        if (idx < n_tasks) {
            auto _ij = tmp[idx];
            int i = _ij[0];
            int j = _ij[1];
            // int vi = vilist[i];
            // int fj = fjlist[j];

            // compute the distance and type
            auto v{ vis[i] };
            auto f{ fjs[j] };
            auto put = cnt[tid] + tid * max_pairs_per_thread;
            auto d = vf_distance(v, f, pt_types[put]);
            if (d < dhat) {
                ij[put] = { i, j };
                cnt[tid]++;
            }
        }
    }
}
__global__ void squeeze_ij_kernel(i2* ij, int* cnt, i2* tmp, PointTriangleDistanceType* pt_types, PointTriangleDistanceType* tmp_pt_types)
{
    // squeeze again and copy back to ij matrix
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto start = tid == 0 ? 0 : cnt[tid - 1];
    auto copy_size = cnt[tid] - start;
    for (int i = 0; i < copy_size; i++) {
        int dst = i + start;
        int src = tid * max_pairs_per_thread + i;

        tmp[dst] = ij[src];
        tmp_pt_types[dst] = pt_types[src];
        // tmp is now dense
    }
}

void vf_col_set_cuda(
    // vector<int>& vilist, vector<int>& fjlist,
    // const std::vector<std::unique_ptr<AffineBody>>& cubes,
    // int I, int J,
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

    thrust::device_vector<PointTriangleDistanceType> pt_types(n_cuda_threads_per_block * max_pairs_per_thread, PointTriangleDistanceType::P_T), pt_types_buffer(n_cuda_threads_per_block * max_pairs_per_thread, PointTriangleDistanceType::P_T);

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
    auto chunk_int = (int*)((char*)cuda_globals.buffer_chunk + tid * cuda_globals.per_stream_buffer_size);
    auto ij_size = n_cuda_threads_per_block * max_pairs_per_thread * 2;
    i2* ij_ptr = (i2*)(chunk_int);
    i2* tmp_ptr = (i2*)(chunk_int + ij_size);
    PointTriangleDistanceType* pt_types_ptr = (PointTriangleDistanceType*)(chunk_int + ij_size * 2);
    PointTriangleDistanceType* tmp_pt_types_ptr = (PointTriangleDistanceType*)(pt_types_ptr + ij_size / 2);
    int* cnt_ptr = (int*)(chunk_int + ij_size * 3);
    int *tmp_cnt = cnt_ptr + n_cuda_threads_per_block * 2;
    vec3f* vis_ptr = (vec3f*)(cnt_ptr + n_cuda_threads_per_block * 3);
    Facef* fjs_ptr = (Facef*)(vis_ptr + max_aabb_list_size);
    luf* aabbs_ptr = (luf*)(fjs_ptr + max_aabb_list_size);
    // vec3f* vis_ptr;
    // Facef* fjs_ptr;
    // luf* aabbs_ptr;
    // cudaMallocManaged(&vis_ptr, vis.size() * sizeof(vec3f));
    // cudaMallocManaged(&fjs_ptr, fjs.size() * sizeof(Facef));
    // cudaMallocManaged(&aabbs_ptr, aabbs.size() * sizeof(luf));
    // CUDA_CALL(cudaGetLastError());
    // CUDA_CALL(cudaMemset(cnt_ptr, 0, n_cuda_threads_per_block));

    
    
    auto &stream {cuda_globals.streams[tid]};
    CUDA_CALL(cudaMemcpyAsync((void*)vis_ptr, vis.data(), vis.size() * sizeof(vec3f), cudaMemcpyHostToDevice), stream);
    CUDA_CALL(cudaMemcpyAsync((void*)fjs_ptr, fjs.data(), fjs.size() * sizeof(Facef), cudaMemcpyHostToDevice), stream);
    
    CUDA_CALL(cudaMemcpyAsync((void*)aabbs_ptr, aabbs.data(), aabbs.size() * sizeof(luf), cudaMemcpyHostToDevice), stream);
    
    // CUDA_CALL(cudaMemcpy((void*)vis_ptr, vis.data(), vis.size() * sizeof(vec3f), cudaMemcpyHostToDevice));
    // CUDA_CALL(cudaMemcpy((void*)fjs_ptr, fjs.data(), fjs.size() * sizeof(Facef), cudaMemcpyHostToDevice));
    
    // CUDA_CALL(cudaMemcpy((void*)aabbs_ptr, aabbs.data(), aabbs.size() * sizeof(luf), cudaMemcpyHostToDevice));
    
    // spdlog::warn("copy complete");
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
                // pass
                filter_distance_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, cnt_ptr, tmp_ptr,
                    // vilist_ptr, fjlist_ptr,
                    vis_ptr, fjs_ptr, pt_types_ptr, tmp_pt_types_ptr);

                CUDA_CALL(cudaGetLastError());
                inclusive_scan_kernel<<<1, 1, 0, stream>>>(cnt_ptr);
                // thrust::inclusive_scan(thrust::cuda::par_nosync, cnt_ptr, cnt_ptr + n_cuda_threads_per_block, cnt_ptr);

                // thrust::inclusive_scan(thrust::cuda::par_nosync, dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());

                squeeze_ij_kernel<<<1, n_cuda_threads_per_block, 0, stream>>>(ij_ptr, cnt_ptr, tmp_ptr, pt_types_ptr, tmp_pt_types_ptr);
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
        int n_collision_set = cnt_ptr[n_cuda_threads_per_block - 1];
        thrust::host_vector<i2> host_ij(tmp_ptr, tmp_ptr + n_collision_set);
        for (int i = 0; i < n_collision_set; i++) {
            auto vi = host_ij[i][0], fj = host_ij[i][1];
            
            idx.push_back({ I, vilist[vi], J, fjlist[fj] });
        }
    }
}

#include "cuda_globals.cuh"
#include <assert.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <thrust/set_operations.h>
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
        auto &cset{ cuda_globals.collision_sets.pt_set[i] }, &bset{ cuda_globals.collision_sets.pt_set_body_index[i] };
        // cset.clear();
        // bset.clear();
        thrust::copy_if(
            thrust::make_zip_iterator(thrust::make_tuple(pt_idx.begin(), pt_body_idx.begin(), pt_types.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(pt_idx.end(), pt_body_idx.end(), pt_types.end())),
            thrust::make_zip_iterator(thrust::make_tuple(cset.end(), bset.end(), thrust::make_discard_iterator())),
            [=] __device__(const thrust::tuple<i2, i2, PointTriangleDistanceType>& tup) {
                return static_cast<cuda::std::underlying_type_t<PointTriangleDistanceType>>(thrust::get<2>(tup)) == i;
            });
    }
    // FIXME: make sure i < j before merging into lut? 
    {
    // static thrust::device_vector<i2> merged_lut;
        // // generate look up table for spare hess 
        // merged_lut .resize(0);
        // thrust::set_union(thrust::device, pt_body_idx.begin(), pt_body_idx.end(), cuda_globals.lut.begin(), cuda_globals.lut.end(), merged_lut.begin());
        // cuda_globals.lut = merged_lut;

        // should be sorted, just not worth it

        thrust::copy(pt_body_idx.begin(), pt_body_idx.end(), cuda_globals.lut.end());
    }
}

CudaGlobals::CudaGlobals()
{
    cudaGetDevice(&device_id);
    int n_proc = omp_get_num_procs();
    per_stream_buffer_size = n_cuda_threads_per_block * max_pairs_per_thread * sizeof(i2) * 4;
    cudaMallocManaged(&buffer_chunk, per_stream_buffer_size * n_proc);
    streams = new cudaStream_t[n_proc];
    for (int i = 0; i < n_proc; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
}

CudaGlobals::~CudaGlobals()
{
    cudaFree(buffer_chunk);
    for (int i = 0 ; i < omp_get_num_procs(); i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}
