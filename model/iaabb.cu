
// #include "iaabb.h"
#include "cuda_header.cuh"
#include "timer.h"
#include <cuda/std/array>
#include <omp.h>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

using namespace std;
using namespace ipc;
// using namespace cuda::std;
// using namespace Eigen;
using luf = cuda::std::array<float, 6>;
using vec3f = cuda::std::array<float, 3>;
using Facef = cuda::std::array<vec3f, 3>;

static const int max_pairs_per_thread = 512, max_aabb_list_size = 256;
__host__ __device__ vec3f operator+(const vec3f& a, const vec3f& b)
{
    return { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}
__host__ __device__ vec3f operator-(const vec3f& a, const vec3f& b)
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}
__host__ __device__ vec3f operator/(const vec3f& a, float k)
{
    return { a[0] / k, a[1] / k, a[2] / k };
}
__host__ __device__ vec3f operator*(const vec3f& a, float k)
{
    return { a[0] * k, a[1] * k, a[2] * k };
}

__host__ __device__ float dot(const vec3f& a, const vec3f& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ float norm(const vec3f& a)
{
    return CUDA_SQRT(dot(a, a));
}
__host__ __device__ vec3f normalize(const vec3f& a)
{
    return a / norm(a);
}

__host__ __device__ bool intersects(const luf& a, const luf& b)
{
    return a[0] <= b[3] && a[3] >= b[0] && a[1] <= b[4] && a[4] >= b[1] && a[2] <= b[5] && a[5] >= b[2];
}

__host__ __device__ vec3f cross(const vec3f& a, const vec3f& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}
__host__ __device__ vec3f unit_normal(const Facef& f)
{
    return normalize(cross(f[1] - f[0], f[2] - f[0]));
}
__host__ __device__ float area(const vec3f& a, const vec3f& b, const vec3f& c)
{
    return norm(cross(c - a, b - a)) / 2.0f;
}

__host__ __device__ float ab(const vec3f& a, const vec3f& b)
{
    return norm(b - a);
}

__host__ __device__ float h(const vec3f& e0, const vec3f& e1, const vec3f& p)
{
    return area(e0, e1, p) / ab(e0, e1);
}
__host__ __device__ float ab_sqr(const vec3f& a, const vec3f& b)
{
    return dot(b - a, b - a);
}

__host__ __device__ bool is_obtuse_triangle(const vec3f& e0, const vec3f& e1, const vec3f& p)
{
    auto ab = ab_sqr(e0, e1);
    auto bc = ab_sqr(e1, p);
    auto ca = ab_sqr(e0, p);
    return CUDA_ABS(ca - bc) > ab;
}

__device__ float vf_distance(const vec3f& _v, const Facef& f, PointTriangleDistanceType& pt_type)
{
    auto n = unit_normal(f);
    auto d = dot(n, _v - f[0]);
    auto a1 = area(f[1], f[0], f[2]);
    auto v = _v - n * d;
    d = d * d;
    // float a2 = ((f[0] - v).cross(f[1] - v).norm() + (f[1] - v).cross(f[2] - v).norm() + (f[2] - v).cross(f[0] - v).norm());
    auto a2 = area(f[0], f[1], v) + area(f[1], f[2], v) + area(f[2], f[0], v);
    if (a2 > a1 + 1e-8) {
        // projection outside of triangle

        auto d_ab = h(f[0], f[1], v);
        auto d_bc = h(f[1], f[2], v);
        auto d_ac = h(f[0], f[2], v);

        auto d_a = ab(v, f[0]);
        auto d_b = ab(v, f[1]);
        auto d_c = ab(v, f[2]);

        auto dab = is_obtuse_triangle(f[0], f[1], v) ? CUDA_MIN(d_a, d_b) : d_ab;
        auto dbc = is_obtuse_triangle(f[2], f[1], v) ? CUDA_MIN(d_c, d_b) : d_bc;
        auto dac = is_obtuse_triangle(f[0], f[2], v) ? CUDA_MIN(d_a, d_c) : d_ac;

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

__global__ void aabb_intersection_kernel(luf* dev_aabbs, int nvi, int nfj, int* ij, int* cnt)
{

    __shared__ luf aabbs[max_aabb_list_size];
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_task_per_thread = (nvi * nfj + blockDim.x - 1) / blockDim.x;
    int n_copies_per_thread = (nvi + nfj + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < n_copies_per_thread; i++) {
        int idx = tid * n_copies_per_thread + i;
        if (idx < nvi + nfj)
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
                auto put = cnt[tid]++ + tid * max_pairs_per_thread;
                ij[put * 2] = i;
                ij[put * 2 + 1] = j;
            }
        }
    }
}

__global__ void squeeze_kernel(int* ij, int* cnt, int* tmp, int* vilist, int* fjlist, vec3f* vis, Facef* fjs, float dhat = 1e-4)
{
    // squeeze the ij list according to a prefix sum array cnt
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid == 0 ? 0 : cnt[tid - 1];
    int n_tasks = cnt[blockDim.x - 1];
    int copy_size = cnt[tid] - start;
    for (int i = 0; i < copy_size; i++) {
        int dst = i + start;
        int src = tid * max_pairs_per_thread + i;

        tmp[dst * 2] = ij[src * 2];
        tmp[dst * 2 + 1] = ij[src * 2 + 1];
    }

    __syncthreads();

    // do the vf distance and type computation
    cnt[tid] = 0;
    int n_task_per_thread = (n_tasks + blockDim.x - 1) / blockDim.x;
    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int idx = tid * n_task_per_thread + _i;
        if (idx < n_tasks) {
            int i = tmp[idx * 2];
            int j = tmp[idx * 2 + 1];
            int vi = vilist[i];
            int fj = fjlist[j];
            // compute the distance and type
            auto& v{ vis[i] };
            auto& f{ fjs[j] };
            PointTriangleDistanceType pt_type;
            auto d = vf_distance(v, f, pt_type);
            if (d < dhat) {
                auto put = cnt[tid]++ + tid * max_pairs_per_thread;
                ij[put * 2] = vi;
                ij[put * 2 + 1] = fj;
            }
        }
    }
}
__global__ void squeeze_ij_kernel(int* ij, int* cnt, int* tmp)
{
    // squeeze again and copy back to ij matrix
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto start = tid == 0 ? 0 : cnt[tid - 1];
    auto copy_size = cnt[tid] - start;
    for (int i = 0; i < copy_size; i++) {
        int dst = i + start;
        int src = tid * max_pairs_per_thread + i;

        tmp[dst * 2] = ij[src * 2];
        tmp[dst * 2 + 1] = ij[src * 2 + 1];
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
    int I, int J)
{
    // auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    // int nvi = vilist.size(), nfj = fjlist.size();

    // thrust::host_vector<luf> aabbs(nvi + nfj);
    // thrust::host_vector<vec3f> vis(nvi);
    // thrust::host_vector<Facef> fjs(nfj);

    // for (int k = 0; k < nvi; k++) {
    //     auto vi{ vilist[k] };
    //     vec3 v{ ci.v_transformed[vi] };
    //     vis[k] = to_vec3f(v);
    //     lu lud{ v.array() - barrier::d_sqrt, v.array() + barrier::d_sqrt };
    //     aabbs[k] = to_luf(lud);
    // }
    // for (int k = 0; k < nfj; k++) {
    //     auto fj{ fjlist[k] };
    //     Face f{ cj, unsigned(fj), true, true };
    //     fjs[k] = to_facef(f);
    //     lu lud{ compute_aabb(f) };
    //     aabbs[k + nvi] = to_luf(lud);
    // }
    // auto tid = omp_get_thread_num();

    thrust::device_vector<luf> dev_aabbs(aabbs.begin(), aabbs.end());
    thrust::host_vector<vec3f> dev_vis(vis.begin(), vis.end());
    thrust::host_vector<Facef> dev_fjs(fjs.begin(), fjs.end());
    thrust::device_vector<int>
        dev_vilist(vilist.begin(), vilist.end()),
        dev_fjlist(fjlist.begin(), fjlist.end());

    // allocate memory on device
    thrust::device_vector<int> dev_cnt(n_cuda_threads_per_block, 0),
        ij(n_cuda_threads_per_block * 2 * max_pairs_per_thread, 0),
        tmp(n_cuda_threads_per_block * 2 * max_pairs_per_thread, 0);

    auto ij_ptr = thrust::raw_pointer_cast(ij.data());
    auto cnt_ptr = thrust::raw_pointer_cast(dev_cnt.data());
    auto tmp_ptr = thrust::raw_pointer_cast(tmp.data());
    auto vilist_ptr = thrust::raw_pointer_cast(dev_vilist.data());
    auto fjlist_ptr = thrust::raw_pointer_cast(dev_fjlist.data());
    auto vis_ptr = thrust::raw_pointer_cast(dev_vis.data());
    auto fjs_ptr = thrust::raw_pointer_cast(dev_fjs.data());
    auto aabbs_ptr = thrust::raw_pointer_cast(dev_aabbs.data());

    // {
    //     // copying
    //     dev_aabbs = aabbs;
    //     dev_vis = vis;
    //     dev_fjs = fjs;
    // }

    {
        // cuda kernels
        aabb_intersection_kernel<<<1, n_cuda_threads_per_block, max_aabb_list_size * sizeof(luf)>>>(aabbs_ptr, nvi, nfj, ij_ptr, cnt_ptr);
        CUDA_CALL(cudaGetLastError());
        thrust::inclusive_scan(dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());
        CUDA_CALL(cudaGetLastError());
        squeeze_kernel<<<1, n_cuda_threads_per_block>>>(ij_ptr, cnt_ptr, tmp_ptr, vilist_ptr, fjlist_ptr, vis_ptr, fjs_ptr);
        CUDA_CALL(cudaGetLastError());

        // thrust::inclusive_scan(dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());
        CUDA_CALL(cudaDeviceSynchronize());

        thrust::host_vector<int> host_cnt(n_cuda_threads_per_block);
        host_cnt = dev_cnt;
        for (int i = 1; i < n_cuda_threads_per_block; i++) {
            host_cnt[i] = host_cnt[i - 1] + host_cnt[i];
        }
        dev_cnt = host_cnt;
        squeeze_ij_kernel<<<1, n_cuda_threads_per_block>>>(ij_ptr, cnt_ptr, tmp_ptr);
        CUDA_CALL(cudaGetLastError());
    }

    cudaDeviceSynchronize();

    {
        // generate collision set in the same syntax as the CPU version
        int n_collision_set = dev_cnt.back();
        thrust::host_vector<int> host_idx(n_collision_set * 2);
        thrust::copy_n(tmp.begin(), n_collision_set * 2, host_idx.begin());
        for (int i = 0; i < n_collision_set; i++) {
            auto vi = host_idx[i * 2], fj = host_idx[i * 2 + 1];
            idx.push_back({ I, vi, J, fj });
        }
    }
}
