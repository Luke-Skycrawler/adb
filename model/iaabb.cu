
#include "iaabb.h"
#include "cuda_header.h"
#include "timer.h"
#include <cuda/std/array>
#include <omp.h>
#include <tbb/parallel_sort.h>
#include "collision.h"
#include "../view/global_variables.h"
using namespace std;
using namespace ipc;
// using namespace cuda::std;
// using namespace Eigen;
using luf = cuda::std::array<float, 6>;
using vec3f = cuda::std::array<float, 3>;
using Facef = cuda::std::array<vec3f, 3>;

static const int max_pairs_per_thread = 512, max_aabb_list_size = 256;
__host__ __device__ inline vec3f operator+(const vec3f& a, const vec3f& b)
{
    return { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}
__host__ __device__ inline vec3f operator-(const vec3f& a, const vec3f& b)
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}
__host__ __device__ inline vec3f operator/(const vec3f& a, float k)
{
    return { a[0] / k, a[1] / k, a[2] / k };
}
__host__ __device__ inline vec3f operator*(const vec3f& a, float k)
{
    return { a[0] * k, a[1] * k, a[2] * k };
}

__host__ __device__ inline float dot(const vec3f& a, const vec3f& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ inline float norm(const vec3f& a)
{
    return CUDA_SQRT(dot(a, a));
}
__host__ __device__ inline vec3f normalize(const vec3f& a)
{
    return a / norm(a);
}

__host__ __device__ inline bool intersects(const luf& a, const luf& b)
{
    return a[0] <= b[3] && a[3] >= b[0] && a[1] <= b[4] && a[4] >= b[1] && a[2] <= b[5] && a[5] >= b[2];
}

__host__ __device__ inline vec3f cross(const vec3f& a, const vec3f& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}
__host__ __device__ inline vec3f unit_normal(const Facef& f)
{
    return normalize(cross(f[1] - f[0], f[2] - f[0]));
}
__host__ __device__ inline float area(const vec3f& a, const vec3f& b, const vec3f& c)
{
    return norm(cross(c - a, b - a)) / 2.0f;
}

__host__ __device__ inline float ab(const vec3f& a, const vec3f& b)
{
    return norm(b - a);
}

__host__ __device__ inline float h(const vec3f& e0, const vec3f& e1, const vec3f& p)
{
    return area(e0, e1, p) / ab(e0, e1);
}
__host__ __device__ inline float ab_sqr(const vec3f& a, const vec3f& b)
{
    return dot(b - a, b - a);
}

__host__ __device__ inline bool is_obtuse_triangle(const vec3f& e0, const vec3f& e1, const vec3f& p)
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

inline luf to_luf(const lu& a)
{
    return {
        float(a[0][0]),
        float(a[0][1]),
        float(a[0][2]),
        float(a[1][0]),
        float(a[1][1]),
        float(a[1][2])
    };
}
inline vec3f to_vec3f(const vec3& a)
{
    return {
        float(a[0]),
        float(a[1]),
        float(a[2])
    };
}
inline Facef to_facef(const Face& f)
{
    return {
        to_vec3f(f.t0),
        to_vec3f(f.t1),
        to_vec3f(f.t2)
    };
}
__global__ void aabb_intersection_kernel(luf* dev_aabbs, int nvi, int nfj, int* dev_ij, int* dev_cnt)
{

    __shared__ luf aabbs[max_aabb_list_size];
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_task_per_thread = (nvi * nfj + blockDim.x - 1) / blockDim.x;
    int n_copies_per_thread = (nvi + nfj + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < n_copies_per_thread; i++) {
        int idx = tid * n_copies_per_thread + i;
        if (idx < nvi + nfj)
            aabbs[i] = dev_aabbs[idx];
    }
    __syncthreads();
    // copys the bounding boxes to shared memory

    for (int _i = 0; _i < n_task_per_thread; _i++) {
        int I = tid * n_task_per_thread + _i;
        if (I < nvi * nfj) {
            int i = I / nfj;
            int j = I % nfj;
            if (intersects(aabbs[i], aabbs[nvi + j])) {
                auto put = dev_cnt[tid]++ * 2;
                dev_ij[put] = i;
                dev_ij[put + 1] = j;
            }
        }
    }
}

__global__ void squeeze_kernel(int* ij, int* cnt, int* tmp, int* vilist, int* fjlist, vec3f* vis, Facef* fjs, float dhat = 1e-4)
{
    // squeeze the ij list according to a prefix sum array cnt
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid == 0 ?0: cnt[tid - 1];     
    int copy_size = cnt[tid] - start;
    for (int i = 0; i < copy_size; i ++ ) {
        int dst = i + start;
        int src = tid + max_pairs_per_thread + i;

        tmp[dst * 2] = ij[src * 2];
        tmp[dst * 2 + 1] = ij[src * 2 + 1];
    }

    // do the vf distance and type computation
    int n_tasks = cnt[blockDim.x];
    __syncthreads();
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
                auto t = cnt[tid]++;
                tmp[t * 2] = vi;
                tmp[t * 2 + 1] = fj;
                // reuse the tmp array
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
        int src = tid + max_pairs_per_thread + i;

        ij[dst * 2] = tmp[src * 2];
        ij[dst * 2 + 1] = tmp[src * 2 + 1];
    }
}
double primitive_brute_force_thrust(
    int n_cubes,
    std::vector<Intersection>& overlaps, // assert sorted
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int vtn,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx)
{

    double toi_global = 1.0, toi_ee_pt = 1.0;
    bool cull_trajectory = vtn == 3;
    int n_overlap = overlaps.size();
    if (!cull_trajectory) {
        pts.resize(0);
        idx.resize(0);
        ees.resize(0);
        eidx.resize(0);
        vidx.resize(0);
    }

    static PList* lists = new PList[n_overlap];
    static int allocated = n_overlap;
    if (n_overlap > allocated) {
        delete []lists;
        lists = new PList[n_overlap];
        allocated = n_overlap;
    }
    for (int i = 0; i < n_overlap; i++) {
        lists[i].vi.resize(0);
        lists[i].vj.resize(0);
        lists[i].ei.resize(0);
        lists[i].ej.resize(0);
        lists[i].fi.resize(0);
        lists[i].fj.resize(0);
    }
    vector<int> starting;
    starting.resize(n_cubes + 1);
    for (int i = 0; i <= n_cubes; i++)
        starting[i] = n_overlap;
    starting[0] = 0;

    int old_cube_index = 0;
    for (int i = 0; i < n_overlap; i++) {

        auto& o{ overlaps[i] };
        if (o.i != old_cube_index) {
            for (int j = old_cube_index + 1; j <= o.i; j++)
                starting[j] = i;
            old_cube_index = o.i;
        }
    }
#pragma omp parallel for schedule(static)
    for (int I = 0; I < n_cubes; I++) {
        // construct the vertex, edge, and triangle list inside each overlap
        auto& c{ *cubes[I] };
        switch (vtn) {
        case 0: c.project_vt0(); break;
        case 1: c.project_vt1(); break;
        default: c.project_vt2();
        }
    }

    static vector<array<int, 2>>* vidx_thread_local = new vector<array<int, 2>>[omp_get_max_threads()];

    if (globals.ground)
    // #pragma omp parallel
    {
        double toi_thread_local = 1.0;
        auto tid = omp_get_thread_num();
        vidx_thread_local[tid].resize(0);
// #pragma omp for schedule(static)
        for (int I = 0; I < n_cubes; I++) {
            auto& c{ *cubes[I] };
            for (int v = 0; v < c.n_vertices; v++) {
                auto& p{ c.v_transformed[v] };
                // handling vertex-ground collision
                if (!cull_trajectory) {
                    double d = vg_distance(p);
                    d = d * d;
                    if (d < barrier::d_hat) {
                        vidx_thread_local[tid].push_back({ I, v });
                    }
                }
                else {
                    double t = collision_time(c, v);
                    toi_thread_local = min(toi_thread_local, t);
                }
            }
        }
        if (cull_trajectory) {
#pragma omp critical
            toi_global = min(toi_global, toi_thread_local);
        }
        else {
#pragma omp critical
            vidx.insert(vidx.end(), vidx_thread_local[tid].begin(), vidx_thread_local[tid].end());
        }
    }

// debugging code
    if (cull_trajectory) {
        if (toi_global < 1e-6) {
            spdlog::error("vertex ground toi_global = {}", toi_global);
            if (globals.params_int.find("g_cnt") != globals.params_int.end())
                globals.params_int["g_cnt"]++;
            else
                globals.params_int["g_cnt"] = 0;
            if (globals.params_int["g_cnt"] > 1) exit(1);
        }
        else
            globals.params_int["g_cnt"] = 0;
    }
    int n_points = globals.points.size(), n_triangles = globals.triangles.size(), n_edges = globals.edges.size();

    static omp_lock_t* locks = nullptr;
    static int allocated_locks = 0;
    static bool init = true;
    if (n_overlap > allocated_locks) {
        if (!init) {
            for (int i = 0; i < allocated_locks; i++)
                omp_destroy_lock(locks + i);
            delete[] locks;
        }
        locks = new omp_lock_t[n_overlap];
        allocated_locks = n_overlap;
        for (int i = 0; i < n_overlap; i++)
            omp_init_lock(locks + i);
    }

    for (int i = 0; i < n_overlap; i ++) overlaps[i].plist = lists + i;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        auto idx{ globals.points[i] };
        auto I{ idx[0] };
        auto v{ idx[1] };
        auto& c{ *cubes[I] };
        vec3 p{ c.v_transformed[v] };
        vec3 p0{ c.vt1(v) };

        lu aabb = cull_trajectory ? compute_aabb(p, p0) : lu{p, p};
        for (int o = starting[I]; o < starting[I + 1]; o++) {
            lu cull = overlaps[o].cull;
            if (intersects(cull, aabb)) {
                auto& t{ overlaps[o] };
                assert(t.i == I);
                omp_set_lock(locks + o);
                if (t.i < t.j)
                    lists[o].vi.push_back(v);
                else
                    lists[o].vj.push_back(v);
                omp_unset_lock(locks + o);
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_edges; i++) {
        auto idx{ globals.edges[i] };
        auto I{ idx[0] };
        auto ei{ idx[1] };
        auto& c{ *cubes[I] };
        Edge e{ c, ei, true, true };
        Edge e0{ c, ei };

        lu aabb = cull_trajectory ? compute_aabb(e, e0) : compute_aabb(e);
        for (int o = starting[I]; o < starting[I + 1]; o++) {
            lu cull = overlaps[o].cull;
            if (intersects(cull, aabb)) {
                auto& t{ overlaps[o] };
                omp_set_lock(locks + o);
                if (t.i < t.j)
                    lists[o].ei.push_back(ei);
                else
                    lists[o].ej.push_back(ei);
                omp_unset_lock(locks + o);
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_triangles; i++) {
        auto idx{ globals.triangles[i] };
        auto I{ idx[0] };
        auto fi{ idx[1] };
        auto& c{ *cubes[I] };
        Face f{ c, fi, true, true };
        Face f0 {c, fi};

        lu aabb= cull_trajectory ? compute_aabb(f, f0): compute_aabb(f);
        for (int o = starting[I]; o < starting[I + 1]; o++) {
            lu cull = overlaps[o].cull;
            if (intersects(aabb, cull)){
                auto& t{ overlaps[o] };
                omp_set_lock(locks + o);
                if (t.i < t.j)
                    lists[o].fi.push_back(fi);
                else
                    lists[o].fj.push_back(fi);
                omp_unset_lock(locks + o);
            }
        }
    }
    tbb::parallel_sort(overlaps.begin(), overlaps.end(), [](const Intersection& a, const Intersection& b) -> bool{
        auto ad = a.i + a.j, am = abs(a.i - a.j);
        auto bd = b.i + b.j, bm = abs(b.i - b.j);

        return ad < bd || (ad == bd && am < bm) || (ad == bd && am == bm && a.i < b.i);
    });
    // spdlog::info("ground toi  = {}", toi_global);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_overlap / 2; i++) {
        int i0 = overlaps[i * 2].i, j0 = overlaps[i * 2].j;
        int i1 = overlaps[i * 2 + 1].i, j1 = overlaps[i * 2 + 1].j;
        assert(i0 == j1 && j0 == i1);
        
        auto &p0{ overlaps[i * 2].plist }, &p1{ overlaps[i * 2 + 1].plist };
        auto
            &vi0{ p0->vi },
            &vi1{ p1->vi },
            &ei0{ p0->ei }, &ei1{ p1->ei },
            &fi0{ p0->fi }, &fi1{ p1->fi },

            &vj0{ p0->vj }, &vj1{ p1->vj },
            &ej0{ p0->ej }, &ej1{ p1->ej },
            &fj0{ p0->fj }, &fj1{ p1->fj };

        assert((vi0.size() == 0 && vj1.size() == 0) || (vi1.size() == 0 && vj0.size() == 0));

        vi0.reserve(vi0.size() + vi1.size());
        ei0.reserve(ei0.size() + ei1.size());
        fi0.reserve(fi0.size() + fi1.size());
        
        vi0.insert(vi0.end(), vi1.begin(), vi1.end());
        ei0.insert(ei0.end(), ei1.begin(), ei1.end());
        fi0.insert(fi0.end(), fi1.begin(), fi1.end());

        vj0.insert(vj0.end(), vj1.begin(), vj1.end());
        ej0.insert(ej0.end(), ej1.begin(), ej1.end());
        fj0.insert(fj0.end(), fj1.begin(), fj1.end());
    }

    static vector<vec3> vt1_buffer;
    static vector<int> vertex_starting_index;

    if (vertex_starting_index.size() == 0) {
        // initialization
        vertex_starting_index.resize(n_cubes);
        vt1_buffer.resize(globals.points.size());
        vertex_starting_index[0] = 0;

        for (int i = 0; i < n_cubes - 1; i++) {
            auto& c{ *cubes[i] };
            vertex_starting_index[i + 1] = vertex_starting_index[i] + c.n_vertices;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n_cubes; i++) {
        auto& c{ *cubes[i] };
        auto offset = vertex_starting_index[i];
        mat3 a;
        vec3 b = c.q[0];
        a << c.q[1], c.q[2], c.q[3];
        for (int j = 0; j < c.n_vertices; j++) {
            vt1_buffer[j + offset] = a * c.vertices(j) + b;
        }
    }

    int n_proc = omp_get_max_threads();
    vector<cudaStream_t> cuda_streams;
    cuda_streams.resize(n_proc);
    for (int i = 0; i < n_proc; i++) {
        cudaStreamCreate(&cuda_streams[i]);
    }

    const auto vf_col_set_cuda = [&](vector<int>& vilist, vector<int>& fjlist,
                                     const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                     int I, int J,
                                     vector<array<vec3, 4>>& pts,
                                     vector<array<int, 4>>& idx) {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        int nvi = vilist.size(), nfj = fjlist.size();

        thrust::host_vector<luf> aabbs(nvi + nfj);
        thrust::host_vector<vec3f> vis(nvi);
        thrust::host_vector<Facef> fjs(nfj);

        for (int k = 0; k < nvi; k++) {
            auto vi{ vilist[k] };
            vec3 v{ ci.v_transformed[vi] };
            vis[k] = to_vec3f(v);
            lu lud{ v.array() - barrier::d_sqrt, v.array() + barrier::d_sqrt };
            aabbs[k] = to_luf(lud);
        }
        for (int k = 0; k < nfj; k++) {
            auto fj{ fjlist[k] };
            Face f{ cj, unsigned(fj), true, true };
            fjs[k] = to_facef(f);
            lu lud{ compute_aabb(f) };
            aabbs[k + nvi] = to_luf(lud);
        }
        auto tid = omp_get_thread_num();

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

            thrust::inclusive_scan(dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());

            squeeze_kernel<<<1, n_cuda_threads_per_block, 0, cuda_streams[tid]>>>(ij_ptr, cnt_ptr, tmp_ptr, vilist_ptr, fjlist_ptr, vis_ptr, fjs_ptr);

            thrust::inclusive_scan(dev_cnt.begin(), dev_cnt.end(), dev_cnt.begin());
            
            squeeze_ij_kernel<<<1, n_cuda_threads_per_block>>>(ij_ptr, cnt_ptr, tmp_ptr);
        }

        cudaDeviceSynchronize();

        {
            // generate collision set in the same syntax as the CPU version
            int n_collision_set = dev_cnt.back();
            thrust::host_vector<int> host_idx(n_collision_set);
            thrust::copy_n(ij.begin(), n_collision_set * 2, host_idx.begin());
            for (int i = 0; i < n_collision_set; i++) {
                auto vi = host_idx[i * 2], fj = host_idx[i * 2 + 1];
                idx.push_back({ I, vi, J, fj });
            }
        }

    };

    const auto ee_col_set = [&](vector<int>& eilist, vector<int>& ejlist,
                                const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                int I, int J,
                                vector<array<vec3, 4>>& ees,
                                vector<array<int, 4>>& eidx) {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        vector<lu> eiaabbs, ejaabbs;
        vector<Edge> eis, ejs;
        for (auto& ei : eilist) {
            Edge eii{ ci, unsigned(ei), true, true };
            eis.push_back(eii);
            eiaabbs.push_back(compute_aabb(eii, barrier::d_sqrt));
        }
        for (auto& ej : ejlist) {
            Edge ejj{ cj, unsigned(ej), true, true };
            ejs.push_back(ejj);
            ejaabbs.push_back(compute_aabb(ejj));
        }
        for (int i = 0; i < eilist.size(); i++)
            for (int j = 0; j < ejlist.size(); j++) {

                bool ee_intersects = intersects(eiaabbs[i], ejaabbs[j]);
                if (!ee_intersects) continue;
                auto &eii{ eis[i] }, &ejj{ ejs[j] };
                int ei = eilist[i], ej = ejlist[j];
                auto ee_type = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
                double d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
                if (d < barrier::d_hat) {
                    array<vec3, 4> ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
                    array<int, 4> ij = { I, ei, J, ej };
                    {
                        ees.push_back(ee);
                        eidx.push_back(ij);
                    }
                }
            }
    };
    const auto vf_col_time = [&](vector<int>& vilist, vector<int>& fjlist,
                                 const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                 int I, int J) -> double {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        int offi{ vertex_starting_index[I] }, offj{ vertex_starting_index[J] };
        double toi = 1.0;

        vector<lu> viaabbs, fjaabbs;
        vector<vec3> v0s, v1s;
        vector<Face> f0s, f1s;
        for (auto& vi : vilist) {
            vec3 v1{ ci.v_transformed[vi] };
            vec3 v0{ vt1_buffer[offi + vi] };
            v0s.push_back(v0);
            v1s.push_back(v1);
            viaabbs.push_back(compute_aabb(v0, v1));
        }
        for (auto& fj : fjlist) {
            Face f{ cj, unsigned(fj), true, true };
            int _a, _b, _c;
            _a = cj.indices[fj * 3 + 0],
            _b = cj.indices[fj * 3 + 1],
            _c = cj.indices[fj * 3 + 2];

            Face f0{ vt1_buffer[offj + _a], vt1_buffer[offj + _b], vt1_buffer[offj + _c] };

            f0s.push_back(f0);
            f1s.push_back(f);
            fjaabbs.push_back(compute_aabb(f0, f));
        }
        for (int i = 0; i < vilist.size(); i++)
            for (int j = 0; j < fjlist.size(); j++) {
                int vi = vilist[i], fj = fjlist[j];
                auto &v0{ v0s[i] }, &v{ v1s[i] };
                auto &f0{ f0s[j] }, &f{ f1s[j] };
                if (intersects(viaabbs[i], fjaabbs[j])) {
                    double t = pt_collision_time(v0, f0, v, f);
#ifdef TESTING
                    if (t < 1.0) {
                        idx.push_back({ I, vi, J, fj });
                        pt_tois.push_back({ t, int(pt_tois.size()) });
                    }
#endif
                    toi = min(toi, t);
                }
            }
        return toi;
    };

    const auto ee_col_time = [&](vector<int>& eilist, vector<int>& ejlist,
                                 const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                 int I, int J) -> double {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        int offi{ vertex_starting_index[I] }, offj{ vertex_starting_index[J] };

        double toi = 1.0;
        vector<lu> eiaabbs, ejaabbs;
        vector<Edge> ei0s, ej0s, ei1s, ej1s;
        for (auto& ei : eilist) {
            Edge ei1(ci, ei, true, true);
            int i0, i1;
            i0 = ci.edges[ei * 2];
            i1 = ci.edges[ei * 2 + 1];
            Edge ei0{ vt1_buffer[offi + i0], vt1_buffer[offi + i1] };
            ei0s.push_back(ei0);
            ei1s.push_back(ei1);
            eiaabbs.push_back(compute_aabb(ei0, ei1));
        }
        for (auto& ej : ejlist) {
            Edge ej1(cj, ej, true, true);
            int j0, j1;
            j0 = cj.edges[ej * 2];
            j1 = cj.edges[ej * 2 + 1];
            Edge ej0{ vt1_buffer[offj + j0], vt1_buffer[offj + j1] };
            ej0s.push_back(ej0);
            ej1s.push_back(ej1);
            ejaabbs.push_back(compute_aabb(ej0, ej1));
        }
        for (int i = 0; i < eilist.size(); i++)
            for (int j = 0; j < ejlist.size(); j++) {
                if (intersects(eiaabbs[i], ejaabbs[j])) {
                    auto &ei0{ ei0s[i] }, &ei1{ ei1s[i] }, &ej0{ ej0s[j] }, &ej1{ ej1s[j] };
                    int ei = eilist[i], ej = ejlist[j];
                    double t = ee_collision_time(ei0, ej0, ei1, ej1);
#ifdef TESTING
                    if (t < 1.0) {
                        if (I < J)
                            eidx.push_back({ I, ei, J, ej });
                        else
                            eidx.push_back({ J, ej, I, ei });
                        ee_tois.push_back({ t, int(ee_tois.size()) });
                    }
#endif
                    toi = min(toi, t);
                }
            }
        return toi;
    };

    double ee_global = 1.0, pt_global = 1.0;
    static vector<array<int, 4>>*idx_private = new vector<array<int, 4>>[omp_get_max_threads()],
                             *eidx_private = new vector<array<int, 4>>[omp_get_max_threads()];
    static vector<array<vec3, 4>>*pts_private = new vector<array<vec3, 4>>[omp_get_max_threads()], *ees_private = new vector<array<vec3, 4>>[omp_get_max_threads()];

    if (!cull_trajectory)
#pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        idx_private[tid].resize(0);
        eidx_private[tid].resize(0);
        pts_private[tid].resize(0);
        ees_private[tid].resize(0);
#pragma omp for schedule(guided) nowait
        for (int _i = 0; _i < n_overlap / 2; _i++) {
            int i = _i * 2;
            int I{ overlaps[i].i }, J{ overlaps[i].j };
            auto& p{ *overlaps[i].plist };
            auto& vilist{ p.vi };
            auto& vjlist{ p.vj };
            auto& eilist{ p.ei };
            auto& ejlist{ p.ej };
            auto& filist{ p.fi };
            auto& fjlist{ p.fj };

            ee_col_set(eilist, ejlist, cubes, I, J, ees_private[tid], eidx_private[tid]);
            vf_col_set_cuda(vilist, fjlist, cubes, I, J, pts_private[tid], idx_private[tid]);
            vf_col_set_cuda(vjlist, filist, cubes, J, I, pts_private[tid], idx_private[tid]);
        }
#pragma omp critical
        {
            pts.insert(pts.end(), pts_private[tid].begin(), pts_private[tid].end());
            idx.insert(idx.end(), idx_private[tid].begin(), idx_private[tid].end());
            ees.insert(ees.end(), ees_private[tid].begin(), ees_private[tid].end());
            eidx.insert(eidx.end(), eidx_private[tid].begin(), eidx_private[tid].end());
        }
    }
    else
#pragma omp parallel
    {
        double toi = 1.0;
#pragma omp for schedule(guided) nowait
        for (int _i = 0; _i < n_overlap / 2; _i++) {
            int i = _i * 2;
            int I{ overlaps[i].i }, J{ overlaps[i].j };
            auto& p{ *overlaps[i].plist };
            auto& vilist{ p.vi };
            auto& vjlist{ p.vj };
            auto& eilist{ p.ei };
            auto& ejlist{ p.ej };
            auto& filist{ p.fi };
            auto& fjlist{ p.fj };
            double t1 = vf_col_time(vilist, fjlist, cubes, I, J);
            double t2 = vf_col_time(vjlist, filist, cubes, J, I);
            double t3 = ee_col_time(eilist, ejlist, cubes, I, J);

            toi = min(toi, min({t1, t2, t3}));
        }
#pragma omp critical
        {
            toi_ee_pt = min(toi_ee_pt, toi);
        }
    }
    if (cull_trajectory) {
        if (toi_ee_pt < 1e-6) {
            spdlog::error("pt/ee toi_global = {}", toi_ee_pt);
            if (globals.params_int.find("p_cnt") != globals.params_int.end())
                globals.params_int["p_cnt"]++;
            else
                globals.params_int["p_cnt"] = 0;
            if (globals.params_int["p_cnt"] > 1) exit(1);
        }
        else
            globals.params_int["p_cnt"] = 0;
    }
    toi_global = min(toi_global, toi_ee_pt);
    return cull_trajectory? toi_global: 1.0;
}
