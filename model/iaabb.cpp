
#include "iaabb.h"
#include "geometry.h"
#include "collision.h"
#include <memory>
#include <array>
#include <algorithm>
#include "barrier.h"
#include "time_integrator.h"
#include <omp.h>
#include <tbb/parallel_sort.h>
// #define _FULL_PARALLEL_

#ifndef TESTING
#include "../view/global_variables.h"
#define DT globals.dt
#else
#include "../iAABB/pch.h"
#define DT 1e-2
#endif
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace utils;
using namespace std::chrono;
using lu = std::array<vec3, 2>;
#define DURATION_TO_DOUBLE(X) duration_cast<duration<double>>(high_resolution_clock::now() - (X)).count()

void glue_vf_col_set(
    vector<int>& vilist, vector<int>& fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    int tid = 0);

#include <cuda/std/array>
#include <thrust/host_vector.h>
#include <type_traits>
#include "cuda_header.cuh"

void make_lut_glue(vector<Intersection> &os) {
    int ni = os.size();
    thrust::host_vector<i2> host_lut(ni);
    host_lut.resize(ni);
    #pragma omp parallel for
    for (int i = 0; i < ni; i++) {
        host_lut[i] = { os[i].i, os[i].j };
    }
    make_lut(host_lut);
}
float vf_distance(vec3f _v, Facef f, ipc::PointTriangleDistanceType &pt_type);
tuple<float, ipc::PointTriangleDistanceType> vf_distance(vec3f vf, Facef ff); 

void vf_col_set_cuda(
    // vector<int>& vilist, vector<int>& fjlist,
    // const std::vector<std::unique_ptr<AffineBody>>& cubes,
    // int I, int J,
    int nvi, int nfj,
    const thrust::host_vector<luf>& aabbs,
    const thrust::host_vector<vec3f>& vis,
    const thrust::host_vector<Facef>& fjs,
    const std::vector<int>& vilist, const std::vector<int>& fjlist,
    std::vector<std::array<int, 4>>& idx,
    int I, int J,
    int tid = 0);

inline luf to_luf(const lu& a)
{
    return {
        make_float3(a[0][0], a[0][1], a[0][2]),
        make_float3(a[1][0], a[1][1], a[1][2])
    };
}
inline vec3f to_vec3f(const vec3& a)
{
    return make_float3(a[0], a[1], a[2]);
}
inline Facef to_facef(const Face& f)
{
    return {
        to_vec3f(f.t0),
        to_vec3f(f.t1),
        to_vec3f(f.t2)
    };
}

void glue_vf_col_set(
    vector<int>& vilist, vector<int>& fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    int tid)
{
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
    vf_col_set_cuda(nvi, nfj, aabbs, vis, fjs, vilist, fjlist, idx, I, J, tid);
    pts.resize(idx.size());
}
inline bool les(const Intersection& a, const Intersection& b)
{
    return a.i < b.i || (a.i == b.i && a.j < b.j);
};
lu compute_aabb(const AffineBody& c)
{
    vec3 l, u;
    for (int v = 0; v < c.n_vertices; v++) {
        auto p = c.vertices(v);
        if (v == 0) {
            l = u = p;
        }
        else {
            l = l.array().min(p.array());
            u = u.array().max(p.array());
        }
    }
    l.array() -= barrier::d_sqrt;
    u.array() += barrier::d_sqrt;
    return { l, u };
}
lu affine(const lu& aabb, q4& q)
{
    Matrix<double, 3, 8> cull, _cull;
    vec3 l{ aabb[0] }, u{ aabb[1] };
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                auto I = (i << 2) | (j << 1) | k;
                _cull(0, I) = i ? u(0) : l(0);
                _cull(1, I) = j ? u(1) : l(1);
                _cull(2, I) = k ? u(2) : l(2);
            }
    mat3 A;
    A << q[1], q[2], q[3];
    cull = A * _cull;
    l = cull.rowwise().minCoeff();
    u = cull.rowwise().maxCoeff();
    return { l + q[0], u + q[0] };
}

inline bool intersection(const lu& a, const lu& b, lu& ret)
{
    vec3 l, u;
    l = a[0].array().max(b[0].array());
    u = a[1].array().min(b[1].array());
    bool intersects = (l.array() <= u.array()).all();
    ret = { l, u };
    return intersects;
}

lu affine(lu aabb, AffineBody& c, int vtn)
{
    auto qi = vtn == 0 ? c.q0 : c.q;
    if (vtn >= 2)
        for (int i = 0; i < 4; i++) qi[i] += c.dq.segment<3>(i * 3);
    if (vtn == 3) {
        auto bt0{ affine(aabb, c.q) }, bt1{ affine(aabb, qi) };
        return { merge(bt0, bt1) };
    }
    return affine(aabb, qi);
};

void intersect_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    std::vector<Intersection>& ret,
    int vtn)
{
    ret.resize(0);

    for (int i = 0; i < n_cubes; i++)
        for (int j = i + 1; j < n_cubes; j++) {
            auto &ci{ *cubes[i] }, &cj{ *cubes[j] };
            auto &ai{ aabbs[i] }, &aj{ aabbs[j] };

            auto bi = affine(ai, ci, vtn);
            auto bj = affine(aj, cj, vtn);
            lu c;
            if (intersection(bi, bj, c)) {
                ret.push_back({ i, j, c, nullptr });
                ret.push_back({ j, i, c, nullptr });
            }
        }
}

void aabb_culling_sort(
    const vector<lu>& aabbsi, const vector<lu>& aabbsj,
    vector<array<int, 2>>& ret)
{
    vector<lu> affine_bb;
    float sep = aabbsi.size() - 0.5;
    affine_bb.reserve(aabbsi.size() + aabbsj.size());
    affine_bb.insert(affine_bb.end(), aabbsi.begin(), aabbsi.end());
    affine_bb.insert(affine_bb.end(), aabbsj.begin(), aabbsj.end());
    int n_bodies = affine_bb.size();
    vector<BoundingBox> bounds[3];

    vector<vector<int>> intersected_body_per_dim[3], intersected_body_joint;
    auto* ret_tmp = new vector<array<int, 2>>[n_bodies];
    intersected_body_joint.resize(n_bodies);
    intersected_body_per_dim[0].resize(n_bodies);
    intersected_body_per_dim[1].resize(n_bodies);
    intersected_body_per_dim[2].resize(n_bodies);
    ret.resize(0);

    for (int dim = 0; dim < 3; dim++) {
        bounds[dim].resize(n_bodies * 2);
        for (int i = 0; i < n_bodies; i++) {
            auto& t{ affine_bb[i] };
            auto &l{ t[0] }, &u{ t[1] };
            bounds[dim][i * 2] = { i, l(dim), true };
            bounds[dim][i * 2 + 1] = { i, u(dim), false };
        }
        sort(bounds[dim].begin(), bounds[dim].end(), [](const BoundingBox& a, const BoundingBox& b) { return a.p < b.p; });
        vector<int> active;
        for (int i = 0; i < n_bodies * 2; i++) {
            auto& b{ bounds[dim][i] };
            auto body = b.body;
            if (b.true_for_l_false_for_u)
                active.push_back(body);
            else {
                auto it = find(active.begin(), active.end(), body);
                active.erase(it);
                // intersected_body_per_dim[dim][body].insert(intersected_body_per_dim[dim][body].end(), active.begin(), active.end());
                for (auto c : active)
                    if ((c - sep) * (body - sep) < 0) {
                        intersected_body_per_dim[dim][body].push_back(c);
                        intersected_body_per_dim[dim][c].push_back(body);
                    }
            }
        }
    }
    vector<unsigned> bucket;
    bucket.resize(n_bodies);
    // #pragma omp parallel
    {
        //         auto tid = omp_get_thread_num();
        // #pragma omp for schedule(guided)
        for (int i = 0; i < n_bodies; i++) {
            fill(bucket.begin(), bucket.end(), 0);
            for (int dim = 0; dim < 3; dim++) {
                auto& l{ intersected_body_per_dim[dim][i] };
                for (auto j : l) {
                    if (dim < 2)
                        bucket[j] += 1;
                    else if (bucket[j] == 2) {
                        intersected_body_joint[i].push_back(j);
                    }
                }
            }
            sort(intersected_body_joint[i].begin(), intersected_body_joint[i].end());
        }
    }
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n_bodies; i++) {
        auto& l{ intersected_body_joint[i] };
        for (int j : l) {
            ret_tmp[i].push_back({ i, j });
        }
    }

    for (int i = 0; i < n_bodies; i++) {
        ret.insert(ret.end(), ret_tmp[i].begin(), ret_tmp[i].end());
    }
}
void intersect_sort(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    std::vector<Intersection>& ret,
    int vtn)
{

    /*
    vtn = {
        1: A = q,
        2: A = q + dq,
        3: trajectory
    }

    */
    static vector<BoundingBox> bounds[3];
    static vector<int>*intersected_body_per_dim[3] = {
        new vector<int>[n_cubes],
        new vector<int>[n_cubes],
        new vector<int>[n_cubes]
    },
           *intersected_body_joint = new vector<int>[n_cubes];
    static vector<Intersection>* ret_tmp = new vector<Intersection>[n_cubes];

    static vector<lu> affine_bb;
    // static vector<Intersection> intersected_body_joint;
    ret.resize(0);

    affine_bb.resize(n_cubes);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_cubes; i++) {
        auto t{ affine(aabbs[i], *cubes[i], vtn) };
        affine_bb[i] = t;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_cubes; i++) {
        intersected_body_joint[i].resize(0);
        ret_tmp[i].resize(0);
        for (int dim = 0; dim < 3; dim++)
            intersected_body_per_dim[dim][i].resize(0);
    }
#pragma omp parallel for schedule(static, 1)
    for (int dim = 0; dim < 3; dim++) {
        bounds[dim].resize(n_cubes * 2);
        for (int i = 0; i < n_cubes; i++) {
            auto& t{ affine_bb[i] };
            auto &l{ t[0] }, &u{ t[1] };
            bounds[dim][i * 2] = { i, l(dim), true };
            bounds[dim][i * 2 + 1] = { i, u(dim), false };
        }
        sort(bounds[dim].begin(), bounds[dim].end(), [](const BoundingBox& a, const BoundingBox& b) { return a.p < b.p; });
        vector<int> active;
        for (int i = 0; i < n_cubes * 2; i++) {
            auto& b{ bounds[dim][i] };
            auto body = b.body;
            if (b.true_for_l_false_for_u)
                active.push_back(body);
            else {
                auto it = find(active.begin(), active.end(), body);
                active.erase(it);
                intersected_body_per_dim[dim][body].insert(intersected_body_per_dim[dim][body].end(), active.begin(), active.end());
                for (auto c : active) {
                    intersected_body_per_dim[dim][c].push_back(body);
                }
            }
        }
    }
    static vector<unsigned> buckets;
    buckets.resize(omp_get_max_threads() * n_cubes);

#pragma omp parallel
    {
        // unsigned char* bucket = new unsigned char[n_cubes];
        auto tid = omp_get_thread_num();
#pragma omp for schedule(guided)
        for (int i = 0; i < n_cubes; i++) {
            auto bucket{ buckets.data() + tid * n_cubes };
            fill(bucket, bucket + n_cubes, 0);
            for (int dim = 0; dim < 3; dim++) {
                auto& l{ intersected_body_per_dim[dim][i] };
                for (auto j : l) {
                    if (dim < 2)
                        bucket[j] += 1;
                    else if (bucket[j] == 2) {
                        intersected_body_joint[i].push_back(j);
                    }
                }
            }
            sort(intersected_body_joint[i].begin(), intersected_body_joint[i].end());
        }
    }

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n_cubes; i++) {
        auto& l{ intersected_body_joint[i] };
        auto& bi = affine_bb[i];
        for (int j : l) {
            auto& bj = affine_bb[j];
            lu cull;
            intersection(bi, bj, cull);
            ret_tmp[i].push_back({ i, j, cull, nullptr });
        }
    }

    for (int i = 0; i < n_cubes; i++) {
        ret.insert(ret.end(), ret_tmp[i].begin(), ret_tmp[i].end());
    }
    // TODO: O(n) insertion sort
}
inline void pt_col_set_task(
    int vi, int fj, int I, int J,
    // const AffineBody& ci, const AffineBody& cj,
    const vec3& v, const Face& f,
    const lu& aabb_i, const lu& aabb_j,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx)
{
    bool pt_intersects = intersects(aabb_i, aabb_j);
    if (!pt_intersects) return;
    auto [d, pt_type] = vf_distance(v, f);
    if (d < barrier::d_hat) {
        array<vec3, 4> pt = { v, f.t0, f.t1, f.t2 };
        array<int, 4> ij = { I, vi, J, fj };
        {
            pts.push_back(pt);
            idx.push_back(ij);
        }
    }
};
inline void ee_col_set_task(
    int ei, int ej, int I, int J,
    // const AffineBody& ci, const AffineBody& cj,
    const Edge& eii, const Edge& ejj,
    const lu& aabb_i, const lu& aabb_j, vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx)
{
    bool ee_intersects = intersects(aabb_i, aabb_j);
    if (!ee_intersects) return;
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
};

double primitive_brute_force(
    int n_cubes,
    std::vector<Intersection>& overlaps, // assert sorted
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int vtn,
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#ifndef _BODY_WISE_
    Globals& globals,
#endif
#endif
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
        delete[] lists;
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
#ifdef _BODY_WISE_
#pragma omp parallel for schedule(guided)
    for (int I = 0; I < n_cubes; I++) {
        auto& c{ *cubes[I] };
        for (int v = 0; v < c.n_vertices; v++) {
            auto& p{ c.v_transformed[v] };
            for (int o = starting[I]; o < starting[I + 1]; o++) {
                lu& cull = overlaps[o].cull;
                overlaps[o].plist = lists + o;
                if (filter_if_inside(cull, p, cull_trajectory, c, v)) {
                    auto& t{ overlaps[o] };
                    assert(t.i == I);
                    if (t.i < t.j)
                        lists[o].vi.push_back(v);
                    else
                        lists[o].vj.push_back(v);
                }
            }
        }

        for (int e = 0; e < c.n_edges; e++) {
            Edge ei{ c, unsigned(e), true, true };
            // TODO: always intialize from v_transformed

            for (int o = starting[I]; o < starting[I + 1]; o++) {
                lu& cull = overlaps[o].cull;

                if (filter_if_inside(cull, ei, cull_trajectory, c, e)) {
                    auto& t{ overlaps[o] };
                    if (t.i < t.j)
                        lists[o].ei.push_back(e);
                    else
                        lists[o].ej.push_back(e);
                }
            }
        }

        for (int f = 0; f < c.n_faces; f++) {
            Face fi{ c, unsigned(f), true, true };
            // TODO: always intialize from v_transformed
            for (int o = starting[I]; o < starting[I + 1]; o++) {
                lu& cull = overlaps[o].cull;

                if (filter_if_inside(cull, fi, cull_trajectory, c, f)) {
                    auto t{ overlaps[o] };
                    if (t.i < t.j)
                        lists[o].fi.push_back(f);
                    else
                        lists[o].fj.push_back(f);
                }
            }
        }
    }

#else
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

    for (int i = 0; i < n_overlap; i++) overlaps[i].plist = lists + i;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        auto idx{ globals.points[i] };
        auto I{ idx[0] };
        auto v{ idx[1] };
        auto& c{ *cubes[I] };
        vec3 p{ c.v_transformed[v] };
        vec3 p0{ c.vt1(v) };

        lu aabb = cull_trajectory ? compute_aabb(p, p0) : lu{ p, p };
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
        Face f0{ c, fi };

        lu aabb = cull_trajectory ? compute_aabb(f, f0) : compute_aabb(f);
        for (int o = starting[I]; o < starting[I + 1]; o++) {
            lu cull = overlaps[o].cull;
            if (intersects(aabb, cull)) {
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
#endif

    if (globals.params_int["cuda_direct"]) {
        make_lut_glue(overlaps);
    }
    tbb::parallel_sort(overlaps.begin(), overlaps.end(), [](const Intersection& a, const Intersection& b) -> bool {
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

    const auto vf_col_set = [&](vector<int>& vilist, vector<int>& fjlist,
                                const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                int I, int J,
                                vector<array<vec3, 4>>& pts,
                                vector<array<int, 4>>& idx) {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        vector<lu> viaabbs, fjaabbs;
        vector<vec3> vis;
        vector<Face> fjs;
        for (auto& vi : vilist) {
            vec3 v{ ci.v_transformed[vi] };
            vis.push_back(v);
            viaabbs.push_back({ v.array() - barrier::d_sqrt, v.array() + barrier::d_sqrt });
        }
        for (auto& fj : fjlist) {
            Face f{ cj, unsigned(fj), true, true };
            fjs.push_back(f);
            fjaabbs.push_back(compute_aabb(f));
        }
#define BRUTE_FORCE
#ifdef BRUTE_FORCE
        for (int i = 0; i < vilist.size(); i++)
            for (int j = 0; j < fjlist.size(); j++) {
                int vi = vilist[i], fj = fjlist[j];
                bool pt_intersects = intersects(viaabbs[i], fjaabbs[j]);
                if (!pt_intersects) continue;
                auto& v{ vis[i] };
                auto& f{ fjs[j] };
                auto [d, pt_type] = vf_distance(v, f);
                if (d < barrier::d_hat) {
                    array<vec3, 4> pt = { v, f.t0, f.t1, f.t2 };
                    array<int, 4> ij = { I, vi, J, fj };
                    {
                        pts.push_back(pt);
                        idx.push_back(ij);
                    }
                }
            }
#else
        vector<array<int, 2>> vfpairs;
        aabb_culling_sort(viaabbs, fjaabbs, vfpairs);
        for (auto vf : vfpairs) {
            int i = vf[0], j = vf[1];
            int vi = vilist[i], fj = fjlist[j];
            auto& v{ vis[i] };
            auto& f{ fjs[j] };
            auto [d, pt_type] = vf_distance(v, f);
            if (d < barrier::d_hat) {
                array<vec3, 4> pt = { v, f.t0, f.t1, f.t2 };
                array<int, 4> ij = { I, vi, J, fj };
                {
                    pts.push_back(pt);
                    idx.push_back(ij);
                }
            }
        }
#endif
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
    
#ifdef _FULL_PARALLEL_
    static vector<int> inner_presum;
    auto n_lists = n_overlap / 2;
    inner_presum.resize(n_lists + 1);

    const auto compute_thread_starting = [&](int pt_ee_tp, vector<array<int, 2>>& ret) -> int {
        int inner, outer;
        int n_threads = omp_get_num_procs();
        ret.resize(n_threads);
        inner_presum[0] = 0;
        for (int _i = 0; _i < n_lists; _i++) {
            int i = _i * 2;
            auto& p{ *overlaps[i].plist };
            auto& pi{
                pt_ee_tp == 0 ? p.vi : pt_ee_tp == 1 ? p.ei
                                                     : p.fi
            };
            auto& pj{
                pt_ee_tp == 0 ? p.fj : pt_ee_tp == 1 ? p.ej
                                                     : p.vj
            };
            auto ni = pi.size(), nj = pj.size();
            inner_presum[_i + 1] = inner_presum[_i] + ni * nj;
            // presum
        }
        int n_task = inner_presum[n_lists];
        int k_threads = (n_task + n_threads - 1) / n_threads;
        for (int j = 0; j < n_threads; j++) {
            int kj = j * k_threads;
            int i0 = lower_bound(inner_presum.begin(), inner_presum.end(), kj, [](int a, int b) -> bool {
                return a <= b;
            }) - inner_presum.begin()
                - 1;

            ret[j] = { i0 * 2, kj - inner_presum[i0] };
        }
        return k_threads;
    };

    static vector<array<int, 2>> thread_starting_index;
    if (!cull_trajectory) {
        // point-triangle
        int k_threads = compute_thread_starting(0, thread_starting_index);
#pragma omp parallel
        {
            int k_tasks_done = 0;
            auto tid = omp_get_thread_num();
            idx_private[tid].resize(0);
            pts_private[tid].resize(0);

            auto& a = thread_starting_index[tid];
            int outer = a[0], inner = a[1];
            do {
                int I{ overlaps[outer].i }, J{ overlaps[outer].j };
                auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
                auto& p{ *overlaps[outer].plist };
                auto& vilist{ p.vi };
                auto& fjlist{ p.fj };
                int nv = vilist.size(), nf = fjlist.size();
                int n_pt = nv * nf;
                for (int i = inner; i < n_pt; i++) {
                    int vi{ i / nf }, fj{ i % nf };
                    pt_col_set_task(vi, fj, I, J, ci, cj, pts_private[tid], idx_private[tid]);
                    k_tasks_done++;
                    if (k_tasks_done >= k_threads) break;
                }
                if (outer >= n_overlap || k_tasks_done >= k_threads) break;
                outer += 2;
                inner = 0;
            } while (1);
#pragma omp critical
            {
                pts.insert(pts.end(), pts_private[tid].begin(), pts_private[tid].end());
                idx.insert(idx.end(), idx_private[tid].begin(), idx_private[tid].end());
            }
        }
        // triangle-point
        k_threads = compute_thread_starting(2, thread_starting_index);
#pragma omp parallel
        {
            int k_tasks_done = 0;
            auto tid = omp_get_thread_num();
            idx_private[tid].resize(0);
            pts_private[tid].resize(0);

            auto& a = thread_starting_index[tid];
            int outer = a[0], inner = a[1];
            do {
                int I{ overlaps[outer].i }, J{ overlaps[outer].j };
                auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
                auto& p{ *overlaps[outer].plist };
                auto& vjlist{ p.vj };
                auto& filist{ p.fi };
                int nv = vjlist.size(), nf = filist.size();
                int n_pt = nv * nf;
                for (int i = inner; i < n_pt; i++) {
                    int vj{ i / nf }, fi{ i % nf };
                    pt_col_set_task(vj, fi, I, J, ci, cj, pts_private[tid], idx_private[tid]);
                    k_tasks_done++;
                    if (k_tasks_done >= k_threads) break;
                }
                if (outer >= n_overlap || k_tasks_done >= k_threads) break;
                outer += 2;
                inner = 0;
            } while (1);
#pragma omp critical
            {
                pts.insert(pts.end(), pts_private[tid].begin(), pts_private[tid].end());
                idx.insert(idx.end(), idx_private[tid].begin(), idx_private[tid].end());
            }
        }
        // edge-edge
        k_threads = compute_thread_starting(1, thread_starting_index);
#pragma omp parallel
        {
            int k_tasks_done = 0;
            auto tid = omp_get_thread_num();
            ees_private[tid].resize(0);
            eidx_private[tid].resize(0);

            auto& a = thread_starting_index[tid];
            int outer = a[0], inner = a[1];
            do {
                int I{ overlaps[outer].i }, J{ overlaps[outer].j };
                auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
                auto& p{ *overlaps[outer].plist };
                auto& eilist{ p.ei };
                auto& ejlist{ p.ej };
                int nei = eilist.size(), nej = ejlist.size();
                int n_ee = nei * nej;
                for (int i = inner; i < n_ee; i++) {
                    int ei{ i / nej }, ej{ i % nej };
                    ee_col_set_task(ei, ej, I, J, ci, cj, ees_private[tid], eidx_private[tid]);
                    k_tasks_done++;
                    if (k_tasks_done >= k_threads) break;
                }
                if (outer >= n_overlap || k_tasks_done >= k_threads) break;
                outer += 2;
                inner = 0;
            } while (1);
#pragma omp critical
            {
                ees.insert(ees.end(), ees_private[tid].begin(), ees_private[tid].end());
                eidx.insert(eidx.end(), eidx_private[tid].begin(), eidx_private[tid].end());
            }
        }
    }
#else
    if (!cull_trajectory)
    // #pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        idx_private[tid].resize(0);
        eidx_private[tid].resize(0);
        pts_private[tid].resize(0);
        ees_private[tid].resize(0);
        // #pragma omp for schedule(guided) nowait
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
            // else
            if (!globals.params_int["cuda_direct"])
            {

                vf_col_set(vilist, fjlist, cubes, I, J, pts_private[tid], idx_private[tid]);
                vf_col_set(vjlist, filist, cubes, J, I, pts_private[tid], idx_private[tid]);
            }
            else {
                glue_vf_col_set(vilist, fjlist, cubes, I, J, pts_private[tid], idx_private[tid], tid);
                glue_vf_col_set(vjlist, filist, cubes, J, I, pts_private[tid], idx_private[tid], tid);
            }
            if (globals.params_int["cuda_supervised"]) {
                vector<array<int, 4>>&idx_ref{ idx_private[tid] }, idx_cuda{}, diff{}, cuda_ref{};
                // glue_vf_col_set(vilist, fjlist, cubes, I, J, pts_private[tid], idx_private[tid]);
                // glue_vf_col_set(vjlist, filist, cubes, J, I, pts_private[tid], idx_private[tid]);
                glue_vf_col_set(vilist, fjlist, cubes, I, J, pts_private[tid], idx_cuda, tid);
                glue_vf_col_set(vjlist, filist, cubes, J, I, pts_private[tid], idx_cuda, tid);
                sort(idx_cuda.begin(), idx_cuda.end());
                sort(idx_ref.begin(), idx_ref.end());

                set_difference(idx_ref.begin(), idx_ref.end(), idx_cuda.begin(), idx_cuda.end(), back_inserter(diff));
                set_difference(idx_cuda.begin(), idx_cuda.end(), idx_ref.begin(), idx_ref.end(), back_inserter(cuda_ref));
                if (globals.params_int["strict"]) {
                    assert(idx_cuda.size() == idx_ref.size());
                    assert(diff.size() == 0);
                }
                else if (diff.size() > 0 || idx_cuda.size() != idx_ref.size()) {
                    spdlog::error("diff size = {}, cuda size = {}, ref size = {}", diff.size(), idx_cuda.size(), idx_ref.size());
                    for (auto a : diff) {
                        Face f{ *cubes[a[2]], unsigned(a[3]), true, true };
                        Facef ff{ to_facef(f) };
                        vec3 p{
                            cubes[a[0]]->v_transformed[a[1]]
                        };
                        vec3f pf{ to_vec3f(
                            cubes[a[0]]->v_transformed[a[1]]) };
                        ipc::PointTriangleDistanceType ptt;
                        auto d = vf_distance(pf, ff, ptt);
                        auto [d_ref, type_ref] = vf_distance(p, f);
                        auto [d_ref2, type_ref2] = vf_distance(pf, ff);
                        spdlog::error("in diff set: pt distance = {}, ref = {}, ref2 = {}", d, d_ref, d_ref2);

                        spdlog::error ("type, cuda = {}, ref = {}", static_cast<underlying_type_t<ipc::PointTriangleDistanceType>>(ptt), static_cast<underlying_type_t<ipc::PointTriangleDistanceType>>(type_ref));
                    }
                    for (auto a: cuda_ref) {
                        Face f{ *cubes[a[2]], unsigned(a[3]), true, true };
                        Facef ff{ to_facef(f) };
                        vec3 p{
                            cubes[a[0]]->v_transformed[a[1]]
                        };
                        vec3f pf{ to_vec3f(
                            cubes[a[0]]->v_transformed[a[1]]) };
                        ipc::PointTriangleDistanceType ptt;
                        auto d = vf_distance(pf, ff, ptt);
                        auto [d_ref, type_ref] = vf_distance(p, f);
                        auto [d_ref2, type_ref2] = vf_distance(pf, ff);
                        spdlog::error("in diff (cuda - ref): pt distance = {}, ref = {}, ref2 = {}", d, d_ref, d_ref2);
                        spdlog::error ("type, cuda = {}, ref = {}", static_cast<underlying_type_t<ipc::PointTriangleDistanceType>>(ptt), static_cast<underlying_type_t<ipc::PointTriangleDistanceType>>(type_ref));

                    }
                }
            }
        }
        // #pragma omp critical
        {
            pts.insert(pts.end(), pts_private[tid].begin(), pts_private[tid].end());
            idx.insert(idx.end(), idx_private[tid].begin(), idx_private[tid].end());
            ees.insert(ees.end(), ees_private[tid].begin(), ees_private[tid].end());
            eidx.insert(eidx.end(), eidx_private[tid].begin(), eidx_private[tid].end());
        }
    }
#endif
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

            toi = min(toi, min({ t1, t2, t3 }));
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
    return cull_trajectory ? toi_global : 1.0;
}

double iaabb_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    int vtn,
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#ifndef _BODY_WISE_
    Globals& globals,
#endif
#endif
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx)
{
    auto start = high_resolution_clock::now();
    vector<Intersection> ret;
    intersect_sort(n_cubes, cubes, aabbs, ret, vtn);
    double toi;

    toi = primitive_brute_force(n_cubes, ret, cubes, vtn,
#ifdef TESTING
        pt_tois, ee_tois,
#ifndef _BODY_WISE_
        globals,
#endif
#endif
        pts,
        idx,
        ees,
        eidx,
        vidx);
    auto t = DURATION_TO_DOUBLE(start);
    spdlog::info("time: {} = {:0.6f} ms", vtn == 3 ? "iaabb upper bound" : "iAABB", t * 1000);
    return toi;
}

tuple<float, ipc::PointTriangleDistanceType> vf_distance(vec3f vf, Facef ff){
    Eigen::Vector3f v {vf.x, vf.y, vf.z};
    Eigen::Vector3f f[3] {{ff.t0.x, ff.t0.y, ff.t0.z}, {ff.t1.x, ff.t1.y, ff.t1.z}, {ff.t2.x, ff.t2.y, ff.t2.z}};
    auto pt_type = ipc::point_triangle_distance_type(v, f[0], f[1], f[2]);
    float d = ipc::point_triangle_distance(v, f[0], f[1], f[2], pt_type);
    return {d, pt_type};
}
