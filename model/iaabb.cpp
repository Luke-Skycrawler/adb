
#include "iaabb.h"
#include "geometry.h"
#include "collision.h"
#include <memory>
#include <array>
#include <algorithm>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "barrier.h"
#include "time_integrator.h"
#include <omp.h>
#include <tbb/parallel_sort.h>
#include "ipc_extension.h"
// #define _FULL_PARALLEL_

#ifndef TESTING
#include "settings.h"
#else
#include "../iAABB/pch.h"
#endif
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace utils;
using namespace std::chrono;

#define DURATION_TO_DOUBLE(X) duration_cast<duration<scalar>>(high_resolution_clock::now() - (X)).count()
inline bool les(const Intersection& a, const Intersection& b){
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
inline lu merge(const lu& a, const lu& b)
{
    vec3 l, u;
    l = a.lower.array().min(b.lower.array());
    u = a.upper.array().max(b.upper.array());
    return { l, u };
}
inline lu compute_aabb(const vec3& p0, const vec3& p1)
{
    vec3 l, u;
    auto e0{ p0.array() }, e1{ p1.array() };
    l = e0.min(e1);
    u = e0.max(e1);
    return { l, u };
}
inline lu compute_aabb(const Edge& e1, const Edge& e2)
{
    vec3 l, u;
    auto e10{ e1.e0.array() }, e11{ e1.e1.array() };
    auto e20{ e2.e0.array() }, e21{ e2.e1.array() };
    l = e10.min(e11).min(e20).min(e21);
    u = e10.max(e11).max(e20).max(e21);
    return { l, u };
}
inline lu compute_aabb(const Face& f1, const Face& f2)
{
    return merge(compute_aabb(f1), compute_aabb(f2));
}
lu affine(const lu& aabb, q4& q)
{
    Matrix<scalar, 3, 8> cull, _cull;
    vec3 l{ aabb.lower }, u{ aabb.upper };
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
    l = a.lower.array().max(b.lower.array());
    u = a.upper.array().min(b.upper.array());
    bool intersects = (l.array() <= u.array()).all();
    ret = { l, u };
    return intersects;
}

inline bool intersects(const lu& a, const lu& b)
{
    const auto overlaps = [&](int i) -> bool {
        return a.lower[i] <= b.upper[i] && a.upper[i] >= b.lower[i];
    };
    return overlaps(0) && overlaps(1) && overlaps(2);
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
        intersected_body_joint[i].resize(0);
        ret_tmp[i].resize(0);
        for (int dim = 0; dim < 3; dim ++)
        intersected_body_per_dim[dim][i].resize(0);
    }
    #pragma omp parallel for schedule(static, 1)
    for (int dim = 0; dim < 3; dim++) {
        bounds[dim].resize(n_cubes * 2);
        for (int i = 0; i < n_cubes; i++) {
            auto& t{ affine_bb[i] };
            auto &l{ t.lower }, &u{ t.upper };
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
    for (int i = 0; i < n_cubes; i ++) {
        auto& l{ intersected_body_joint[i] };
        auto& bi = affine_bb[i];
        for (int j: l) {
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
    scalar d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
    if (d < barrier::d_hat) {
        array<vec3, 4> ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
        array<int, 4> ij = { I, ei, J, ej };

        {
            ees.push_back(ee);
            eidx.push_back(ij);
        }
    }
};

scalar ee_col_time(
    vector<int>& eilist, vector<int>& ejlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J, vector<int>& vertex_starting_index, vector<vec3>& vt1_buffer)
{
    auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    int offi{ vertex_starting_index[I] }, offj{ vertex_starting_index[J] };

    scalar toi = 1.0;
    vector<lu> eiaabbs, ejaabbs;
    vector<Edge> ei0s, ej0s, ei1s, ej1s;

    eiaabbs.clear();
    ejaabbs.clear();
    ei0s.clear();
    ej0s.clear();
    ei1s.clear();
    ej1s.clear();

    eiaabbs.reserve(eilist.size());
    ejaabbs.reserve(ejlist.size());
    ei0s.reserve(eilist.size());
    ej0s.reserve(ejlist.size());
    ei1s.reserve(eilist.size());
    ej1s.reserve(ejlist.size());
    
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
    int nei = eilist.size(), nej = ejlist.size();   
    if (nei > globals.params_int["thres"] || nej > globals.params_int["thres"]) {
        auto &EIs{nei > nej? eiaabbs: ejaabbs}, &EJs{nei > nej? ejaabbs: eiaabbs};
        auto bvh = bvh_create(EIs.data(), EIs.size());
        for (int J = 0; J < EJs.size(); J ++){
            int I;
            auto &aabbj {EJs[J]};
            auto query = bvh_query_aabb(uint64_t(&bvh), aabbj.lower, aabbj.upper);
            while (bvh_query_next(query, I)){
                int i, j;
                if (nei > nej) 
                    {i = I; j = J;}
                else 
                    {i = J; j = I;}

                auto &ei0{ ei0s[i] }, &ei1{ ei1s[i] }, &ej0{ ej0s[j] }, &ej1{ ej1s[j] };
                int ei = eilist[i], ej = ejlist[j];
                scalar t = ee_collision_time(ei0, ej0, ei1, ej1);
                toi = min(toi, t);
            }
        }
        bvh_destroy_host(bvh);
    } else
    for (int i = 0; i < eilist.size(); i++)
        for (int j = 0; j < ejlist.size(); j++) {
            if (intersects(eiaabbs[i], ejaabbs[j])) {
                auto &ei0{ ei0s[i] }, &ei1{ ei1s[i] }, &ej0{ ej0s[j] }, &ej1{ ej1s[j] };
                int ei = eilist[i], ej = ejlist[j];
                scalar t = ee_collision_time(ei0, ej0, ei1, ej1);
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
}

scalar vf_col_time(
    vector<int>& vilist, vector<int>& fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J, vector<int> &vertex_starting_index, vector<vec3> &vt1_buffer){
    auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    int offi{ vertex_starting_index[I] }, offj{ vertex_starting_index[J] };
    scalar toi = 1.0;

    vector<lu> viaabbs, fjaabbs;
    vector<vec3> v0s, v1s;
    vector<Face> f0s, f1s;

    viaabbs.clear();
    fjaabbs.clear();
    v0s.clear();
    v1s.clear();
    f0s.clear();
    f1s.clear();

    viaabbs.reserve(vilist.size());
    fjaabbs.reserve(fjlist.size());
    v0s.reserve(vilist.size());
    v1s.reserve(vilist.size());
    f0s.reserve(fjlist.size());
    f1s.reserve(fjlist.size());
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

    int nvi = vilist.size(), nfj = fjlist.size();   
    if (nvi > globals.params_int["thres"] || nfj > globals.params_int["thres"]) {
        auto &EIs{nvi > nfj? viaabbs: fjaabbs}, &EJs{nvi > nfj? fjaabbs: viaabbs};
        auto bvh = bvh_create(EIs.data(), EIs.size());
        for (int J = 0; J < EJs.size(); J ++){
            int I;
            auto &aabbj {EJs[J]};
            auto query = bvh_query_aabb(uint64_t(&bvh), aabbj.lower, aabbj.upper);
            while (bvh_query_next(query, I)){
                int i, j;
                if (nvi > nfj) 
                    {i = I; j = J;}
                else 
                    {i = J; j = I;}

                auto &v0{ v0s[i] }, &v{ v1s[i] };
                auto &f0{ f0s[j] }, &f{ f1s[j] };
                scalar t = pt_collision_time(v0, f0, v, f);
                toi = min(toi, t);
            }
        }
        bvh_destroy_host(bvh);
    } else
    for (int i = 0; i < vilist.size(); i++)
        for (int j = 0; j < fjlist.size(); j++) {
            int vi = vilist[i], fj = fjlist[j];
            if (intersects(viaabbs[i], fjaabbs[j])) {
                auto &v0{ v0s[i] }, &v{ v1s[i] };
                auto &f0{ f0s[j] }, &f{ f1s[j] };
                scalar t = pt_collision_time(v0, f0, v, f);
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
}

void vf_col_set(vector<int>& vilist, vector<int>& fjlist,
                            const std::vector<std::unique_ptr<AffineBody>>& cubes,
                            int I, int J,
                            vector<array<vec3, 4>>& pts,
                            vector<array<int, 4>>& idx) {
    auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    vector<lu> viaabbs, fjaabbs;
    vector<vec3> vis;
    vector<Face> fjs;
    
    viaabbs.clear();
    fjaabbs.clear();
    vis.clear();
    fjs.clear();

    viaabbs.reserve(vilist.size());
    fjaabbs.reserve(fjlist.size());
    vis.reserve(vilist.size());
    fjs.reserve(fjlist.size());

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

    int nvi = vilist.size(), nfj = fjlist.size();   
    if (nvi > globals.params_int["thres"] || nfj > globals.params_int["thres"]) {
        auto &EIs{nvi > nfj? viaabbs: fjaabbs}, &EJs{nvi > nfj? fjaabbs: viaabbs};
        auto bvh = bvh_create(EIs.data(), EIs.size());
        for (int jj = 0; jj < EJs.size(); jj ++){
            int ii;
            auto &aabbj {EJs[jj]};
            auto query = bvh_query_aabb(uint64_t(&bvh), aabbj.lower, aabbj.upper);
            while (bvh_query_next(query, ii)){
                int i, j;
                if (nvi > nfj) 
                    {i = ii; j = jj;}
                else 
                    {i = jj; j = ii;}


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
        }
        bvh_destroy_host(bvh);
    } else

    for (int i = 0; i < vilist.size(); i ++)
        for (int j = 0; j < fjlist.size(); j ++){
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
}

void ee_col_set(vector<int>& eilist, vector<int>& ejlist,
                            const std::vector<std::unique_ptr<AffineBody>>& cubes,
                            int I, int J,
                            vector<array<vec3, 4>>& ees,
                            vector<array<int, 4>>& eidx) {
    auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    vector<lu> eiaabbs, ejaabbs;
    vector<Edge> eis, ejs;

    eiaabbs.clear();
    ejaabbs.clear();
    eis.clear();
    ejs.clear();
    
    eiaabbs.reserve(eilist.size());
    ejaabbs.reserve(ejlist.size());
    eis.reserve(eilist.size());
    ejs.reserve(ejlist.size());
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

    int nei = eilist.size(), nej = ejlist.size();   
    if (nei > globals.params_int["thres"] || nej > globals.params_int["thres"]) {
        auto &EIs{nei > nej? eiaabbs: ejaabbs}, &EJs{nei > nej? ejaabbs: eiaabbs};
        auto bvh = bvh_create(EIs.data(), EIs.size());
        for (int jj = 0; jj < EJs.size(); jj ++){
            int ii;
            auto &aabbj {EJs[jj]};
            auto query = bvh_query_aabb(uint64_t(&bvh), aabbj.lower, aabbj.upper);
            while (bvh_query_next(query, ii)){
                int i, j;
                if (nei > nej) 
                    {i = ii; j = jj;}
                else 
                    {i = jj; j = ii;}

                auto &eii{ eis[i] }, &ejj{ ejs[j] };
                int ei = eilist[i], ej = ejlist[j];
                auto ee_type = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
                scalar d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
                if (d < barrier::d_hat) {
                    array<vec3, 4> ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
                    array<int, 4> ij = { I, ei, J, ej };
                    {
                        ees.push_back(ee);
                        eidx.push_back(ij);
                    }
                }

            }
        }
        bvh_destroy_host(bvh);
    } else
    for (int i = 0; i < eilist.size(); i++)
        for (int j = 0; j < ejlist.size(); j++) {

            bool ee_intersects = intersects(eiaabbs[i], ejaabbs[j]);
            if (!ee_intersects) continue;
            auto &eii{ eis[i] }, &ejj{ ejs[j] };
            int ei = eilist[i], ej = ejlist[j];
            auto ee_type = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
            scalar d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
            if (d < barrier::d_hat) {
                array<vec3, 4> ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
                array<int, 4> ij = { I, ei, J, ej };
                {
                    ees.push_back(ee);
                    eidx.push_back(ij);
                }
            }
        }
}


scalar primitive_brute_force(
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

    scalar toi_global = 1.0, toi_ee_pt = 1.0;
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
        scalar toi_thread_local = 1.0;
        auto tid = omp_get_thread_num();
        vidx_thread_local[tid].resize(0);
// #pragma omp for schedule(static)
        for (int I = 0; I < n_cubes; I++) {
            auto& c{ *cubes[I] };
            for (int v = 0; v < c.n_vertices; v++) {
                auto& p{ c.v_transformed[v] };
                // handling vertex-ground collision
                if (!cull_trajectory) {
                    scalar d = vg_distance(p);
                    d = d * d;
                    if (d < barrier::d_hat) {
                        vidx_thread_local[tid].push_back({ I, v });
                    }
                }
                else {
                    scalar t = collision_time(c, v);
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
                    auto &t{ overlaps[o] };
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
#endif
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

    scalar ee_global = 1.0, pt_global = 1.0;
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
            }) - inner_presum.begin() - 1;

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
            vf_col_set(vilist, fjlist, cubes, I, J, pts_private[tid], idx_private[tid]);
            vf_col_set(vjlist, filist, cubes, J, I, pts_private[tid], idx_private[tid]);
        }
#pragma omp critical
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
        scalar toi = 1.0;
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
            scalar t1 = vf_col_time(vilist, fjlist, cubes, I, J, vertex_starting_index, vt1_buffer);
            scalar t2 = vf_col_time(vjlist, filist, cubes, J, I, vertex_starting_index, vt1_buffer);
            scalar t3 = ee_col_time(eilist, ejlist, cubes, I, J, vertex_starting_index, vt1_buffer);

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

scalar iaabb_brute_force(
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
    scalar toi = primitive_brute_force(n_cubes, ret, cubes, vtn,
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
    spdlog::info("time: {} = {:0.6f} ms", vtn == 3 ? "iaabb upper bound": "iAABB", t * 1000);
    return toi;
}
