#include "affine_body.h"
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include "thread_local_culling.h"
#include "geometry.h"
#include "barrier.h"
#include "../view/global_variables.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "collision.h"

using namespace Eigen;
using namespace std;

void prepare_aabbs_ei(vector<int> eilist, AffineBody &ci, vector<vec3>& vt1_buffer, int offi, vector<Edge>& ei0s, vector<Edge>& ei1s, vector<lu>& eiaabbs, scalar dialation = 0.0){

    eiaabbs.clear();
    ei0s.clear();
    ei1s.clear();

    eiaabbs.reserve(eilist.size());
    ei0s.reserve(eilist.size());
    ei1s.reserve(eilist.size());

    for (auto& ei : eilist) {
        Edge ei1(ci, ei, true, true);
        int i0, i1;
        i0 = ci.edges[ei * 2];
        i1 = ci.edges[ei * 2 + 1];
        Edge ei0{ vt1_buffer[offi + i0], vt1_buffer[offi + i1] };
        ei0s.push_back(ei0);
        ei1s.push_back(ei1);
        eiaabbs.push_back(dialate(compute_aabb(ei0, ei1), dialation));
    }
}

void prepare_aabbs_v(vector<int> vilist, AffineBody &ci, vector<vec3>& vt1_buffer, int offi, vector<vec3>& v0s, vector<vec3> &v1s, vector<lu> &viaabbs, scalar dialation = 0.0){

    viaabbs.clear();
    v0s.clear();
    v1s.clear();

    viaabbs.reserve(vilist.size());
    v0s.reserve(vilist.size());
    v1s.reserve(vilist.size());

    for (auto& vi : vilist) {
        vec3 v1{ ci.v_transformed[vi] };
        vec3 v0{ vt1_buffer[offi + vi] };
        v0s.push_back(v0);
        v1s.push_back(v1);
        viaabbs.push_back(dialate(compute_aabb(v0, v1), dialation));
    }
}

void prepare_aabbs_f(vector<int> fjlist, AffineBody &cj, vector<vec3> &vt1_buffer, int offj, vector<Face> &f0s, vector<Face> &f1s, vector<lu> &fjaabbs, scalar dialation = 0.0){
    fjaabbs.clear();
    f0s.clear();
    f1s.clear();
    fjaabbs.reserve(fjlist.size());
    f0s.reserve(fjlist.size());
    f1s.reserve(fjlist.size());

    for (auto& fj : fjlist) {
        Face f{ cj, unsigned(fj), true, true };
        int _a, _b, _c;
        _a = cj.indices[fj * 3 + 0],
        _b = cj.indices[fj * 3 + 1],
        _c = cj.indices[fj * 3 + 2];

        Face f0{ vt1_buffer[offj + _a], vt1_buffer[offj + _b], vt1_buffer[offj + _c] };

        f0s.push_back(f0);
        f1s.push_back(f);
        fjaabbs.push_back(dialate(compute_aabb(f0, f), dialation));
    }

}


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

    prepare_aabbs_ei(eilist, ci, vt1_buffer, offi, ei0s, ei1s, eiaabbs);
    prepare_aabbs_ei(ejlist, cj, vt1_buffer, offj, ej0s, ej1s, ejaabbs);
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

    prepare_aabbs_v(vilist, ci, vt1_buffer, offi, v0s, v1s, viaabbs);
    prepare_aabbs_f(fjlist, cj, vt1_buffer, offj, f0s, f1s, fjaabbs);

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
                toi = min(toi, t);
            }
        }
    return toi;
}

void vf_col_set(vector<int>& vilist, vector<int>& fjlist,
                            const std::vector<std::unique_ptr<AffineBody>>& cubes,
                            int I, int J,
                            vector<q4>& pts,
                            vector<i4>& idx) {
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

                pt_col_set_task(vilist[i], fjlist[j], I, J, vis[i], fjs[j], viaabbs[i], fjaabbs[j], pts, idx);
            }
        }
        bvh_destroy_host(bvh);
    } else

    for (int i = 0; i < vilist.size(); i ++)
        for (int j = 0; j < fjlist.size(); j ++){
            pt_col_set_task(vilist[i], fjlist[j], I, J, vis[i], fjs[j], viaabbs[i], fjaabbs[j], pts, idx);
        }
}

void ee_col_set(vector<int>& eilist, vector<int>& ejlist,
                            const std::vector<std::unique_ptr<AffineBody>>& cubes,
                            int I, int J,
                            vector<q4>& ees,
                            vector<i4>& eidx) {
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

                ee_col_set_task(eilist[i], ejlist[j], I, J, eis[i], ejs[j], eiaabbs[i], ejaabbs[j], ees, eidx);
            }
        }
        bvh_destroy_host(bvh);
    } else
    for (int i = 0; i < eilist.size(); i++)
        for (int j = 0; j < ejlist.size(); j++) {
            ee_col_set_task(eilist[i], ejlist[j], I, J, eis[i], ejs[j], eiaabbs[i], ejaabbs[j], ees, eidx);
        }
}

scalar pt_col_set_and_time(
    std::vector<int> vilist, std::vector<int> fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J,
    std::vector<int> &vertex_starting_index,
    std::vector<vec3> &vt1_buffer,
    std::vector<q4> &pts,
    std::vector<i4> &idx
) {
    auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    int offi{ vertex_starting_index[I] }, offj{ vertex_starting_index[J] }; 

    scalar toi = 1.0;
    vector<lu> viaabbs, fjaabbs;
    vector<vec3> v0s, v1s;
    vector<Face> f0s, f1s;

    prepare_aabbs_v(vilist, ci, vt1_buffer, offi, v0s, v1s, viaabbs, barrier::d_sqrt * 0.5);
    prepare_aabbs_f(fjlist, cj, vt1_buffer, offj, f0s, f1s, fjaabbs, barrier::d_sqrt * 0.5);

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

                pt_task(vilist[i], fjlist[j], I, J, v0, f0, v, f, pts, idx);
            }
        }
        bvh_destroy_host(bvh);
    } else
    for (int i = 0; i < vilist.size(); i++)
        for (int j = 0; j < fjlist.size(); j++) {
            if (intersects(viaabbs[i], fjaabbs[j])) {
                auto &v0{ v0s[i] }, &v{ v1s[i] };
                auto &f0{ f0s[j] }, &f{ f1s[j] };
                scalar t = pt_collision_time(v0, f0, v, f);
                toi = min(toi, t);

                pt_task(vilist[i], fjlist[j], I, J, v0, f0, v, f, pts, idx);
            }
        }
    return toi;

}

scalar ee_col_set_and_time(
    std::vector<int> eilist, std::vector<int> ejlist, 
    const std::vector<std::unique_ptr<AffineBody>>& cubes,  
    int I, int J,   
    std::vector<int> &vertex_starting_index, 
    std::vector<vec3> &vt1_buffer,
    std::vector<q4> &ees, 
    std::vector<i4> &eidx
) {
    auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
    int offi{ vertex_starting_index[I] }, offj{ vertex_starting_index[J] };

    scalar toi = 1.0;
    vector<lu> eiaabbs, ejaabbs;
    vector<Edge> ei0s, ej0s, ei1s, ej1s;

    
    prepare_aabbs_ei(eilist, ci, vt1_buffer, offi, ei0s, ei1s, eiaabbs, barrier:: d_sqrt * 0.5);
    prepare_aabbs_ei(ejlist, cj, vt1_buffer, offj, ej0s, ej1s, ejaabbs, barrier:: d_sqrt * 0.5);
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
                ee_task(ei, ej, I, J, ei0, ej0, ei1, ej1, ees, eidx);
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
                toi = min(toi, t);
                ee_task(ei, ej, I, J, ei0, ej0, ei1, ej1, ees, eidx);
            }
        }

    return toi;
}
void pt_col_set_task(
    int vi, int fj, int I, int J,
    // const AffineBody& ci, const AffineBody& cj,
    const vec3& v, const Face& f,
    const lu& aabb_i, const lu& aabb_j,
    vector<q4>& pts,
    vector<i4>& idx)
{
    bool pt_intersects = intersects(aabb_i, aabb_j);
    if (!pt_intersects) return;
    auto [d, pt_type] = vf_distance(v, f);
    if (d < barrier::d_hat) {
        q4 pt = { v, f.t0, f.t1, f.t2 };
        i4 ij = { I, vi, J, fj };
        {
            pts.push_back(pt);
            idx.push_back(ij);
        }
    }
}
void ee_col_set_task(
    int ei, int ej, int I, int J,
    // const AffineBody& ci, const AffineBody& cj,
    const Edge& eii, const Edge& ejj,
    const lu& aabb_i, const lu& aabb_j, vector<q4>& ees,
    vector<i4>& eidx)
{
    bool ee_intersects = intersects(aabb_i, aabb_j);
    if (!ee_intersects) return;
    auto ee_type = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
    scalar d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
    if (d < barrier::d_hat) {
        q4 ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
        i4 ij = { I, ei, J, ej };

        {
            ees.push_back(ee);
            eidx.push_back(ij);
        }
    }
}

void pt_task(
    int vi, int fj, int I, int J,
    const vec3 &v0, const Face &f0,
    const vec3 &v1, const Face &f1,
    std::vector<q4> pts,
    std::vector<i4> idx
) {
    auto [d0, pt_type0] = vf_distance(v0, f0);
    auto [d1, pt_type1] = vf_distance(v1, f1);
    scalar d_p = (v1 - v0).norm();
    scalar d_f0 = (f1.t0 - f0.t0).norm();
    scalar d_f1 = (f1.t1 - f0.t1).norm();
    scalar d_f2 = (f1.t2 - f0.t2).norm();

    scalar L = d_p + max(d_f0, max(d_f1, d_f2));
    scalar min_d = min(d0, d1) - 0.5 * (L - abs(d1 - d0));
    if (min_d < barrier::d_hat) {
        q4 pt = { v0, f0.t0, f0.t1, f0.t2 };
        i4 ij = { I, vi, J, fj };
        {
            pts.push_back(pt);
            idx.push_back(ij);
        }
    }
}
void ee_task(
    int ei, int ej, int I, int J, 
    const Edge &eii, const Edge &ejj,
    const Edge &ei1, const Edge &ej1,
    std::vector<q4> ees,
    std::vector<i4> eidx
) {
    auto ee_type0 = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
    auto ee_type1 = ipc::edge_edge_distance_type(ei1.e0, ei1.e1, ej1.e0, ej1.e1);   
    
    scalar d0 = sqrt(ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type0));
    scalar d1 = sqrt(ipc::edge_edge_distance(ei1.e0, ei1.e1, ej1.e0, ej1.e1, ee_type1));
    
    scalar d_ei0 = (ei1.e0 - eii.e0).norm();
    scalar d_ei1 = (ei1.e1 - eii.e1).norm();
    scalar d_ej0 = (ej1.e0 - ejj.e0).norm();
    scalar d_ej1 = (ej1.e1 - ejj.e1).norm();

    scalar L = max(d_ei0, d_ei1) + max(d_ej0, d_ej1);
    scalar min_d = min(d0, d1) - 0.5 * (L - abs(d1 - d0));

    if (min_d < barrier::d_sqrt) {
        q4 ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
        i4 ij = { I, ei, J, ej };
        {
            ees.push_back(ee);
            eidx.push_back(ij);
        }
    }
}