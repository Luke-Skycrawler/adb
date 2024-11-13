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
#include "ipc_extension.h"
#define CUDA_ENABLED
using namespace Eigen;
using namespace std;

namespace cuda {

scalar cuda_pt_list_toi(int nvi, int nfj, int* vilist, int* fjlist, lu* viaabbs, lu* fjaabbs, vec3* v0s, vec3* v1s, Face* f0s, Face* f1s);

scalar cuda_pt_list_toi(vector<int> vilist, vector<int> fjlist, vector<lu> viaabbs, vector<lu> fjaabbs, vector<vec3> v0s, vector<vec3> v1s, vector<Face> f0s, vector<Face> f1s)
{
    return cuda_pt_list_toi(vilist.size(), fjlist.size(), vilist.data(), fjlist.data(), viaabbs.data(), fjaabbs.data(), v0s.data(), v1s.data(), f0s.data(), f1s.data());
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
        Edge ei1(ci.edge(ei, true, true));
        int i0, i1;
        i0 = ci.edges[ei * 2];
        i1 = ci.edges[ei * 2 + 1];
        Edge ei0{ vt1_buffer[offi + i0], vt1_buffer[offi + i1] };
        ei0s.push_back(ei0);
        ei1s.push_back(ei1);
        eiaabbs.push_back(compute_aabb(ei0, ei1));
    }
    for (auto& ej : ejlist) {
        Edge ej1(cj.edge(ej, true, true));
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
        Face f{ cj.face(int(fj), true, true) };
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
    #ifdef CUDA_ENABLED 
    toi = cuda::cuda_pt_list_toi(vilist, fjlist, viaabbs, fjaabbs, v0s, v1s, f0s, f1s);
    #else
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
    #endif
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
        Face f{ cj.face(int(fj), true, true) };
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
                    q4 pt = { v, f.t0, f.t1, f.t2 };
                    i4 ij = { I, vi, J, fj };
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
        Edge eii{ ci.edge(int(ei), true, true) };
        eis.push_back(eii);
        eiaabbs.push_back(compute_aabb(eii, barrier::d_sqrt));
    }
    for (auto& ej : ejlist) {
        Edge ejj{ cj.edge(int(ej), true, true) };
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
                    q4 ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
                    i4 ij = { I, ei, J, ej };
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
            ee_col_set_task(eilist[i], ejlist[j], I, J, eis[i], ejs[j], eiaabbs[i], ejaabbs[j], ees, eidx);
        }
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

