
#include "iaabb.h"
#include "geometry.h"
#include <memory>
#include <array>
#include <algorithm>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "barrier.h"
#include "time_integrator.h"
#ifndef TESTING
#include "../view/global_variables.h"
#endif
using namespace std;
using namespace Eigen;
using namespace utils;
using lu = std::array<vec3, 2>;
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
    return { l, u };
}

lu affine(const lu& aabb, q4& q)
{
    Matrix<double, 3, 8> cull;
    vec3 l{ aabb[0] }, u{ aabb[1] };
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                auto I = (i << 2) | (j << 1) | k;
                cull(0, I) = i ? u(0) : l(0);
                cull(1, I) = j ? u(1) : l(1);
                cull(2, I) = k ? u(2) : l(2);
            }
    mat3 A;
    A << q[1], q[2], q[3];
    cull = A * cull;
    l = cull.rowwise().minCoeff();
    u = cull.rowwise().maxCoeff();
    return { l + q[0], u + q[0] };
}

bool intersection(const lu& a, const lu& b, lu& ret)
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
    if (vtn == 2)
        for (int i = 0; i < 4; i++) qi[i] += c.dq.segment<3>(i * 3);
    return affine(aabb, qi);
};

void intersect_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    std::vector<Intersection> &ret,
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
                ret.push_back({ i, j, c });
                ret.push_back({ j, i, c });
            }
        }
}

void intersect_sort(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu> &aabbs,
    std::vector<Intersection> &ret,
    int vtn)
{
    ret.resize(0);
    vector<BoundingBox> bounds[3];
    vector<Intersection> ilists[3];
    for (int dim = 0; dim < 3; dim++) {
        bounds[dim].resize(n_cubes * 2);
        for (int i = 0; i < n_cubes; i++) {
            auto t{ affine(aabbs[i], *cubes[i], vtn) };
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
                for (auto it = active.begin(); it != active.end(); ) {
                    if (*it == body) {
                        it = active.erase(it);
                        continue;
                    }
                    auto c{ *it };
                    lu cull;
                    auto bi = affine(aabbs[c], *cubes[c], vtn);
                    auto bj = affine(aabbs[body], *cubes[body], vtn);
                    
                    intersection(bi, bj, cull);
                    // ret.push_back({ min(c, body), max(c, body), cull });
                    ilists[dim].push_back({ c, body, cull });
                    ilists[dim].push_back({ body, c, cull });
                    ++it;
                }
            }
        }
        const auto les = [](const Intersection& a, const Intersection& b) -> bool {
            return a.i < b.i || (a.i == b.i && a.j < b.j);
        };
        sort(ilists[dim].begin(), ilists[dim].end(), les);
    }
    vector<Intersection> tmp;
    std::set_intersection(ilists[0].begin(), ilists[0].end(), ilists[1].begin(), ilists[1].end(), std::back_inserter(tmp)); 
    std::set_intersection(tmp.begin(), tmp.end(), ilists[2].begin(), ilists[2].end(), std::back_inserter(ret)); 
}

inline bool filter_if_inside(lu& a, vec3& p)
{
    vec3 l = a[0], u = a[1];
    return (l.array() <= p.array()).all() && (u.array() >= p.array()).all();
}
inline bool filter_if_inside(lu& a, Edge& e)
{
    bool b1 = filter_if_inside(a, e.e0),
         b2 = filter_if_inside(a, e.e1);
    return b1 || b2;
    // FIXME: true if bounding box intersects
}
inline bool filter_if_inside(lu& a, Face& f)
{
    return filter_if_inside(a, f.t0) || filter_if_inside(a, f.t1) || filter_if_inside(a, f.t2);
    // FIXME:
}

//void primitive_brute_force(
//    int n_cubes,
//    const std::vector<Intersection>& overlaps,
//    const std::vector<std::unique_ptr<AffineBody>>& cubes,
//    int vtn,
//    std::vector<int>* ovi,
//    std::vector<int>* ovj,
//    std::vector<int>* oei,
//    std::vector<int>* oej,
//    std::vector<int>* ofi,
//    std::vector<int>* ofj,
//    vector<array<vec3, 4>>& pts,
//    vector<array<int, 4>>& idx,
//    vector<array<vec3, 4>>& ees,
//    vector<array<int, 4>>& eidx,
//    vector<array<int, 2>>& vidx,
//    vector<Matrix<double, 2, 12>>& pt_tk,
//    vector<Matrix<double, 2, 12>>& ee_tk,
//    bool gen_basis)
//{
//    Intersection a{ 0, 0, lu{ vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0) } }, b;
//    b = a;
//    bool t = b == a, t1 = b < a;
//    //std::sort(overlaps.begin(), overlaps.end(),
//    //    [](const Intersection& a, const Intersection &b) -> bool{
//    //        return a.i < b.i || (a.i == b.i && a.j < b.j);
//    //    });
//
//    int n_overlap = overlaps.size();
//    // vector<int>*ov, *oei, *oej, *of;
//
//    ovi = new vector<int>[n_overlap];
//    ovj = new vector<int>[n_overlap];
//    oei = new vector<int>[n_overlap];
//    oej = new vector<int>[n_overlap];
//    ofi = new vector<int>[n_overlap];
//    ofj = new vector<int>[n_overlap];
//
//    int* starting = new int[n_cubes + 1];
//    starting[n_cubes] = n_overlap;
//
//    int old_cube_index = 0;
//    for (int i = 0; i < n_overlap; i++) {
//
//        auto& o{ overlaps[i] };
//        if (o.i != old_cube_index) {
//            for (int j = old_cube_index + 1; j <= o.i; j++)
//                starting[j] = i;
//            old_cube_index = o.i;
//        }
//    }
//
//    for (int I = 0; I < n_cubes; I++) {
//        // construct the vertex, edge, and triangle list inside each overlap
//        auto& c{ *cubes[I] };
//        switch (vtn) {
//        case 0: c.project_vt0(); break;
//        case 1: c.project_vt1(); break;
//        default: c.project_vt2();
//        }
//        for (int v = 0; v < c.n_vertices; v++) {
//            auto& p{ c.v_transformed[v] };
//            for (int o = starting[I]; o < starting[I + 1]; o++) {
//                lu cull = overlaps[o].cull;
//
//                if (filter_if_inside(cull, p)) {
//                    auto t{ overlaps[o] };
//
//                    if (t.i < t.j)
//                        ovi[o].push_back(v);
//                    else
//                        ovj[o].push_back(v);
//                }
//            }
//        }
//
//        for (int e = 0; e < c.n_edges; e++) {
//            Edge ei{ c, unsigned(e), true, true };
//            // TODO: always intialize from v_transformed
//
//            for (int o = starting[I]; o < starting[I + 1]; o++) {
//                lu cull = overlaps[o].cull;
//
//                if (filter_if_inside(cull, ei)) {
//                    auto t{ overlaps[o] };
//                    if (t.i < t.j)
//                        oei[o].push_back(e);
//                    else
//                        oej[o].push_back(e);
//                }
//            }
//        }
//
//        for (int f = 0; f < c.n_vertices; f++) {
//            Face fi{ c, unsigned(f), true, true };
//            // TODO: always intialize from v_transformed
//            for (int o = starting[I]; o < starting[I + 1]; o++) {
//                lu cull = overlaps[o].cull;
//
//                if (filter_if_inside(cull, fi)) {
//                    auto t{ overlaps[o] };
//                    if (t.i < t.j)
//                        ofi[o].push_back(f);
//                    else
//                        ofj[o].push_back(f);
//                }
//            }
//        }
//    }
//
//    for (int i = 0; i < n_overlap; i++) {
//        int I{ overlaps[i].i }, J{ overlaps[i].j };
//        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
//        auto& vilist{ ovi[i] };
//        auto& vjlist{ ovi[i] };
//        auto& eilist{ oei[i] };
//        auto& ejlist{ oej[i] };
//        auto& filist{ ofi[i] };
//        auto& fjlist{ ofj[i] };
//        const auto vf_col_set = [&](vector<int>& vilist, vector<int>& fjlist,
//                                    const std::vector<std::unique_ptr<AffineBody>>& cubes) {
//            for (int vi : vilist)
//                for (int fj : fjlist) {
//                    vec3 v{ ci.v_transformed[vi] };
//                    Face f{ cj, unsigned(fj), true, true };
//                    ipc::PointTriangleDistanceType pt_type;
//                    double d = vf_distance(v, f, pt_type);
//                    if (d < barrier::d_hat) {
//                        array<vec3, 4> pt = { v, f.t0, f.t1, f.t2 };
//                        array<int, 4> ij = { I, vi, J, fj };
//                        Matrix<double, 2, 12> Tk_T;
//                        Tk_T.setZero(2, 12);
//                        Vector2d uk;
//                        if (gen_basis)
//                            pt_uktk(*cubes[I], cj, pt, ij, pt_type, Tk_T, uk, d, globals.dt);
//                        {
//                            pts.push_back(pt);
//                            idx.push_back(ij);
//#ifdef _FRICTION_
//                            pt_tk.push_back(Tk_T);
//#endif
//                        }
//                    }
//                }
//        };
//        for (auto ei : eilist)
//            for (auto ej : ejlist) {
//                Edge eii{ ci, unsigned(ei), true, true };
//                Edge ejj{ cj, unsigned(ej), true, true };
//
//                auto ee_type = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
//                double d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
//                if (d < barrier::d_hat) {
//                    array<vec3, 4> ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
//                    array<int, 4> ij = { I, ei, J, ej };
//                    Matrix<double, 2, 12> Tk_T;
//                    Tk_T.setZero(2, 12);
//                    Vector2d uk;
//                    if (gen_basis)
//                        ee_uktk(*cubes[I], cj, ee, ij, ee_type, Tk_T, uk, d, globals.dt);
//                    {
//                        ees.push_back(ee);
//                        eidx.push_back(ij);
//#ifdef _FRICTION_
//                        ee_tk.push_back(Tk_T);
//#endif
//                    }
//                }
//                vf_col_set(vilist, fjlist, cubes);
//                vf_col_set(vjlist, filist, cubes);
//            }
//    }
//}
