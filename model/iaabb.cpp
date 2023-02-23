
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
#ifndef TESTING
#include "../view/global_variables.h"
#define DT globals.dt
#else
#define DT 1e-2
#endif
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace utils;
using namespace std::chrono;
using lu = std::array<vec3, 2>;
#define DURATION_TO_DOUBLE(X) duration_cast<duration<double>>(high_resolution_clock::now() - (X)).count()
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
lu compute_aabb(const Edge& e)
{
    vec3 l, u;
    l = e.e0.array().min(e.e1.array());
    u = e.e0.array().max(e.e1.array());
    return { l, u };
}

lu compute_aabb(const Face& f)
{
    vec3 l, u;
    l = f.t0.array().min(f.t1.array()).min(f.t2.array());
    u = f.t0.array().max(f.t1.array()).max(f.t2.array());
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
    ret.resize(0);
    vector<BoundingBox> bounds[3];
    vector<Intersection> ilists[3];
#pragma omp parallel for schedule(static, 1)
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
                for (auto it = active.begin(); it != active.end();) {
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
                    ilists[dim].push_back({ c, body, cull, nullptr });
                    ilists[dim].push_back({ body, c, cull, nullptr });
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
    // TODO: change bounds[3] to static for reuse;
    // TODO: O(n) insertion sort
}

inline bool filter_if_inside(lu& a, vec3& p)
{
    vec3 l = a[0], u = a[1];
    return (l.array() <= p.array()).all() && (u.array() >= p.array()).all();
}
inline bool filter_if_inside(lu& a, Edge& e)
{
    lu ret;
    return intersection(a, compute_aabb(e), ret);
    // bool b1 = filter_if_inside(a, e.e0),
    //      b2 = filter_if_inside(a, e.e1);
    // return b1 || b2;
}
inline bool filter_if_inside(lu& a, Face& f)
{
    lu ret;
    return intersection(a, compute_aabb(f), ret);
    // return filter_if_inside(a, f.t0) || filter_if_inside(a, f.t1) || filter_if_inside(a, f.t2);
}

void primitive_brute_force(
    int n_cubes,
    std::vector<Intersection>& overlaps, // assert sorted
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int vtn,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx,
    vector<Matrix<double, 2, 12>>& pt_tk,
    vector<Matrix<double, 2, 12>>& ee_tk,
    bool gen_basis)
{
    pts.resize(0);
    idx.resize(0);
    ees.resize(0);
    eidx.resize(0);
    vidx.resize(0);
    pt_tk.resize(0);
    ee_tk.resize(0);
    int n_overlap = overlaps.size();

    auto lists = new PList[n_overlap];
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

//#pragma omp parallel for schedule(guided)
    for (int I = 0; I < n_cubes; I++) {
        // construct the vertex, edge, and triangle list inside each overlap
        auto& c{ *cubes[I] };
        switch (vtn) {
        case 0: c.project_vt0(); break;
        case 1: c.project_vt1(); break;
        default: c.project_vt2();
        }
        for (int v = 0; v < c.n_vertices; v++) {
            auto& p{ c.v_transformed[v] };
            for (int o = starting[I]; o < starting[I + 1]; o++) {
                lu cull = overlaps[o].cull;
                overlaps[o].plist = lists + o;
                if (filter_if_inside(cull, p)) {
                    auto &t{ overlaps[o] };
                    assert(t.i == I);
                    if (t.i < t.j)
                        lists[o].vi.push_back(v);
                    else
                        lists[o].vj.push_back(v);
                }
            }
            double d = vg_distance(p);
            d = d * d;
            if (d < barrier::d_hat) {
                vidx.push_back({ I, v });
            }
        }

        for (int e = 0; e < c.n_edges; e++) {
            Edge ei{ c, unsigned(e), true, true };
            // TODO: always intialize from v_transformed

            for (int o = starting[I]; o < starting[I + 1]; o++) {
                lu cull = overlaps[o].cull;

                if (filter_if_inside(cull, ei)) {
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
                lu cull = overlaps[o].cull;

                if (filter_if_inside(cull, fi)) {
                    auto t{ overlaps[o] };
                    if (t.i < t.j)
                        lists[o].fi.push_back(f);
                    else
                        lists[o].fj.push_back(f);
                }
            }
        }
    }
    sort(overlaps.begin(), overlaps.end(), [](const Intersection& a, const Intersection& b) -> bool{
        auto ad = a.i + a.j, am = abs(a.i - a.j);
        auto bd = b.i + b.j, bm = abs(b.i - b.j);

        return ad < bd || (ad == bd && am < bm) || (ad == bd && am == bm && a.i < b.i);
    });
    

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

    //#pragma omp parallel for schedule(guided)
    for (int _i = 0; _i < n_overlap / 2; _i++) {
        int i = _i * 2;
        int I{ overlaps[i].i }, J{ overlaps[i].j };
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        auto& p{ *overlaps[i].plist };
        auto& vilist{ p.vi };
        auto& vjlist{ p.vj };
        auto& eilist{ p.ei };
        auto& ejlist{ p.ej };
        auto& filist{ p.fi };
        auto& fjlist{ p.fj };
        const auto vf_col_set = [&](vector<int>& vilist, vector<int>& fjlist,
                                    const std::vector<std::unique_ptr<AffineBody>>& cubes, int I, int J) {
            auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
            for (int vi : vilist)
                for (int fj : fjlist) {
                    vec3 v{ ci.v_transformed[vi] };
                    Face f{ cj, unsigned(fj), true, true };
                    ipc::PointTriangleDistanceType pt_type;
                    double d = vf_distance(v, f, pt_type);
                    if (d < barrier::d_hat) {
                        array<vec3, 4> pt = { v, f.t0, f.t1, f.t2 };
                        array<int, 4> ij = { I, vi, J, fj };
                        Matrix<double, 2, 12> Tk_T;
                        Tk_T.setZero(2, 12);
                        Vector2d uk;
#ifndef TESTING
                        if (gen_basis)
                            pt_uktk(*cubes[I], cj, pt, ij, pt_type, Tk_T, uk, d, DT);
#endif
                        {
                            pts.push_back(pt);
                            idx.push_back(ij);
#ifdef _FRICTION_
                            pt_tk.push_back(Tk_T);
#endif
                        }
                    }
                }
        };
        for (auto ei : eilist)
            for (auto ej : ejlist) {
                Edge eii{ ci, unsigned(ei), true, true };
                Edge ejj{ cj, unsigned(ej), true, true };

                auto ee_type = ipc::edge_edge_distance_type(eii.e0, eii.e1, ejj.e0, ejj.e1);
                double d = ipc::edge_edge_distance(eii.e0, eii.e1, ejj.e0, ejj.e1, ee_type);
                if (d < barrier::d_hat) {
                    array<vec3, 4> ee = { eii.e0, eii.e1, ejj.e0, ejj.e1 };
                    array<int, 4> ij = { I, ei, J, ej };
                    Matrix<double, 2, 12> Tk_T;
                    Tk_T.setZero(2, 12);
                    Vector2d uk;
#ifndef TESTING
                    if (gen_basis)
                        ee_uktk(*cubes[I], cj, ee, ij, ee_type, Tk_T, uk, d, DT);
#endif
                    {
                        ees.push_back(ee);
                        eidx.push_back(ij);
#ifdef _FRICTION_
                        ee_tk.push_back(Tk_T);
#endif
                    }
                }
            }
        vf_col_set(vilist, fjlist, cubes, I, J);
        vf_col_set(vjlist, filist, cubes, J, I);
    }
}

void iaabb_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    int vtn,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx,
    std::vector<Eigen::Matrix<double, 2, 12>>& pt_tk,
    std::vector<Eigen::Matrix<double, 2, 12>>& ee_tk,
    bool gen_basis)
{
    auto start = high_resolution_clock::now();
    vector<Intersection> ret;
    intersect_sort(n_cubes, cubes, aabbs, ret, vtn);
    primitive_brute_force(n_cubes, ret, cubes, vtn,
        pts,
        idx,
        ees,
        eidx,
        vidx,
        pt_tk,
        ee_tk,
        gen_basis);
    auto t = DURATION_TO_DOUBLE(start);
    spdlog::info("time: iAABB = {:0.6f} ms", t * 1000);
}
