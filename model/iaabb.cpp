
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
    l = a[0].array().min(b[0].array());
    u = a[1].array().max(b[1].array());
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
    static vector<int>*ilists[3] = {
        new vector<int>[n_cubes],
        new vector<int>[n_cubes],
        new vector<int>[n_cubes]
    },
           *tmp = new vector<int>[n_cubes];
    static vector<Intersection>* ret_tmp = new vector<Intersection>[n_cubes];

    static vector<lu> affine_bb;
    // static vector<Intersection> tmp;
    ret.resize(0);
    
    affine_bb.resize(n_cubes);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_cubes; i++) {
        auto t{ affine(aabbs[i], *cubes[i], vtn) };
        affine_bb[i] = t;
        tmp[i].resize(0);
        ret_tmp[i].resize(0);
        for (int dim = 0; dim < 3; dim ++)
        ilists[dim][i].resize(0);
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
                ilists[dim][body].insert(ilists[dim][body].end(), active.begin(), active.end());
                for (auto c : active) {
                    ilists[dim][c].push_back(body);
                }
            }
        }
    }

#pragma omp parallel
    {
        unsigned char* bucket = new unsigned char[n_cubes];
#pragma omp for schedule(guided)
        for (int i = 0; i < n_cubes; i++) {
            fill(bucket, bucket + n_cubes, 0);
            for (int dim = 0; dim < 3; dim++) {
                auto& l{ ilists[dim][i] };
                for (auto j : l) {
                    if (dim < 2)
                        bucket[j] += 1;
                    else if (bucket[j] == 2) {
                        tmp[i].push_back(j);
                    }
                }
            }
            sort(tmp[i].begin(), tmp[i].end());
        }
        delete[] bucket;
    }

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < n_cubes; i ++) {
        auto& l{ tmp[i] };
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

inline bool filter_if_inside(lu& a, vec3& p, bool trajectory, AffineBody& c, int pid)
{
    if (!trajectory) {
        vec3 l = a[0], u = a[1];
        return (l.array() <= p.array()).all() && (u.array() >= p.array()).all();
    }
    auto p1{ c.vt1(pid) }, &p2{ p };
    lu ret;
    return intersection(a, compute_aabb(p1, p2), ret);
}
inline bool filter_if_inside(lu& a, Edge& e, bool trajectory, AffineBody& c, int pid)
{
    lu ret;
    if (!trajectory) {
        return intersection(a, compute_aabb(e), ret);
    }
    Edge e1{ c, unsigned(pid) }, &e2{ e };
    return intersection(a, compute_aabb(e1, e2), ret);
}
inline bool filter_if_inside(lu& a, Face& f, bool trajectory, AffineBody& c, int pid)
{
    lu ret;
    if (!trajectory)
        return intersection(a, compute_aabb(f), ret);
    Face f1{ c, unsigned(pid) }, &f2{ f };
    return intersection(a, compute_aabb(f1, f2), ret);
}

double primitive_brute_force(
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
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#endif
    bool gen_basis)
{

    double toi_global = 1.0;
    bool cull_trajectory = vtn == 3;
    int n_overlap = overlaps.size();
    if (!cull_trajectory) {
        pts.resize(0);
        idx.resize(0);
        ees.resize(0);
        eidx.resize(0);
        vidx.resize(0);
        pt_tk.resize(0);
        ee_tk.resize(0);
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
#pragma omp parallel for schedule(guided)
    for (int I = 0; I < n_cubes; I++) {
        // construct the vertex, edge, and triangle list inside each overlap
        auto& c{ *cubes[I] };
        switch (vtn) {
        case 0: c.project_vt0(); break;
        case 1: c.project_vt1(); break;
        default: c.project_vt2();
        }
    }
#pragma omp parallel for schedule(guided)
    for (int I = 0; I < n_cubes; I++) {
        auto& c{ *cubes[I] };
        for (int v = 0; v < c.n_vertices; v++) {
            auto& p{ c.v_transformed[v] };
            for (int o = starting[I]; o < starting[I + 1]; o++) {
                lu& cull = overlaps[o].cull;
                overlaps[o].plist = lists + o;
                if (filter_if_inside(cull, p, cull_trajectory, c, v)) {
                    auto &t{ overlaps[o] };
                    assert(t.i == I);
                    if (t.i < t.j)
                        lists[o].vi.push_back(v);
                    else
                        lists[o].vj.push_back(v);
                }
            }

            // handling vertex-ground collision
            if (!cull_trajectory) {
                double d = vg_distance(p);
                d = d * d;
                if (d < barrier::d_hat) {
                    #pragma omp critical
                    vidx.push_back({ I, v });
                }
            }
            else {
#ifndef TESTING
                double t = collision_time(c, v);
                #pragma omp critical
                toi_global = min(toi_global, t);
#endif
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
    sort(overlaps.begin(), overlaps.end(), [](const Intersection& a, const Intersection& b) -> bool{
        auto ad = a.i + a.j, am = abs(a.i - a.j);
        auto bd = b.i + b.j, bm = abs(b.i - b.j);

        return ad < bd || (ad == bd && am < bm) || (ad == bd && am == bm && a.i < b.i);
    });
    // spdlog::info("ground toi  = {}", toi_global);

#pragma omp parallel for schedule(guided)
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

    const auto vf_col_set = [&](vector<int>& vilist, vector<int>& fjlist,
                                const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                int I, int J,
                                vector<array<vec3, 4>>& pts,
                                vector<array<int, 4>>& idx,
                                vector<Matrix<double, 2, 12>>& pt_tk) {
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

    const auto vf_col_time = [&](vector<int>& vilist, vector<int>& fjlist,
                                 const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                 int I, int J) -> double {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        double toi = 1.0;
        for (int vi : vilist)
            for (int fj : fjlist) {
                vec3 v{ ci.v_transformed[vi] };
                Face f{ cj, unsigned(fj), true, true };
                double t = pt_collision_time(ci.vt1(vi), Face{ cj, unsigned(fj) }, v, f);
#ifdef TESTING
                if (t < 1.0) {
                    idx.push_back({ I, vi, J, fj });
                    pt_tois.push_back({ t, int(pt_tois.size()) });
                }
#endif
                toi = min(toi, t);
            }
        return toi;
    };

    const auto ee_col_time = [&](vector<int>& eilist, vector<int>& ejlist,
                                 const std::vector<std::unique_ptr<AffineBody>>& cubes,
                                 int I, int J) -> double {
        auto &ci{ *cubes[I] }, &cj{ *cubes[J] };
        double toi = 1.0;
        for (int ei : eilist)
            for (int ej : ejlist) {
                Edge ei0(ci, ei), ei1(ci, ei, true, true);
                Edge ej0(cj, ej), ej1(cj, ej, true, true);
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
        return toi;
    };

    double ee_global = 1.0, pt_global = 1.0;
    if (!cull_trajectory)
#pragma omp parallel
    {
        vector<array<int, 4>> idx_private, eidx_private;
        vector<array<vec3, 4>> pts_private, ees_private;
        vector<Matrix<double, 2, 12>> pt_tk_private, ee_tk_private;
#pragma omp for schedule(guided) nowait
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
                            ees_private.push_back(ee);
                            eidx_private.push_back(ij);
#ifdef _FRICTION_
                            ee_tk_private.push_back(Tk_T);
#endif
                        }
                    }
                }
            vf_col_set(vilist, fjlist, cubes, I, J, pts_private, idx_private, pt_tk_private);
            vf_col_set(vjlist, filist, cubes, J, I, pts_private, idx_private, pt_tk_private);
        }
#pragma omp critical
        {
            pts.insert(pts.end(), pts_private.begin(), pts_private.end());
            idx.insert(idx.end(), idx_private.begin(), idx_private.end());
            pt_tk.insert(pt_tk.end(), pt_tk_private.begin(), pt_tk_private.end());
            ees.insert(ees.end(), ees_private.begin(), ees_private.end());
            eidx.insert(eidx.end(), eidx_private.begin(), eidx_private.end());
            ee_tk.insert(ee_tk.end(), ee_tk_private.begin(), ee_tk_private.end());
        }
    }
    else
#pragma omp parallel
    {
        double toi = 1.0;
        double ee_toi = 1.0;
        double pt_toi = 1.0;
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

            pt_toi = min(toi, min(t1, t2));
            ee_toi = min(ee_toi, t3);
            toi = min(toi, min(t1, t2));
            toi = min(toi, t3);
        }
#pragma omp critical
        {
            toi_global = min(toi_global, toi);
            ee_global = min(ee_global, ee_toi);
            pt_global = min(pt_global, pt_toi);
        }
    }
    if (cull_trajectory)
    spdlog::info("pt toi = {}, ee toi = {}", pt_global, ee_global);
    return cull_trajectory? toi_global: 1.0;
}

double iaabb_brute_force(
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
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#endif

    bool gen_basis)
{
    auto start = high_resolution_clock::now();
    vector<Intersection> ret;
    intersect_sort(n_cubes, cubes, aabbs, ret, vtn);
    double toi = primitive_brute_force(n_cubes, ret, cubes, vtn,
        pts,
        idx,
        ees,
        eidx,
        vidx,
        pt_tk,
        ee_tk,
#ifdef TESTING
        pt_tois, ee_tois,
#endif
        gen_basis);
    auto t = DURATION_TO_DOUBLE(start);
    spdlog::info("time: {} = {:0.6f} ms", vtn == 3 ? "iaabb upper bound": "iAABB", t * 1000);
    return toi;
}
