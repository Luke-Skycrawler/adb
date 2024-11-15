
#include "iaabb.h"
#include "geometry.h"
#include "collision.h"
#include <memory>
#include <array>
#include <algorithm>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "barrier.h"
#include "timer.h"
#include "time_integrator.h"
#include <tbb/parallel_sort.h>
#include "ipc_extension.h"
#include "thread_local_culling.h"


using namespace std;
using namespace Eigen;
using namespace utils;
using namespace std::chrono;

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

void IAABB::intersect_brute_force(
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

void IAABB::intersect_sort(
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
    static vector<int>*intersected_body_per_dim[3] = {
        new vector<int>[n_cubes],
        new vector<int>[n_cubes],
        new vector<int>[n_cubes]
    },
           *intersected_body_joint = new vector<int>[n_cubes];
    static vector<Intersection>* ret_tmp = new vector<Intersection>[n_cubes];

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
    buckets.resize(omp_get_max_threads() * n_cubes);

#pragma omp parallel
    {
        // int char* bucket = new int char[n_cubes];
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

scalar IAABB::primitive_brute_force(
    int n_cubes,
    std::vector<Intersection>& overlaps, // assert sorted
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int vtn,
    vector<q4>& pts,
    vector<i4>& idx,
    vector<q4>& ees,
    vector<i4>& eidx,
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

    // static PList* lists = new PList[n_overlap];
    // static int allocated = n_overlap;
    // if (n_overlap > allocated) {
    //     delete []lists;
    //     lists = new PList[n_overlap];
    //     allocated = n_overlap;
    // }
    lists.resize(n_overlap);
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

    if(ground)
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

            g_cnt++;
            if(g_cnt > 1) exit(1);
        }
        else
            g_cnt = 0;
    }

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

    for(int i = 0; i < n_overlap; i++) overlaps[i].plist = &lists[i];

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; i++) {
        auto idx{ points[i] };
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
        auto idx{ edges[i] };
        auto I{ idx[0] };
        auto ei{ idx[1] };
        auto& c{ *cubes[I] };
        Edge e{ c.edge(ei, true, true) };
        Edge e0{ c.edge(ei) };

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
        auto idx{ triangles[i] };
        auto I{ idx[0] };
        auto fi{ idx[1] };
        auto& c{ *cubes[I] };
        Face f{ c.face(fi, true, true) };
        Face f0{ c.face(fi) };

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


    if (vertex_starting_index.size() == 0) {
        // initialization
        vertex_starting_index.resize(n_cubes);
        vt1_buffer.resize(points.size());
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
    static vector<vector<i4>> idx_private(omp_get_max_threads()),eidx_private(omp_get_max_threads());
    static vector<vector<q4>> pts_private(omp_get_max_threads()), ees_private(omp_get_max_threads());

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
            p_cnt++;
            if(p_cnt > 1) exit(1);
        }
        else
            p_cnt = 0;
    }
    toi_global = min(toi_global, toi_ee_pt);
    return cull_trajectory? toi_global: 1.0;
}

scalar IAABB::iaabb_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    int vtn,
    std::vector<q4>& pts,
    std::vector<i4>& idx,
    std::vector<q4>& ees,
    std::vector<i4>& eidx,
    std::vector<std::array<int, 2>>& vidx)
{
    auto start = high_resolution_clock::now();
    vector<Intersection> ret;
    intersect_sort(n_cubes, cubes, aabbs, ret, vtn);
    scalar toi = primitive_brute_force(n_cubes, ret, cubes, vtn,
        pts,
        idx,
        ees,
        eidx,
        vidx);
    auto t = DURATION_TO_DOUBLE(start);
    spdlog::info("time: {} = {:0.6f} ms", vtn == 3 ? "iaabb upper bound": "iAABB", t * 1000);
    return toi;
}

IAABB::IAABB(std::vector<std::unique_ptr<AffineBody>>& cubes, bool ground)
    : points(utils::gen_point_list(cubes, cubes.size())), edges(utils::gen_edge_list(cubes, cubes.size())), triangles(utils::gen_triangle_list(cubes, cubes.size())), n_points(points.size()), n_triangles(triangles.size()), n_edges(edges.size()), ground(ground), vidx_thread_local(omp_get_max_threads()) {}
