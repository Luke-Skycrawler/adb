
#include "iaabb.h"
#include "cuda_header.h"
#include "timer.h"

using namespace std;
using namespace Eigen;

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
