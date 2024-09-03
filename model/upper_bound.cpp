#include "time_integrator.h"
#include "../view/global_variables.h"
#include "barrier.h"
#include <chrono>
#include <spdlog/spdlog.h>
#include "collision.h"
#include "spatial_hashing.h"

using namespace std;
using namespace std::chrono;
using namespace utils;
#define DURATION_TO_DOUBLE(X) duration_cast<duration<scalar>>(high_resolution_clock::now() - (X)).count()
scalar step_size_upper_bound(Vector<scalar, -1>& dq, vector<unique_ptr<AffineBody>>& cubes,
    int n_cubes, int n_pt, int n_ee, int n_g,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx
)
 {
    auto start = high_resolution_clock::now();
    scalar toi = 1.0;
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_cubes; k++) {
        cubes[k]->project_vt2();
    }
    if (globals.full_ccd) {
        if (globals.ground) {

#pragma omp parallel
            {
                scalar toi_private = 1.0;
#pragma omp for schedule(static)
                for (int i = 0; i < n_cubes; i++) {
                    // cubes[i]->project_vt1();
                    for (int v = 0; v < cubes[i]->n_vertices; v++) {
                        auto p = cubes[i]->v_transformed[v];
                        scalar d = vg_distance(p);
                        if (d < 0) {
                            scalar t = collision_time(*cubes[i], v);
                            toi_private = min(toi_private, t);
                        }
                    }
                }
#pragma omp critical
                toi = min(toi, toi_private);
            }
        }
        spdlog::info("sh: ground toi = {}", toi);
        if (globals.pt) {
#ifdef _BODY_LEVEL_
#pragma omp parallel for schedule(static)
            for (int I = 0; I < n_cubes; I++) {
                auto& ci(*cubes[I]);
                for (unsigned v = 0; v < ci.n_vertices; v++) {
                    vec3 p1 = ci.v_transformed[v];
                    vec3 p0 = ci.vt1(v);
                    globals.sh->register_edge(p0, p1, I, v);
                }
            }

#pragma omp parallel for schedule(static)
            for (int J = 0; J < n_cubes; J++) {
                auto& cj(*cubes[J]);
                scalar toi_private = 1.0;
                for (unsigned f = 0; f < cj.n_faces; f++) {
                    Face f0(cj, f, false);
                    Face f1(cj, f, true, true);
                    auto collisions = globals.sh->query_triangle_trajectory(
                        f0.t0, f0.t1, f0.t2,
                        f1.t0, f1.t1, f1.t2,
                        J);
                    for (auto& c : collisions) {
                        unsigned I = c.body, v = c.pid;
                        vec3 p1 = cubes[I]->v_transformed[v];
                        vec3 p0 = cubes[I]->vt1(v);

                        scalar t = pt_collision_time(p0, f0, p1, f1);
                        toi_private = min(toi_private, t);
                    }
                }
                #pragma omp critical
                toi = min(toi_private, toi);
            }
#else
            int n_points = globals.points.size(), n_edges = globals.edges.size(), n_triangles = globals.triangles.size();
#pragma omp parallel for schedule(static)
            for (int I = 0; I < n_points; I++) {
                auto& idx = globals.points[I];
                int v = idx[1];
                auto& ci(*cubes[idx[0]]);
                vec3 p1 = ci.v_transformed[v];
                vec3 p0 = ci.vt1(v);
                globals.sh->register_edge(p0, p1, idx[0], v);
            }
            globals.sh->remove_all_entries();

            scalar pt_toi = 1.0;
#pragma omp parallel
            {
                scalar toi_private = 1.0;
                vector<Primitive> & collisions {globals.sh -> collisions[omp_get_thread_num()]};
#pragma omp for schedule(guided)
                for (int J = 0; J < n_triangles; J++) {
                    auto& idx = globals.triangles[J];

                    auto& cj(*cubes[idx[0]]);
                    unsigned f = idx[1];
                    Face f0(cj, f, false);
                    Face f1(cj, f, true, true);
                    // vector<Primitive> collisions;
                    globals.sh->query_triangle_trajectory(
                        f0.t0, f0.t1, f0.t2,
                        f1.t0, f1.t1, f1.t2,
                        idx[0], collisions);
                    for (auto& _c : collisions) {
                        auto& c{ _c.pbody };
                        unsigned I = c.body, v = c.pid;
                        vec3 p1 = cubes[I]->v_transformed[v];
                        vec3 p0 = cubes[I]->vt1(v);

                        scalar t = pt_collision_time(p0, f0, p1, f1);
                        toi_private = min(toi_private, t);
                    }
                }
#pragma omp critical
                {
                    toi = min(toi_private, toi);
                    pt_toi = min(toi_private, pt_toi);
                }
            }
#endif
            // globals.sh->remove_all_entries();
            spdlog::info("sh pt toi = {}", pt_toi);
        }
        if (globals.ee) {
#ifdef _BODY_LEVEL_

#pragma omp parallel for schedule(static)
            for (int I = 0; I < n_cubes; I++) {
                auto& ci(*cubes[I]);
                for (unsigned ei = 0; ei < ci.n_edges; ei++) {
                    Edge e1{ ci, ei, true, true };
                    Edge e0{ ci, ei, false };
                    globals.sh->register_edge_trajectory(e0.e0, e0.e1, e1.e0, e1.e1, I, ei);
                }
            }

#pragma omp parallel for schedule(static)
            for (int J = 0; J < n_cubes; J++) {
                auto& cj(*cubes[J]);

                scalar toi_private = 1.0;
                for (unsigned ej = 0; ej < cj.n_edges; ej++) {
                    Edge e1{ cj, ej, true, true };

                    Edge e0{ cj, ej, false };
                    auto collisions = globals.sh->query_edge_trajectory(e0.e0, e0.e1, e1.e0, e1.e1, J);

                    for (auto& c : collisions) {
                        unsigned I = c.body, ei = c.pid;
                        if (I > J) continue;
                        Edge ei1{ *cubes[I], ei, true, true };
                        Edge ei0{ *cubes[I], ei, false };
                        auto t = ee_collision_time(ei0, e0, ei1, e1);
                        toi_private = min(toi_private, t);
                    }
                }
                #pragma omp critical
                toi = min(toi, toi_private);
            }
#else
            int n_edges = globals.edges.size();
#pragma omp parallel for schedule(static)
            for (int I = 0; I < n_edges; I++) {
                auto& idx(globals.edges[I]);
                auto& ci(*cubes[idx[0]]);
                auto ei{ idx[1] };
                Edge e1{ ci, ei, true, true };
                Edge e0{ ci, ei, false };
                globals.sh->register_edge_trajectory(e0.e0, e0.e1, e1.e0, e1.e1, idx[0], ei);
            }
            globals.sh->remove_all_entries();
            scalar ee_toi = 1.0;
#pragma omp parallel
            {
                scalar toi_private = 1.0;
                vector<Primitive> & collisions {globals.sh -> collisions[omp_get_thread_num()]};
#pragma omp for schedule(guided)
                for (int J = 0; J < n_edges; J++) {
                    auto& idx(globals.edges[J]);

                    auto& cj(*cubes[idx[0]]);
                    auto ej{ idx[1] };
                    Edge e1{ cj, ej, true, true };

                    Edge e0{ cj, ej, false };
                    // vector<Primitive> collisions;

                    globals.sh->query_edge_trajectory(e0.e0, e0.e1, e1.e0, e1.e1, idx[0], collisions);

                    for (auto& _c : collisions) {
                        auto &c{ _c.pbody };
                        unsigned I = c.body, ei = c.pid;
                        if (I > idx[0]) continue;
                        Edge ei1{ *cubes[I], ei, true, true };
                        Edge ei0{ *cubes[I], ei, false };
                        auto t = ee_collision_time(ei0, e0, ei1, e1);
                        toi_private = min(toi_private, t);
                    }
                }
#pragma omp critical
                {
                    toi = min(toi, toi_private);
                    ee_toi = min(toi_private, ee_toi);
                }
            }

#endif
            // globals.sh->remove_all_entries();
            spdlog::info("sh ee toi = {}", ee_toi);
        }
        auto _duration = DURATION_TO_DOUBLE(start);
        spdlog::info("time: step size upper bound = {:0.6f} ms", _duration * 1000);
        return toi;
    }

    toi = barrier::d_sqrt / 2.0 / norm_1(dq, n_cubes) * globals.safe_factor;
    toi = min(1.0, toi);
    vector<scalar> tois;
    tois.resize(n_pt + n_ee + n_g);

#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_g; k++) {
        auto& v{ vidx[k] };
        scalar t = collision_time(*cubes[v[0]], v[1]);
        tois[k + n_pt + n_ee] = t;
        // toi = min(t, toi);
    }
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_pt; k++) {
        auto& pt(pts[k]);
        const auto& ij(idx[k]);

        int i = ij[0], j = ij[2];
        // Face f(cubes[j], ij[3], true);
        vec3 p_t2 = cubes[i]->v_transformed[ij[1]];
        vec3 p_t1 = cubes[i]->vt1(ij[1]);
        // scalar t = vf_collision_detect(p_t1, p_t2,
        //     *cubes[j], ij[3]);
        scalar t = pt_collision_time(p_t1, Face(*cubes[j], ij[3]), p_t2, Face(*cubes[j], ij[3], true, true));
        // scalar t = vf_collision_detect(p_t1, p_t2, Face(*cubes[j], ij[3]), Face(*cubes[j], ij[3], true));
        tois[k] = t;
        // toi = min(toi, t);
    }
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_ee; k++) {
        auto& ee(ees[k]);
        const auto& ij(eidx[k]);

        auto &ci(*cubes[ij[0]]), &cj(*cubes[ij[2]]);
        // scalar t = ee_collision_detect(ci, cj, ij[1], ij[3]);
        Edge ei0(ci, ij[1]), ei1(ci, ij[1], true, true);
        Edge ej0(cj, ij[3]), ej1(cj, ij[3], true, true);
        scalar t = ee_collision_time(ei0, ej0, ei1, ej1);
        tois[k + n_pt] = t;
        // toi = min(toi, t);
    }
    for (auto t : tois) toi = min(toi, t);
    auto _duration = DURATION_TO_DOUBLE(start);
    spdlog::info("time: step size upper bound = {:0.6f} ms", _duration * 1000);
    return toi;
}
