#include "../view/global_variables.h"
#include "barrier.h"
#include "affine_body.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <chrono>
#include "collision.h"
#include "time_integrator.h"
#include "spatial_hashing.h"
using namespace std;
using namespace barrier;
using namespace Eigen;
using namespace std::chrono;
using namespace utils;
#define DURATION_TO_DOUBLE(X) duration_cast<duration<double>>((X)).count()

void gen_collision_set(
    bool vt2, int n_cubes,
    const vector<unique_ptr<AffineBody>>& cubes,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx,
    vector<Matrix<double, 2, 12>>& pt_tk,
    vector<Matrix<double, 2, 12>>& ee_tk)
{
    auto start = high_resolution_clock::now();

    // const auto blend_e = [=](const vec3& x, const vec3& y, double lam) -> vec3 {
    //     return y * lam + (1.0 - lam) * x;
    // };
    // const auto blend_t = [=](const vec3& x, const vec3& y, const vec3& z, double lam0, double lam1) -> vec3 {
    //     return z * lam1 + y * lam0 + x * (1 - lam0 - lam1);
    // };
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_cubes; i++) {
        if (vt2)
            cubes[i]->project_vt2();
        else
            cubes[i]->project_vt1();
    }

    if (globals.ground) {

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_cubes; i++) {
            for (int v = 0; v < cubes[i]->n_vertices; v++) {
                auto p = cubes[i]->v_transformed[v];
                double d = vg_distance(p);
                d = d * d;
                if (d < barrier::d_hat * (globals.safe_factor * globals.safe_factor)) {
#pragma omp critical
                    vidx.push_back({ i, v });
                }
            }
        }
    }
    if (globals.pt) {
#ifdef SPATIAL_HASHING_H
#ifdef _BODY_LEVEL_
#pragma omp parallel for schedule(static)
        for (int I = 0; I < n_cubes; I++) {
            auto& ci(*cubes[I]);
            for (unsigned v = 0; v < ci.n_vertices; v++) {
                vec3 p = ci.v_transformed[v];
                globals.sh -> register_vertex(p, I, v);
            }
        }
#pragma omp parallel
        {
            vector<array<vec3, 4>> pts_private;
            vector<array<int, 4>> idx_private;
            vector<Matrix<double, 2, 12>> pt_tk_private;

#pragma omp for schedule(static) nowait
            for (int J = 0; J < n_cubes; J++) {
                auto& cj(*cubes[J]);
                for (unsigned f = 0; f < cj.n_faces; f++) {
                    Face _f(cj, f, false, true);
                    auto collisions = globals.sh -> query_triangle(_f.t0, _f.t1, _f.t2, J, barrier::d_sqrt * globals.safe_factor);
                    for (auto& c : collisions) {
                        unsigned I = c.body, v = c.pid;
                        vec3 p = cubes[I]->v_transformed[v];
                        ipc::PointTriangleDistanceType pt_type;
                        double d = vf_distance(p, _f, pt_type);
                        if (d < barrier::d_hat * (globals.safe_factor * globals.safe_factor)) {
                            array<vec3, 4> pt = { p, _f.t0, _f.t1, _f.t2 };
                            array<int, 4> ij = { I, v, J, f };
                            {
                                pts_private.push_back(pt);
                                idx_private.push_back(ij);
#ifdef _FRICTION_
                                pt_tk_private.push_back(MatrixXd::Zero(2, 12));
#endif
                            }
                        }
                    }
                }
            }
#pragma omp critical
            pts.insert(pts.end(), pts_private.begin(), pts_private.end());
#pragma omp critical
            idx.insert(idx.end(), idx_private.begin(), idx_private.end());
#ifdef _FRICTION_
#pragma omp critical
            pt_tk.insert(pt_tk.end(), pt_tk_private.begin(), pt_tk_private.end());
#endif
        }
#else
        int n_points = globals.points.size(), n_triangles = globals.triangles.size();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_points; i++) {
            auto& idx{ globals.points[i] };
            auto I{ idx[0] };
            auto v{ idx[1] };
            auto& ci(*cubes[I]);
            vec3 p = ci.v_transformed[v];
            globals.sh->register_vertex(p, I, v);
        }
#pragma omp parallel
        {
            vector<array<vec3, 4>> pts_private;
            vector<array<int, 4>> idx_private;
            vector<Matrix<double, 2, 12>> pt_tk_private;

#pragma omp for schedule(static) nowait
            for (int j = 0; j < n_triangles; j++) {
                auto idx{ globals.triangles[j] };
                auto J{ idx[0] };
                auto f{ idx[1] };
                auto& cj(*cubes[J]);
                Face _f(cj, f, false, true);
                auto collisions = globals.sh->query_triangle(_f.t0, _f.t1, _f.t2, J, barrier::d_sqrt * globals.safe_factor);
                for (auto& c : collisions) {
                    unsigned I = c.body, v = c.pid;
                    vec3 p = cubes[I]->v_transformed[v];
                    ipc::PointTriangleDistanceType pt_type;
                    double d = vf_distance(p, _f, pt_type);
                    if (d < barrier::d_hat * (globals.safe_factor * globals.safe_factor)) {
                        array<vec3, 4> pt = { p, _f.t0, _f.t1, _f.t2 };
                        array<int, 4> ij = { I, v, J, f };
                        Matrix<double, 2, 12> Tk_T;
                        Tk_T.setZero(2, 12);
                        Vector2d uk;
                        pt_uktk(*cubes[I], cj, pt, ij, pt_type, Tk_T, uk, d, globals.dt);
                        {
                            pts_private.push_back(pt);
                            idx_private.push_back(ij);
#ifdef _FRICTION_
                            pt_tk_private.push_back(Tk_T);
#endif
                        }
                    }
                }
            }
#pragma omp critical
            pts.insert(pts.end(), pts_private.begin(), pts_private.end());
#pragma omp critical
            idx.insert(idx.end(), idx_private.begin(), idx_private.end());
#ifdef _FRICTION_
#pragma omp critical
            pt_tk.insert(pt_tk.end(), pt_tk_private.begin(), pt_tk_private.end());
#endif
        }

#endif
        globals.sh -> remove_all_entries();
#else
#pragma omp parallel for schedule(static)
        for (int I = 0; I < nsqr; I++) {
            int i = I / n_cubes, j = I % n_cubes;
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            if (i == j) continue;
            for (int v = 0; v < ci.n_vertices; v++)
                for (int f = 0; f < cj.n_faces; f++) {
                    Face _f(cj, f);
                    vec3 p = ci.vt1(v);
                    auto fu = _f.t0.cwiseMax(_f.t1).cwiseMax(_f.t2).array() + barrier ::d_sqrt;
                    auto fl = _f.t0.cwiseMin(_f.t1).cwiseMin(_f.t2).array() - barrier::d_sqrt;
                    if ((p.array() <= fu.array()).all() && (p.array() >= fl.array()).all()) {
                    }
                    else
                        continue;

                    auto pt_type = ipc::point_triangle_distance_type(p, _f.t0, _f.t1, _f.t2);

                    double d = ipc::point_triangle_distance(p, _f.t0, _f.t1, _f.t2, pt_type);
                    if (d < barrier::d_hat * globals.safe_factor) {
                        array<vec3, 4> pt = { p, _f.t0, _f.t1, _f.t2 };
                        array<int, 4> ij = { i, v, j, f };

#pragma omp critical
                        {
                            pts.push_back(pt);
                            idx.push_back(ij);
                        }
                    }
                }
        }
#endif
    }
    if (globals.ee) {

#ifdef SPATIAL_HASHING_H
#ifdef _BODY_LEVEL_
#pragma omp parallel for schedule(static)
        for (int I = 0; I < n_cubes; I++) {
            auto& ci(*cubes[I]);
            for (unsigned ei = 0; ei < ci.n_edges; ei++) {
                Edge e{ ci, ei, false, true };
                globals.sh -> register_edge(e.e0, e.e1, I, ei);
            }
        }

        size_t* cnt;
#pragma omp parallel
        {
            int ithread = omp_get_thread_num();
            int nthreads = omp_get_num_threads();

            vector<array<vec3, 4>> ees_private;
            vector<array<int, 4>> eidx_private;
            vector<Matrix<double, 2, 12>> ee_tk_private;
#pragma omp single
            {
                cnt = new size_t[nthreads + 1];
                cnt[0] = 0;
            }
#pragma omp for schedule(static) nowait
            for (int J = 0; J < n_cubes; J++) {
                auto& cj(*cubes[J]);
                for (unsigned ej = 0; ej < cj.n_edges; ej++) {
                    Edge e{ cj, ej, false, true };
                    auto collisions = globals.sh -> query_edge(e.e0, e.e1, J, barrier::d_sqrt * globals.safe_factor);
                    for (auto& c : collisions) {
                        unsigned I = c.body, ei = c.pid;
                        if (I > J) continue;
                        Edge _ei{ *cubes[I], ei };

                        double d = ipc::edge_edge_distance(_ei.e0, _ei.e1, e.e0, e.e1);
                        if (d < barrier::d_hat * (globals.safe_factor * globals.safe_factor)) {
                            array<vec3, 4> ee = { _ei.e0, _ei.e1, e.e0, e.e1 };
                            array<int, 4> ij = { I, ei, J, ej };
                            {
                                ees_private.push_back(ee);
                                eidx_private.push_back(ij);
#ifdef _FRICTION_
                                ee_tk_private.push_back(MatrixXd::Zero(2, 12));
#endif
                            }
                        }
                    }
                }
    
            }
            cnt[ithread + 1] = ees_private.size();
#pragma omp barrier
#pragma omp single
            {
                for (int i = 1; i < nthreads + 1; i++) cnt[i] += cnt[i - 1];
                ees.resize(cnt[nthreads]);
                eidx.resize(cnt[nthreads]);
#ifdef _FRICTION_
                ee_tk.resize(cnt[nthreads]);
#endif
            }
            std::copy(ees_private.begin(), ees_private.end(), ees.begin() + cnt[ithread]);
            std::copy(eidx_private.begin(), eidx_private.end(), eidx.begin() + cnt[ithread]);
            #ifdef _FRICTION_
            std::copy(ee_tk_private.begin(), ee_tk_private.end(), ee_tk.begin() + cnt[ithread]);
            #endif
            // #pragma omp critical
            //             ees.insert(ees.end(), ees_private.begin(), ees_private.end());
            // #pragma omp critical
            //             eidx.insert(eidx.end(), eidx_private.begin(), eidx_private.end());
            // #ifdef _FRICTION_
            // #pragma omp critical
            //             ee_tk.insert(ee_tk.end(), ee_tk_private.begin(), ee_tk_private.end());
            // #endif
        }
        delete[] cnt;
#else

        int n_edges = globals.edges.size();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_edges; i++) {
            auto idx{ globals.edges[i] };
            auto I{ idx[0] }, ei{ idx[1] };
            auto& ci(*cubes[I]);
            Edge e{ ci, ei, false, true };
            globals.sh->register_edge(e.e0, e.e1, I, ei);
        }

        size_t* cnt;
#pragma omp parallel
        {
            int ithread = omp_get_thread_num();
            int nthreads = omp_get_num_threads();

            vector<array<vec3, 4>> ees_private;
            vector<array<int, 4>> eidx_private;
            vector<Matrix<double, 2, 12>> ee_tk_private;
#pragma omp single
            {
                cnt = new size_t[nthreads + 1];
                cnt[0] = 0;
            }
#pragma omp for schedule(static) nowait
            for (int j = 0; j < n_edges; j++) {
                auto idx{ globals.edges[j] };
                auto J{ idx[0] }, ej{ idx[1] };
                auto& cj(*cubes[J]);
                Edge e{ cj, ej, false, true };
                auto collisions = globals.sh->query_edge(e.e0, e.e1, J, barrier::d_sqrt * globals.safe_factor);
                for (auto& c : collisions) {
                    unsigned I = c.body, ei = c.pid;
                    if (I > J) continue;
                    Edge _ei{ *cubes[I], ei };
                    auto ee_type = ipc::edge_edge_distance_type(_ei.e0, _ei.e1, e.e0, e.e1);
                    double d = ipc::edge_edge_distance(_ei.e0, _ei.e1, e.e0, e.e1, ee_type);
                    if (d < barrier::d_hat * (globals.safe_factor * globals.safe_factor)) {
                        array<vec3, 4> ee = { _ei.e0, _ei.e1, e.e0, e.e1 };
                        array<int, 4> ij = { I, ei, J, ej };
                        Matrix<double, 2, 12> Tk_T;
                        Vector2d uk;
                        ee_uktk(*cubes[I], cj, ee, ij, ee_type, Tk_T, uk, d, globals.dt);
                        {
                            ees_private.push_back(ee);
                            eidx_private.push_back(ij);
#ifdef _FRICTION_
                            ee_tk_private.push_back(Tk_T);
#endif
                        }
                    }
                }
            }
            cnt[ithread + 1] = ees_private.size();
#pragma omp barrier
#pragma omp single
            {
                for (int i = 1; i < nthreads + 1; i++) cnt[i] += cnt[i - 1];
                ees.resize(cnt[nthreads]);
                eidx.resize(cnt[nthreads]);
#ifdef _FRICTION_
                ee_tk.resize(cnt[nthreads]);
#endif
            }
            std::copy(ees_private.begin(), ees_private.end(), ees.begin() + cnt[ithread]);
            std::copy(eidx_private.begin(), eidx_private.end(), eidx.begin() + cnt[ithread]);
#ifdef _FRICTION_
            std::copy(ee_tk_private.begin(), ee_tk_private.end(), ee_tk.begin() + cnt[ithread]);
#endif
        }
        delete[] cnt;
#endif
        globals.sh -> remove_all_entries();
#else
        for (int i = 0; i < n_cubes; i++)
            for (int j = i + 1; j < n_cubes; j++) {
                auto &ci(*cubes[i]), &cj(*cubes[j]);
                for (int _ei = 0; _ei < ci.n_edges; _ei++)
                    for (int _ej = 0; _ej < cj.n_edges; _ej++) {
                        Edge ei(ci, _ei), ej(cj, _ej);
                        auto iu = ei.e0.cwiseMax(ei.e1).array() + barrier ::d_sqrt / 2;
                        auto il = ei.e0.cwiseMin(ei.e1).array() - barrier ::d_sqrt / 2;

                        auto ju = ej.e0.cwiseMax(ej.e1).array() + barrier ::d_sqrt / 2;
                        auto jl = ej.e0.cwiseMin(ej.e1).array() - barrier ::d_sqrt / 2;
                        if ((iu.array() >= jl.array()).all() && (ju.array() >= il.array()).all()) {}
                        else
                            continue;
                        double d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
                        if (d < barrier::d_hat * globals.safe_factor) {
                            array<vec3, 4> ee = { ei.e0, ei.e1, ej.e0, ej.e1 };
                            array<int, 4> ij = { i, _ei, j, _ej };

#pragma omp critical
                            {
                                ees.push_back(ee);
                                eidx.push_back(ij);
                            }
                        }
                    }
            }
#endif
    }
    double _duration
        = DURATION_TO_DOUBLE(high_resolution_clock::now() - start);
    spdlog::info("collision set : time = {:0.6f} ms", _duration * 1000);
}
