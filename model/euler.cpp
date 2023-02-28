#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include <assert.h>
#include <array>
// #include <ipc/distance/point_triangle.hpp>
// #include <ipc/distance/edge_edge.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include <chrono>
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif
#include <ipc/distance/edge_edge_mollifier.hpp>
// #define IAABB_COMPARING
#define IAABB_INTERNSHIP
#ifdef IAABB_COMPARING

#include <algorithm>
#endif
using namespace std;
using namespace barrier;
using namespace Eigen;
using namespace std::chrono;
using namespace utils;
#define DURATION_TO_DOUBLE(X) duration_cast<duration<double>>((X)).count()
double E_global(const VectorXd& q_plus_dq, const VectorXd& dq, int n_cubes, int n_pt, int n_ee, int n_g,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<double, 2, 12>>& pt_tk,
    const vector<Matrix<double, 2, 12>>& ee_tk,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt, double& _ef, bool _vt2)
{
    double e = 0.0;
    double ef = 0.0;
// inertia energy
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : e)
    for (int i = 0; i < n_cubes; i++) {
        auto& c(*cubes[i]);
        c.dq = dq.segment<12>(i * 12);

        auto q_tiled = c.q_tile(dt, globals.gravity);
        auto _q = q_plus_dq.segment<12>(12 * i);
        double e_inert = E(_q, q_tiled, c, dt);
        // if (_vt2)
        //     c.project_vt2();
        // else
        // c.project_vt1();
        c.project_vt2();
        // used for pt_vstack. vt1 and vt2 should not use buffer
        e += e_inert;
    }
    if (globals.pt)
// point-triangle energy
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) reduction(+ \
                                                                   : ef)
        for (int k = 0; k < n_pt; k++) {
            auto& ij = idx[k];
            Face f(*cubes[ij[2]], ij[3], _vt2, _vt2);
            vec3 v(cubes[ij[0]]->v_transformed[ij[1]]);
            if (!_vt2) v = cubes[ij[0]]->vt1(ij[1]);
            // array<vec3, 4> a = {v, f.t0, f.t1, f.t2};
            ipc::PointTriangleDistanceType pt_type;
            double d = vf_distance(v, f, pt_type);
            // double d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2);
            e += barrier::barrier_function(d);
#ifdef _FRICTION_
            auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
            auto v_stack = pt_vstack(*cubes[ij[0]], *cubes[ij[2]], ij[1], ij[3]);
            auto uk = (pt_tk[k] * v_stack).norm();
            if (globals.pt_fric)
                ef += D_f0(uk, contact_force);
#endif
        }

    // ee ipc energy
    if (globals.ee)
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) reduction(+ \
                                                                   : ef)
        for (int k = 0; k < n_ee; k++) {
            auto& ij(eidx[k]);
            Edge ei(*cubes[ij[0]], ij[1], _vt2, _vt2), ej(*cubes[ij[2]], ij[3], _vt2, _vt2);
            double eps_x = (ei.e0 - ei.e1).squaredNorm() * (ej.e0 - ej.e1).squaredNorm() * globals.eps_x;
            double p = ipc::edge_edge_mollifier(ei.e0, ei.e1, ej.e0, ej.e1, eps_x);
            double d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
#ifdef NO_MOLLIFIER
            e += barrier_function(d);
#else
            e += barrier_function(d) * p;
#endif
#ifdef _FRICTION_
            auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
            auto v_stack = ee_vstack(*cubes[ij[0]], *cubes[ij[2]], ij[1], ij[3]);
            auto uk = (ee_tk[k] * v_stack).norm();
            if (globals.ee_fric)
                ef += D_f0(uk, contact_force);
#endif
        }

    // vertex-ground ipc energy
    if (globals.ground)
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) reduction(+ \
                                                                   : ef)
        for (int k = 0; k < n_g; k++) {
            auto& v{
                vidx[k]
            };
            vec3 vt2 = cubes[v[0]]->v_transformed[v[1]];
            vec3 vd = _vt2 ? vt2 : cubes[v[0]]->vt1(v[1]);
            double e_ground = E_ground(vd);
            e += e_ground;
#ifdef _FRICTION_
            auto& c{ *cubes[v[0]] };
            auto vt0 = c.vt0(v[1]);
            double d = vg_distance(vd);
            if (d * d < barrier::d_hat) {
                auto contact_force = -barrier_derivative_d(d * d) / (dt * dt) * 2 * d;
                vec3 _uk = vt2 - vt0;
                double uk = sqrt(_uk(0) * _uk(0) + _uk(1) * _uk(1));
                if (globals.vg_fric)
                    ef += D_f0(uk, contact_force);
            }
#endif
        }
    _ef = ef;
    return e;
}
double line_search(const VectorXd& dq, const VectorXd& grad, VectorXd& q0, double& E0, double& E1,
    int n_cubes, int n_pt, int n_ee, int n_g,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx,
    vector<Matrix<double, 2, 12>>& pt_tk,
    vector<Matrix<double, 2, 12>>& ee_tk,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt)
{
    const double c1 = 1e-4;
    double alpha = 1.0;
    bool wolfe = false;
    double ef0 = 0.0;
    E0 = E_global(q0, 0.0 * dq,
        n_cubes, n_pt, n_ee, n_g,
        idx,
        eidx,
        vidx,
        pt_tk,
        ee_tk,
        cubes, dt, ef0, false);
    E0 += ef0;
    double qdg = dq.dot(grad);
    VectorXd q1;
    vector<array<vec3, 4>> pts_new, pts_iaab;
    vector<array<int, 4>> idx_new, idx_iaab;

    vector<array<vec3, 4>> ees_new, ees_iaab;
    vector<array<int, 4>> eidx_new, eidx_iaab;

    vector<array<int, 2>> vidx_new, vidx_iaab;

    vector<Matrix<double, 2, 12>> pt_tk_new, pt_tk_iaab;
    vector<Matrix<double, 2, 12>> ee_tk_new, ee_tk_iaab;

    auto dq_norm = dq.norm();
    do {
        q1 = q0 + dq * alpha;
        auto dqk = dq * alpha;
        for (int i = 0; i < n_cubes; i++) {
            auto& c(*cubes[i]);
            c.dq = dqk.segment<12>(i * 12);
        }

        if (globals.iaabb)
            iaabb_brute_force(n_cubes, cubes, globals.aabbs, 2,
        #ifdef IAABB_COMPARING
                pts_iaab,
                idx_iaab,
                ees_iaab,
                eidx_iaab,
                vidx_iaab,
                pt_tk_iaab,
                ee_tk_iaab);
        #else
                pts_new,
                idx_new,
                ees_new,
                eidx_new,
                vidx_new,
                pt_tk_new,
                ee_tk_new);
        else
        #endif
            gen_collision_set(true, n_cubes, cubes,
                pts_new,
                idx_new,
                ees_new,
                eidx_new,
                vidx_new,
                pt_tk_new,
                ee_tk_new);
        double ef1 = 0.0, E2 = 0.0, ef2 = 0.0;
        E1 = E_global(q1, dqk,
            n_cubes, n_pt, n_ee, n_g,
            idx,
            eidx,
            vidx,
            pt_tk,
            ee_tk,
            cubes, dt, ef1, false);
        E2 = E_global(q1, dqk, n_cubes, pts_new.size(), ees_new.size(), vidx_new.size(),
            idx_new, eidx_new, vidx_new,
            pt_tk_new,
            ee_tk_new,
            cubes, dt, ef2, true);
        E1 = E2 + ef1;
        wolfe = E1 <= E0 + c1 * alpha * qdg;
        // spdlog::info("wanted descend = {}, E1 - E0 = {}, E1 = {}, E0 = {}, alpha = {}", c1 * alpha * qdg, E1 - E0, E1, E0, alpha);
        alpha /= 2;
        if (!(!wolfe && grad.norm() > 1e-3)) break;
        if (dq_norm * alpha * 2 < 1e-4) {
            // smaller than Newton iter convergence condition, clip it
            alpha = 0.0;
            break;
        }
    } while (true);
    pts = pts_new;
    idx = idx_new;
    ees = ees_new;
    eidx = eidx_new;
    vidx = vidx_new;
    pt_tk = pt_tk_new;
    ee_tk = ee_tk_new;

    return alpha * 2;
}

void implicit_euler(vector<unique_ptr<AffineBody>>& cubes, double dt)
{
    if (!globals.log)
        spdlog::set_level(spdlog::level::err);
    bool term_cond;
    int& ts = globals.ts;
    int iter = 0;
    double sup_dq = 0.0;
    for (int k = 0; k < cubes.size(); k++) {
        auto& c(*cubes[k]);
        for (int i = 0; i < 4; i++) {
            c.q[i] = c.q0[i];
        }
    }

    vector<array<vec3, 4>> pts;
    vector<array<int, 4>> idx;

    vector<array<vec3, 4>> ees;
    vector<array<int, 4>> eidx;

    vector<array<int, 2>> vidx;

    vector<Matrix<double, 2, 12>> pt_tk;
    vector<Matrix<double, 2, 12>> ee_tk;
#ifdef IAABB_COMPARING
    vector<array<vec3, 4>> pts_iaabb;
    vector<array<int, 4>> idx_iaabb;

    vector<array<vec3, 4>> ees_iaabb;
    vector<array<int, 4>> eidx_iaabb;

    vector<array<int, 2>> vidx_iaabb;

    vector<Matrix<double, 2, 12>> pt_tk_iaabb;
    vector<Matrix<double, 2, 12>> ee_tk_iaabb;
#endif

    const int n_cubes = cubes.size(), nsqr = n_cubes * n_cubes, hess_dim = n_cubes * 12;

    if (globals.col_set) {
        if (globals.iaabb)
            iaabb_brute_force(n_cubes, cubes, globals.aabbs, 1,
#ifdef IAABB_COMPARING
                pts_iaabb,
                idx_iaabb,
                ees_iaabb,
                eidx_iaabb,
                vidx_iaabb,
                pt_tk_iaabb,
                ee_tk_iaabb,
                true);
#else
                pts,
                idx,
                ees,
                eidx,
                vidx,
                pt_tk,
                ee_tk,
                true);
        else
#endif
        gen_collision_set(false, n_cubes, cubes,
            pts,
            idx,
            ees,
            eidx,
            vidx,
            pt_tk,
            ee_tk,
            true);

        if (globals.iaabb) {

#ifdef IAABB_COMPARING
            const auto compare_collision = [](
                                               vector<array<int, 4>>& aidx,

                                               vector<array<int, 4>>& bidx) -> bool {
                // FIXME: make copy before sorting;
                auto a_copy = aidx;
                vector<array<int, 4>> adb, bda;

                sort(aidx.begin(), aidx.end());
                sort(bidx.begin(), bidx.end());
                set_difference(aidx.begin(), aidx.end(), bidx.begin(), bidx.end(), back_inserter(adb));
                set_difference(bidx.begin(), bidx.end(), aidx.begin(), aidx.end(), back_inserter(bda));

                if (adb.size()) {
                    spdlog::error("detection not detected: ");
                    for (int i = 0; i < adb.size(); i++) {
                        auto& a{ adb[i] };
                        spdlog::warn("( {}, {}, {}, {})", a[0], a[1], a[2], a[3]);
                    }
                }
                if (bda.size()) {
                    spdlog::error("False positive: ");
                    for (int i = 0; i < bda.size(); i++) {
                        auto& a{ bda[i] };
                        spdlog::warn("( {}, {}, {}, {})", a[0], a[1], a[2], a[3]);
                    }
                }
                spdlog::info("sizes: ref = {}, iaabb = {}", aidx.size(), bidx.size());
                aidx = a_copy;
                return !(bda.size() || adb.size());
            };
            spdlog::info("PT");
            bool pt_success = compare_collision(
                idx, idx_iaabb
            );

            spdlog::info("EE");
            bool ee_success = compare_collision(
                eidx, eidx_iaabb
            );
            if (!pt_success) spdlog::error("pt fails, exiting");
            if (!ee_success) spdlog::error("ee fails, exiting");
            if (!(pt_success && ee_success)) exit(1);
            else spdlog::info("pt and ee set matched");

#endif
        }
    }

#ifdef _SM_
    map<array<int, 2>, int> lut;
    // look-up table
    SparseMatrix<double> sparse_hess(hess_dim, hess_dim);
    gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);
#endif
#ifdef _TRIPLETS_
    globals.hess_triplets.reserve(((n_pt + n_ee) * 2 + n_cubes) * 12);
#endif

    int n_pt = idx.size(), n_ee = eidx.size(), n_g = vidx.size();
    spdlog::info("constraint size = {}, {}", n_pt, n_ee);

    do {
#ifdef _TRIPLETS_
        globals.hess_triplets.clear();
#endif
        auto newton_iter_start = high_resolution_clock::now();
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_cubes; k++) {
            auto& c(*cubes[k]);
            c.grad = grad_residue_per_body(c, dt);
            c.hess = hess_inertia_per_body(c, dt);
            c.project_vt1();
        }
        for (auto _v : vidx) {
            int i = _v[0], v = _v[1];
            auto& c{ *cubes[i] };
            vec3 p = c.v_transformed[v];
            double _d = vg_distance(p);
            double d = _d * _d;
            if (d < barrier::d_hat) {
#ifdef _FRICTION_
                Matrix<double, 3, 2> Pk;
                Pk.col(0) = vec3(1.0, 0.0, 0.0);
                Pk.col(1) = vec3(0.0, 0.0, 1.0);

                Vector2d uk = Pk.transpose() * (p - c.vt0(v));
                auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * _d;

                ipc_term_vg(c, v, uk, contact_force, Pk);
#else
                ipc_term_vg(c, v);
#endif
            }
        }

#ifdef _SM_
        if (iter) {
            lut.clear();
            sparse_hess.setZero();
            gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);
            n_pt = idx.size();
            n_ee = eidx.size();
            n_g = vidx.size();
        }
        // clear(sparse_hess);
#endif

        auto ipc_start = high_resolution_clock::now();

#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pt; k++) {
            // auto& pt(pts[k]);
            auto& ij(idx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            Face f{ cj, unsigned(ij[3]), false, true };
            array<vec3, 4> pt{ ci.v_transformed[ij[1]], f.t0, f.t1, f.t2 };
            // auto pt_type = ipc::point_triangle_distance_type(pt[0], pt[1], pt[2], pt[3]);
            ipc::PointTriangleDistanceType pt_type;
            double d = vf_distance(pt[0], f, pt_type);
            if (d < barrier::d_hat) {
#ifdef _FRICTION_

                Vector2d uk;
                double contact_force = pt_uktk(ci, cj, pt, ij, pt_type, pt_tk[k], uk, d, dt);

                ipc_term(
                    pt, ij, pt_type, d,
#ifdef _SM_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
                    ci.grad, cj.grad,
                    uk, contact_force, pt_tk[k].transpose());
#else
                ipc_term(pt, ij, pt_type, d,
#ifdef _SM_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
                    ci.grad, cj.grad);
#endif
            }
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_ee; k++) {
            // auto& ee(ees[k]);
            auto& ij(eidx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            Edge ei{ ci, unsigned(ij[1]), false, true }, ej{ cj, unsigned(ij[3]), false, true };
            array<vec3, 4> ee{ ei.e0, ei.e1, ej.e0, ej.e1 };
            auto ee_type = ipc::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
            double d = edge_edge_distance(ee[0], ee[1], ee[2], ee[3], ee_type);
            if (d < barrier::d_hat) {

#ifdef _FRICTION_
                Vector2d uk;
                double contact_force = ee_uktk(ci, cj, ee, ij, ee_type, ee_tk[k], uk, d, dt);

                ipc_term_ee(
                    ee, ij, ee_type, d,
#ifdef _SM_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
                    ci.grad, cj.grad,
                    uk, contact_force, ee_tk[k].transpose());

#else
                ipc_term_ee(ee, ij, ee_type, d,
#ifdef _SM_
                    lut,
                    sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
                    ci.grad, cj.grad);
#endif
            }
        }

        auto ipc_duration = DURATION_TO_DOUBLE(high_resolution_clock::now() - ipc_start);

        double toi = 1.0, factor = 1.0, alpha = 1.0;

        {
            MatrixXd big_hess;
            if (globals.dense)
                big_hess.setZero(hess_dim, hess_dim);
            VectorXd r, q0_cat, q_tile_cat, dq;
            r.setZero(hess_dim);
            dq.setZero(hess_dim);
            q0_cat.setZero(hess_dim);
            q_tile_cat.setZero(hess_dim);
            for (int k = 0; k < n_cubes; k++) {
                auto& c = *cubes[k];
                r.segment<12>(k * 12) = c.grad;
                auto t = cat(c.q);
                q0_cat.segment<12>(k * 12) = t;
                q_tile_cat.segment<12>(k * 12) = c.q_tile(dt, globals.gravity);
            }

#ifdef _TRIPLETS_
            SparseMatrix<double> sparse_hess_trip(hess_dim, hess_dim);
            build_from_triplets(sparse_hess_trip, big_hess, hess_dim, n_cubes);
#endif
#ifdef _SM_
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_cubes; k++) {
                auto& c = *cubes[k];
                double* values = sparse_hess.valuePtr();
                int* outers = sparse_hess.outerIndexPtr();
                int offset = starting_offset(k, k, lut, outers);
                int _stride = stride(k, outers);
                for (int j = 0; j < 12; j++)
                    for (int i = 0; i < 12; i++) {
                        values[offset + _stride * j + i] += c.hess(i, j);
                    }
            }

#endif
            if (globals.damp)
                damping_sparse(sparse_hess, dt, n_cubes);
            auto solver_start = high_resolution_clock::now();

            if (globals.dense)
                dq = -big_hess.ldlt().solve(r);
            else if (globals.sparse) {
                // sparse_hess.setFromTriplets(bht.begin(), bht.end());
                // sparse_hess.finalize();
#ifdef EIGEN_USE_MKL_ALL
                PardisoLLT<SparseMatrix<double>> ldlt_solver;
#else
                SimplicialLLT<SparseMatrix<double>> ldlt_solver;
#endif
                ldlt_solver.compute(sparse_hess);
                dq = -ldlt_solver.solve(r);
#ifdef _TRIPLET_
                sparse_hess_trip.finalize();
                double _dif = (sparse_hess - sparse_hess_trip).norm();
                if (_dif > 1e-6) {
                    cout << "error: dif = " << _dif << "\n\n";
                    cout << sparse_hess_trip << "\n\n"
                         << sparse_hess;
                    exit(0);
                }
#endif
            }
            auto solver_duration = DURATION_TO_DOUBLE(high_resolution_clock::now() - solver_start);
            spdlog::info("solver time = {:0.6f} ms", solver_duration);

            spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm() : big_hess.norm());
            // spdlog::warn("dense norms: dq = {}, grad = {}, big_hess = {}, difference = {}", dq.norm(), r.norm(), big_hess.norm(), dif);
            spdlog::info("dq dot grad = {}, cos = {}", dq.dot(r), dq.dot(r) / (dq.norm() * r.norm()));
            if (globals.sparse && globals.dense && (sparse_hess - big_hess).norm() > 1e-6) {
                spdlog::error("diff too large");
                cout << big_hess << "\n\n"
                     << sparse_hess;
            }

            toi = 1.0;
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_cubes; k++) {
                auto& c(*cubes[k]);
                c.dq = dq.segment<12>(k * 12);
            }

            if (globals.upper_bound) {
                double toi_iaabb;
                if (globals.iaabb)
                    toi_iaabb = iaabb_brute_force(n_cubes, cubes, globals.aabbs, 3, pts, idx, ees, eidx, vidx, pt_tk, ee_tk, false);
#ifndef IAABB_INTERNSHIP
                else
#endif
                    toi = step_size_upper_bound(dq, cubes, n_cubes, n_pt, n_ee, n_g, pts, idx, ees, eidx, vidx);
#ifdef IAABB_INTERNSHIP
                if (toi != toi_iaabb)
                    spdlog::error("step size upper bound not match, toi = {}, iaabb = {}", toi, toi_iaabb);
#else
                toi = toi_iaabb;
#endif
            }

            if (toi < 1.0) {
                spdlog::warn("collision at {}, toi = {}", iter, toi);
                factor = globals.backoff;
            }

            dq *= factor * toi;

            alpha = 1.0;
            double E0 = 0.0, E1 = 0.0;
            if (globals.line_search)
                alpha = line_search(dq, r, q0_cat, E0, E1,
                    n_cubes, n_pt, n_ee, n_g,
                    pts,
                    idx,
                    ees,
                    eidx,
                    vidx,
                    pt_tk,
                    ee_tk,
                    cubes, dt);
            spdlog::info("alpha = {}", alpha);
            dq *= alpha;
            if (alpha < 2e-8) {
                spdlog::error("iter, ts ({}, {}), alpha = {}, E0 = {}, E1 = {}", iter, ts, alpha, E0, E1);
            }
            double norm_dq = dq.norm();
            sup_dq = norm_dq;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_cubes; i++) {
                for (int j = 0; j < 4; j++)
                    cubes[i]->q[j] += dq.segment<3>(i * 12 + j * 3);
            }
            spdlog ::info("step size upper = {}, alpha = {}", toi, alpha);

            auto iter_duration = DURATION_TO_DOUBLE(high_resolution_clock::now() - newton_iter_start);
            spdlog::info("iter {}, time = {} ms, IPC term time = {} \n e0 = {}, e1 = {}, norm_dq = {}\n", iter, iter_duration * 1000, ipc_duration * 1000, E0, E1, norm_dq);
        }
        //         {
        //             // updating collision set
        //             pts.resize(idx.size());
        //             ees.resize(eidx.size());
        // #pragma omp parallel for schedule(static)
        //             for (int k = 0; k < n_cubes; k++) {
        //                 cubes[k]->project_vt1();
        //             }
        // #pragma omp parallel for schedule(static)
        //             for (int k = 0; k < n_pt; k++) {
        //                 auto& ij = idx[k];
        //                 Face f(*cubes[ij[2]], ij[3], false, true);
        //                 vec3 v(cubes[ij[0]]->v_transformed[ij[1]]);
        //                 pts[k] = { v, f.t0, f.t1, f.t2 };
        //             }
        // #pragma omp parallel for schedule(static)
        //             for (int k = 0; k < n_ee; k++) {
        //                 auto& ij = eidx[k];
        //                 Edge ei(*cubes[ij[0]], ij[1], false, true), ej(*cubes[ij[2]], ij[3], false, true);
        //                 ees[k] = { ei.e0, ei.e1, ej.e0, ej.e1 };
        //             }
        //         }
        term_cond = sup_dq < 1e-4 || ++iter >= globals.max_iter;
        sup_dq = 0.0;
    } while (!term_cond);
    spdlog::info("\n  converge at iter {}, ts = {} \n", iter, ts++);
    globals.tot_iter += iter;
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_cubes; k++) {
        auto& c(*cubes[k]);
        for (int i = 0; i < 4; i++) {
            c.dqdt[i] = (c.q[i] - c.q0[i]) / dt;
            c.q0[i] = c.q[i];
        }
        c.p = c.q0[0];
        c.A << c.q0[1], c.q0[2], c.q0[3];
    }
}
