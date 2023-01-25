#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include <assert.h>
#include <array>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include <chrono>
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif
#include <ipc/distance/edge_edge_mollifier.hpp>

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
    double dt
)
{
    double e = 0.0;
// inertia energy
#pragma omp parallel for schedule(static) reduction(+ : e)
    for (int i = 0; i < n_cubes; i++) {
        auto& c(*cubes[i]);
        c.dq = dq.segment<12>(i * 12);

        auto q_tiled = c.q_tile(dt, globals.gravity);
        auto _q = q_plus_dq.segment<12>(12 * i);
        double e_inert = E(_q, q_tiled, c, dt);
        c.project_vt2();
        e += e_inert;
    }

// point-triangle energy
#pragma omp parallel for schedule(static) reduction(+ : e)
    for (int k = 0; k < n_pt; k++) {
        auto& ij = idx[k];
        Face f(*cubes[ij[2]], ij[3], true, true);
        vec3 v(cubes[ij[0]]->v_transformed[ij[1]]);
        // array<vec3, 4> a = {v, f.t0, f.t1, f.t2};
        ipc::PointTriangleDistanceType pt_type;
        double d = vf_distance(v, f, pt_type);
        // double d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2);
        e += barrier::barrier_function(d);
#ifdef _FRICTION_
        auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
        auto v_stack = pt_vstack(*cubes[ij[0]], *cubes[ij[0]], ij[1], ij[3]);
        auto uk = (pt_tk[k] * v_stack).norm();
        e += D_f0(uk, contact_force);
#endif
    }

    // ee ipc energy
#pragma omp parallel for schedule(static) reduction(+ : e)
    for (int k = 0; k < n_ee; k++) {
        auto& ij(eidx[k]);
        Edge ei(*cubes[ij[0]], ij[1], true, true), ej(*cubes[ij[2]], ij[3], true, true);
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
        auto v_stack = ee_vstack(*cubes[ij[0]], *cubes[ij[0]], ij[1], ij[3]);
        auto uk = (ee_tk[k] * v_stack).norm();
        e += D_f0(uk, contact_force);
#endif
    }

    // vertex-ground ipc energy
#pragma omp parallel for schedule(static) reduction(+ : e)
    for (int k = 0; k < n_g; k++) {
        auto& v{
            vidx[k]
        };
        vec3 vt2 = cubes[v[0]]->v_transformed[v[1]];
        double e_ground = E_ground(vt2);
        e += e_ground;
#ifdef _FRICTION_
        auto& c{ *cubes[v[0]] };
        auto vt0 = c.vt0(v[1]);
        double d = vg_distance(vt2);
        if (d * d < barrier::d_hat) {
            auto contact_force = -barrier_derivative_d(d * d) / (dt * dt) * 2 * d;
            vec3 _uk = vt2 - vt0;
            double uk = sqrt(_uk(0) * _uk(0) + _uk(1) * _uk(1));
            e += D_f0(uk, contact_force);
        }
#endif
    }
    return e;
}
double line_search(const VectorXd& dq, const VectorXd& grad, VectorXd& q0, double& E0, double& E1,
    int n_cubes, int n_pt, int n_ee, int n_g,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<double, 2, 12>>& pt_tk,
    const vector<Matrix<double, 2, 12>>& ee_tk,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt
)
{
    const double c1 = 1e-4;
    double alpha = 1.0;
    bool wolfe = false;
    E0 = E_global(q0, 0.0 * dq,
        n_cubes, n_pt, n_ee, n_g,
        idx,
        eidx,
        vidx,
        pt_tk,
        ee_tk,
        cubes, dt);
    double qdg = dq.dot(grad);
    // double E0 = E(q0, q_tiled, c);
    VectorXd q1;
    do {
        q1 = q0 + dq * alpha;
        auto dqk = dq * alpha;

        E1 = E_global(q1, dqk,
            n_cubes, n_pt, n_ee, n_g,
            idx,
            eidx,
            vidx,
            pt_tk,
            ee_tk,
            cubes, dt

        );
        wolfe = E1 <= E0 + c1 * alpha * qdg;
        // spdlog::info("wanted descend = {}, E1 - E0 = {}, E1 = {}, E0 = {}, alpha = {}", c1 * alpha * qdg, E1 - E0, E1, E0, alpha);
        alpha /= 2;
        if (alpha < 1e-8) break;
    } while (!wolfe && grad.norm() > 1e-3);

    return alpha * 2;
}


void implicit_euler(vector<unique_ptr<AffineBody>>& cubes, double dt)
{
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

    double D_friction;
    const int n_cubes = cubes.size(), nsqr = n_cubes * n_cubes, hess_dim = n_cubes * 12;

    if (globals.col_set)
        gen_collision_set(n_cubes, cubes,
            pts,
            idx,
            ees,
            eidx,
            vidx,
            pt_tk,
            ee_tk);

#ifdef _SM_
    map<array<int, 2>, int> lut;
    // look-up table
    SparseMatrix<double> sparse_hess(hess_dim, hess_dim);
    gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);
#endif
#ifdef _TRIPLETS_
    globals.hess_triplets.reserve(((n_pt + n_ee) * 2 + n_cubes) * 12);
#endif

    const int n_pt = idx.size(), n_ee = eidx.size(), n_g = vidx.size();
    spdlog::info("constraint size = {}, {}", n_pt, n_ee);

    

    do {
        globals.hess_triplets.clear();
        auto newton_iter_start = high_resolution_clock::now();
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_cubes; k++) {
            auto& c(*cubes[k]);
            c.grad = grad_residue_per_body(c, dt);
            c.hess = hess_inertia_per_body(c, dt);
        }
        for (auto _v : vidx) {
            int i = _v[0], v = _v[1];
            auto& c{ *cubes[i] };
            vec3 p = c.vt1(v);
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
        clear(sparse_hess);
#endif

        auto ipc_start = high_resolution_clock::now();

#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pt; k++) {
            auto& pt(pts[k]);
            auto& ij(idx[k]);

            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            // auto pt_type = ipc::point_triangle_distance_type(pt[0], pt[1], pt[2], pt[3]);
            ipc::PointTriangleDistanceType pt_type;
            double d = vf_distance(pt[0], Face{pt[1], pt[2], pt[3]}, pt_type);
            if (d < barrier::d_hat) {
#ifdef _FRICTION_

                Vector<double, 12> v_stack = pt_vstack(ci, cj, ij[1], ij[3]);

                auto lams = ipc::point_triangle_closest_point(pt[0], pt[1], pt[2], pt[3]);
                array<double, 3> tlams = { 1 - lams(0) - lams(1), lams(0), lams(1) };

                if (pt_type == ipc::PointTriangleDistanceType::P_T)
                    ; // do nothing
                else if (pt_type == ipc::PointTriangleDistanceType::P_T0)
                    tlams = { 1.0, 0.0, 0.0 };
                else if (pt_type == ipc::PointTriangleDistanceType::P_T1)
                    tlams = { 0.0, 1.0, 0.0 };
                else if (pt_type == ipc::PointTriangleDistanceType::P_T2)
                    tlams = { 0.0, 0.0, 1.0 };
                else if (pt_type == ipc::PointTriangleDistanceType::P_E0) {
                    auto elam = ipc::point_edge_closest_point(pt[0], pt[1], pt[2]);
                    tlams = { 1.0 - elam, elam, 0.0 };
                }
                else if (pt_type == ipc::PointTriangleDistanceType::P_E1) {
                    auto elam = ipc::point_edge_closest_point(pt[0], pt[2], pt[3]);
                    tlams = { 0.0, 1.0 - elam, elam };
                }
                else if (pt_type == ipc::PointTriangleDistanceType::P_E2) {
                    auto elam = ipc::point_edge_closest_point(pt[0], pt[3], pt[1]);
                    tlams = { elam, 0.0, 1.0 - elam };
                }

                auto tp = (pt[1] * tlams[0] + pt[2] * tlams[1] + pt[3] * tlams[2]);
                auto closest = (pt[0] - tp).squaredNorm();
                assert(abs(d - closest) < 1e-8);
                auto Pk = ipc::point_triangle_tangent_basis(pt[0], pt[1], pt[2], pt[3]);
                Matrix<double, 3, 12> gamma;
                gamma.setZero(3, 12);
                for (int i = 0; i < 3; i++) {
                    gamma(i, i) = -1.0;
                    for (int j = 0; j < 3; j++)
                        gamma(i, i + 3 + 3 * j) = tlams[j];
                }
                // gamma.block<3, 3>(0, 0) = Matrix3d::Identity(3, 3) * -1.0;
                // gamma.block<3, 3>(0, 3) = Matrix3d::Identity(3, 3) * tlams[0];
                // gamma.block<3, 3>(0, 6) = Matrix3d::Identity(3, 3) * tlams[1];
                // gamma.block<3, 3>(0, 9) = Matrix3d::Identity(3, 3) * tlams[2];

                Matrix<double, 2, 12> Tk_T = Pk.transpose() * gamma;

                // Matrix<double, 12, 24> jacobian;
                // auto _0 = ci.indices[f * 3], _1 = ci.indices[f * 3 + 1], _2 = ci.indices[f * 3 + 2];
                // jacobian.setZero(12, 24);
                // jacobian.block<3, 12>(0, 0) = x_jacobian_q(ci.vertices(v));
                // jacobian.block<3, 12>(3, 12) = x_jacobian_q(cj.vertices(_0));
                // jacobian.block<3, 12>(6, 12) = x_jacobian_q(cj.vertices(_1));
                // jacobian.block<3, 12>(9, 12) = x_jacobian_q(cj.vertices(_2));

                // auto Tq_k = Tk_T * jacobian;

                // auto contact_force_lam = - barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
                Vector2d uk = Tk_T * v_stack;
                auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
                pt_tk[k] = Tk_T;

                ipc_term(
                    pt, ij, pt_type, d,
#ifdef _SM_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
                    ci.grad, cj.grad,
                    uk, contact_force, Tk_T.transpose());
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
            auto& ee(ees[k]);
            auto& ij(eidx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            auto ee_type = ipc::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
            double d = edge_edge_distance(ee[0], ee[1], ee[2], ee[3], ee_type);
            if (d < barrier::d_hat) {

#ifdef _FRICTION_
                auto v_stack = ee_vstack(ci, cj, ij[1], ij[3]);
                auto ei0 = ee[0], ei1 = ee[1], ej0 = ee[2], ej1 = ee[3];
                auto rei = ei0 - ei1, rej = ej0 - ej1;
                auto cnorm = rei.cross(rej).squaredNorm();
                auto sin2 = cnorm / rei.squaredNorm() / rej.squaredNorm();
                Matrix<double, 3, 2> degeneracy;
                degeneracy.col(0) = rei.normalized();
                degeneracy.col(1) = (ej0 - ei0).cross(rei).normalized();
                bool par = sin2 < 1e-8;

                auto lams = ipc::edge_edge_closest_point(ei0, ei1, ej0, ej1);

                if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB)
                    ;

                else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB0)
                    lams = { 0.0, 0.0 };
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB1)
                    lams = { 0.0, 1.0 };
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB0)
                    lams = { 1.0, 0.0 };
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB1)
                    lams = { 1.0, 1.0 };
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB0) {
                    auto pe = ipc::point_edge_closest_point(ej0, ei0, ei1);
                    lams = { pe, 0.0 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB1) {
                    auto pe = ipc::point_edge_closest_point(ej1, ei0, ei1);
                    lams = { pe, 1.0 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB) {
                    auto pe = ipc::point_edge_closest_point(ei0, ej0, ej1);
                    lams = { 0.0, pe };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB) {
                    auto pe = ipc::point_edge_closest_point(ei1, ej0, ej1);
                    lams = { 1.0, pe };
                }
                const auto clip = [&](double& a, double l, double u) {
                    a = max(min(u, a), l);
                };
                clip(lams(0), 0.0, 1.0);
                clip(lams(1), 0.0, 1.0);
                array<double, 4> lambdas = { 1 - lams(0), lams(0), 1 - lams(1), lams(1) };
                if (par) {
                    // ignore this friction, already handled in point-triangle pair
                    lambdas = { 0.0, 0.0, 0.0, 0.0 };
                    // uk will be set to zero
                }
                auto pei = ei0 * lambdas[0] + ei1 * lambdas[1];
                auto pej = ej0 * lambdas[2] + ej1 * lambdas[3];
                auto closest = (pei - pej).squaredNorm();
                assert(par || abs(d - closest) < 1e-8);

                auto Pk = par ? degeneracy : ipc::edge_edge_tangent_basis(ei0, ei1, ej0, ej1);
                Matrix<double, 3, 12> gamma;
                gamma.setZero(3, 12);
                for (int i = 0; i < 3; i++) {
                    gamma(i, i) = -lambdas[0];
                    gamma(i, i + 3) = -lambdas[1];
                    gamma(i, i + 6) = lambdas[2];
                    gamma(i, i + 9) = lambdas[3];
                }
                // gamma.block<3, 3>(0, 0) = Matrix3d::Identity(3, 3) * -lambdas[0];
                // gamma.block<3, 3>(0, 3) = Matrix3d::Identity(3, 3) * -lambdas[1];
                // gamma.block<3, 3>(0, 6) = Matrix3d::Identity(3, 3) * lambdas[2];
                // gamma.block<3, 3>(0, 9) = Matrix3d::Identity(3, 3) * lambdas[3];

                Matrix<double, 2, 12> Tk_T = Pk.transpose() * gamma;

                // Matrix<double, 12, 24> jacobian;
                // auto _i0 = ci.edges[_ei * 2], _i1 = ci.edges[_ei * 2 + 1],
                //      _j0 = cj.edges[_ej * 2], _j1 = cj.edges[_ej * 2 + 1];

                // jacobian.setZero(12, 24);
                // jacobian.block<3, 12>(0, 0) = x_jacobian_q(ci.vertices(_i0));
                // jacobian.block<3, 12>(3, 0) = x_jacobian_q(cj.vertices(_i1));
                // jacobian.block<3, 12>(6, 12) = x_jacobian_q(cj.vertices(_j0));
                // jacobian.block<3, 12>(9, 12) = x_jacobian_q(cj.vertices(_j1));

                // auto Tq_k = Tk_T * jacobian;

                // auto contact_force_lam = barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
                Vector2d uk = Tk_T * v_stack;
                auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
                ee_tk[k] = Tk_T;

                ipc_term_ee(
                    ee, ij, ee_type, d,
#ifdef _SM_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
                    ci.grad, cj.grad,
                    uk, contact_force, Tk_T.transpose());

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

        const auto step_size_upper_bound = [&](VectorXd& dq, vector<unique_ptr<AffineBody>>& cubes) -> double {
            auto start = high_resolution_clock::now();
            double toi = barrier::d_sqrt / 2.0 / norm_1(dq, n_cubes) * globals.safe_factor;
            toi = min(1.0, toi);

#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_cubes; k ++ ){
                cubes[k] -> project_vt2();
            }
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_g; k ++) {
                auto & v {vidx[k]};
                double t = collision_time(*cubes[v[0]], v[1]);
                toi = min(t, toi);
            }
            vector<double> tois;
            tois.resize(n_pt + n_ee);
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_pt; k++) {
                auto& pt(pts[k]);
                const auto& ij(idx[k]);

                int i = ij[0], j = ij[2];
                // Face f(cubes[j], ij[3], true);
                vec3 p_t2 = cubes[i]->v_transformed[ij[1]];
                vec3 p_t1 = cubes[i]->vt1(ij[1]);
                // double t = vf_collision_detect(p_t1, p_t2,
                //     *cubes[j], ij[3]);
                double t = pt_collision_time(p_t1, Face(*cubes[j], ij[3]), p_t2, Face(*cubes[j], ij[3], true, true));
                // double t = vf_collision_detect(p_t1, p_t2, Face(*cubes[j], ij[3]), Face(*cubes[j], ij[3], true));
                tois[k] = t;
                // toi = min(toi, t);
            }
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_ee; k++) {
                auto& ee(ees[k]);
                const auto& ij(eidx[k]);

                auto &ci(*cubes[ij[0]]), &cj(*cubes[ij[2]]);
                // double t = ee_collision_detect(ci, cj, ij[1], ij[3]);
                Edge ei0 (ci, ij[1]), ei1 (ci, ij[1], true, true);
                Edge ej0 (cj, ij[3]), ej1 (cj, ij[3], true, true);
                double t = ee_collision_time(ei0, ej0, ei1, ej1);
                tois[k + n_pt] = t;
                // toi = min(toi, t);
            }
            for (auto t : tois) toi = min(toi, t);
            auto _duration = DURATION_TO_DOUBLE(high_resolution_clock::now() - start);
            spdlog::info("time: step size upper bound = {:0.6f} ms", _duration * 1000);
            return toi;
        };
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
                double * values = sparse_hess.valuePtr();
                int * outers = sparse_hess.outerIndexPtr();
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
                PardisoLDLT<SparseMatrix<double>> ldlt_solver;
#else
                SimplicialLDLT<SparseMatrix<double>> ldlt_solver;
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
            auto solver_duration = DURATION_TO_DOUBLE (high_resolution_clock::now() - solver_start);
            spdlog::info("solver time = {:0.6f} ms", solver_duration);

            spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm(): big_hess.norm());
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

            if (globals.upper_bound)
                toi = step_size_upper_bound(dq, cubes);

            if (toi < 1.0) {
                spdlog::warn("collision at {}, toi = {}", iter, toi);
                factor = 0.95;
            }

            dq *= factor * toi;

            alpha = 1.0;
            double E0 = 0.0, E1 = 0.0;
            if (globals.line_search)
                alpha = line_search(dq, r, q0_cat, E0, E1,
                    n_cubes, n_pt, n_ee, n_g,
                    idx,
                    eidx,
                    vidx,
                    pt_tk,
                    ee_tk,
                    cubes, dt);
            spdlog::info("alpha = {}", alpha);
            dq *= alpha;
            if (alpha < 1e-6) {
                spdlog::error("alpha = {}, E0 = {}, E1 = {}", alpha, E0, E1);
            }
            double norm_dq = dq.norm();
            sup_dq = norm_dq;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_cubes; i++) {
                for (int j = 0; j < 4; j++)
                    cubes[i]->q[j] += dq.segment<3>(i * 12 + j * 3);
            }
            spdlog ::info("step size upper = {}, alpha = {}", toi, alpha);

            auto iter_duration = DURATION_TO_DOUBLE (high_resolution_clock::now() - newton_iter_start);
            spdlog::info("iter {}, time = {} ms, IPC term time = {} \n e0 = {}, e1 = {}, norm_dq = {}\n", iter, iter_duration * 1000, ipc_duration
                 * 1000, E0, E1, norm_dq);
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_cubes; k++){
            cubes[k]->project_vt1();
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pt; k++) {
            auto& ij = idx[k];
            Face f(*cubes[ij[2]], ij[3], false, true);
            vec3 v(cubes[ij[0]]->v_transformed[ij[1]]);
            pts[k] = { v, f.t0, f.t1, f.t2 };
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_ee; k++) {
            auto& ij = eidx[k];
            Edge ei(*cubes[ij[0]], ij[1], false, true), ej(*cubes[ij[2]], ij[3], false, true);
            ees[k] = { ei.e0, ei.e1, ej.e0, ej.e1 };
        }
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
