// #include "IpcFrictionConstraint.h"
#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include <assert.h>
#include <array>
#include "timer.h"
#include "../iAABB/finitediff.hpp"

// #include <ipc/distance/point_triangle.hpp>
// #include <ipc/distance/edge_edge.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif
#include <ipc/distance/edge_edge_mollifier.hpp>
//#define IAABB_COMPARING
// #define IAABB_INTERNSHIP
#ifdef IAABB_COMPARING
#include <algorithm>
#endif
using namespace std;
using namespace barrier;
using namespace Eigen;
using namespace utils;

#define __IPC__ 0
#define __SOLVER__ 1
#define __CCD__ 2
#define __LINE_SEARCH__ 3
#include "cuda_glue.h"
void hess_cuda(int n_cubes, float dt, float* grads, float* hess);
static const bool strict = false;
void cuda_hess_glue(
    int n_cubes,
    const vector<unique_ptr<AffineBody>>& cubes,
    float dt)
{
    vector<cudaAffineBody> cabs(n_cubes);
    float* grads = new float[n_cubes * 12];
    float* hess = new float[n_cubes * 144];
    for (int i = 0; i < n_cubes; i++) cabs[i] = to_cabd(*cubes[i]);
    cudaMemcpy(host_cuda_globals.cubes, cabs.data(), sizeof(cudaAffineBody) * cabs.size(), cudaMemcpyHostToDevice);
    hess_cuda(n_cubes, dt, grads, hess);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_cubes; k++) {
        auto& c(*cubes[k]);
        c.grad = grad_residue_per_body(c, dt);
        c.hess = hess_inertia_per_body(c, dt);
        c.project_vt1();
    }
    
    for (int k = 0; k < n_cubes; k++) {
        auto& c{ *cubes[k] };
        vec12 ref = c.grad, act = Map<Vector<float, 12>>(grads + k * 12).cast<double>();
        mat12 h_ref = c.hess, h_act = Map<Matrix<float, 12, 12>>(hess + k * 144).cast<double>();
        auto norm = (ref - act).norm();
        auto h_norm = (h_ref - h_act).norm();
        if (!strict) {
            if (norm > 1e-1)
                spdlog::warn("norm too large: {}", norm);
            if (! h_act.isApprox(h_ref, 1e-3))
                spdlog::warn("hess norm too large: {}", h_norm);
        }
        else {
            assert(norm < 1e-2);
            assert(h_norm < 1e-2);
        }
    }
    delete [] grads; 
    delete[] hess;
}
void cuda_solve(Eigen::VectorXd& dq, Eigen::SparseMatrix<double>& sparse_hess, Eigen::VectorXd& r);

double E_global(const VectorXd& q_plus_dq, const VectorXd& dq, int n_cubes, int n_pt, int n_ee, int n_g,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<double, 2, 12>>& pt_tk,
    const vector<Matrix<double, 2, 12>>& ee_tk,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt, double& _ef, bool _vt2);


double E_barrier_plus_inert(const VectorXd& q_plus_dq, const VectorXd& dq, int n_cubes,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt);

double E_fric(
    const VectorXd& dq, int n_cubes, 
    int n_pt, int n_ee, int n_g,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<double, 2, 12>>& pt_tk,
    const vector<Matrix<double, 2, 12>>& ee_tk,
    const vector<double>& pt_contact_forces,
    const vector<double>& ee_contact_forces,
    const vector<double>& g_contact_forces,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt);

double line_search(const VectorXd& dq, const VectorXd& grad, VectorXd& q0, double& E0, double& E1,
    int n_cubes, int n_pt, int n_ee, int n_g,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx,
    const vector<Matrix<double, 2, 12>>& pt_tk,
    const vector<Matrix<double, 2, 12>>& ee_tk,
    const vector<double> &pt_contact_forces,
    const vector<double> &ee_contact_forces,
    const vector<double> &g_contact_forces,
    const vector<unique_ptr<AffineBody>>& cubes,
    double dt)
{
    static double tol
    {
        globals.params_double["tol"]
    };
    static const double c1 = globals.params_double["c1"];
    double alpha = 1.0;
    bool wolfe = false;
    double ef0 = 0.0;
    E0 = E_barrier_plus_inert(q0, 0.0 * dq, n_cubes, idx, eidx, vidx, cubes, dt) 
    + E_fric(0.0 * dq, n_cubes, n_pt, n_ee, n_g,
        idx, eidx, vidx,
        pt_tk, ee_tk, 
        pt_contact_forces, ee_contact_forces, g_contact_forces, 
        cubes, dt    
    );
    double qdg = dq.dot(grad);
    VectorXd q1;
    static vector<array<vec3, 4>> pts_new, pts_iaab;
    static vector<array<int, 4>> idx_new, idx_iaab;

    static vector<array<vec3, 4>> ees_new, ees_iaab;
    static vector<array<int, 4>> eidx_new, eidx_iaab;

    static vector<array<int, 2>> vidx_new, vidx_iaab;

    auto dq_norm = dq.norm();
    do {
        q1 = q0 + dq * alpha;
        auto dqk = dq * alpha;
        for (int i = 0; i < n_cubes; i++) {
            auto& c(*cubes[i]);
            c.dq = dqk.segment<12>(i * 12);
        }

        if (globals.iaabb % 2)
            iaabb_brute_force(n_cubes, cubes, globals.aabbs, 2,
#ifdef IAABB_COMPARING
                pts_iaab,
                idx_iaab,
                ees_iaab,
                eidx_iaab,
                vidx_iaab);
#else
                pts_new,
                idx_new,
                ees_new,
                eidx_new,
                vidx_new);
        else
#endif
        gen_collision_set(true, n_cubes, cubes,
            pts_new,
            idx_new,
            ees_new,
            eidx_new,
            vidx_new);
        double ef1 = 0.0, E2 = 0.0, ef2 = 0.0;
        double E3 = E_barrier_plus_inert(q1, dqk, n_cubes, idx_new, eidx_new, vidx_new, cubes, dt);
        double ef = E_fric(dqk, n_cubes, n_pt, n_ee, n_g, idx, eidx, vidx, pt_tk, ee_tk, pt_contact_forces, ee_contact_forces, g_contact_forces, cubes,  dt);
        E1 = E3 + ef;
        wolfe = E1 <= E0 + c1 * alpha * qdg;
        // spdlog::info("wanted descend = {}, E1 - E0 = {}, E1 = {}, E0 = {}, alpha = {}", c1 * alpha * qdg, E1 - E0, E1, E0, alpha);
        alpha /= 2;
        if (!(!wolfe && grad.norm() > 1e-3)) break;
        if (dq_norm * alpha * 2 < tol) {
            // smaller than Newton iter convergence condition, clip it
            if (globals.params_int["clip"]) {
                alpha = 0.0;
                break;
            }
            else {
                // continue to loop until & line search
            }
        }
    } while (true);
    pts = pts_new;
    idx = idx_new;
    ees = ees_new;
    eidx = eidx_new;
    vidx = vidx_new;

    return alpha * 2;
}

void implicit_euler(vector<unique_ptr<AffineBody>>& cubes, double dt)
{
    bool term_cond;
    int& ts = globals.ts;
    static double tol = globals.params_double["tol"];
    static VectorXd lastdq;
    int iter = 0;
    double sup_dq = 0.0;
    for (int k = 0; k < cubes.size(); k++) {
        auto& c(*cubes[k]);
        for (int i = 0; i < 4; i++) {
            c.q[i] = c.q0[i];
        }
    }

    static vector<array<vec3, 4>> pts;
    static vector<array<int, 4>> idx;

    static vector<array<vec3, 4>> ees;
    static vector<array<int, 4>> eidx;

    static vector<array<int, 2>> vidx;

    static vector<Matrix<double, 2, 12>> pt_tk;
    static vector<Matrix<double, 2, 12>> ee_tk;
    static Vector4d times;
    static int n_pt, n_ee, n_g;
    static vector<double> pt_contact_forces, ee_contact_forces, g_contact_forces;
    times.setZero(4);
    auto frame_start = high_resolution_clock::now();
#ifdef IAABB_COMPARING
    vector<array<vec3, 4>> pts_iaabb;
    vector<array<int, 4>> idx_iaabb;

    vector<array<vec3, 4>> ees_iaabb;
    vector<array<int, 4>> eidx_iaabb;

    vector<array<int, 2>> vidx_iaabb;

    vector<Matrix<double, 2, 12>> pt_tk_iaabb;
    vector<Matrix<double, 2, 12>> ee_tk_iaabb;
#endif

    const int n_cubes = cubes.size(), hess_dim = n_cubes * 12;

    if (globals.col_set) {
        if (globals.iaabb % 2)
            iaabb_brute_force(n_cubes, cubes, globals.aabbs, 1,
#ifdef IAABB_COMPARING
                pts_iaabb,
                idx_iaabb,
                ees_iaabb,
                eidx_iaabb,
                vidx_iaabb);
#else
                pts,
                idx,
                ees,
                eidx,
                vidx);
        else
#endif
        gen_collision_set(false, n_cubes, cubes,
            pts,
            idx,
            ees,
            eidx,
            vidx);

        if (globals.iaabb % 2) {

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
                idx, idx_iaabb);

            spdlog::info("EE");
            bool ee_success = compare_collision(
                eidx, eidx_iaabb);
            if (!pt_success) spdlog::error("pt fails, exiting");
            if (!ee_success) spdlog::error("ee fails, exiting");
            if (!(pt_success && ee_success))
                exit(1);
            else
                spdlog::info("pt and ee set matched");

#endif
        }
    }

#ifdef _SM_
    map<array<int, 2>, int> lut;
    // look-up table
    SparseMatrix<double> sparse_hess(hess_dim, hess_dim);
#endif
#ifdef _TRIPLETS_
    globals.hess_triplets.reserve(((n_pt + n_ee) * 2 + n_cubes) * 12);
#endif

    ///////////////////////// MAIN LOOP /////////////////////
    do {
#ifdef _TRIPLETS_
        globals.hess_triplets.clear();
#endif
        auto newton_iter_start = high_resolution_clock::now();

        if (globals.params_int["cuda_hess"]) {
            cuda_hess_glue(n_cubes, cubes, dt);
        }
        else
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_cubes; k++) {
            auto& c(*cubes[k]);
            c.grad = grad_residue_per_body(c, dt);
            c.hess = hess_inertia_per_body(c, dt);
            c.project_vt1();
        }

#ifdef _SM_
        {
            lut.clear();
            sparse_hess.setZero();
            if (globals.params_int["gen_sm_test"] && iter ==  0)
                gen_empty_sm_glue(n_cubes, idx, eidx, sparse_hess, lut);
            else
                gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);
  
            n_pt = idx.size();
            n_ee = eidx.size();
            n_g = vidx.size();
            pt_tk.resize(n_pt);
            pt_contact_forces.resize(n_pt);
            ee_tk.resize(n_ee);
            ee_contact_forces.resize(n_ee);
            g_contact_forces.resize(n_g);
            spdlog::info("constraint size = {}, {}", n_pt, n_ee);
        }
#endif
        double evh = globals.evh;
        auto ipc_start = high_resolution_clock::now();
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n_pt; k++) {
            // auto& pt(pts[k]);
            auto& ij(idx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            Face f{ cj, unsigned(ij[3]), false, true };
            vec3 p{ ci.v_transformed[ij[1]] };
            array<vec3, 4> pt{ p, f.t0, f.t1, f.t2 };
            auto [d, pt_type] = vf_distance(pt[0], f);
            if (d < barrier::d_hat) {
#ifdef _PLUG_IN_LAN_

                Vector4i cid_p{ 0, 0, 0, 0 };
                Vector4i cid_t0{ 4, 4, 4, 4 };
                Vector4i cid_t1{ 4, 4, 4, 4 };
                Vector4i cid_t2{ 4, 4, 4, 4 };

                auto pr{ ci.vertices(ij[1]) },
                    t0r{ cj.vertices(cj.indices[ij[3] * 3]) },
                    t1r{ cj.vertices(cj.indices[ij[3] * 3 + 1]) },
                    t2r{ cj.vertices(cj.indices[ij[3] * 3 + 2]) };

                auto p0{ ci.vt0(ij[1]) },
                    t00{ cj.vt0(cj.indices[ij[3] * 3]) },
                    t10{ cj.vt0(cj.indices[ij[3] * 3 + 1]) },
                    t20{ cj.vt0(cj.indices[ij[3] * 3 + 2]) };

                Vector4d w_p{ 1.0, pr[0], pr[1], pr[2] };
                Vector4d w_t0{ 1.0, t0r[0], t0r[1], t0r[2] };
                Vector4d w_t1{ 1.0, t1r[0], t1r[1], t1r[2] };
                Vector4d w_t2{ 1.0, t2r[0], t2r[1], t2r[2] };

                vector<vec3> surface_x{ p, f.t0, f.t1, f.t2 }, surface_xhat{ p0, t00, t10, t20 }, surface_X{ pr, t0r, t1r, t2r };

                vector<pair<Vector4i, Vector4d>> dpdx{ { cid_p, w_p }, { cid_t0, w_t0 }, { cid_t1, w_t1 }, { cid_t2, w_t2 } };
                vec12 ga, gb;
                ga.setZero(12);
                gb.setZero(12);
                mat12 ha, hb, hab;
                ha.setZero(12, 12);
                hb.setZero(12, 12);
                hab.setZero(12, 12);
                VectorXd gaf, gbf;
                gaf.setZero(12);
                gbf.setZero(12);
                MatrixXd haf, hbf, habf;
                haf.setZero(12, 12);
                hbf.setZero(12, 12);
                habf.setZero(12, 12);
                VectorXd gac, gbc;
                gac.setZero(12);
                gbc.setZero(12);
                MatrixXd hac, hbc, habc;
                hac.setZero(12, 12);
                hbc.setZero(12, 12);
                habc.setZero(12, 12);
                AIPC::IpcFrictionConstraintOp3D* friction_constraint;
                AIPC::IpcConstraintOp3D* constraint;
                if (pt_type == ipc::PointTriangleDistanceType::P_T0) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPPConstraint(0, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }

                else if (pt_type == ipc::PointTriangleDistanceType::P_T1) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPPConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }
                else if (pt_type == ipc::PointTriangleDistanceType::P_T2) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPPConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }
                else if (pt_type == ipc::PointTriangleDistanceType::P_E0) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPEConstraint(0, 0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }
                else if (pt_type == ipc::PointTriangleDistanceType::P_E1) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPEConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }
                else if (pt_type == ipc::PointTriangleDistanceType::P_E2) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 3, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPEConstraint(0, 0, 3, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }
                else {
                    friction_constraint = new AIPC::IpcPTFConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    constraint = new AIPC::IpcPTConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                }

#endif

                vec12 gradp, gradt;
                mat12 hess_p, hess_t, off_diag;
                ipc_term(
                    pt, ij, pt_type, d,
#ifdef _SM_OUT_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
#ifdef _DIRECT_OUT_
                    hess_p, hess_t, off_diag,
                    gradp, gradt
#else
                    ci.grad, cj.grad
#endif
#ifdef _FRICTION_
                    ,
                    pt_contact_forces[k], pt_tk[k]
#endif
                );
#ifdef _PLUG_IN_LAN_
                constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gac, gbc);
                constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hac, hbc, habc);

                friction_constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gaf, gbf);
                friction_constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, haf, hbf, habf);

              
                ga = gaf + gac;
                gb = gbf + gbc;
                ha = haf + hac;
                hb = hbf + hbc;
                hab = habf + habc;

                ga /= barrier::d_hat;
                gb /= barrier::d_hat;
                ha /= barrier::d_hat;
                hb /= barrier::d_hat;
                hab /= barrier::d_hat;
                output_hessian_gradient(
                    lut, sparse_hess,
                    i, j, ci.mass > 0.0, cj.mass > 0.0,
                    ci.grad, cj.grad,

                    // gradp, gradt, hess_p, hess_t, off_diag, off_diag.transpose()
                    ga, gb, ha, hb, hab, hab.transpose()
                    
                );
                bool b0 = ::fd::compare_gradient(ga, gradp);
                bool b1 = ::fd::compare_gradient(gb, gradt);

                bool b2 = fd::compare_hessian(ha, hess_p);
                bool b3 = fd::compare_hessian(hb, hess_t);
                bool b4 = fd::compare_hessian(hab, off_diag);

                if (!b0) {
                    spdlog::error("gradient p error");
                }
                if (!b1) {
                    spdlog::error("gradient t error");
                }
                if (!b2) {
                    spdlog::error("hessian p error");
                }
                if (!b3) {
                    spdlog::error("hessian t error");
                }
                if (!b4) {
                    spdlog::error("hessian off_diag error");
                }
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
                mat12 hess_0, hess_1, off_diag;
                vec12 grad_0, grad_1;
                double eps_x = globals.eps_x * (ee[0] - ee[1]).squaredNorm() * (ee[2] - ee[3]).squaredNorm();

                double mollifier = ipc::edge_edge_mollifier(ee[0], ee[1], ee[2], ee[3], eps_x);
                ipc_term_ee(
                    ee, ij, ee_type, d,
#ifdef _SM_OUT_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    globals.hess_triplets,
#endif
#ifdef _DIRECT_OUT_
                    hess_0, hess_1, off_diag,
                    grad_0, grad_1

#else
                    ci.grad, cj.grad
#endif
#ifdef _FRICTION_
                    ,
                    ee_contact_forces[k], ee_tk[k]
#endif
                );

#ifdef _PLUG_IN_LAN_

                Vector4i cid_ei0{ 0, 0, 0, 0 };
                Vector4i cid_ei1{ 0, 0, 0, 0 };
                Vector4i cid_ej0{ 4, 4, 4, 4 };
                Vector4i cid_ej1{ 4, 4, 4, 4 };

                auto ei0r{ ci.vertices(ci.edges[ij[1] * 2]) },
                    ei1r{ ci.vertices(ci.edges[ij[1] * 2 + 1]) },
                    ej0r{ cj.vertices(cj.edges[ij[3] * 2]) },
                    ej1r{ cj.vertices(cj.edges[ij[3] * 2 + 1]) };

                auto ei00{ ci.vt0(ci.edges[ij[1] * 2]) },
                    ei10{ ci.vt0(ci.edges[ij[1] * 2 + 1]) },
                    ej00{ cj.vt0(cj.edges[ij[3] * 2]) },
                    ej10{ cj.vt0(cj.edges[ij[3] * 2 + 1]) };

                Vector4d w_ei0{ 1.0, ei0r[0], ei0r[1], ei0r[2] };
                Vector4d w_ei1{ 1.0, ei1r[0], ei1r[1], ei1r[2] };
                Vector4d w_ej0{ 1.0, ej0r[0], ej0r[1], ej0r[2] };
                Vector4d w_ej1{ 1.0, ej1r[0], ej1r[1], ej1r[2] };

                vector<vec3> surface_x{ ee[0], ee[1], ee[2], ee[3] }, surface_xhat{ ei00, ei10, ej00, ej10 }, surface_X{ ei0r, ei1r, ej0r, ej1r };
                vector<vec3> sx = surface_x, sX = surface_X;
                vector<pair<Vector4i, Vector4d>> dpdx{ { cid_ei0, w_ei0 }, { cid_ei1, w_ei1 }, { cid_ej0, w_ej0 }, { cid_ej1, w_ej1 } }, px = dpdx;
                vec12 ga, gb;
                ga.setZero(12);
                gb.setZero(12);
                mat12 ha, hb, hab;
                ha.setZero(12, 12);
                hb.setZero(12, 12);
                hab.setZero(12, 12);
                VectorXd gaf, gbf;
                gaf.setZero(12);
                gbf.setZero(12);
                MatrixXd haf, hbf, habf;
                haf.setZero(12, 12);
                hbf.setZero(12, 12);
                habf.setZero(12, 12);
                VectorXd gac, gbc;
                gac.setZero(12);
                gbc.setZero(12);
                MatrixXd hac, hbc, habc;
                hac.setZero(12, 12);
                hbc.setZero(12, 12);
                habc.setZero(12, 12);
                AIPC::IpcFrictionConstraintOp3D* friction_constraint;
                AIPC::IpcConstraintOp3D* constraint;
                if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB0) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPPConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                    constraint = new AIPC::IpcPPMConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB1) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPPConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

                    constraint = new AIPC::IpcPPMConstraint(0, 0, 1, 3, 2, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                    // perm = { 0, 1, 3, 2 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB0) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPPConstraint(0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

                    constraint = new AIPC::IpcPPMConstraint(0, 1, 0, 2, 3, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                    // perm = { 1, 0, 2, 3 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB1) {
                    friction_constraint = new AIPC::IpcPPFConstraint(0, 1, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPPConstraint(0, 1, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

                    constraint = new AIPC::IpcPPMConstraint(0, 1, 0, 3, 2, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                    // perm = { 1, 0, 3, 2 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB0) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 2, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPEConstraint(0, 2, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

                    constraint = new AIPC::IpcPEMConstraint(0, 2, 3, 0, 1, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                    // perm = { 2, 3, 0, 1 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB1) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 3, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPEConstraint(0, 3, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                    
                    constraint = new AIPC::IpcPEMConstraint(0, 3, 2, 0, 1, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                    // perm = { 3, 2, 0, 1 };
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPEConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                    constraint = new AIPC::IpcPEMConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                }
                else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB) {
                    friction_constraint = new AIPC::IpcPEFConstraint(0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcPEConstraint(0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                    
                    constraint = new AIPC::IpcPEMConstraint(0, 1, 0, 2, 3, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
                    // perm = { 1, 0, 2, 3 };
                }
                else {
                    friction_constraint = new AIPC::IpcEEFConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, evh, 1.0, 1.0);
                    // constraint = new AIPC::IpcEEConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
                    constraint = new AIPC::IpcEEMConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);

                }
                // if (false) {
                if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB0 || ee_type == ipc::EdgeEdgeDistanceType::EA_EB1) {
                    MatrixXd _habf, _habc;
                    _habf.setZero(12, 12);
                    _habc.setZero(12, 12);
                    friction_constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gbf, gaf);
                    friction_constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hbf, haf, _habf);
                    constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gbc, gac);
                    constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hbc, hac, _habc);
                    habf = _habf; // .transpose();
                    habc = _habc; //.transpose();
                }
                else {
                    friction_constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gaf, gbf);
                    friction_constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, haf, hbf, habf);
                    constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gac, gbc);
                    constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hac, hbc, habc);
                }

                ga = gaf + gac;
                gb = gbf + gbc;
                ha = haf + hac;
                hb = hbf + hbc;
                hab = habf + habc;

                ga /= barrier::d_hat;
                gb /= barrier::d_hat;
                ha /= barrier::d_hat;
                hb /= barrier::d_hat;
                hab /= barrier::d_hat;

                output_hessian_gradient(
                    lut, sparse_hess,
                    i, j, ci.mass > 0.0, cj.mass > 0.0,
                    ci.grad, cj.grad,
// #define LANS_DIRECT
#ifdef LANS_DIRECT
                    ga, gb, ha, hb, hab, hab.transpose()
#else
                    grad_0, grad_1, hess_0, hess_1, off_diag, off_diag.transpose()
#endif
                );

                bool b0 = ::fd::compare_gradient(ga, grad_0);
                bool b1 = ::fd::compare_gradient(gb, grad_1);

                bool b2 = fd::compare_hessian(ha, hess_0);
                bool b3 = fd::compare_hessian(hb, hess_1);
                bool b4 = fd::compare_hessian(hab, off_diag);
                const auto to_int = [](const ipc::EdgeEdgeDistanceType& type) -> int{
                    if (type == ipc::EdgeEdgeDistanceType::EA0_EB) {
                        return 0;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA1_EB) {
                        return 1;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA_EB0) {
                        return 2;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA_EB1) {
                        return 3;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA0_EB0) {
                        return 4;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA0_EB1) {
                        return 5;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA1_EB0) {
                        return 6;
                    }
                    else if (type == ipc::EdgeEdgeDistanceType::EA1_EB1) {
                        return 7;
                    }
                    else {
                        return 8;
                    }
                };
#ifndef LANS_DIRECT
                // b4 = b4 || mollifier != 1.0;
                // b3 = b3 || mollifier != 1.0;
                // b2 = b2 || mollifier != 1.0;
                // b1 = b1 || mollifier != 1.0;
                // b0 = b0 || mollifier != 1.0;
                

                if (!b0) {
                    spdlog::error("ee gradient p error, {}, molli = {}", to_int(ee_type), mollifier);
                }
                if (!b1) {
                    spdlog::error("ee gradient t error, {}, molli = {}", to_int(ee_type), mollifier);
                }
                if (!b2) {
                    spdlog::error("ee hessian p error, {}, molli = {}", to_int(ee_type), mollifier);
                }
                if (!b3) {
                    spdlog::error("ee hessian t error, {}, molli = {}", to_int(ee_type), mollifier);
                }
                if (!b4) {
                    globals.params_int["ee error " + to_string(to_int(ee_type))] ++;
                    spdlog::error("ee hessian off_diag error, {}, molli = {}", to_int(ee_type), mollifier);
                }
                
#endif
#endif
            }
        }

        for (int k = 0; k < n_g; k ++) {
            auto &_v {vidx[k]};
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
                g_contact_forces[k] = -barrier_derivative_d(d) * 2 * _d;

                ipc_term_vg(c, v, uk, g_contact_forces[k], Pk);
#else
                ipc_term_vg(c, v);
#endif
            }
        }
        auto ipc_duration = DURATION_TO_DOUBLE(ipc_start);
        times[__IPC__] += ipc_duration;
        double toi = 1.0, factor = 1.0, alpha = 1.0;

        {
            MatrixXd big_hess;
            if (globals.dense)
                big_hess.setZero(hess_dim, hess_dim);
            VectorXd r, q0_cat, dq;
            r.setZero(hess_dim);
            dq.setZero(hess_dim);
            q0_cat.setZero(hess_dim);
            for (int k = 0; k < n_cubes; k++) {
                auto& c = *cubes[k];
                r.segment<12>(k * 12) = c.grad;
                q0_cat.segment<12>(k * 12) = cat(c.q);
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

                if (globals.params_int["cuda_solver"]) {
                    cuda_solve(dq, sparse_hess, r);
                }
                else {

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
            }

            auto solver_duration = DURATION_TO_DOUBLE(solver_start);
            times[__SOLVER__] += solver_duration;
            spdlog::info("solver time = {:0.6f} ms", solver_duration);
            if (isnan(dq.norm())) {
                spdlog::error("solver nan");
                exit(1);
            }
            spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm() : big_hess.norm());
            // spdlog::warn("dense norms: dq = {}, grad = {}, big_hess = {}, difference = {}", dq.norm(), r.norm(), big_hess.norm(), dif);
            spdlog::info("dq dot grad = {}, cos = {}", dq.dot(r), dq.dot(r) / (dq.norm() * r.norm()));
            if (globals.sparse && globals.dense && (sparse_hess - big_hess).norm() > 1e-6) {
                spdlog::error("diff too large");
                cout << big_hess << "\n\n"
                     << sparse_hess;
            }
            sup_dq = dq.norm();
            toi = 1.0;
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n_cubes; k++) {
                auto& c(*cubes[k]);
                c.dq = dq.segment<12>(k * 12);
            }
            auto ccd_start = high_resolution_clock::now();
            if (globals.upper_bound) {
                double toi_iaabb;
                if (globals.iaabb > 1)
                    toi_iaabb = iaabb_brute_force(n_cubes, cubes, globals.aabbs, 3, pts, idx, ees, eidx, vidx);
#ifndef IAABB_INTERNSHIP
                else
#endif
                    toi = step_size_upper_bound(dq, cubes, n_cubes, n_pt, n_ee, n_g, pts, idx, ees, eidx, vidx);
#ifdef IAABB_INTERNSHIP
                if (globals.iaabb > 1 && toi != toi_iaabb) {
                    spdlog::error("step size upper bound not match, toi = {}, iaabb = {}", toi, toi_iaabb);
                    dump_states(cubes);
                    exit(1);
                }
#else
                if (globals.iaabb > 1)
                    toi = toi_iaabb;
#endif
            }
            double ccd_duration = DURATION_TO_DOUBLE(ccd_start);
            times[__CCD__] += ccd_duration;
            if (toi < 1.0) {
                spdlog::warn("collision at {}, toi = {}", iter, toi);
                factor = globals.backoff;
            }
            if (iter) {
                double cos_dq_lastdq = dq.dot(lastdq) / (dq.norm() * lastdq.norm());
                if (abs(cos_dq_lastdq) > 1.0 - globals.params_double["dq_tol"]) {
                    double ddq = (dq - lastdq).norm() / dq.norm();
                    if (ddq < globals.params_double["dq_tol"]) {
                        spdlog::error("same direction, cos = {}", cos_dq_lastdq);
                        exit(1);
                    }
                }
            }
            lastdq = dq;
            dq *= factor * toi;

            alpha = 1.0;
            double E0 = 0.0, E1 = 0.0;
            auto line_search_start = high_resolution_clock::now();
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
                    pt_contact_forces,
                    ee_contact_forces,
                    g_contact_forces,
                    cubes, dt);
            spdlog::info("alpha = {}", alpha);
            dq *= alpha;
            if (alpha < 2e-8) {
                spdlog::error("iter, ts ({}, {}), alpha = {}, E0 = {}, E1 = {}", iter, ts, alpha, E0, E1);
            }
            auto line_search_duration = DURATION_TO_DOUBLE(line_search_start);
            times[__LINE_SEARCH__] += line_search_duration;
            double norm_dq = dq.norm();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_cubes; i++) {
                for (int j = 0; j < 4; j++)
                    cubes[i]->q[j] += dq.segment<3>(i * 12 + j * 3);
            }
            spdlog::info("step size upper = {}, alpha = {}", toi, alpha);

            auto iter_duration = DURATION_TO_DOUBLE(newton_iter_start);
            // spdlog::warn("iter {}, time = {} ms, IPC term time = {} \n e0 = {}, e1 = {}, norm_dq = {}\n", iter, iter_duration, ipc_duration, E0, E1, norm_dq);
            spdlog::warn("Newton iter #{}, time = {} ms, upper bound = {}, line search = {}", iter + 1, iter_duration, toi, alpha);
        }

        term_cond = sup_dq < tol || ++iter >= globals.max_iter;
        sup_dq = 0.0;
    } while (!term_cond);

    double frame_duration = DURATION_TO_DOUBLE(frame_start) / 100.0;
    spdlog::warn("converge #iter {}, ts = {}, time = {} ms\n\\
    time breakdown :\n\\
    \tipc: {:.3f} ms, percentage = {:.3f}% \n\\
    \tsolver: {:.3f}, percentage = {:.3f}%\n\\
    \tccd: {:.3f}, percentage = {:.3f}%\n\\
    \tline search: {:.3f}, percentage = {:.3f}%\n\n\n",
        ++iter, ts++, frame_duration * 100.0,
        times[__IPC__],
        times[__IPC__] / frame_duration,
        times[__SOLVER__], times[__SOLVER__] / frame_duration,
        times[__CCD__], times[__CCD__] / frame_duration,
        times[__LINE_SEARCH__], times[__LINE_SEARCH__] / frame_duration);
    globals.tot_iter += iter;
    globals.aggregate_time += times;
    if (globals.params_int.find("worst case iter") == globals.params_int.end())
        globals.params_int["worst case iter"] = iter;
    else
        globals.params_int["worst case iter"] = max(iter, globals.params_int["worst case iter"]);
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
