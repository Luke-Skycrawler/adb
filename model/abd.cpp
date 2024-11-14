#include "abd.h"
#include "time_integrator.h"
#include "sparse.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "ipc.h"
#include <assert.h>
#include <array>
#include "geometry.h"
#include "timer.h"
#include "ipc_extension.h"
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif
#include <ipc/distance/edge_edge_mollifier.hpp>



using namespace std; 
using namespace Eigen;
using namespace barrier;
using namespace utils;

mat3 RQ(const mat3 &a) {
    Eigen::HouseholderQR<mat3> qr(a.transpose()); // Decompose A^T

    mat3 R = qr.matrixQR().triangularView<Eigen::Upper>().transpose(); // Extract R and transpose back
    Eigen::MatrixXd Q = qr.householderQ().transpose(); // Extract Q and transpose back

    // std::cout << "Original Matrix A:\n"
    //         << a << "\n";
    // std::cout << "R (on the left):\n"
    //         << R << "\n";
    // std::cout << "Q:\n"
    //         << Q << "\n";

    // // Verify decomposition A = Q * R
    // std::cout << "A - R * Q:\n"
    //         << a - R * Q << "\n";
    return R;
}
void AffineBody::compute_R()
{
    mat3 a;
    a << q[1], q[2], q[3];
    R = RQ(a);
}
void AffineBody::compute_R0()
{
    mat3 a;
    a << q0[1], q0[2], q0[3];
    R0 = RQ(a);
}

void ABD::implicit_euler(scalar dt) {
    times.setZero(4);
    auto frame_start = high_resolution_clock::now();

    for (int k = 0; k < cubes.size(); k++) {
        auto& c(*cubes[k]);
        for (int i = 0; i < 4; i++) {
            c.q[i] = c.q0[i];
        }
    }
    
    if (globals.col_set) {

        auto &pts_arg = pts;
        auto &idx_arg = idx;
        auto &ees_arg = ees;
        auto &eidx_arg = eidx;
        auto &vidx_arg = vidx;
        if (globals.iaabb % 2)
            culling.iaabb_brute_force(n_cubes, cubes, globals.aabbs, 1, pts_arg, idx_arg, ees_arg, eidx_arg, vidx_arg);
        else {
            gen_collision_set(false, n_cubes, cubes, pts, idx, ees, eidx, vidx);
        }
    }

    ////// MAIN LOOP /////////////////////
    do {
        auto newton_iter_start = high_resolution_clock::now();

        n_pt = idx.size();
        n_ee = eidx.size();
        n_g = vidx.size();

#ifdef _TRIPLETS_
        hess_triplets.reserve(((n_pt + n_ee) * 2 + n_cubes) * 12);
        hess_triplets.clear();
#endif

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
            gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);

            pt_tk.resize(n_pt);
            pt_contact_forces.resize(n_pt);
            ee_tk.resize(n_ee);
            ee_contact_forces.resize(n_ee);
            g_contact_forces.resize(n_g);
            spdlog::info("constraint size = {}, {}", n_pt, n_ee);
        }
#endif
        ipc();
        solve(dt);
        ccd();
        line_search(dt);

        scalar norm_dq = dq.norm();
#pragma omp parallel for schedule(static)
        for(int i = 0; i < n_cubes; i++) {
            for(int j = 0; j < 4; j++)
                cubes[i]->q[j] += dq.segment<3>(i * 12 + j * 3);
        }

        auto iter_duration = DURATION_TO_DOUBLE(newton_iter_start);
        spdlog::warn("Newton iter #{}, time = {} ms, upper bound = {}, line search = {}", iter + 1, iter_duration, toi, alpha);

        term_cond = sup_dq < tol || ++iter >= globals.max_iter;
        sup_dq = 0.0;
    } while(!term_cond);

    scalar frame_duration = DURATION_TO_DOUBLE(frame_start) / 100.0;
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
    if(globals.params_int.find("worst case iter") == globals.params_int.end())
        globals.params_int["worst case iter"] = iter;
    else
        globals.params_int["worst case iter"] = max(iter, globals.params_int["worst case iter"]);
#pragma omp parallel for schedule(static)
    for(int k = 0; k < n_cubes; k++) {
        auto& c(*cubes[k]);
        for(int i = 0; i < 4; i++) {
            c.dqdt[i] = (c.q[i] - c.q0[i]) / dt;
            c.q0[i] = c.q[i];
        }
        // c.p = c.q0[0];
        // c.A << c.q0[1], c.q0[2], c.q0[3];
        // c.project_vt1();
        c.predraw();
    }
}

void ABD::ccd()
{

    auto ccd_start = high_resolution_clock::now();
    {
        toi = 1.0;
        factor = 1.0;
        sup_dq = dq.norm();

#pragma omp parallel for schedule(static)
        for(int k = 0; k < n_cubes; k++) {
            auto& c(*cubes[k]);
            c.dq = dq.segment<12>(k * 12);
        }

        if(globals.upper_bound) {
            scalar toi_iaabb;
            if(globals.iaabb > 1)
                toi_iaabb = culling.iaabb_brute_force(n_cubes, cubes, globals.aabbs, 3, pts, idx, ees, eidx, vidx);
#ifndef IAABB_INTERNSHIP
            else
#endif
                toi = step_size_upper_bound(dq, cubes, n_cubes, n_pt, n_ee, n_g, pts, idx, ees, eidx, vidx);
#ifdef IAABB_INTERNSHIP
            if(globals.iaabb > 1 && toi != toi_iaabb) {
                spdlog::error("step size upper bound not match, toi = {}, iaabb = {}", toi, toi_iaabb);
                dump_states(cubes);
                exit(1);
            }
#else
            if(globals.iaabb > 1)
                toi = toi_iaabb;
#endif
        }

        if(toi < 1.0) {
            spdlog::warn("collision at {}, toi = {}", iter, toi);
            factor = globals.backoff;
        }
        if(iter) {
            scalar cos_dq_lastdq = dq.dot(lastdq) / (dq.norm() * lastdq.norm());
            if(abs(cos_dq_lastdq) > 1.0 - globals.params_double["dq_tol"]) {
                scalar ddq = (dq - lastdq).norm() / dq.norm();
                if(ddq < globals.params_double["dq_tol"]) {
                    spdlog::error("same direction, cos = {}", cos_dq_lastdq);
                    exit(1);
                }
            }
        }
        lastdq = dq;
        dq *= factor * toi;
    }
    scalar ccd_duration = DURATION_TO_DOUBLE(ccd_start);
    times[__CCD__] += ccd_duration;
}

void ABD::line_search(scalar dt)
{
    auto line_search_start = high_resolution_clock::now();
    {
        alpha = 1.0;
        scalar E0 = 0.0, E1 = 0.0;
        if(globals.line_search)
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
        if(alpha < 2e-8) {
            spdlog::error("iter, ts ({}, {}), alpha = {}, E0 = {}, E1 = {}", iter, ts, alpha, E0, E1);
        }
        spdlog::info("step size upper = {}, alpha = {}", toi, alpha);
    }
    auto line_search_duration = DURATION_TO_DOUBLE(line_search_start);
    times[__LINE_SEARCH__] += line_search_duration;
}

void ABD::solve(scalar dt)
{
    auto solver_start = high_resolution_clock::now();
    {
        if(globals.dense)
            big_hess.setZero(hess_dim, hess_dim);
        r.setZero(hess_dim);
        dq.setZero(hess_dim);
        q0_cat.setZero(hess_dim);

        for(int k = 0; k < n_cubes; k++) {
            auto& c = *cubes[k];
            r.segment<12>(k * 12) = c.grad;
            q0_cat.segment<12>(k * 12) = cat(c.q);
        }

#ifdef _TRIPLETS_
        build_from_triplets(sparse_hess, big_hess, hess_dim, n_cubes, hess_triplets);
#endif
#ifdef _SM_
#pragma omp parallel for schedule(static)
        for(int k = 0; k < n_cubes; k++) {
            auto& c = *cubes[k];
            scalar* values = sparse_hess.valuePtr();
            int* outers = sparse_hess.outerIndexPtr();
            int offset = starting_offset(k, k, lut, outers);
            int _stride = stride(k, outers);
            for(int j = 0; j < 12; j++)
                for(int i = 0; i < 12; i++) {
                    values[offset + _stride * j + i] += c.hess(i, j);
                }
        }

#endif
        if(globals.damp)
            damping_sparse(sparse_hess, dt, n_cubes);

        if(globals.dense)
            dq = -big_hess.ldlt().solve(r);
        else if(globals.sparse) {
#ifdef EIGEN_USE_MKL_ALL
            PardisoLLT<SparseMatrix<scalar>> ldlt_solver;
            // SimplicialLLT<SparseMatrix<scalar, ColMajor>> ldlt_solver;
#else
            SimplicialLLT<SparseMatrix<scalar, ColMajor>> ldlt_solver;
#endif
            ldlt_solver.compute(sparse_hess);
            dq = -ldlt_solver.solve(r);
#ifdef _TRIPLET_
            sparse_hess_trip.finalize();
            scalar _dif = (sparse_hess - sparse_hess_trip).norm();
            if(_dif > 1e-6) {
                cout << "error: dif = " << _dif << "\n\n";
                cout << sparse_hess_trip << "\n\n"
                     << sparse_hess;
                exit(0);
            }
#endif
        }
        if(isnan(dq.norm())) {
            spdlog::error("solver nan");
            exit(1);
        }
        spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm() : big_hess.norm());
        spdlog::info("dq dot grad = {}, cos = {}", dq.dot(r), dq.dot(r) / (dq.norm() * r.norm()));
        if(globals.sparse && globals.dense && (sparse_hess - big_hess).norm() > 1e-6) {
            spdlog::error("diff too large");
            cout << big_hess << "\n\n"
                 << sparse_hess;
        }
    }
    auto solver_duration = DURATION_TO_DOUBLE(solver_start);
    times[__SOLVER__] += solver_duration;
    spdlog::info("solver time = {:0.6f} ms", solver_duration);
}

void ABD::ipc()
{
    // ipc hessian & gradient
    auto ipc_start = high_resolution_clock::now();
    {
#pragma omp parallel for schedule(static)
        for(int k = 0; k < n_pt; k++) {

            // auto& pt(pts[k]);
            auto& ij(idx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            Face f{ cj.face(int(ij[3]), false, true) };
            vec3 p{ ci.v_transformed[ij[1]] };
            q4 pt{ p, f.t0, f.t1, f.t2 };
            auto [d, pt_type] = vf_distance(pt[0], f);
            if(d < barrier::d_hat) {
                vec12 gradp, gradt;
                mat12 hess_p, hess_t, off_diag;
                ipc_assembler.ipc_term(
                    pt, ij, pt_type, d,
#ifdef _SM_OUT_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    hess_triplets,
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
            }
        }
#pragma omp parallel for schedule(static)
        for(int k = 0; k < n_ee; k++) {
            // auto& ee(ees[k]);
            auto& ij(eidx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            Edge ei{ ci.edge(int(ij[1]), false, true) }, ej{ cj.edge(int(ij[3]), false, true) };
            q4 ee{ ei.e0, ei.e1, ej.e0, ej.e1 };
            auto ee_type = ipc::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
            scalar d = edge_edge_distance(ee[0], ee[1], ee[2], ee[3], ee_type);
            if(d < barrier::d_hat) {
                mat12 hess_0, hess_1, off_diag;
                vec12 grad_0, grad_1;
                ipc_assembler.ipc_term_ee(
                    ee, ij, ee_type, d,
#ifdef _SM_OUT_
                    lut, sparse_hess,
#endif
#ifdef _TRIPLETS_
                    hess_triplets,
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
            }
        }

        for(int k = 0; k < n_g; k++) {
            auto& _v{ vidx[k] };
            int i = _v[0], v = _v[1];
            auto& c{ *cubes[i] };
            vec3 p = c.v_transformed[v];
            scalar _d = vg_distance(p);
            scalar d = _d * _d;
            if(d < barrier::d_hat) {
#ifdef _FRICTION_
                Matrix<scalar, 3, 2> Pk;
                Pk.col(0) = vec3(1.0, 0.0, 0.0);
                Pk.col(1) = vec3(0.0, 0.0, 1.0);

                Vector<scalar, 2> uk = Pk.transpose() * (p - c.vt0(v));
                g_contact_forces[k] = -barrier_derivative_d(d) * 2 * _d;

                ipc_assembler.ipc_term_vg(c, v, uk, g_contact_forces[k], Pk);
#else
                ipc_assembler.ipc_term_vg(c, v);
#endif
            }
        }
    }
    auto ipc_duration = DURATION_TO_DOUBLE(ipc_start);
    times[__IPC__] += ipc_duration;
}

void ABD::vibrate(scalar dt) {
    //static const scalar kappa = 1e-3;
    scalar vib_kappa = globals.params_double["vib_kappa"];
    scalar thres_print = globals.params_double["thres_print"];
    // compute excitement
    for (int i = 0; i < n_g; i++) {
        int bid = vidx[i][0], pid = vidx[i][1];
        auto &c{ *cubes[bid] };
        vec3 p = c.v_transformed[pid];

        vec3 F = vec3(0.0, 0.0, 1.0) * g_contact_forces[i];
        int n_modes = c.Phi.cols();
        for (int j = 0; j < n_modes; j ++) {

            c.excitement[j] = F.dot( c.displacement(pid, j)) * vib_kappa;
        }
    }


    for (int i = 0; i < n_cubes; i++){
        auto &c{ *cubes[i] };
        int n_modes = c.Phi.cols();
        for (int j = 0; j < n_modes; j++) {
            scalar lj = c.lam[j];
            scalar divd = dt * dt * lj + 1.0;
            scalar a = 1.0 / divd;
            scalar b = dt / divd;
            scalar cc = 0.5 * dt * dt / divd;

            c.qq[j] = a * c.qq0[j] + b * c.dqqdt[j] + cc * c.excitement[j];
            if (c.qq[j] > thres_print) {
                spdlog::error("qq[{}] = {}, excitement = {}", j, c.qq[j], c.excitement[j]);
            }

            c.dqqdt[j] = (c.qq[j] - c.qq0[j]) / dt;
            c.qq0[j] = c.qq[j];
        }
        c.project_vib();
    }
}

vec3 AffineBody::displacement(int i, int j) {
    // displacement of vertex i in mode j
    compute_R();
    auto &Phij {Phi.col(j)};
    vec3 phi(Phij[i * 3], Phij[i * 3 + 1], Phij[i * 3 + 2]);
    phi = R * phi;
    return phi;
}