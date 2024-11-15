// #include "time_integrator.h"
// #include "sparse.h"
// #include "barrier.h"
// #include "spdlog/spdlog.h"
// #include "collision.h"
// #include "settings.h"
// #include <assert.h>
// #include <array>
// #include "geometry.h"
// #include "timer.h"
// #include "ipc_extension.h"
// // #include <ipc/distance/point_triangle.hpp>
// // #include <ipc/distance/edge_edge.hpp>
// #include <ipc/friction/closest_point.hpp>
// #include <ipc/friction/tangent_basis.hpp>
// #ifdef EIGEN_USE_MKL_ALL
// #include <Eigen/PardisoSupport>
// #endif
// #include <ipc/distance/edge_edge_mollifier.hpp>
// //#define IAABB_COMPARING
// // #define IAABB_INTERNSHIP
// #ifdef IAABB_COMPARING
// #include <algorithm>
// #endif
// using namespace std;
// using namespace barrier;
// using namespace Eigen;
// using namespace utils;

// #define __IPC__ 0
// #define __SOLVER__ 1
// #define __CCD__ 2
// #define __LINE_SEARCH__ 3

// #ifdef PLUG_IN_LAN
// #include "IpcFrictionConstraint.h"
// #endif
// scalar line_search(const Vector<scalar, -1>& dq, const Vector<scalar, -1>& grad, Vector<scalar, -1>& q0, scalar& E0, scalar& E1,
//     int n_cubes, int n_pt, int n_ee, int n_g,
//     vector<q4>& pts,
//     vector<i4>& idx,
//     vector<q4>& ees,
//     vector<i4>& eidx,
//     vector<array<int, 2>>& vidx,
//     const vector<Matrix<scalar, 2, 12>>& pt_tk,
//     const vector<Matrix<scalar, 2, 12>>& ee_tk,
//     const vector<scalar> &pt_contact_forces,
//     const vector<scalar> &ee_contact_forces,
//     const vector<scalar> &g_contact_forces,
//     const vector<unique_ptr<AffineBody>>& cubes,
//     scalar dt);

// void implicit_euler(vector<unique_ptr<AffineBody>>& cubes, scalar dt)
// {
//     static bool term_cond;
//     int& ts = globals.ts;
//     static scalar tol = globals.params_double["tol"];
//     static Vector<scalar, -1> lastdq;
//     static int iter = 0;

//     static Vector<scalar, -1> r, q0_cat, dq;
//     static scalar sup_dq = 0.0;
//     static Matrix<scalar, -1, -1> big_hess;

//     static scalar toi = 1.0, factor = 1.0;
//     static scalar alpha = 1.0;


//     static const int n_cubes = cubes.size(), hess_dim = n_cubes * 12;

//     // collision set
//     static vector<q4> pts;
//     static vector<i4> idx;

//     static vector<q4> ees;
//     static vector<i4> eidx;

//     static vector<array<int, 2>> vidx;

//     static vector<Matrix<scalar, 2, 12>> pt_tk;
//     static vector<Matrix<scalar, 2, 12>> ee_tk;
//     static Vector<scalar, 4> times;
//     static int n_pt, n_ee, n_g;
//     static vector<scalar> pt_contact_forces, ee_contact_forces, g_contact_forces;

// #ifdef _TRIPLETS_
//     static vector<HessBlock> hess_triplets;
// #endif
// #ifdef _SM_
//     // look-up table for sparse matrix
//     static map<array<int, 2>, int> lut;
//     static SparseMatrix<scalar> sparse_hess(hess_dim, hess_dim);
// #endif


//     times.setZero(4);
//     auto frame_start = high_resolution_clock::now();

//     for (int k = 0; k < cubes.size(); k++) {
//         auto& c(*cubes[k]);
//         for (int i = 0; i < 4; i++) {
//             c.q[i] = c.q0[i];
//         }
//     }
// #ifdef IAABB_COMPARING
//     vector<q4> pts_iaabb;
//     vector<i4> idx_iaabb;

//     vector<q4> ees_iaabb;
//     vector<i4> eidx_iaabb;

//     vector<array<int, 2>> vidx_iaabb;

//     vector<Matrix<scalar, 2, 12>> pt_tk_iaabb;
//     vector<Matrix<scalar, 2, 12>> ee_tk_iaabb;
// #endif


//     if (globals.col_set) {

//         auto &pts_arg = pts;
//         auto &idx_arg = idx;
//         auto &ees_arg = ees;
//         auto &eidx_arg = eidx;
//         auto &vidx_arg = vidx;

// #ifdef IAABB_COMPARING
//         pts_arg = pts_iaabb;
//         idx_arg = idx_iaabb;
//         ees_arg = ees_iaabb;
//         eidx_arg = eidx_iaabb;
//         vidx_arg = vidx_iaabb;

//         gen_collision_set(false, n_cubes, cubes, pts, idx, ees, eidx, vidx);
//         if (globals.iaabb % 2) {
//             iaabb_brute_force(n_cubes, cubes, globals.aabbs, 1, pts_arg, idx_arg, ees_arg, eidx_arg, vidx_arg);
//             const auto compare_collision = [](
//                                                vector<i4>& aidx,

//                                                vector<i4>& bidx) -> bool {
//                 // FIXME: make copy before sorting;
//                 auto a_copy = aidx;
//                 vector<i4> adb, bda;

//                 sort(aidx.begin(), aidx.end());
//                 sort(bidx.begin(), bidx.end());
//                 set_difference(aidx.begin(), aidx.end(), bidx.begin(), bidx.end(), back_inserter(adb));
//                 set_difference(bidx.begin(), bidx.end(), aidx.begin(), aidx.end(), back_inserter(bda));

//                 if (adb.size()) {
//                     spdlog::error("detection not detected: ");
//                     for (int i = 0; i < adb.size(); i++) {
//                         auto& a{ adb[i] };
//                         spdlog::warn("( {}, {}, {}, {})", a[0], a[1], a[2], a[3]);
//                     }
//                 }
//                 if (bda.size()) {
//                     spdlog::error("False positive: ");
//                     for (int i = 0; i < bda.size(); i++) {
//                         auto& a{ bda[i] };
//                         spdlog::warn("( {}, {}, {}, {})", a[0], a[1], a[2], a[3]);
//                     }
//                 }
//                 spdlog::info("sizes: ref = {}, iaabb = {}", aidx.size(), bidx.size());
//                 aidx = a_copy;
//                 return !(bda.size() || adb.size());
//             };
//             spdlog::info("PT");
//             bool pt_success = compare_collision(
//                 idx, idx_iaabb);

//             spdlog::info("EE");
//             bool ee_success = compare_collision(
//                 eidx, eidx_iaabb);
//             if (!pt_success) spdlog::error("pt fails, exiting");
//             if (!ee_success) spdlog::error("ee fails, exiting");
//             if (!(pt_success && ee_success))
//                 exit(1);
//             else
//                 spdlog::info("pt and ee set matched");

//         }   
// #else
//         if (globals.iaabb % 2)
//             iaabb_brute_force(n_cubes, cubes, globals.aabbs, 1, pts_arg, idx_arg, ees_arg, eidx_arg, vidx_arg);
//         else {
//             gen_collision_set(false, n_cubes, cubes, pts, idx, ees, eidx, vidx);
//         }

// #endif
//     }


//     ///////////////////////// MAIN LOOP /////////////////////
//     do {
//         auto newton_iter_start = high_resolution_clock::now();

//         n_pt = idx.size();
//         n_ee = eidx.size();
//         n_g = vidx.size();

// #ifdef _TRIPLETS_
//         hess_triplets.reserve(((n_pt + n_ee) * 2 + n_cubes) * 12);
//         hess_triplets.clear();
// #endif

// #pragma omp parallel for schedule(static)
//         for (int k = 0; k < n_cubes; k++) {
//             auto& c(*cubes[k]);
//             c.grad = grad_residue_per_body(c, dt);
//             c.hess = hess_inertia_per_body(c, dt);
//             c.project_vt1();
//         }

// #ifdef _SM_
//         {
//             lut.clear();
//             sparse_hess.setZero();
//             gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);

//             pt_tk.resize(n_pt);
//             pt_contact_forces.resize(n_pt);
//             ee_tk.resize(n_ee);
//             ee_contact_forces.resize(n_ee);
//             g_contact_forces.resize(n_g);
//             spdlog::info("constraint size = {}, {}", n_pt, n_ee);
//         }
// #endif
//         // ipc hessian & gradient
//         auto ipc_start = high_resolution_clock::now();
//         {
// #pragma omp parallel for schedule(static)
//             for (int k = 0; k < n_pt; k++) {

//                 // auto& pt(pts[k]);
//                 auto& ij(idx[k]);
//                 int i = ij[0], j = ij[2];
//                 auto &ci(*cubes[i]), &cj(*cubes[j]);
//                 Face f{ cj, int(ij[3]), false, true };
//                 vec3 p{ ci.v_transformed[ij[1]] };
//                 q4 pt{ p, f.t0, f.t1, f.t2 };
//                 auto [d, pt_type] = vf_distance(pt[0], f);
//                 if (d < barrier::d_hat) {
//                     vec12 gradp, gradt;
//                     mat12 hess_p, hess_t, off_diag;
//                     ipc_term(
//                         pt, ij, pt_type, d,
//     #ifdef _SM_OUT_
//                         lut, sparse_hess,
//     #endif
//     #ifdef _TRIPLETS_
//                         hess_triplets,
//     #endif
//     #ifdef _DIRECT_OUT_
//                         hess_p, hess_t, off_diag,
//                         gradp, gradt
//     #else
//                         ci.grad, cj.grad
//     #endif
//     #ifdef _FRICTION_
//                         ,
//                         pt_contact_forces[k], pt_tk[k]
//     #endif
//                     );

//     #ifdef _PLUG_IN_LAN_
//                     [ga, gb, ha, hb, hab] = pt_ipc_friction_constraint(ci, cj, f, p, pt_type);

//                     output_hessian_gradient(
//                         lut, sparse_hess,
//                         i, j, ci.mass > 0.0, cj.mass > 0.0,
//                         ci.grad, cj.grad,

//                         // gradp, gradt, hess_p, hess_t, off_diag, off_diag.transpose()
//                         ga, gb, ha, hb, hab, hab.transpose()
                        
//                     );
//                     compare_lan(ga, gb, ha, hb, hab, gradp, gradt, hess_p, hess_t, off_diag);
//     #endif
//                 }
//         }
// #pragma omp parallel for schedule(static)
//             for (int k = 0; k < n_ee; k++) {
//                 // auto& ee(ees[k]);
//                 auto& ij(eidx[k]);
//                 int i = ij[0], j = ij[2];
//                 auto &ci(*cubes[i]), &cj(*cubes[j]);
//                 Edge ei{ ci, int(ij[1]), false, true }, ej{ cj, int(ij[3]), false, true };
//                 q4 ee{ ei.e0, ei.e1, ej.e0, ej.e1 };
//                 auto ee_type = ipc::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
//                 scalar d = edge_edge_distance(ee[0], ee[1], ee[2], ee[3], ee_type);
//                 if (d < barrier::d_hat) {
//                     mat12 hess_0, hess_1, off_diag;
//                     vec12 grad_0, grad_1;
//                     ipc_term_ee(
//                         ee, ij, ee_type, d,
//     #ifdef _SM_OUT_
//                         lut, sparse_hess,
//     #endif
//     #ifdef _TRIPLETS_
//                         hess_triplets,
//     #endif
//     #ifdef _DIRECT_OUT_
//                         hess_0, hess_1, off_diag,
//                         grad_0, grad_1
//     #else
//                         ci.grad, cj.grad
//     #endif
//     #ifdef _FRICTION_
//                         ,
//                         ee_contact_forces[k], ee_tk[k]
//     #endif
//                     );

//     #ifdef _PLUG_IN_LAN_
//                     scalar eps_x = globals.eps_x * (ee[0] - ee[1]).squaredNorm() * (ee[2] - ee[3]).squaredNorm();

//                     scalar mollifier = ipc::edge_edge_mollifier(ee[0], ee[1], ee[2], ee[3], eps_x);
//                     [ga, gb, ha, hb, hab] = ee_ipc_friction_constraint(ci, cj, ee_type, eps_x, mollifier);

//                     output_hessian_gradient(
//                         lut, sparse_hess,
//                         i, j, ci.mass > 0.0, cj.mass > 0.0,
//                         ci.grad, cj.grad,
//     // #define LANS_DIRECT
//     #ifdef LANS_DIRECT
//                         ga, gb, ha, hb, hab, hab.transpose()
//     #else
//                         grad_0, grad_1, hess_0, hess_1, off_diag, off_diag.transpose()
//     #endif
//                     );
//                     compare_lan_ee(ga, gb, ha, hb, hab, grad_0, grad_1, hess_0, hess_1, off_diag, ee_type, mollifier);
//     #endif
//                 }
//             }

//             for (int k = 0; k < n_g; k ++) {
//             auto &_v {vidx[k]};
//             int i = _v[0], v = _v[1];
//             auto& c{ *cubes[i] };
//             vec3 p = c.v_transformed[v];
//             scalar _d = vg_distance(p);
//             scalar d = _d * _d;
//             if (d < barrier::d_hat) {
// #ifdef _FRICTION_
//                 Matrix<scalar, 3, 2> Pk;
//                 Pk.col(0) = vec3(1.0, 0.0, 0.0);
//                 Pk.col(1) = vec3(0.0, 0.0, 1.0);

//                 Vector<scalar, 2> uk = Pk.transpose() * (p - c.vt0(v));
//                 g_contact_forces[k] = -barrier_derivative_d(d) * 2 * _d;

//                 ipc_term_vg(c, v, uk, g_contact_forces[k], Pk);
// #else
//                 ipc_term_vg(c, v);
// #endif
//             }
//         }
//         }
//         auto ipc_duration = DURATION_TO_DOUBLE(ipc_start);
//         times[__IPC__] += ipc_duration;

//         auto solver_start = high_resolution_clock::now();
//         {
//             if (globals.dense)
//                 big_hess.setZero(hess_dim, hess_dim);
//             r.setZero(hess_dim);
//             dq.setZero(hess_dim);
//             q0_cat.setZero(hess_dim);

//             for (int k = 0; k < n_cubes; k++) {
//                 auto& c = *cubes[k];
//                 r.segment<12>(k * 12) = c.grad;
//                 q0_cat.segment<12>(k * 12) = cat(c.q);
//             }

// #ifdef _TRIPLETS_
//             build_from_triplets(sparse_hess, big_hess, hess_dim, n_cubes, hess_triplets);
// #endif
// #ifdef _SM_
// #pragma omp parallel for schedule(static)
//             for (int k = 0; k < n_cubes; k++) {
//                 auto& c = *cubes[k];
//                 scalar* values = sparse_hess.valuePtr();
//                 int* outers = sparse_hess.outerIndexPtr();
//                 int offset = starting_offset(k, k, lut, outers);
//                 int _stride = stride(k, outers);
//                 for (int j = 0; j < 12; j++)
//                     for (int i = 0; i < 12; i++) {
//                         values[offset + _stride * j + i] += c.hess(i, j);
//                     }
//             }

// #endif
//             if (globals.damp)
//                 damping_sparse(sparse_hess, dt, n_cubes);

//             if (globals.dense)
//                 dq = -big_hess.ldlt().solve(r);
//             else if (globals.sparse) {
// #ifdef EIGEN_USE_MKL_ALL
//                 PardisoLLT<SparseMatrix<scalar>> ldlt_solver;
//                  //SimplicialLLT<SparseMatrix<scalar, ColMajor>> ldlt_solver;
// #else
//                 SimplicialLLT<SparseMatrix<scalar, ColMajor>> ldlt_solver;
// #endif
//                 ldlt_solver.compute(sparse_hess);
//                 dq = -ldlt_solver.solve(r);
// #ifdef _TRIPLET_
//                 sparse_hess_trip.finalize();
//                 scalar _dif = (sparse_hess - sparse_hess_trip).norm();
//                 if (_dif > 1e-6) {
//                     cout << "error: dif = " << _dif << "\n\n";
//                     cout << sparse_hess_trip << "\n\n"
//                          << sparse_hess;
//                     exit(0);
//                 }
// #endif
//             }
//             if (isnan(dq.norm())) {
//                 spdlog::error("solver nan");
//                 exit(1);
//             }
//             spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm() : big_hess.norm());
//             spdlog::info("dq dot grad = {}, cos = {}", dq.dot(r), dq.dot(r) / (dq.norm() * r.norm()));
//             if (globals.sparse && globals.dense && (sparse_hess - big_hess).norm() > 1e-6) {
//                 spdlog::error("diff too large");
//                 cout << big_hess << "\n\n"
//                      << sparse_hess;
//             }
//         }
//         auto solver_duration = DURATION_TO_DOUBLE(solver_start);
//         times[__SOLVER__] += solver_duration;
//         spdlog::info("solver time = {:0.6f} ms", solver_duration);

//         auto ccd_start = high_resolution_clock::now();
//         {
//             toi = 1.0;
//             factor = 1.0;
//             sup_dq = dq.norm();

// #pragma omp parallel for schedule(static)
//             for (int k = 0; k < n_cubes; k++) {
//                 auto& c(*cubes[k]);
//                 c.dq = dq.segment<12>(k * 12);
//             }

//             if (globals.upper_bound) {
//                 scalar toi_iaabb;
//                 if (globals.iaabb > 1)
//                     toi_iaabb = iaabb_brute_force(n_cubes, cubes, globals.aabbs, 3, pts, idx, ees, eidx, vidx);
// #ifndef IAABB_INTERNSHIP
//                 else
// #endif
//                     toi = step_size_upper_bound(dq, cubes, n_cubes, n_pt, n_ee, n_g, pts, idx, ees, eidx, vidx);
// #ifdef IAABB_INTERNSHIP
//                 if (globals.iaabb > 1 && toi != toi_iaabb) {
//                     spdlog::error("step size upper bound not match, toi = {}, iaabb = {}", toi, toi_iaabb);
//                     dump_states(cubes);
//                     exit(1);
//                 }
// #else
//                 if (globals.iaabb > 1)
//                     toi = toi_iaabb;
// #endif
//             }
            
//             if (toi < 1.0) {
//                 spdlog::warn("collision at {}, toi = {}", iter, toi);
//                 factor = globals.backoff;
//             }
//             if (iter) {
//                 scalar cos_dq_lastdq = dq.dot(lastdq) / (dq.norm() * lastdq.norm());
//                 if (abs(cos_dq_lastdq) > 1.0 - globals.params_double["dq_tol"]) {
//                     scalar ddq = (dq - lastdq).norm() / dq.norm();
//                     if (ddq < globals.params_double["dq_tol"]) {
//                         spdlog::error("same direction, cos = {}", cos_dq_lastdq);
//                         exit(1);
//                     }
//                 }
//             }
//             lastdq = dq;
//             dq *= factor * toi;
//         }
//         scalar ccd_duration = DURATION_TO_DOUBLE(ccd_start);
//         times[__CCD__] += ccd_duration;

//         auto line_search_start = high_resolution_clock::now();
//         {
//             alpha = 1.0;
//             scalar E0 = 0.0, E1 = 0.0;
//             if (globals.line_search)
//                 alpha = line_search(dq, r, q0_cat, E0, E1,
//                     n_cubes, n_pt, n_ee, n_g,
//                     pts,
//                     idx,
//                     ees,
//                     eidx,
//                     vidx,
//                     pt_tk,
//                     ee_tk,
//                     pt_contact_forces,
//                     ee_contact_forces,
//                     g_contact_forces,
//                     cubes, dt);
//             spdlog::info("alpha = {}", alpha);
//             dq *= alpha;
//             if (alpha < 2e-8) {
//                 spdlog::error("iter, ts ({}, {}), alpha = {}, E0 = {}, E1 = {}", iter, ts, alpha, E0, E1);
//             }
//             spdlog::info("step size upper = {}, alpha = {}", toi, alpha);
//         }
//         auto line_search_duration = DURATION_TO_DOUBLE(line_search_start);
//         times[__LINE_SEARCH__] += line_search_duration;



//         scalar norm_dq = dq.norm();
// #pragma omp parallel for schedule(static)
//         for (int i = 0; i < n_cubes; i++) {
//             for (int j = 0; j < 4; j++)
//                 cubes[i]->q[j] += dq.segment<3>(i * 12 + j * 3);
//         }

//         auto iter_duration = DURATION_TO_DOUBLE(newton_iter_start);
//         spdlog::warn("Newton iter #{}, time = {} ms, upper bound = {}, line search = {}", iter + 1, iter_duration, toi, alpha);

//         term_cond = sup_dq < tol || ++iter >= globals.max_iter;
//         sup_dq = 0.0;
//     } while (!term_cond);

//     scalar frame_duration = DURATION_TO_DOUBLE(frame_start) / 100.0;
//     spdlog::warn("converge #iter {}, ts = {}, time = {} ms\n\\
//     time breakdown :\n\\
//     \tipc: {:.3f} ms, percentage = {:.3f}% \n\\
//     \tsolver: {:.3f}, percentage = {:.3f}%\n\\
//     \tccd: {:.3f}, percentage = {:.3f}%\n\\
//     \tline search: {:.3f}, percentage = {:.3f}%\n\n\n",
//         ++iter, ts++, frame_duration * 100.0,
//         times[__IPC__],
//         times[__IPC__] / frame_duration,
//         times[__SOLVER__], times[__SOLVER__] / frame_duration,
//         times[__CCD__], times[__CCD__] / frame_duration,
//         times[__LINE_SEARCH__], times[__LINE_SEARCH__] / frame_duration);
//     globals.tot_iter += iter;
//     globals.aggregate_time += times;
//     if (globals.params_int.find("worst case iter") == globals.params_int.end())
//         globals.params_int["worst case iter"] = iter;
//     else
//         globals.params_int["worst case iter"] = max(iter, globals.params_int["worst case iter"]);
// #pragma omp parallel for schedule(static)
//     for (int k = 0; k < n_cubes; k++) {
//         auto& c(*cubes[k]);
//         for (int i = 0; i < 4; i++) {
//             c.dqdt[i] = (c.q[i] - c.q0[i]) / dt;
//             c.q0[i] = c.q[i];
//         }
//         c.p = c.q0[0];
//         c.A << c.q0[1], c.q0[2], c.q0[3];
//     }
// }
