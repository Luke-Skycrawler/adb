#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include <assert.h>
#include <array>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <algorithm>

using namespace std;
using namespace barrier;
using namespace Eigen;

VectorXd cat(const q4 &q)
{
    Vector<double, 12> ret;
    for (int i = 0; i < 4; i++) {
        ret.segment<3>(i * 3) = q[i];
    }
    return ret;
}

VectorXd AffineBody::q_tile(double dt, const vec3& f) const
{
    auto _q = cat(q0);
    auto _dqdt = cat(dqdt);
    _q = _q + dt * _dqdt;
    _q.head(3) += dt * dt * f;
    return _q;
}

void implicit_euler(vector<Cube>& cubes, double dt)
{

    bool term_cond;
    static int ts = 0;
    int iter = 0;
    double sup_dq = 0.0;
    for (auto& c : cubes) {
        for (int i = 0; i < 4; i++) {
            c.q[i] = c.q0[i];
        }
    }
    const auto grad_residue_per_body = [&](Cube& c) -> VectorXd {
        VectorXd grad = othogonal_energy::grad(c.q);
        const auto M = [&](const VectorXd& dq) -> VectorXd {
            auto ret = dq;
            ret.head(3) *= c.mass;
            ret.tail(9) *= c.Ic;
            return ret;
        };
        return dt * dt * grad + M(cat(c.q) - c.q_tile(dt, globals.gravity));
    };

    const auto hess_inertia_per_body = [&](Cube& c) -> MatrixXd {
        MatrixXd H = MatrixXd::Identity(12, 12) * c.Ic;
        H.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3) * c.mass;
        auto hess_otho = othogonal_energy::hessian(c.q);
        // cout << hess_otho << endl;
        return H + hess_otho * dt * dt;
    };

    const auto norm_M = [&](const VectorXd& x, const Cube& c) -> double {
        // assert shape of x
        auto p = x.head(3);
        auto q = x.tail(9);
        return p.dot(p) * c.mass + q.dot(q) * c.Ic;
    };

    vector<array<vec3, 4>> pts;
    vector<array<int, 2>> vidx;
    vector<array<int, 4>> idx;
    vector<array<vec3, 4>> ees;
    vector<array<int, 4>> eidx;
    const auto E = [&](const VectorXd& q, const VectorXd& q_tiled, const Cube& c) -> double {
        return othogonal_energy::otho_energy(q) * dt * dt + 0.5 * norm_M(q - q_tiled, c);
    };

    const auto E_ground = [&](const Cube& c, int i) -> double {
        double e = 0.0;
        const vec3& v_tile(c.vertices(i));
        const vec3 v(c.vt2(i));
        double d = vg_distance(v);
        d = d * d;
        if (d < d_hat) {
            e = barrier::barrier_function(d);
        }
        return e;
    };

    const auto E_global = [&](const VectorXd& q_plus_dq, const VectorXd& dq) -> double {
        double e = 0.0;
        const int n = cubes.size(), m = idx.size(), l = eidx.size();
        // inertia energy
        #pragma omp parallel for schedule(dynamic) reduction(+:e)
        for (int i = 0; i < n; i++) {
            auto& c(cubes[i]);
            c.dq = dq.segment<12>(i * 12);

            auto q_tiled = c.q_tile(dt, globals.gravity);
            auto _q = q_plus_dq.segment<12>(12 * i);
            double e_inert = E(_q, q_tiled, c);
            e += e_inert;
        }
        
        // point-triangle energy
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < m; k++) {
            auto& ij = idx[k];
            Face f(cubes[ij[2]], ij[3], true);
            vec3 v(cubes[ij[0]].vt2(ij[1]));
            // array<vec3, 4> a = {v, f.t0, f.t1, f.t2};
            double d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2);
            e += barrier::barrier_function(d);
        }

        // ee ipc energy
        for (int k = 0; k < l; k++) {
            auto &ij(eidx[k]);
            Edge ei(cubes[ij[0]], ij[1], true), ej(cubes[ij[2]], ij[3], true);
            double d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
            e += barrier_function(d);
        }

        // vertex-ground ipc energy
        for (auto v : vidx) {
            double e_ground = E_ground(cubes[v[0]], v[1]);
            e += e_ground;
        }
        return e;
    };

    const auto line_search = [&](const VectorXd& dq, const VectorXd& grad, VectorXd& q0) -> double {
        const double c1 = 1e-4;
        double alpha = 1.0;
        bool wolfe = false;
        double E0 = E_global(q0, 0.0 * dq);
        double qdg = dq.dot(grad);
        // double E0 = E(q0, q_tiled, c);
        VectorXd q1;
        do {
            q1 = q0 + dq * alpha;
            auto dqk = dq * alpha;

            double E1 = E_global(q1, dqk);
            wolfe = E1 <= E0 + c1 * alpha * qdg;
            // spdlog::info("wanted descend = {}, E1 - E0 = {}, E1 = {}, E0 = {}, alpha = {}", c1 * alpha * qdg, E1 - E0, E1, E0, alpha);
            alpha /= 2;
            if (alpha < 1e-8) break;
        } while (!wolfe && grad.norm() > 1e-3);

        return alpha * 2;
    };

    const auto gen_collision_set = [&](const vector<Cube>& cubes) {
        const int n = cubes.size(), nsqr = n * n;
        if (globals.ground)
            for (int i = 0; i < n; i++)
                for (int v = 0; v < Cube::n_vertices; v++) {
                    auto p = cubes[i].vt1(v);
                    double d = vg_distance(p);
                    d = d * d;
                    if (d < barrier::d_hat * globals.safe_factor) {
                        vidx.push_back({ i, v });
                    }
                }
        if (globals.pt) {
        #pragma omp parallel for schedule(dynamic)
        for (int I = 0; I < nsqr; I++) {
            int i = I / n, j = I % n;
                auto &ci(cubes[i]), &cj(cubes[j]);
                if (i == j) continue;
                for (int v = 0; v < Cube::n_vertices; v++)
                    for (int f = 0; f < Cube::n_faces; f++) {
                        Face _f(cj, f);
                        vec3 p = ci.vt1(v);
                        auto fu = _f.t0.cwiseMax(_f.t1).cwiseMax(_f.t2).array() + barrier ::d_sqrt;
                        auto fl = _f.t0.cwiseMin(_f.t1).cwiseMin(_f.t2).array() - barrier::d_sqrt;
                        if ((p.array() <= fu.array()).all() && (p.array() >= fl.array()).all()) {
                        }
                        else
                            continue;
                        double d = ipc::point_triangle_distance(p, _f.t0, _f.t1, _f.t2);
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
        }
        if (globals.ee)
        for(int i = 0; i < n; i++) for(int j = i + 1; j < n; j ++){
                auto &ci(cubes[i]), &cj(cubes[j]);
                if (i == j) continue;
                for (int _ei = 0; _ei < Cube::n_edges; _ei++)
                    for (int _ej = 0; _ej < Cube::n_edges; _ej++) {
                        Edge ei(ci, _ei), ej(cj, _ej);
                        auto iu = ei.e0.cwiseMax(ei.e1).array() + barrier :: d_sqrt / 2;
                        auto il = ei.e0.cwiseMin(ei.e1).array() - barrier :: d_sqrt / 2;

                        auto ju = ej.e0.cwiseMax(ej.e1).array() + barrier :: d_sqrt / 2;
                        auto jl = ej.e0.cwiseMin(ej.e1).array() - barrier :: d_sqrt / 2;
                        if ((iu.array() >= jl.array()).all() && (ju.array() >= il.array() ).all()) {} 
                        else continue;
                        double d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
                        if (d < barrier::d_hat * globals.safe_factor) {
                            array<vec3, 4> ee = {ei.e0, ei.e1, ej.e0, ej.e1};
                            array<int, 4> ij = { i, _ei, j, _ej };

                            #pragma omp critical
                            {
                                ees.push_back(ee);
                                eidx.push_back(ij);
                            }
                        }
                    }
            }    
    };
    if (globals.col_set)
        gen_collision_set(cubes);
    spdlog::info("constraint size = {}, {}", pts.size(), idx.size());

    do {
        const int nc = cubes.size();
        globals.hess_triplets.clear();
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < nc; k++) {
            auto& c(cubes[k]);
            VectorXd r = grad_residue_per_body(c);
            MatrixXd hess = hess_inertia_per_body(c);
            // barrier_grad_hess_per_body(c, r, hess);

            c.grad = r;
            c.hess = hess;
        }
        for (auto _v : vidx) {
            int i = _v[0], v = _v[1];
            vec3 p = cubes[i].vt1(v);
            vec3 v_tile = cubes[i].vertices(v);
            double d = vg_distance(p);
            d = d * d;
            if (d < barrier::d_hat) {
                cubes[i].grad += barrier_gradient_q(v_tile, p);
                cubes[i].hess += barrier_hessian_q(v_tile, p);
            }
        }
        for (int k = 0; k < pts.size(); k++) {
            auto& pt(pts[k]);
            auto& ij(idx[k]);

            int i = ij[0], j = ij[2];
            auto &ci(cubes[i]), &cj(cubes[j]);
            double d = ipc::point_triangle_distance(pt[0], pt[1], pt[2], pt[3]);
            if (d < barrier::d_hat) {
                ipc_term(pt, ij, globals.hess_triplets ,ci.grad, cj.grad);
            }
        }
        
        for (int k = 0; k< ees.size(); k ++) {
            auto &ee(ees[k]);
            auto &ij(eidx[k]);
            int i = ij[0], j = ij[2];
            auto &ci(cubes[i]), &cj(cubes[j]);
            double d = ipc::edge_edge_distance(ee[0], ee[1], ee[2], ee[3]);
            if(d < barrier::d_hat) {
                ipc_term_ee(ee, ij, globals.hess_triplets, ci.grad, cj.grad);
            }
        }
        

        const auto collision_time = [&](Cube& c, int i) {
            double toi = 1.0;
            const vec3 v_t2(c.vt2(i));
            const vec3 v_t1(c.vt1(i));

            double d2 = vg_distance(v_t2);
            double d1 = vg_distance(v_t1);
            assert(d1 > 0);
            if (d2 < 0) {

                double t = d1 / (d1 - d2);
                auto vtoi = v_t2 * (t * 0.8) + v_t1 * (1 - t * 0.8);
                double dtoi = vg_distance(vtoi);
                spdlog::error("dtoi = {}, d0 = {}, d1 = {}, toi = {}", dtoi, d1, d2, t);

                assert(dtoi > 0.0);
                assert(t > 0.0 && t < 1.0);
                toi = min(toi, t);
            }
            return toi;
        };
        const auto step_size_upper_bound = [&](VectorXd& dq, vector<Cube> cubes) -> double {
            double toi = 1.0;
            const int nc = cubes.size(), nidx = idx.size(), ne = eidx.size();

            for (auto v : vidx) {
                double t = collision_time(cubes[v[0]], v[1]);
                toi = min(t, toi);
            }
            // #pragma omp parallel for schedule(dynamic) reduction(min: toi)
            for (int k = 0; k < nidx; k++) {
                auto& pt(pts[k]);
                const auto& ij(idx[k]);

                int i = ij[0], j = ij[2];
                // Face f(cubes[j], ij[3], true);
                vec3 p_t2 = cubes[i].vt2(ij[1]);
                vec3 p_t1 = cubes[i].vt1(ij[1]);
                double t = vf_collision_detect(p_t1, p_t2, cubes[j], ij[3]);
                toi = min(toi, t);
            }
            //#pragma omp parallel for schedule(dynamic) reduction(min: toi)
            for (int k = 0; k < ne; k++) {
                auto& ee(ees[k]);
                const auto& ij(eidx[k]);

                auto & ci(cubes[ij[0]]), &cj(cubes[ij[2]]);
                double t = ee_collision_detect(ci, cj, ij[1], ij[2]);
                toi = min(toi, t);
            }
            return toi;
        };
        double toi = 1.0, factor = 1.0, alpha = 1.0;

        {
            static const int n_cubes = cubes.size();

            vector<int> starting_point;
            starting_point.resize(n_cubes * 12);
            static const auto merge_triplets = [&](vector<HessBlock>& triplets) {
                sort(triplets.begin(), triplets.end(), [&](const HessBlock& a, const HessBlock& b) -> bool {
                    return a.j < b.j || (a.j == b.j && a.i < b.i);
                });
                int n = triplets.size();
                for (int i = 0; i < n; i++) {
                    if (triplets[i].i == -1) continue;
                    for (int j = i + 1; j < n; j++)
                        if (triplets[i].i == triplets[j].i && triplets[i].j == triplets[j].j) {
                            // disable triplet j
                            triplets[j].i = -1;
                            triplets[i].block += triplets[j].block;
                        }
                        else {
                            if (triplets[i].j != triplets[j].j)
                                starting_point[triplets[j].j] = j;
                            break;
                        }
                }
                starting_point[0] = 0;
            };
            const int hess_dim = n_cubes * 12;
            MatrixXd big_hess;
            //vector<Triplet<double>> bht;
            SparseMatrix<double> sparse_hess(hess_dim, hess_dim);
            //static const auto insert = [&](vector<Triplet<double>>& bht, const Matrix<double, 12, 12>& m, int r, int c) {
            //    for (int i = 0; i < 12; i++)
            //        for (int j = 0; j < 12; j++) {
            //            bht.push_back({ r + i, c + j, m(i, j) });
            //        }
            //};
            static const auto insert2 = [&](SparseMatrix<double>& sm, int tid) {
                auto& triplet = globals.hess_triplets[tid];
                bool new_col = tid == 0 || globals.hess_triplets[tid - 1].j != triplet.j;

                int c = triplet.j, r = triplet.i;
                if (new_col)
                    sm.startVec(c);
                for (int i = 0; i < 12; i++) {
                    sm.insertBack(i + r, c) = triplet.block(i, 0);
                }
            };

            if (globals.sparse){
                int n_ele = (n_cubes + globals.hess_triplets.size())* 12 * 12;
                //bht.resize(n_ele);
                //sparse_hess.reserve(n_ele);
            }

            big_hess.setZero(hess_dim, hess_dim);
            VectorXd r, q0_cat, q_tile_cat, dq;
            r.setZero(hess_dim);
            dq.setZero(hess_dim);
            q0_cat.setZero(hess_dim);
            q_tile_cat.setZero(hess_dim);

            for (int k = 0; k < n_cubes; k++) {
                // if (globals.dense)
                //     big_hess.block<12, 12>(k * 12, k * 12) += cubes[k].hess;
                
                // if(globals.sparse)
                //     insert(bht, cubes[k].hess, k * 12, k * 12);

                for (int i = 0; i < 12; i++)
                    globals.hess_triplets.push_back({ k * 12, k * 12 + i, cubes[k].hess.block<12, 1>(0, i) });
                // globals.hess_triplets.push_back({k * 12, k * 12, cubes[k].hess});

                r.segment<12>(k * 12) = cubes[k].grad;
                auto t = cat(cubes[k].q);
                q0_cat.segment<12>(k * 12) = t;
                q_tile_cat.segment<12>(k * 12) = cubes[k].q_tile(dt, globals.gravity);
            }

            merge_triplets(globals.hess_triplets);

            const int nt = globals.hess_triplets.size();

            for (int k = 0; k < nt; k++) {
                auto& triplet(globals.hess_triplets[k]);
                if (triplet.i == -1) continue;
                if (globals.dense)
                    big_hess.block<12, 1>(triplet.i, triplet.j) = triplet.block;
                    // big_hess.block<12, 12>(triplet.i, triplet.j) = triplet.block;
                if (globals.sparse)
                    // insert(bht, triplet.block, triplet.i * 12, triplet.j * 12);
                    insert2(sparse_hess, k);
            }

            const auto damping = [&]() {
                MatrixXd D = globals.beta * big_hess;
                for (int i = 0; i < n_cubes; i++) {
                    for (int j = 0; j < 3; j++) { D(i * 12 + j, i * 12 + j) += cubes[i].mass; }
                    for (int j = 3; j < 12; j++) { D(i * 12 + j, i * 12 + j) += cubes[i].Ic; }
                }
                big_hess += D / dt;
            };

            damping();
            if (globals.dense)
                dq = -big_hess.ldlt().solve(r);
            else if (globals.sparse) {
                // sparse_hess.setFromTriplets(bht.begin(), bht.end());
                sparse_hess.finalize();
                SimplicialLDLT<SparseMatrix<double>> ldlt_solver;
                ldlt_solver.compute(sparse_hess);
                dq = -ldlt_solver.solve(r);
            }

            double dif = (sparse_hess - big_hess).norm();
            spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm(): big_hess.norm());
            // spdlog::warn("dense norms: dq = {}, grad = {}, big_hess = {}, difference = {}", dq.norm(), r.norm(), big_hess.norm(), dif);
            spdlog::info("dq dot grad = {}, cos = {}", dq.dot(r), dq.dot(r) / (dq.norm() * r.norm()));
            if (globals.sparse && globals.dense && dif > 1e-6) {
                spdlog::error("diff too large");
                cout << big_hess << "\n\n"
                     << sparse_hess;
            }

            toi = 1.0;
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < n_cubes; k++) {
                auto& c(cubes[k]);
                c.dq = dq.segment<12>(k * 12);
            }

            if (globals.upper_bound)
                toi = step_size_upper_bound(dq, cubes);

            if (toi < 1.0) {
                spdlog::warn("collision at {}, toi = {}", iter, toi);
                factor = 0.8;
            }

            dq *= factor * toi;

            alpha = 1.0;
            if (globals.line_search)
                alpha = line_search(dq, r, q0_cat);
            spdlog::info("alpha = {}", alpha);
            dq *= alpha;
            double E0 = E_global(q0_cat, dq * 0.0), E1 = E_global(q0_cat + dq, dq);
            double norm_dq = dq.norm();
            sup_dq = norm_dq;
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n_cubes; i++) {
                for (int j = 0; j < 4; j++)
                    cubes[i].q[j] += dq.segment<3>(i * 12 + j * 3);
            }
            spdlog ::info("step size upper = {}, alpha = {}", toi, alpha);
            spdlog::info("iter {}, e0 = {}, e1 = {}, norm_dq = {}\n", iter, E0, E1, norm_dq);
        }
        const int npts = pts.size();
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < npts; k++) {
            auto& ij = idx[k];
            Face f(cubes[ij[2]], ij[3]);
            vec3 v(cubes[ij[0]].vt1(ij[1]));
            pts[k] = { v, f.t0, f.t1, f.t2 };
        }
        term_cond = sup_dq < 1e-6 || iter++ > globals.max_iter;
        sup_dq = 0.0;
    } while (!term_cond);
    spdlog::info("\n  converge at iter {}, ts = {} \n", iter, ts++);
    const int nc = cubes.size();
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < nc; k++) {
        auto &c(cubes[k]);
        for (int i = 0; i < 4; i++) {
            c.dqdt[i] = (c.q[i] - c.q0[i]) / dt;
            c.q0[i] = c.q[i];
        }
        c.p = c.q0[0];
        c.A << c.q0[1], c.q0[2], c.q0[3];
    }
}

