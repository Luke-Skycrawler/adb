#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include "marcros_settings.h"
#include <assert.h>
#include <array>
#include <ipc/distance/point_triangle.hpp>
using namespace std;
using namespace barrier;
using namespace Eigen;

VectorXd cat(array<vec3, 4> q)
{
    Vector<double, 12> ret;
    for (int i = 0; i < 4; i++) {
        ret.segment<3>(i * 3) = q[i];
    }
    return ret;
}

VectorXd Cube::q_tile(double dt, const vec3& f) const
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
    vector<array<int, 4>> idx;
    const auto E = [&](const VectorXd& q, const VectorXd& q_tiled, const Cube& c) -> double {
        return othogonal_energy::otho_energy(q) * dt * dt + 0.5 * norm_M(q - q_tiled, c);
    };

    const auto E_ground = [&](const Cube& c) -> double {
        double e = 0.0;
        for (int i = 0; i < Cube::n_vertices; i++) {
            const vec3& v_tile(c.vertices()[i]);
            const vec3 v(c.vt2(i));
            double d = vg_distance(v);
            d = d * d;
            if (d < d_hat) {
                e += barrier::barrier_function(d);
            }
        }
        return e;
    };

    const auto E_global = [&](const VectorXd& q_plus_dq, const VectorXd& dq) -> double {
        double e = 0.0;
        const int n = cubes.size(), m = idx.size();
        #pragma omp parallel for schedule(dynamic) reduction(+:e)
        for (int i = 0; i < n; i++) {
            auto& c(cubes[i]);
            c.dq = dq.segment<12>(i * 12);

            auto q_tiled = c.q_tile(dt, globals.gravity);
            auto _q = q_plus_dq.segment<12>(12 * i);
            double e_ground_inert = E(_q, q_tiled, c) + E_ground(c);

            e += e_ground_inert;
        }
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < m; k++) {
            auto& ij = idx[k];
            Face f(cubes[ij[2]], ij[3], true);
            vec3 v(cubes[ij[0]].vt2(ij[1]));
            // array<vec3, 4> a = {v, f.t0, f.t1, f.t2};
            double d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2);
            e += barrier::barrier_function(d);
        }
        return e;
    };
    const auto barrier_grad_hess_per_body = [&](Cube& c, VectorXd& grad, MatrixXd& hess) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < Cube::n_vertices; i++) {
            const vec3& v_tile(c.vertices()[i]);
            const vec3 v(c.vt1(i));
            double d = vg_distance(v);
            d = d * d;
            if (d < d_hat) {
                assert(d > 0);
                grad += barrier_gradient_q(v_tile, v);
                hess += barrier_hessian_q(v_tile, v);
                // FIXME: not debuged
            }
        }
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
        #pragma omp parallel for schedule(dynamic)
        for (int I = 0; I < nsqr; I++) {
            int i = I / n, j = I % n;
                auto &ci(cubes[i]), &cj(cubes[j]);
                if (i == j) continue;
                for (int v = 0; v < Cube::n_vertices; v++)
                    for (int f = 0; f < Cube::n_faces; f++) {
                        Face _f(cj, f);
                        vec3 p = ci.vt1(v);
                        auto fu = _f.t0.cwiseMax(_f.t1).cwiseMax(_f.t2).array() + barrier :: d_sqrt;
                        auto fl = _f.t0.cwiseMin(_f.t1).cwiseMin(_f.t2).array() - barrier::d_sqrt;
                        if ((p.array() <= fu.array()).all() && (p.array() >= fl.array()).all()) {
                            
                        }
                        else continue;
                        double d = ipc::point_triangle_distance(p, _f.t0, _f.t1, _f.t2);
                        if (d < barrier::d_hat) {
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
    };
    if (globals.col_set)
        gen_collision_set(cubes);
    spdlog::info("constraint size = {}, {}", pts.size(), idx.size());

    do {
        const int nc = cubes.size();
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < nc; k++) {
            auto& c(cubes[k]);
            VectorXd r = grad_residue_per_body(c);
            MatrixXd hess = hess_inertia_per_body(c);
            barrier_grad_hess_per_body(c, r, hess);

            c.grad = r;
            c.hess = hess;
        }
#ifdef _VF_
        const int npts = pts.size();
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < npts; k++) {
            auto& pt(pts[k]);
            auto& ij(idx[k]);

            int i = ij[0], j = ij[2];
            auto &ci(cubes[i]), &cj(cubes[j]);
            double d = ipc::point_triangle_distance(pt[0], pt[1], pt[2], pt[3]);
            
            if (d < barrier::d_hat) {
                Matrix<double, 12, 12> hess_p, hess_t;
                Vector<double, 12> grad_p, grad_t;
                ipc_term(hess_p, hess_t, grad_p, grad_t, pt, ij);
                #pragma omp critical
                {
                    ci.hess += hess_p;
                    cj.hess += hess_t;
                    ci.grad += grad_p;
                    cj.grad += grad_t;
                }
            }
        }

#endif
#ifdef _EE_
        e0i = ee_colliding_response(k, 1 - k);
        e1i = ee_colliding_response(1 - k, k);
#endif

        const auto step_size_upper_bound = [&](VectorXd& dq, vector<Cube> cubes) -> double {
            double toi = 1.0;
            const int nc = cubes.size(), nidx = idx.size();
            //#pragma omp parallel for schedule(dynamic) reduction(min: toi)
            for (int i = 0; i < nc; i++){
                double _t = cubes[i].vg_collision_time();
                toi = min(toi, _t);
            }
            //#pragma omp parallel for schedule(dynamic) reduction(min: toi)
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
            return toi;
        };
        double toi = 1.0, factor = 1.0, alpha = 1.0;
        static const auto insert = [&](vector<Triplet<double>> &bht, const Matrix<double, 12, 12> &m, int r, int c) {
            for(int i = 0; i< 12; i ++)for(int j = 0; j < 12; j ++){
                bht.push_back({r + i, c + j, m(i, j)});
            }
        };
        static const auto insert1 = [&](SparseMatrix<double>& sm, const Matrix<double, 12, 12>& m, int r, int c) {
            for (int j = 0; j < 12; j++) {
                sm.startVec(j + c);
                for (int i = 0; i < 12; i++) {
                    sm.insertBack(i + r, j + c) = m(i, j);
                }
            }
        }; 
        {
            const int n_cubes = cubes.size();
            const int hess_dim = n_cubes * 12;
            MatrixXd big_hess;
            vector<Triplet<double>> bht;
            SparseMatrix<double> sparse_hess(hess_dim, hess_dim);
            if (globals.sparse){
                int n_ele = (n_cubes + globals.hess_triplets.size())* 12 * 12;
                bht.resize(n_ele);
                sparse_hess.reserve(n_ele);
            }


            big_hess.setZero(hess_dim, hess_dim);
            VectorXd r, q0_cat, q_tile_cat, dq;
            r.setZero(hess_dim);
            dq.setZero(hess_dim);
            q0_cat.setZero(hess_dim);
            q_tile_cat.setZero(hess_dim);
            //#pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < n_cubes; k++) {
                if(!globals.sparse)
                    big_hess.block<12, 12>(k * 12, k * 12) += cubes[k].hess;
                else 
                    insert(bht, cubes[k].hess, k * 12, k * 12);
                    // insert1(sparse_hess, cubes[k].hess, k * 12, k * 12);
                r.segment<12>(k * 12) = cubes[k].grad;
                auto t = cat(cubes[k].q);
                q0_cat.segment<12>(k * 12) = t;
                q_tile_cat.segment<12>(k * 12) = cubes[k].q_tile(dt, t);
            }

            const int ntrip = globals.hess_triplets.size();
            //#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < ntrip; i++) {
                auto &triplet(globals.hess_triplets[i]);
                if (!globals.sparse)
                big_hess.block<12, 12>(triplet.i * 12, triplet.j * 12) += triplet.block;
                else 
                insert(bht, triplet.block, triplet.i * 12, triplet.j * 12);
            }

            //const auto damping = [&]() {
            //    MatrixXd M, D;
            //    VectorXd q_cat;

            //    M.setZero(hess_dim, hess_dim);
            //    D.setZero(hess_dim, hess_dim);
            //    q_cat.setZero(hess_dim);

            //    for (int i = 0; i < cubes.size(); i++) {
            //        Matrix<double, 12, 12> m;
            //        auto& c(cubes[i]);
            //        m = MatrixXd::Identity(12, 12) * c.Ic;
            //        m.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3) * c.mass;
            //        M.block<12, 12>(i * 12, i * 12) = m;

            //        q_cat.segment<12>(i * 12) = cat(c.q0);
            //    }

            //    D = globals.beta * big_hess + (globals.alpha - globals.beta) * M;

            //    VectorXd damp_term = D * q_cat / dt;
            //    r += damp_term;

            //    big_hess += globals.beta * big_hess / dt;
            //    for (int i = 0; i < cubes.size(); i++) {
            //        for (int j = 0; j < 3; j++) {
            //            big_hess(i * 12 + j, i * 12 + j) += (globals.alpha - globals.beta) * cubes[i].mass / dt;
            //        }
            //        for (int j = 3; j < 12; j++) {
            //            big_hess(i * 12 + j, i * 12 + j) += (globals.alpha - globals.beta) * cubes[i].Ic / dt;
            //        }
            //    }
            //};

            // damping();
            if (globals.sparse){
                sparse_hess.setFromTriplets(bht.begin(), bht.end());
                SimplicialLDLT<SparseMatrix<double>> ldlt_solver;
                ldlt_solver.compute(sparse_hess);
                dq = -ldlt_solver.solve(r);
            }
            else 
                dq = -big_hess.ldlt().solve(r);
            spdlog::info("norms: dq = {}, grad = {}, big_hess = {}", dq.norm(), r.norm(), globals.sparse ? sparse_hess.norm(): big_hess.norm());
            // spdlog::warn("dense norms: dq = {}, grad = {}, big_hess = {}, difference = {}", dq.norm(), r.norm(), big_hess.norm(), (dq - dq_sparse).norm());
            spdlog::info("dq dot grad = {}, cos = {}", dq.dot(r), dq.dot(r) / (dq.norm() * r.norm()));

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

double Cube::vg_collision_time()
{
    double toi = 1.0;
    for (int i = 0; i < n_vertices; i++) {
        const vec3 v_t2(vt2(i));
        const vec3 v_t1(vt1(i));

        double d2 = vg_distance(v_t2);
        double d1 = vg_distance(v_t1);
        assert(d1 > 0);
        if (d2 < 0) {

            double t = d1 / (d1 - d2);
            auto vtoi = v_t2 * (t * 0.8) + v_t1 * (1 - t * 0.8);
            double dtoi = vg_distance(vtoi);
            spdlog::warn("dtoi = {}, d0 = {}, d1 = {}, toi = {}", dtoi, d1, d2, t);

            assert(dtoi > 0.0);
            assert(t > 0.0 && t < 1.0);
            toi = min(toi, t);
        }
    }
    return toi;
}