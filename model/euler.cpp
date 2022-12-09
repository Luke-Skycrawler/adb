#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include "marcros_settings.h"
#include <assert.h>

using namespace std;
using namespace barrier;
using namespace Eigen;

VectorXd cat(vec3 q[])
{
    Vector<double, 12> ret;
    for (int i = 0; i < 4; i++) {
        ret.segment<3>(i * 3) = q[i];
    }
    return ret;
}

mat3 stack(vec3 q[])
{
    mat3 ret;
    ret << q[0], q[1], q[2];
    return ret;
}
VectorXd Cube::q_tile(double dt, const vec3& f)
{
    auto _q = cat(q0);
    auto _dqdt = cat(dqdt);
    _q = _q + dt * _dqdt;
    _q.head(3) += dt* dt* f;
    return _q;
}

// VectorXd& cat_all(vector<Cube>& cubes)
// {
//     VectorXd* ret = new VectorXd;
//     // for (auto &c : cubes) {
//     // }
//     return *ret;
// }

void implicit_euler(vector<Cube> & cubes, double dt) {

    bool term_cond;
    int iter = 0;
    double sup_dq = 0.0;
    for (auto &c: cubes) {
        for (int i = 0; i < 4; i ++) {
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
        //cout << hess_otho << endl;
        return H + hess_otho * dt * dt;
    };

    const auto norm_M = [&](const VectorXd &x, const Cube& c) -> double {
        // assert shape of x
        auto p = x.head(3);
        auto q = x.tail(9);
        return p.dot(p) * c.mass + q.dot(q) * c.Ic;
    };
    const auto E = [&](const VectorXd &q, const VectorXd& q_tiled, const Cube& c) -> double {
        return othogonal_energy::otho_energy(q) * dt * dt + 0.5 * norm_M(q - q_tiled, c);
        // FIXME: ugly, per_body otho energy
    };
    const auto barrier_grad_hess_per_body = [&](Cube& c, VectorXd& grad, MatrixXd& hess) {
        for (int i= 0; i < Cube::n_vertices; i++) {
            const vec3& v_tile(c.vertices()[i]);
            const vec3 v(c.vt1(i));
            double d = vg_distance(v);
            if (d < d_hat) {
                assert(d > 0);
                grad += barrier_gradient_q(v_tile, v);
                hess += barrier_hessian_q(v_tile, v);
                // FIXME: not debuged
            }
        }
    };

    const auto line_search = [&](const VectorXd& dq, const VectorXd& grad, VectorXd& q0, const VectorXd q_tiled, Cube& c) -> double {
        const double c1 = 1e-4;
        double alpha = 1.0;
        bool wolfe = false;
        double E0 = E(q0, q_tiled, c);
        VectorXd q1;
        do {
            q1 = q0 + dq * alpha;
            double E1 = E(q1, q_tiled, c);
            wolfe = E1 <= E0 + c1 * alpha * (dq.dot(grad));
            alpha /= 2;
            if (alpha < 1e-8) break;
        } while (!wolfe && grad.norm() > 1e-3);

        return alpha;
    };

    do {
        for (int k = 0; k < cubes.size(); k++) {
            auto& c(cubes[k]);
            VectorXd r = grad_residue_per_body(c);
            MatrixXd hess = hess_inertia_per_body(c);
            barrier_grad_hess_per_body(c, r, hess);

#ifdef _INDEPENDENT
            {
                VectorXd dq = -hess.ldlt().solve(r);
                auto q_tiled = c.q_tile(dt, globals.gravity);
                auto q = cat(c.q);
                double alpha = line_search(dq, r, q, q_tiled, c);
                c.increment_q = dq; // * alpha;
                c.toi = c.vg_collision_time();
                c.alpha = alpha;
                double n = (hess * dq + r).norm();
                spdlog::info("alpha = {}, solver res = {}, grad = {}", alpha, n, r.norm());
                // cout << hess;
            }
#else
            {
                int fi = 0, vi = 0, e0i = 0, e1i = 0;
                c.grad = r;
                c.hess = hess;
#ifdef _VF_
                fi = vf_colliding_response(1 - k, k);
                vi = vf_colliding_response(k, 1 - k);
#endif
#ifdef _EE_
                e0i = ee_colliding_response(k, 1 - k);
                e1i = ee_colliding_response(1 - k, k);
#endif

                if (fi || vi || e0i || e1i) {
                    spdlog::info("collision detected at {}", iter);
                }
            }
#endif
        }
        {
            const int n_cubes = cubes.size();
            const int hess_dim = n_cubes * 12;
            MatrixXd big_hess;
            big_hess.setZero(hess_dim, hess_dim);
            VectorXd r;
            r.setZero(hess_dim);

            for (int k = 0; k < n_cubes; k++) {
                big_hess.block<12, 12>(k * 12, k * 12) = cubes[k].hess;
                r.segment<12>(k * 12) = cubes[k].grad;
            }
            for (auto& triplet : globals.hess_triplets) {
                big_hess.block<12, 12>(triplet.i * 12, triplet.j * 12) = triplet.block;
            }
            VectorXd dq = - big_hess.ldlt().solve(r);
            for (int k = 0; k < n_cubes; k++) {
                auto &c(cubes[k]);
                c.increment_q = dq.segment<12>(k * 12); 
                c.toi = c.vg_collision_time();
            }
        }

        double toi = 1.0, factor = 1.0;

        for (auto &c : cubes) {
            toi = min(toi, c.toi);
        }
        if (toi < 1.0) {
            spdlog::warn("collision at {}, toi = {}", iter, toi);
            factor = 0.8;
        }

        for (auto& c : cubes) {
            auto tiled_q = c.q_tile(dt, globals.gravity);
            double E0 = E(cat(c.q), tiled_q, c);
            for (int i = 0; i < 4; i++) {
                c.q[i] += c.increment_q.segment<3>(i * 3) * toi * factor;
            }

            double norm_dq = c.increment_q.norm();
            sup_dq = max(sup_dq, norm_dq);
            double E1 = E(cat(c.q), tiled_q, c);
            spdlog::info("iter {}, e0 = {}, e1 = {}, norm_dq = {}, sup_dq = {}, dq = ", iter, E0, E1, norm_dq, sup_dq);
            spdlog::info("toi {}, factor = {}", toi, factor);

            //cout << c.increment_q << endl;
        }
        term_cond = sup_dq < 1e-6 || iter ++ > globals.max_iter;
        sup_dq = 0.0;
    }
    while (! term_cond);
    spdlog::info("\n  converge at iter {}", iter);
    for (auto& c : cubes) {
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
        // const vec3 tile_v(vertices()[i]);
        const vec3 v_t2(vt2(i));
        const vec3 v_t1(vt1(i));

        double d2 = vg_distance(v_t2);
        double d1 = vg_distance(v_t1);
        assert(d1 > 0);
        if (d2 < 0) {

            double t = d1 / (d1 - d2);
            auto vtoi = v_t2 * (t * 0.8) + v_t1 * (1 - t * 0.8);
            double dtoi = vg_distance(vtoi);
            spdlog::warn("dtoi = {}, d0 = {}, d1 = {}, toi = {}", dtoi, d1, d2, toi);

            assert(dtoi > 0.0);
            assert(t > 0.0 && t < 1.0);
            toi = min(toi, t);
        }
    }
    return toi;
}

void Cube::prepare_q_array(){
    q[0] = p_next;
    q0[0] = p;
    dqdt[0] = p_dot;
    for (int i = 1; i < 4; i++) {
        q[i] = q_next.col(i - 1);
        q0[i] = A.col(i - 1);
        dqdt[i] = q_dot.col(i - 1);
    }
}