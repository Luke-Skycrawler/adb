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

VectorXd Cube::q_tile(double dt, const vec3& f)
{
    auto _q = cat(q0);
    auto _dqdt = cat(dqdt);
    return _q + dt * _dqdt + dt * dt * f;
}

VectorXd cat(vec3 q[])
{
    Vector<double, 12> ret;
    for (int i = 0; i < 4; i++) {
        ret.segment<3>(i * 3) = q[i];
    }
    return ret;
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
    double sup_dq;

    const auto grad_residue_per_body = [&](Cube& c) -> VectorXd {
        auto grad = othogonal_energy::grad(c.q);
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
        return H + hess_otho * dt * dt;
    };

    const auto grad_barrier_per_body = [&](Cube &c)  -> VectorXd{
        VectorXd ret; 
        for (int i= 0; i < Cube::n_vertices; i++) {
            const vec3& v_tile(c.vertices()[i]);
            const vec3 v(c.v(i));
            double d = vg_distance(v);
            if (d < d_hat) {
                assert(d > 0);
                ret += barrier_gradient_q(v_tile, v);
            }
        }
    };

    const auto hess_barrier_per_body = [&](Cube &c) -> MatrixXd{
        
    };
    do {
        for (int k = 0; k < cubes.size(); k++) {
            auto& c(cubes[k]);
            VectorXd r = grad_residue_per_body(c) + grad_barrier_per_body(c);
            MatrixXd hess = hess_inertia_per_body(c) + hess_barrier_per_body(c);


            #ifdef _INDEPENDENT
            {


            }
            #endif

        }
        
        term_cond = sup_dq > 1e-6 && iter < globals.max_iter;
    }
    while (! term_cond);
}