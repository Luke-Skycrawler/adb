#include "time_integrator.h"
#include "barrier.h"
#include <iostream>
#include <assert.h>
#include "collision.h"

using namespace std;
using namespace barrier;
using namespace Eigen;

static const int max_iters = 10;

// #define _DEBUG_BARRIER_
// #define PER_ITER_RESIDUE_PRINT
VectorXf q_residue_barrier_term(Cube &c)
{
    VectorXf barrier_term;
    barrier_term.setZero(12, 1);

    for (int i = 0; i < 8; i++)
    {
        const vec3 &v0(c.vertices()[i]);
        const vec3 &v(c.q_next * v0 + c.p_next);
        float d = vf_distance(v);

        if (d < d_hat)
        {
            // TODO: ccd restart
            assert(d > 0);
            barrier_term += barrier_gradient_q(v0, v);
        }
    }
    // barrier_term.segment(0, 3).setZero();
    return barrier_term;
}

mat3 q_residue(Cube &c, float dt)
{
    float m = 1.0 / 12.0f * c.mass * c.scale * c.scale / (dt * dt); // to be corrected
    mat3 r = m * c.q_next + othogonal_energy::grad(c.q_next) - (c.A + dt * c.q_dot) * m;
    return r;
}

vec3 p_residue(Cube &c, float dt)
{
    // static const vec3 gravity(0.0f, 0.0f, 0.0f);
    static const vec3 gravity(0.0f, -9.8, 0.0f);
    float m = c.mass / (dt * dt);
    vec3 r = m * c.p_next - gravity * c.mass - (c.p + dt * c.p_dot) * m;
    return r;
}

VectorXf cat(mat3 &rq, const vec3 &rp)
{
    Vector<float, 12> ret;
    Map<VectorXf> _rq(rq.data(), rq.size());
    ret << rp, _rq;
    return ret;
}
void implicit_euler(vector<Cube> &cubes)
{
    // x[t+1] = x[t] + v[t+1] dt
    static const float dt = 1e-4;
    for (auto &c : cubes)
    {
        c.q_next = c.A;
        c.p_next = c.p;
    }
    for (int iter = 0; iter < max_iters; iter++)
    {
        // newton iterations
        for (auto &c : cubes)
        {
            float m = c.mass / (dt * dt);
            float Im = m / 12.0 * c.scale * c.scale;
            mat3 rq(q_residue(c, dt));
            vec3 rp(p_residue(c, dt));
            VectorXf r(cat(rq, rp));

            r += q_residue_barrier_term(c);

            // build hessian
            MatrixXf hess = MatrixXf::Identity(12, 12) * Im;
            hess.block<3, 3>(0, 0) = MatrixXf::Identity(3, 3) * m;

            for (int i = 1; i < 4; i++)
            {
                for (int j = i; j < 4; j++)
                {
                    mat3 block = othogonal_energy::hessian(c.q_next, i - 1, j - 1);
                    if (j != i)
                    {
                        hess.block<3, 3>(j * 3, i * 3) = block;
                        hess.block<3, 3>(i * 3, j * 3) = block;
                    }
                    else
                    {
                        hess.block<3, 3>(i * 3, i * 3) += block;
                        // elements already deployed at the diagonal
                    }
                }
            }

            // hessian for barrier term
            for (int i = 0; i < 8; i++)
            {
                const vec3 &v0(c.vertices()[i]);
                const vec3 &v(c.q_next * v0 + c.p_next);
                hess += barrier_hessian_q(v0, v);
            }

            VectorXf _dq(hess.ldlt().solve(r));

            // Map<MatrixXf> dq(_dq.block<9,9>(3,3).data(), 3, 3);
            VectorXf __dq(_dq.segment<9>(3));
            Map<MatrixXf> dq(__dq.data(), 3, 3);
            vec3 dp(_dq.segment(0, 3));

            float toi = c.vf_collision_detect(dp, dq);

            if (toi < 1.0) {
                cout << "collision detected, resetting i" << toi << endl;
                iter = 0; 
            }
            // only works for single cube

            c.q_next -= dq * toi;
            c.p_next -= dp * toi;
#ifdef PER_ITER_RESIDUE_PRINT
            if (iter == 0 || iter == max_iters - 1)
            {
                cout << "iter " << iter << ", residue = " << r << endl;
                cout << "dp = " << dp << endl;
                cout << "dq = " << dq << endl;
            }
#endif
        }
    }
    for (auto &c : cubes)
    {
        c.q_dot = (c.q_next - c.A) * (1.0 / dt);
        c.A = c.q_next;

        c.p_dot = (c.p_next - c.p) * (1.0 / dt);
        c.p = c.p_next;
#ifdef PER_TIMESTEP_PRINT
        cout << "end of time step, A = " << c.A << endl
             << "q. = " << c.q_dot << endl;
        cout << "energy = " << othogonal_energy ::otho_energy(c.A) << endl;
#endif
    }
}