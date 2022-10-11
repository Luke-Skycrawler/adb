#include "time_integrator.h"

using namespace std;
static const int max_iters = 5;
mat3 compute_residue(Cube &c, float dt)
{
    float m = c.mass / (dt * dt);
    mat3 r = m * c.q_next + othogonal_energy::grad(c.q_next) - (c.A /*+ c.translation + dt * c.vc */ + dt * c.q_dot) * m;
    return r;
}

void implicit_euler(float dt, vector<Cube> &cubes)
{
    // x[t+1] = x[t] + v[t+1] dt
    for (auto &c : cubes)
    {
        c.q_next = c.A;
    }
    for (int iter = 0; iter < max_iters; iter++)
    {
        // newton iterations
        for (auto &c : cubes)
        {
            float m = c.mass / (dt * dt);
            c.q_next = c.A;
            mat3 r = compute_residue(c, dt);

            // build hessian
            MatrixXf hess = MatrixXf::Identity(9, 9);
            hess *= m;
            for (int i = 0; i < 3; i++)
            {
                for (int j = i; j < 3; j++)
                {
                    hess.block<3, 3>(i * 3, j * 3) = othogonal_energy::hessian(c.q_next, i, j);
                }
            }

            VectorXf p = VectorXf::Zero(9, 1); // only for affine matrix
            Map<RowVectorXf> residue(r.data(), r.size());
            p = hess.ldlt().solve(residue);
            Map<MatrixXf> _p(p.data(), 3, 3);
            c.q_next += _p;
        }
    }
}