#include "time_integrator.h"
#include <iostream>
using namespace std;
static const int max_iters = 500;
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
            mat3 r = compute_residue(c, dt);

            // build hessian
            MatrixXf hess = MatrixXf::Identity(9, 9) * m;

            for (int i = 0; i < 3; i++)
            {
                for (int j = i; j < 3; j++)
                {
                    mat3 block = othogonal_energy::hessian(c.q_next, i, j);
                    if (j != i){
                        hess.block<3, 3>(j * 3, i * 3) = block;
                        hess.block<3, 3>(i * 3, j * 3) = block;
                    }
                    else {
                        hess.block<3,3> (i * 3, i *3) += block;
                        // elements already deployed at the diagonal
                    }
                }
            }

            VectorXf p = VectorXf::Zero(9, 1); // only for affine matrix
            Map<VectorXf> residue(r.data(), r.size());
            
            p = hess.ldlt().solve(residue);
            Map<MatrixXf> _p(p.data(), 3, 3);
            c.q_next -= _p;

            if (iter == 0 || iter == max_iters - 1){
                cout<< "iter " << iter << ", residue = " << residue << endl;
                cout << "p = " << p << endl;
                cout << "q = " << c.q_next << endl;
            }
        }
    }
    for (auto & c: cubes){
        c.q_dot = (c.q_next - c.A) * (1.0 / dt);
        c.A = c.q_next;
        cout << "end of time step, A = " << c.A << endl << "q. = " << c.q_dot << endl;
        cout << "energy = " << othogonal_energy :: otho_energy(c.A) << endl;
    }
}