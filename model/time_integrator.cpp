#include "time_integrator.h"
#include "barrier.h"
#include <iostream>
#include <assert.h>
#include "collision.h"
#include "glue.h"

using namespace std;
using namespace barrier;
using namespace Eigen;

static const int max_iters = 10;


// #define _DEBUG_BARRIER_
mat3 q_residue_barrier_term(Cube &c) {
    VectorXf barrier_term;
    barrier_term.setZero(9, 1);
  
    for (int i=0; i< 8; i++) {
        const vec3 &v(c.q_next * c.vertices()[i] + c.p_next);
        float d = vf_distance(v);
        
        if (d < d_hat) {
            // TODO: ccd restart
            assert(d > 0);
            barrier_term += barrier_gradient_q(v);
        }
    }
    Map<MatrixXf> ret(barrier_term.data(), 3, 3);
    return ret;
}

vec3 p_residue_barrier_term(Cube &c) {
    vec3 barrier_term(0.0f,0.0f,0.0f);
    for (int i=0; i <8; i++){
        c.toi[i] = 1.0;
    }
    for (int i=0; i< 8; i++) {
        const vec3 &v(c.q_next * c.vertices()[i] + c.p_next);
        float d = vf_distance(v);
        
        if (d < d_hat) {
            // TODO: ccd restart
            #ifdef _DEBUG_BARRIER_
            cout<< "collision detected, d = " << d << endl;
            #endif
            assert(d > 0);
            if (d < 0){
                float toi = vf_collision_time(c.vertices()[i], c.p, c.A, c.p_next, c.q_next);
                c.toi[i] = toi;
                continue;
            }
            barrier_term += barrier_gradient_x(v);
            #ifdef _DEBUG_BARRIER_
            cout<< "gradient = " << barrier_gradient_x(v) << endl;
            #endif
        }
    }
    float toi = *min_element(c.toi.begin(), c.toi.end());
    if (toi < 1.0f){
        throw toi;
    }
    return barrier_term;
}

mat3 q_residue(Cube &c, float dt)
{
    float m = 1.0 / 12.0f * c.mass * c.scale * c.scale/ (dt * dt);   // to be corrected
    mat3 r = m * c.q_next + othogonal_energy::grad(c.q_next) - (c.A + dt * c.q_dot) * m;
    r += q_residue_barrier_term(c);
    return r;
}

vec3 p_residue(Cube &c, float dt) {
    static const vec3 gravity(0.0f, -9.8e3, 0.0f);
    float m = c.mass / (dt * dt);   
    vec3 r = m * c.p_next - gravity * c.mass - (c.p + dt * c.p_dot) * m;
    r += p_residue_barrier_term(c);
    return r;
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
            mat3 rq;
            vec3 rp;
            try {
                rq = q_residue(c, dt);
                rp = p_residue(c, dt);
            }
            catch (float toi) {
                restart_at(toi * 0.9);
            }
            
            // build hessian
            MatrixXf hess = MatrixXf::Identity(9, 9) * Im;

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

            // hessian for barrier term
            for(int i =0;i<8;i++){
                const vec3 &v(c.q_next * c.vertices()[i] + c.p_next);
                hess += barrier_hessian_q(v);
            }

            VectorXf p = VectorXf::Zero(9, 1); // only for affine matrix
            Map<VectorXf> residue(r.data(), r.size());
            
            p = hess.ldlt().solve(residue);
            Map<MatrixXf> _p(p.data(), 3, 3);
            c.q_next -= _p;
            c.p_next -= rp / m;
            #ifdef PER_ITER_RESIDUE_PRINT
            if (iter == 0 || iter == max_iters - 1){
                cout<< "iter " << iter << ", residue = " << residue << endl;
                cout << "p = " << p << endl;
                cout << "q = " << c.q_next << endl;
            }
            #endif
        }
    }
    for (auto & c: cubes){
        c.q_dot = (c.q_next - c.A) * (1.0 / dt);
        c.A = c.q_next;

        c.p_dot = (c.p_next - c.p) * (1.0 / dt);
        c.p = c.p_next;
        #ifdef PER_TIMESTEP_PRINT
        cout << "end of time step, A = " << c.A << endl << "q. = " << c.q_dot << endl;
        cout << "energy = " << othogonal_energy :: otho_energy(c.A) << endl;
        #endif
    }
}