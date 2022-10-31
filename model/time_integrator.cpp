#include "time_integrator.h"
#include "barrier.h"
#include <iostream>
#include <assert.h>
#include "collision.h"
#include "spatial_hashing.h"
using namespace std;
using namespace barrier;
using namespace Eigen;

static const int max_iters = 10;

// #define _DEBUG_BARRIER_
// #define PER_ITER_RESIDUE_PRINT
VectorXd q_residue_barrier_term(Cube &c)
{
    VectorXd barrier_term;
    barrier_term.setZero(12, 1);

    for (int i = 0; i < 8; i++)
    {
        const vec3 &v0(c.vertices()[i]);
        const vec3 &v(c.q_next * v0 + c.p_next);
        double d = vf_distance(v);

        if (d < d_hat)
        {
            assert(d > 0);
            barrier_term += barrier_gradient_q(v0, v);
        }
    }
    // barrier_term.segment(0, 3).setZero();
    return barrier_term;
}

mat3 q_residue(Cube &c, double dt)
{
    double m = 1.0 / 12.0 * c.mass * c.scale * c.scale / (dt * dt); // to be corrected
    mat3 r = m * c.q_next + othogonal_energy::grad(c.q_next) - (c.A + dt * c.q_dot) * m;
    return r;
}

vec3 p_residue(Cube &c, double dt)
{
    // static const vec3 gravity(0.0f, 0.0f, 0.0f);
    static const vec3 gravity(0.0f, -9.8e4, 0.0f);
    double m = c.mass / (dt * dt);
    vec3 r = m * c.p_next - gravity * c.mass - (c.p + dt * c.p_dot) * m;
    return r;
}

VectorXd cat(mat3 &rq, const vec3 &rp)
{
    Vector<double, 12> ret;
    Map<VectorXd> _rq(rq.data(), rq.size());
    ret << rp, _rq;
    return ret;
}
void implicit_euler(vector<Cube> &cubes)
{
    static int ts = 0;
    // x[t+1] = x[t] + v[t+1] dt
    static const double dt = 1e-4;
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
            double m = c.mass / (dt * dt);
            double Im = m / 12.0 * c.scale * c.scale;
            mat3 rq(q_residue(c, dt));
            vec3 rp(p_residue(c, dt));
            VectorXd r(cat(rq, rp));

            r += q_residue_barrier_term(c);

            // build hessian
            MatrixXd hess = MatrixXd::Identity(12, 12) * Im;
            hess.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3) * m;

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

            VectorXd _dq(hess.ldlt().solve(r));

            // Map<MatrixXd> dq(_dq.block<9,9>(3,3).data(), 3, 3);
            VectorXd __dq(_dq.segment<9>(3));
            Map<MatrixXd> dq(__dq.data(), 3, 3);
            vec3 dp(_dq.segment(0, 3));
            c.dp = dp;
            c.dq = dq;
        }
        int group = 0;
        for (auto &c: cubes){
            for (int i = 0; i < 12; i++) {
                int a = Cube::indices[i * 3 + 0], 
                    b = Cube::indices[i * 3 + 1], 
                    _c = Cube::indices[i * 3 + 2]; 

                auto v = c.vertices();
                const vec3 &v0a(v[a]),
                    &v0b(v[b]),
                    &v0c(v[_c]);
                
                vec3 va(c.q_next * v0a + c.p_next),
                    vb(c.q_next * v0b + c.p_next),
                    vc(c.q_next * v0c + c.p_next);

                vec3 va1((c.q_next - c.dq) * v0a + (c.p_next - c.dp)),
                    vb1((c.q_next - c.dq) * v0b + (c.p_next - c.dp)),
                    vc1((c.q_next - c.dq) * v0c + (c.p_next - c.dp));
                

                Primitive t(ts, i + group * 12, group);
                spatial_hashing::register_triangle_orbit(va, vb, vc, va1, vb1, vc1, t, ts);
                // register_triangle(va, vb, vc, t, ts);
                
            }
            group ++;
        }
        double toi = 1.0;
        group = 0;
        for (auto &c: cubes){
        
            // boundary collision detector
            double body_toi = c.vf_collision_detect(c.dp, c.dq);

            for(int i = 0; i < 8; i ++) {
                const vec3 &v0(c.vertices()[i]);
                vec3 v_start(c.q_next * v0 + c.p_next);
                vec3 v_end((c.q_next - c.dq) * v0 + c.p_next - c.dp);
                auto triangle_list(spatial_hashing::query_edge(v_start, v_end, group, ts));
                
                for (auto tri: triangle_list) {
                    int g = tri.group;
                    int id = tri.id - g * 12;

                    double tri_toi = vf_collision_detect(v_start, v_end, cubes[g], id);
                    if(tri_toi < body_toi) {
                        body_toi = tri_toi;
                    }
                }                
            }
            if (body_toi < toi) {
                toi = body_toi;
                cout << "overall toi changed" << group << toi << endl;
            }
            group ++;
        }
        
        if (toi < 1.0) {
            cout << "collision detected, resetting i, toi = " << toi << endl;
            iter = 0; 
        }
        double factor = toi < 1.0 ? 0.8 : 1.0; 
        for (auto &c: cubes){
            
            c.q_next -= c.dq * toi * factor;
            c.p_next -= c.dp * toi * factor;
#ifdef PER_ITER_RESIDUE_PRINT
            if (iter == 0 || iter == max_iters - 1)
            {
                //cout << "iter " << iter << ", residue = " << r << endl;
                cout << "dp = " << c.dp << endl;
                cout << "dq = " << c.dq << endl;
            }
#endif
        }
        ts ++;
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