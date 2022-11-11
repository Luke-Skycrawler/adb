#include "time_integrator.h"
#include "barrier.h"
#include <iostream>
#include <assert.h>
#include "collision.h"
#include "spatial_hashing.h"
#include "spdlog/spdlog.h"
using namespace std;
using namespace barrier;
using namespace Eigen;

static const int max_iters = 10;

// #define _DEBUG_BARRIER_
// #define PER_ITER_RESIDUE_PRINT
// #define DEBUG_COLLISION
#define _DEBUG_TWO_BLOCKS
#define _VF_CULLING_HACK
#define _EE_CULLING_HACK
VectorXd q_residue_barrier_term(Cube& c)
{
    VectorXd barrier_term;
    barrier_term.setZero(12, 1);

    for (int i = 0; i < 8; i++) {
        const vec3& v0(c.vertices()[i]);
        const vec3& v(c.q_next * v0 + c.p_next);
        double d = vf_distance(v);

        if (d < d_hat) {
            assert(d > 0);
            barrier_term += barrier_gradient_q(v0, v);
        }
    }
    // barrier_term.segment(0, 3).setZero();
    return barrier_term;
}

mat3 q_residue(Cube& c, double dt)
{
    double m = 1.0 / 12.0 * c.mass * c.scale * c.scale / (dt * dt); // to be corrected
    mat3 r = m * c.q_next + othogonal_energy::grad(c.q_next) - (c.A + dt * c.q_dot) * m;
    return r;
}

vec3 p_residue(Cube& c, double dt)
{
    static const vec3 gravity(0.0f, 0.0f, 0.0f);
    // static const vec3 gravity(0.0f, -9.8e3, 0.0f);
    double m = c.mass / (dt * dt);
    vec3 r = m * c.p_next - gravity * c.mass - (c.p + dt * c.p_dot) * m;
    return r;
}

VectorXd cat(mat3& rq, const vec3& rp)
{
    Vector<double, 12> ret;
    Map<VectorXd> _rq(rq.data(), rq.size());
    ret << rp, _rq;
    return ret;
}
void implicit_euler(vector<Cube>& cubes)
{
    static int ts = 0;
    static int cv = 0, cf = 0, ce0 = 0, ce1 = 0;
    // x[t+1] = x[t] + v[t+1] dt
    static const double dt = 1e-4;
    for (auto& c : cubes) {
        c.q_next = c.A;
        c.p_next = c.p;
    }
    for (int iter = 0; iter < max_iters; iter++) {
        // newton iterations
        for (int _i = 0; _i < cubes.size(); _i++) {
            auto& c(cubes[_i]);
            double m = c.mass / (dt * dt);
            double Im = m / 12.0 * c.scale * c.scale;
            mat3 rq(q_residue(c, dt));
            vec3 rp(p_residue(c, dt));
            VectorXd r(cat(rq, rp));
            for (auto& c : cubes) {
                c.hess.setZero(12, 12);
                c.barrier_gradient.setZero(12);
            }
            // build hessian
            MatrixXd hess = MatrixXd::Identity(12, 12) * Im;
            hess.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3) * m;

            for (int i = 1; i < 4; i++) {
                for (int j = i; j < 4; j++) {
                    mat3 block = othogonal_energy::hessian(c.q_next, i - 1, j - 1);
                    if (j != i) {
                        hess.block<3, 3>(j * 3, i * 3) = block;
                        hess.block<3, 3>(i * 3, j * 3) = block;
                    }
                    else {
                        hess.block<3, 3>(i * 3, i * 3) += block;
                        // elements already deployed at the diagonal
                    }
                }
            }

#ifndef DEBUG_COLLISION
            r += q_residue_barrier_term(c);
            c.hess.setZero(12, 12);
            c.barrier_gradient = r;
            // for (int j = 0; j < cubes.size(); j ++) {
            int j = 1 - _i;
            /*if (j == _i) continue;*/
            auto& cj(cubes[j]);
            cf = vf_colliding_response(cj, c);
            cv = vf_colliding_response(c, cj);

            ce0 = ee_colliding_response(c, cj);
            ce1 = ee_colliding_response(cj, c);

            if (cf || cv || ce0 || ce1) {
                if (ts % max_iters == 0)
                    spdlog::warn("updates velocity");
                spdlog::info("collision response at {}", ts);
                // cout << endl << ts << endl;
            }

            r = c.barrier_gradient;

            hess += c.hess;
            // hessian for barrier term
            for (int i = 0; i < 8; i++) {
                const vec3& v0(c.vertices()[i]);
                const vec3& v(c.q_next * v0 + c.p_next);
                hess += barrier_hessian_q(v0, v);
            }
#else
            vec3 t0(-5.0, -0.5, 5.0),
                t1(5.0, -0.5, 5.0),
                t2(0.0, -0.5, -5.0);

            Face f(t0, t1, t2);
            vector<Face> faces;
            faces.push_back(f);
            for (int i = 0; i < Cube::n_faces; i++) {
                faces.push_back(Face(cubes[1 - _i], i));
            }
            for (Face& f : faces) {
                for (int _v = 0; _v < 8; _v++) {
                    vec3 v(c.p_next + c.q_next * c.vertices()[_v]);
                    double d = vf_distance(v, f);
                    if (d < d_hat) {
                        // colliding_set.push_back(make_pair(v, Face(cj, f)));
                        // TODO: directly change it to collision response
                        if (d < 1e-4) {
                            // cout << "restarting " << endl;
                        }

                        VectorXd ci_barrier_term, cj_barrier_term;
                        ci_barrier_term.setZero(12);
                        cj_barrier_term.setZero(12);
                        double gx[12], Hx[144];
                        autogen::point_plane_distance_gradient(
                            v(0),
                            v(1),
                            v(2),
                            f.t0(0),
                            f.t0(1),
                            f.t0(2),
                            f.t1(0),
                            f.t1(1),
                            f.t1(2),
                            f.t2(0),
                            f.t2(1),
                            f.t2(2),
                            gx);
                        autogen::point_plane_distance_hessian(
                            v(0),
                            v(1),
                            v(2),
                            f.t0(0),
                            f.t0(1),
                            f.t0(2),
                            f.t1(0),
                            f.t1(1),
                            f.t1(2),
                            f.t2(0),
                            f.t2(1),
                            f.t2(2),
                            Hx);

                        for (int i = 0; i < 12; i++) {
                            gx[i] /= (2 * d);
                        }
                        for (int i = 0; i < 144; i++) {
                            Hx[i] /= (4 * d * d);
                        }
                        double dbdd = barrier_derivative_d(d);
                        double db2 = barrier_second_derivative(d);

                        // int _t0 = Cube::indices[3 * _f];
                        // int _t1 = Cube::indices[3 * _f + 1];
                        // int _t2 = Cube::indices[3 * _f + 2];

                        auto tile_v = c.vertices()[_v];
                        // auto tile_t0 = cj.vertices()[_t0];
                        // auto tile_t1 = cj.vertices()[_t1];
                        // auto tile_t2 = cj.vertices()[_t2];

                        auto Jv = x_jacobian_q(tile_v);
                        // auto J_t0 = x_jacobian_q(tile_t0);
                        // auto J_t1 = x_jacobian_q(tile_t1);
                        // auto J_t2 = x_jacobian_q(tile_t2);

                        auto gx_v = vec3(gx[0], gx[1], gx[2]);
                        // auto gx_t0 = vec3(gx[3], gx[4], gx[5]);
                        // auto gx_t1 = vec3(gx[6], gx[7], gx[8]);
                        // auto gx_t2 = vec3(gx[9], gx[10], gx[11]);
                        ci_barrier_term += dbdd * gx_v.adjoint() * Jv;
                        // cj_barrier_term += dbdd * (gx_t0.adjoint() * J_t0 + gx_t1.adjoint() * J_t1 + gx_t2.adjoint() * J_t2);

                        // TODO: construct hessian

                        Map<MatrixXd> _Hx(Hx, 12, 12);
                        Matrix<double, 12, 12> hess_ij, hess_i, hess_j;
                        hess_i.setZero(12, 12);
                        hess_j.setZero(12, 12);
                        hess_ij.setZero(12, 12);
                        hess_i += Jv.adjoint() * (db2 * gx_v * gx_v.adjoint() + dbdd * _Hx.block<3, 3>(0, 0)) * Jv;
                        // the hessian term should be zero but add it up anyway.

                        // for (int i = 0; i < 3; i++) {
                        //     const auto& J1(i == 0 ? J_t0 : i == 1 ? J_t1
                        //                                         : J_t2);

                        //     const auto& g1(i == 0 ? gx_t0 : i == 1 ? gx_t1
                        //                                         : gx_t2);
                        //     for (int j = 0; j < 3; j++) {
                        //         const auto& J2(j == 0 ? J_t0 : j == 1 ? J_t1
                        //                                             : J_t2);
                        //         const auto& g2(j == 0 ? gx_t0 : j == 1 ? gx_t1
                        //                                             : gx_t2);
                        //         hess_j += J1.adjoint() * (db2 * g1 * g2.adjoint() + dbdd * _Hx.block<3, 3>(i * 3 + 3, j * 3 + 3)) * J2;
                        //     }
                        // }

                        // hess_ij += dbdd * (J_t0.adjoint() * _Hx.block<3, 3>(0, 3) + J_t1.adjoint() * _Hx.block<3, 3>(0, 6) + J_t2.adjoint() * _Hx.block<3, 3>(0, 9)) * Jv;
                        // hess_ij += db2 * (J_t0.adjoint() * gx_t0 + J_t1.adjoint() * gx_t1 + J_t2.adjoint() * gx_t2) * gx_v.adjoint() * Jv;

                        r += ci_barrier_term;
                        // FIXME: transpose and symetry issues
                        hess += hess_i;
                        // cj.hess += hess_j;

                        // spdlog::info("grad_i {}", ci_barrier_term);
                        // spdlog::info("grad_j {}", cj_barrier_term);
                        cout << "d " << d << " " << ts << endl;
                        // cout << "grad_i " << ci_barrier_term.adjoint() << endl;
                        // cout << "grad_j " << cj_barrier_term.adjoint() << endl;
                        // cout << "r0 " << ci.barrier_gradient.adjoint() << endl;

                        // ci.barrier_gradient += ci_barrier_term;
                        // cj.barrier_gradient += cj_barrier_term;
                        // cout << "r " << ci.barrier_gradient.adjoint() << endl;
                        // cout << endl;
                    }
                }
            }
#endif
            VectorXd _dq(hess.ldlt().solve(r));

            // Map<MatrixXd> dq(_dq.block<9,9>(3,3).data(), 3, 3);
            VectorXd __dq(_dq.segment<9>(3));
            Map<MatrixXd> dq(__dq.data(), 3, 3);
            vec3 dp(_dq.segment(0, 3));
            c.dp = dp;
            c.dq = dq;
        }
#ifdef _DEBUG_TWO_BLOCKS
        double d_min = d_hat, d_t0_min = d_hat, d_f = d_hat, d_f_t0 = d_hat;
        for (int i = 0; i < 2; i++) {
            auto& ci = cubes[i];
            auto& cj = cubes[1 - i];
            auto p = ci.p_next - ci.dp;
            auto q = ci.q_next - ci.dq;

            for (int _v = 0; _v < Cube::n_vertices; _v++) {
                auto v_t1(ci.vi(_v, true)), v_t0(ci.vi(_v));
                for (int _f = 0; _f < Cube::n_faces; _f++) {
                    Face ft1(cj, _f, true), ft0(cj, _f);
                    double d = vf_distance(v_t1, ft1);
                    double d_t0 = vf_distance(v_t0, ft0);
                    if (d < d_min) {
                        d_min = d;
                    }
                    if (d_t0_min > d_t0) d_t0_min = d_t0;
                }
                d_f = min(d_f, vf_distance(v_t1));
                d_f_t0 = min(d_f_t0, vf_distance(v_t0));
            }
        }

        if (cv || cf) {
            spdlog::info("after increment min d = {}, distance at t0 is {}", d_min, d_t0_min);
            spdlog::info("after increment d_floor = {}, distance at t0 is {}", d_f, d_f_t0);
            double normp0 = cubes[0].dp.norm(), normp1 = cubes[1].dp.norm();
            double normq1 = cubes[1].dq.norm(), normq0 = cubes[0].dq.norm();
            spdlog::info("cube 0 norm of dp, dq = {}, {}", normp0, normq0);
            spdlog::info("cube 1 norm of dp, dq = {}, {}", normp1, normq1);
        }
#endif

        double toi = 1.0;
        for (int _i = 0; _i < cubes.size(); _i++) {
            auto& c(cubes[_i]);

            // boundary collision detector
            double body_toi = c.vf_collision_detect(c.dp, c.dq);

            int _j = 1 - _i;
            auto& cj(cubes[_j]);

            for (int ei = 0; ei < Cube::n_edges; ei++) {
                for (int ej = 0; ej < Cube::n_edges; ej++) {
                    Edge _ei(c, ei), _ej(cj, ej);
#ifdef _EE_CULLING_HACK
                    vec3 l0, l1;
                    vec3 u0, u1;

                    l0 = _ei.e0.cwiseMin(_ei.e1);
                    u0 = _ei.e0.cwiseMax(_ei.e1);
                    l1 = _ej.e0.cwiseMin(_ej.e1);
                    u1 = _ej.e0.cwiseMax(_ej.e1);

                    bool b = ((u1.array() >= l0.array()).array() && (u0.array() >= l1.array()).array()).all();
                    if (!b) {
                        continue;
                        cout << 0;
                    }
                    else
                        cout << 1;

#endif
                    double edge_toi = ee_collision_detect(c, cj, ei, ej);
                    body_toi = min(body_toi, edge_toi);
                }
            }
            for (int i = 0; i < Cube::n_vertices; i++) {
                vec3 v_t0(c.vi(i)), v_t1(c.vi(i, true));
                for (int _f = 0; _f < Cube::n_faces; _f++) {
                    Face f(cj, _f);
                    Face f1(cj, _f, true);
                    double tri_toi = 1.0;
#ifdef _VF_CULLING_HACK
                    vec3 fl, fu;
                    fl = f.t0.cwiseMin(f.t1).cwiseMin(f.t2);
                    fl = fl.cwiseMin(f1.t0).cwiseMin(f1.t1).cwiseMin(f1.t2);

                    fu = f.t0.cwiseMax(f.t1).cwiseMax(f.t2);
                    fu = fl.cwiseMax(f1.t0).cwiseMax(f1.t1).cwiseMax(f1.t2);

                    bool term0 = (v_t0.array() <= fu.array()).all() && (v_t0.array() >= fl.array()).all();
                    bool term1 = (v_t1.array() <= fu.array()).all() && (v_t1.array() >= fl.array()).all();

                    if (!(term0 && term1)) {
                        // cout << "0 ";
                        continue;
                    }
                    else
                    // cout << "1 ";
#endif
                        tri_toi = vf_collision_detect(v_t0, v_t1, cj, _f);
                    body_toi = min(body_toi, tri_toi);
                }
                // auto triangle_list(spatial_hashing::query_edge(v_start, v_end, group, ts));

                // for (auto tri: triangle_list) {
                //     int g = tri.group;
                //     int id = tri.id - g * 12;

                //     double tri_toi = vf_collision_detect(v_start, v_end, cubes[g], id);
                //     if(tri_toi < body_toi) {
                //         body_toi = tri_toi;
                //     }
                // }
            }
            if (body_toi < toi) {
                toi = body_toi;
                // cout << "overall toi changed" << group << toi << endl;
            }
        }

        if (toi < 1.0) {
            spdlog::info("collision detected at ", ts);
            // iter = 0;
        }
        double factor = toi < 1.0 ? 0.9 : 1.0;
        for (auto& c : cubes) {

            c.q_next -= c.dq * toi * factor;
            c.p_next -= c.dp * toi * factor;
#ifdef PER_ITER_RESIDUE_PRINT
            if (iter == 0 || iter == max_iters - 1) {
                // cout << "iter " << iter << ", residue = " << r << endl;
                cout << "dp = " << c.dp << endl;
                cout << "dq = " << c.dq << endl;
            }
#endif
        }
        // spatial_hashing::remove_all_entries();
        ts++;
    }
    for (auto& c : cubes) {
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