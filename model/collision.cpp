#include "collision.h"
#include "barrier.h"
#define TIGHT_INCLUSION_DOUBLE
#include <tight_inclusion/ccd.hpp>
#include <tight_inclusion/interval_root_finder.hpp>
#include <iostream>
using namespace barrier;
using namespace std;

double vf_collision_time(const vec3& x, const vec3& p, const mat3& q, const vec3& p_next, const mat3& q_next)
{
    // one-way collision
    // assert that (p, q) is collision-free and (p_next, q_next) is the state after penetration
    vec3 initial_guess(q * x + p);
    double t = 0.0f;
    double distance = barrier::vf_distance(initial_guess);
    vec3 dd_dx(barrier::vf_distance_gradient_x(initial_guess));
    // distance derivative of x

    vec3 x_t = (q_next - q) * x + (p_next - p);
    t -= distance / (dd_dx.dot(x_t));
    // distance function should be linear,
    // leading to an exact solution

    return t;
}

double Cube::vf_collision_detect(const vec3& dp, const mat3& dq)
{
    // TODO: ensure the updated q_next is collision-free
    double min_toi = 1.0f;
    for (int i = 0; i < 8; i++) {
        const vec3 v0(vertices()[i]);
        const vec3& v((q_next - dq) * v0 + (p_next - dp));
        double d = vf_distance(v);
        if (d < 0) {
            double t = vf_collision_time(v0, p, A, p_next - dp, q_next - dq);
            if (t < min_toi) {
                min_toi = t;
            }
        }
    }
    return min_toi;
}

double vf_collision_detect(vec3& p_t0, vec3& p_t1, const Cube& c, int id)
{
    Face f_t0(c, id, false), f_t1(c, id, true);
    ticcd::Scalar toi = 1.0, output_tolerance;
    std::vector<ticcd::Vector3> bounding_box;
    for (int i = 0; i < 8; i++) {
        bounding_box.push_back(c.vertices()[i] * 20.0);
    }
    static const ticcd::Array3 err_vf(ticcd::get_numerical_error(bounding_box, true, false));
    double min_distance = 1e-4, tmax = 1, adjusted_tolerance = 1e-6;
    long max_iterations = 10;

    bool is_impacting = ticcd::vertexFaceCCD(
        p_t0, f_t0.t0, f_t0.t1, f_t0.t2, p_t1, f_t1.t0, f_t1.t1, f_t1.t2,
        Eigen::Array3d::Constant(-1), // rounding error (auto)
        min_distance, // minimum separation distance
        toi, // time of impact
        adjusted_tolerance, // delta
        tmax, // maximum time to check
        max_iterations, // maximum number of iterations
        output_tolerance, // delta_actual
        true);
    if (toi < 1.0) {
        cout << "lowest level collision detected" << endl;
    }
    return toi;
}

// vector<pair<vec3, Face>> vf_colliding_set(const Cube &ci, const Cube &cj) {
void vf_colliding_response(Cube& ci, Cube& cj)
{
    auto p = ci.p_next;
    auto q = ci.q_next;
    for (int _v = 0; _v < Cube::n_vertices; _v++) {
        auto v(q * ci.vertices()[_v] + p);
        for (int _f = 0; _f < Cube::n_faces; _f++) {
            Face f(cj, _f);
            double d = vf_distance(v, f);
            if (d < d_hat) {
                // colliding_set.push_back(make_pair(v, Face(cj, f)));
                // TODO: directly change it to collision response

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
                double dbdd = barrier_derivative_d(d);
                double db2 = barrier_second_derivative(d);

                int _t0 = Cube::indices[3 * _f];
                int _t1 = Cube::indices[3 * _f + 1];
                int _t2 = Cube::indices[3 * _f + 2];

                auto tile_v = cj.vertices()[_v];
                auto tile_t0 = cj.vertices()[_t0];
                auto tile_t1 = cj.vertices()[_t1];
                auto tile_t2 = cj.vertices()[_t2];

                auto Jv = x_jacobian_q(tile_v);
                auto J_t0 = x_jacobian_q(tile_t0);
                auto J_t1 = x_jacobian_q(tile_t1);
                auto J_t2 = x_jacobian_q(tile_t2);

                auto gx_v = vec3(gx[0], gx[1], gx[2]);
                auto gx_t0 = vec3(gx[3], gx[4], gx[5]);
                auto gx_t1 = vec3(gx[6], gx[7], gx[8]);
                auto gx_t2 = vec3(gx[9], gx[10], gx[11]);
                ci_barrier_term += dbdd * gx_v.adjoint() * Jv;
                cj_barrier_term += dbdd * (gx_t0.adjoint() * J_t0 + gx_t1.adjoint() * J_t1 + gx_t2.adjoint() * J_t2);
                ci.barrier_gradient += ci_barrier_term;
                cj.barrier_gradient += cj_barrier_term;

                // TODO: construct hessian

                Map<MatrixXd> _Hx(Hx, 12, 12);
                Matrix<double, 12, 12> hess_ij, hess_i, hess_j;
                hess_i.setZero(12, 12);
                hess_j.setZero(12, 12);
                hess_ij.setZero(12, 12);
                hess_i += db2 * Jv.adjoint() * (db2 * gx_v * gx_v.adjoint() + dbdd * _Hx.block<3, 3>(0, 0)) * Jv;
                // the hessian term should be zero but add it up anyway.

                for (int i = 0; i < 3; i++) {
                    const auto& J1(i == 0 ? J_t0 : i == 1 ? J_t1
                                                          : J_t2);

                    const auto& g1(i == 0 ? gx_t0 : i == 1 ? gx_t1
                                                           : gx_t2);
                    for (int j = 0; j < 3; j++) {
                        const auto& J2(j == 0 ? J_t0 : j == 1 ? J_t1
                                                              : J_t2);
                        const auto& g2(j == 0 ? gx_t0 : j == 1 ? gx_t1
                                                               : gx_t2);
                        hess_j += J1.adjoint() * (db2 * g1 * g2.adjoint() + dbdd * _Hx.block<3, 3>(i * 3 + 3, j * 3 + 3)) * J2;
                    }
                }

                hess_ij += dbdd * (J_t0.adjoint() * _Hx.block<3, 3>(0, 3) + J_t1.adjoint() * _Hx.block<3, 3>(0, 6) + J_t2.adjoint() * _Hx.block<3, 3>(0, 9)) * Jv;
                hess_ij += db2 * (J_t0.adjoint() * gx_t0 + J_t1.adjoint() * gx_t1 + J_t2.adjoint() * gx_t2) * gx_v.adjoint() * Jv;
                // FIXME: transpose and symetry issues
                ci.hess += hess_i;
                cj.hess += hess_j;
            }
        }
    }
}