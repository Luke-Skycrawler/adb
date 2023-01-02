#include "collision.h"
#include "barrier.h"
#define TIGHT_INCLUSION_DOUBLE
#include <tight_inclusion/ccd.hpp>
#include <tight_inclusion/interval_root_finder.hpp>
#include <iostream>
#include <spdlog/spdlog.h>
#include "../view/global_variables.h"

using namespace barrier;
using namespace std;

double vf_collision_detect(vec3& p_t0, vec3& p_t1, const Cube& c, int id)
{
    Face f_t0(c, id, false), f_t1(c, id, true);
    ticcd::Scalar toi = 1.0, output_tolerance;
    std::vector<ticcd::Vector3> bounding_box;
    for (int i = 0; i < 8; i++) {
        bounding_box.push_back(c.vertices(i) * 20.0);
    }
    static const ticcd::Array3 err_vf(ticcd::get_numerical_error(bounding_box, true, false));
    double min_distance = 1e-6, tmax = 1, adjusted_tolerance = 1e-6;
    long max_iterations = 1e6;

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
        spdlog::warn("vf collision detected at toi = {}", toi);
    }
    return toi;
}

double ee_collision_detect(const Cube& ci, const Cube& cj, int eid_i, int eid_j)
{
    ticcd::Scalar toi = 1.0, output_tolerance;
    double min_distance = 1e-6, tmax = 1, adjusted_tolerance = 1e-6;
    long max_iterations = 1e6;
    Edge ei_t1(ci, eid_i, true), ej_t1(cj, eid_j, true), ei_t0(ci, eid_i), ej_t0(cj, eid_j);

    bool is_impacting = ticcd::edgeEdgeCCD(
        ei_t0.e0, ei_t0.e1, ej_t0.e0, ej_t0.e1, ei_t1.e0, ei_t1.e1, ej_t1.e0, ej_t1.e1,
        Eigen::Array3d::Constant(-1), // rounding error (auto)
        min_distance, // minimum separation distance
        toi, // time of impact
        adjusted_tolerance, // delta
        tmax, // maximum time to check
        max_iterations, // maximum number of iterations
        output_tolerance, // delta_actual
        true);
    if (toi < 1.0) {
        spdlog::warn("ee collision detected at toi = {}", toi);
    }
    else
        toi = 1.0;
    return toi;
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

// vector<pair<vec3, Face>> vf_colliding_set(const Cube &ci, const Cube &cj) {

//int vf_colliding_response(int __i, int __j)
//{
//    int ret = 0;
//    Cube &ci(globals.cubes[__i]), &cj(globals.cubes[__j]);
//
//    for (int _v = 0; _v < Cube::n_vertices; _v++) {
//        vec3 v = ci.vt1(_v);
//        for (int _f = 0; _f < Cube::n_faces; _f++) {
//            Face f(cj, _f);
//            #ifndef IPC_TOOLKIT
//            double d = vf_distance(v, f);
//            d = d * d;
//            #else 
//            double d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2);
//            #endif
//            if (d < d_hat) {
//                #define NEW_GEO
//                #ifdef NEW_GEO
//                array<vec3, 4> pt = { v, f.t0, f.t1, f.t2 };
//                array<int, 4> ij = { __i, _v, __j, _f };
//                ipc_term(ci.hess, cj.hess, ci.grad, cj.grad, pt, ij);
//
//                #else
//                VectorXd ci_grad, cj_grad;
//                ci_grad.setZero(12);
//                cj_grad.setZero(12);
//                double gx[12], Hx[144];
//                autogen::point_plane_distance_gradient( 
//                    v(0), v(1), v(2), 
//                    f.t0(0), f.t0(1), f.t0(2), f.t1(0), f.t1(1), f.t1(2), f.t2(0), f.t2(1), f.t2(2),
//                    gx);
//                autogen::point_plane_distance_hessian(
//                    v(0), v(1), v(2), 
//                    f.t0(0), f.t0(1), f.t0(2), f.t1(0), f.t1(1), f.t1(2), f.t2(0), f.t2(1), f.t2(2),
//                    Hx);
//
//                // for (int i = 0; i < 12; i++) {
//                //     gx[i] /= (2 * d);
//                // }
//                // for (int i = 0; i < 144; i++) {
//                //     Hx[i] /= (4 * d * d);
//                // }
//                double dbdd = barrier_derivative_d(d);
//                double db2 = barrier_second_derivative(d);
//
//                int _t0 = Cube::indices[3 * _f];
//                int _t1 = Cube::indices[3 * _f + 1];
//                int _t2 = Cube::indices[3 * _f + 2];
//
//                auto tile_v = ci.vertices()[_v];
//                auto tile_t0 = cj.vertices()[_t0];
//                auto tile_t1 = cj.vertices()[_t1];
//                auto tile_t2 = cj.vertices()[_t2];
//
//                auto Jv = x_jacobian_q(tile_v);
//                auto J_t0 = x_jacobian_q(tile_t0);
//                auto J_t1 = x_jacobian_q(tile_t1);
//                auto J_t2 = x_jacobian_q(tile_t2);
//
//                auto gx_v = vec3(gx[0], gx[1], gx[2]);
//                auto gx_t0 = vec3(gx[3], gx[4], gx[5]);
//                auto gx_t1 = vec3(gx[6], gx[7], gx[8]);
//                auto gx_t2 = vec3(gx[9], gx[10], gx[11]);
//
//                ci_grad += dbdd * gx_v.adjoint() * Jv;
//                cj_grad += dbdd * (gx_t0.adjoint() * J_t0 + gx_t1.adjoint() * J_t1 + gx_t2.adjoint() * J_t2);
//
//                // TODO: construct hessian
//
//                Map<MatrixXd> _Hx(Hx, 12, 12);
//                Matrix<double, 12, 12> hess_ij, hess_i, hess_j;
//                hess_i.setZero(12, 12);
//                hess_j.setZero(12, 12);
//                hess_ij.setZero(12, 12);
//
//
//                hess_i += Jv.adjoint() * (db2 * gx_v * gx_v.adjoint() + dbdd * _Hx.block<3, 3>(0, 0)) * Jv;
//                // the hessian term should be zero but add it up anyway.
//
//                for (int i = 0; i < 3; i++) {
//                    const auto& J1(i == 0 ? J_t0 : i == 1 ? J_t1
//                                                          : J_t2);
//
//                    const auto& g1(i == 0 ? gx_t0 : i == 1 ? gx_t1
//                                                           : gx_t2);
//                    for (int j = 0; j < 3; j++) {
//                        const auto& J2(j == 0 ? J_t0 : j == 1 ? J_t1
//                                                              : J_t2);
//                        const auto& g2(j == 0 ? gx_t0 : j == 1 ? gx_t1
//                                                               : gx_t2);
//                        hess_j += J1.adjoint() * (db2 * g1 * g2.adjoint() + dbdd * _Hx.block<3, 3>(i * 3 + 3, j * 3 + 3)) * J2;
//                    }
//                }
//
//                hess_ij += dbdd * (J_t0.adjoint() * _Hx.block<3, 3>(0, 3) + J_t1.adjoint() * _Hx.block<3, 3>(0, 6) + J_t2.adjoint() * _Hx.block<3, 3>(0, 9)) * Jv;
//                hess_ij += db2 * (J_t0.adjoint() * gx_t0 + J_t1.adjoint() * gx_t1 + J_t2.adjoint() * gx_t2) * gx_v.adjoint() * Jv;
//                // FIXME: transpose and symetry issues
//
//                ci.hess += hess_i;
//                cj.hess += hess_j;
//
//                // spdlog::info("grad_i {}", ci_grad);
//                // spdlog::info("grad_j {}", cj_grad);
//                //  cout << "d " << d << endl;
//                // cout << "grad_i " << ci_grad.adjoint() << endl;
//                // cout << "grad_j " << cj_grad.adjoint() << endl;
//                // cout << "r0 " << ci.barrier_gradient.adjoint() << endl;
//
//                ci.grad += ci_grad;
//                cj.grad += cj_grad;
//
//                // cout << "r " << ci.barrier_gradient.adjoint() << endl;
//                // cout << endl;
//                globals.hess_triplets.push_back(HessBlock(__i, __j, hess_ij));
//                globals.hess_triplets.push_back(HessBlock(__j, __i, hess_ij.adjoint()));
//                ret = 1;
//                #endif 
//            }
//        }
//    }
//    return ret;
//}

//int ee_colliding_response(int __i, int __j)
//{
//    Cube &ci(globals.cubes[__i]), &cj(globals.cubes[__j]);
//    int ret = 0;
//    double gx[12], Hx[144];
//    for (int _i = 0; _i < Cube::n_edges; _i++) {
//        Edge ei(ci, _i);
//        for (int _j = 0; _j < Cube::n_edges; _j++) {
//            Edge ej(cj, _j);
//            double d = ee_distance(ei, ej);
//            if (d < d_hat) {
//                auto ea0 = ei.e0, ea1 = ei.e1, eb0 = ej.e0, eb1 = ej.e1;
//
//                VectorXd ci_barrier_term, cj_barrier_term;
//                ci_barrier_term.setZero(12);
//                cj_barrier_term.setZero(12);
//                autogen::line_line_distance_gradient(
//                    ea0(0), ea0(1), ea0(2), ea1(0), ea1(1), ea1(2), eb0(0), eb0(1), eb0(2), eb1(0), eb1(1), eb1(2),
//                    gx);
//                autogen::line_line_distance_hessian(
//                    ea0(0), ea0(1), ea0(2), ea1(0), ea1(1), ea1(2), eb0(0), eb0(1), eb0(2), eb1(0), eb1(1), eb1(2),
//                    Hx);
//                for (int i = 0; i < 12; i++) {
//                    gx[i] /= (2 * d);
//                }
//                for (int i = 0; i < 144; i++) {
//                    Hx[i] /= (4 * d * d);
//                }
//                double dbdd = barrier_derivative_d(d);
//                double db2 = barrier_second_derivative(d);
//
//                int _ea0 = Cube::edges[2 * _i];
//                int _ea1 = Cube::edges[2 * _i + 1];
//
//                int _eb0 = Cube::edges[2 * _j];
//                int _eb1 = Cube::edges[2 * _j + 1];
//
//                auto tile_ea0 = cj.vertices()[_ea0];
//                auto tile_ea1 = cj.vertices()[_ea1];
//                auto tile_eb0 = cj.vertices()[_eb0];
//                auto tile_eb1 = cj.vertices()[_eb1];
//
//                auto J_ea0 = x_jacobian_q(tile_ea0);
//                auto J_ea1 = x_jacobian_q(tile_ea1);
//                auto J_eb0 = x_jacobian_q(tile_eb0);
//                auto J_eb1 = x_jacobian_q(tile_eb1);
//
//                auto gx_ea0 = vec3(gx[0], gx[1], gx[2]);
//                auto gx_ea1 = vec3(gx[3], gx[4], gx[5]);
//                auto gx_eb0 = vec3(gx[6], gx[7], gx[8]);
//                auto gx_eb1 = vec3(gx[9], gx[10], gx[11]);
//
//                ci_barrier_term += dbdd * (gx_ea0.adjoint() * J_ea0 + gx_ea1.adjoint() * J_ea1);
//                cj_barrier_term += dbdd * (gx_eb0.adjoint() * J_eb0 + gx_eb1.adjoint() * J_eb1);
//
//                // cj_barrier_term += dbdd * (gx_t0.adjoint() * J_t0 + gx_t1.adjoint() * J_t1 + gx_t2.adjoint() * J_t2);
//
//                // TODO: construct hessian
//
//                Map<MatrixXd> _Hx(Hx, 12, 12);
//                Matrix<double, 12, 12> hess_ij, hess_i, hess_j;
//                hess_i.setZero(12, 12);
//                hess_j.setZero(12, 12);
//                hess_ij.setZero(12, 12);
//
//                for (int i = 0; i < 2; i++) {
//                    const auto& J1(i == 0 ? J_ea0 : J_ea1);
//                    const auto& g1(i == 0 ? gx_ea0 : gx_ea1);
//
//                    for (int j = 0; j < 2; j++) {
//                        const auto& J2(j == 0 ? J_ea0 : J_ea1);
//                        const auto& g2(j == 0 ? gx_ea0 : gx_ea1);
//                        hess_i += J1.adjoint() * (db2 * g1 * g2.adjoint() + dbdd * _Hx.block<3, 3>(i * 3, j * 3)) * J2;
//                    }
//                }
//
//                for (int i = 2; i < 4; i++) {
//                    const auto& J1(i == 2 ? J_eb0 : J_eb1);
//                    const auto& g1(i == 2 ? gx_eb0 : gx_eb1);
//
//                    for (int j = 2; j < 4; j++) {
//                        const auto& J2(j == 2 ? J_eb0 : J_eb1);
//                        const auto& g2(j == 2 ? gx_eb0 : gx_eb1);
//                        hess_j += J1.adjoint() * (db2 * g1 * g2.adjoint() + dbdd * _Hx.block<3, 3>(i * 3, j * 3)) * J2;
//                    }
//                }
//
//                // hess_ij += dbdd * (J_t0.adjoint() * _Hx.block<3, 3>(0, 3) + J_t1.adjoint() * _Hx.block<3, 3>(0, 6) + J_t2.adjoint() * _Hx.block<3, 3>(0, 9)) * Jv;
//                // hess_ij += db2 * (J_t0.adjoint() * gx_t0 + J_t1.adjoint() * gx_t1 + J_t2.adjoint() * gx_t2) * gx_v.adjoint() * Jv;
//                // FIXME: transpose and symetry issues
//                // FIXME: worry about hess_ij later
//
//                ci.hess += hess_i;
//                cj.hess += hess_j;
//
//                // spdlog::info("grad_i {}", ci_barrier_term);
//                // spdlog::info("grad_j {}", cj_barrier_term);
//                //  cout << "d " << d << endl;
//                // cout << "grad_i " << ci_barrier_term.adjoint() << endl;
//                // cout << "grad_j " << cj_barrier_term.adjoint() << endl;
//                // cout << "r0 " << ci.barrier_gradient.adjoint() << endl;
//
//                ci.barrier_gradient += ci_barrier_term;
//                cj.barrier_gradient += cj_barrier_term;
//
//                // cout << "r " << ci.barrier_gradient.adjoint() << endl;
//                // cout << endl;
//                ret = 1;
//            }
//        }
//    }
//    return ret;
//}