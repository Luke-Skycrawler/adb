#include "ipc.h"
#include "barrier.h"
#include "affine_body.h"
#ifndef TESTING
#include "settings.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif
#include "collision.h"
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include "time_integrator.h"
#include "ipc_extension.h"
using namespace utils;
using namespace Eigen;

scalar ee_uktk(
    AffineBody& ci, AffineBody& cj,
    q4& ee, i4& ij, const ::ipc::EdgeEdgeDistanceType& ee_type,
    Matrix<scalar, 2, 12>& Tk_T_ret, Vector<scalar, 2>& uk_ret, scalar d, scalar mollifier)
{
    auto v_stack = ee_vstack(ci, cj, ij[1], ij[3]);
    auto ei0 = ee[0], ei1 = ee[1], ej0 = ee[2], ej1 = ee[3];
    auto rei = ei0 - ei1, rej = ej0 - ej1;
    auto cnorm = rei.cross(rej).squaredNorm();
    auto sin2 = cnorm / rei.squaredNorm() / rej.squaredNorm();
    Matrix<scalar, 3, 2> degeneracy;
    degeneracy.col(0) = rei.normalized();
    degeneracy.col(1) = (ej0 - ei0).cross(rei).normalized();
    // bool par = sin2 < 1e-8;
    // bool par = ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB && mollifier != 1.0;
    bool par = false;

    auto lams = ::ipc::edge_edge_closest_point(ei0, ei1, ej0, ej1);
    auto Pk = par ? degeneracy : ::ipc::edge_edge_tangent_basis(ei0, ei1, ej0, ej1);

    if (ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB)
        ;

    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA0_EB0) {
        Pk = ::ipc::point_point_tangent_basis(ei0, ej0);
        lams = { 0.0, 0.0 };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA0_EB1) {
        Pk = ::ipc::point_point_tangent_basis(ei0, ej1);
        lams = { 0.0, 1.0 };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA1_EB0) {
        Pk = ::ipc::point_point_tangent_basis(ei1, ej0);
        lams = { 1.0, 0.0 };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA1_EB1) {
        Pk = ::ipc::point_point_tangent_basis(ei1, ej1);
        lams = { 1.0, 1.0 };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB0) {
        auto pe = ::ipc::point_edge_closest_point(ej0, ei0, ei1);
        lams = { pe, 0.0 };
        Pk = ::ipc::point_edge_tangent_basis(ej0, ei0, ei1);
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB1) {
        auto pe = ::ipc::point_edge_closest_point(ej1, ei0, ei1);
        lams = { pe, 1.0 };
        Pk = ::ipc::point_edge_tangent_basis(ej1, ei0, ei1);
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA0_EB) {
        auto pe = ::ipc::point_edge_closest_point(ei0, ej0, ej1);
        lams = { 0.0, pe };
        Pk = ::ipc::point_edge_tangent_basis(ei0, ej0, ej1);
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA1_EB) {
        auto pe = ::ipc::point_edge_closest_point(ei1, ej0, ej1);
        lams = { 1.0, pe };
        Pk = ::ipc::point_edge_tangent_basis(ei1, ej0, ej1);
    }
    const auto clip = [&](scalar& a, scalar l, scalar u) {
        a = max(min(u, a), l);
    };
    clip(lams(0), 0.0, 1.0);
    clip(lams(1), 0.0, 1.0);
    array<scalar, 4> lambdas = { 1 - lams(0), lams(0), 1 - lams(1), lams(1) };
    // if (par) {
    //     // ignore this friction, already handled in point-triangle pair
    //     lambdas = { 0.0, 0.0, 0.0, 0.0 };
    //     // uk will be set to zero
    // }
    auto pei = ei0 * lambdas[0] + ei1 * lambdas[1];
    auto pej = ej0 * lambdas[2] + ej1 * lambdas[3];
    auto closest = (pei - pej).squaredNorm();
    assert(par || abs(d - closest) < 1e-6);
    if (!(par || abs(d - closest) < 1e-6)) {
        std::cerr << "ee error\n";
        std::cerr << "d: " << d << std::endl;
        std::cerr << "closest: " << closest << std::endl;
        std::cerr << "diff: " << abs(d - closest) << std::endl;
        // exit(1);
    }

    Matrix<scalar, 3, 12> gamma;
    gamma.setZero(3, 12);
    for (int i = 0; i < 3; i++) {
        gamma(i, i) = -lambdas[0];
        gamma(i, i + 3) = -lambdas[1];
        gamma(i, i + 6) = lambdas[2];
        gamma(i, i + 9) = lambdas[3];
    }
    Matrix<scalar, 2, 12> Tk_T = Pk.transpose() * gamma;
    Vector<scalar, 2> uk = Tk_T * v_stack;
    auto contact_force = -barrier::barrier_derivative_d(d) * 2 * sqrt(d);
    Tk_T_ret = Tk_T;
    uk_ret = uk;

    return contact_force;
}

tuple<scalar, Vector<scalar, 2>, Matrix<scalar, 2, 12>> ee_uktk(AffineBody& ci, AffineBody& cj, q4& ee, i4& ij, const ::ipc::EdgeEdgeDistanceType& ee_type, scalar d, scalar mollifier)
{
    Vector<scalar, 2> uk;
    Matrix<scalar, 2, 12> Tk;
    scalar lam = ee_uktk(ci, cj, ee, ij, ee_type, Tk, uk, d, mollifier);
    return { lam, uk, Tk };
}

void IPC::ipc_term_ee(
    q4 ee, i4 ij, ipc::EdgeEdgeDistanceType ee_type, scalar dist,
#ifdef _SM_OUT_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<scalar>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    vector<HessBlock>& triplets,
#endif
#ifdef _DIRECT_OUT_
    mat12& hess_0_ret, mat12& hess_1_ret, mat12& off_diag_ret,
#endif
    Vector<scalar, 12>& grad_0, Vector<scalar, 12>& grad_1
#ifdef _FRICTION_
    ,
    scalar& contact_lambda, Matrix<scalar, 2, 12>& Tk
#endif
)
{
    int _i = ij[0], _ei = ij[1], _j = ij[2], _ej = ij[3];
    auto &ci(*globals.cubes[_i]), &cj(*globals.cubes[_j]);

    const auto &eidxi = ci.edges, &eidxj = cj.edges;

    auto ei0 = ee[0], ei1 = ee[1],
         ej0 = ee[2], ej1 = ee[3];

    int ii = _i, jj = _j;

    auto [ipc_hess, ee_grad, p] = ipc_hess_ee_12x12(ee, ij, ee_type, dist);

    Vector<scalar, 2> _uk;
    contact_lambda = ee_uktk(ci, cj, ee, ij, ee_type, Tk, _uk, dist, p);
    // if (p != 1.0) {
    //     // contact_lambda = contact_lambda * p;
    // }

#ifdef _FRICTION_
    bool ee_parallel = ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB && p != 1.0;

    // if (globals.ee_fric && !ee_parallel)
    //     friction(_uk, contact_lambda, Tk.transpose(), ee_grad, ipc_hess);
    // if (ee_parallel)
    //     contact_lambda = 0.0;
    if (globals.ee_fric) friction(_uk, contact_lambda, Tk.transpose(), ee_grad, ipc_hess);
    
#endif
    auto ei0_tile = ci.vertices(eidxi[2 * _ei]), ei1_tile = ci.vertices(eidxi[2 * _ei + 1]),
         ej0_tile = cj.vertices(eidxj[2 * _ej]), ej1_tile = cj.vertices(eidxj[2 * _ej + 1]);

    Matrix<scalar, 6, 12> J0;
    Matrix<scalar, 6, 12> J1;

    J0.block<3, 12>(0, 0) = barrier::x_jacobian_q(ei0_tile);
    J0.block<3, 12>(3, 0) = barrier::x_jacobian_q(ei1_tile);
    J1.block<3, 12>(0, 0) = barrier::x_jacobian_q(ej0_tile);
    J1.block<3, 12>(3, 0) = barrier::x_jacobian_q(ej1_tile);

    mat12 hess_0 = J0.transpose() * ipc_hess.block<6, 6>(0, 0) * J0;
    mat12 hess_1 = J1.transpose() * ipc_hess.block<6, 6>(6, 6) * J1;
    mat12 off_diag = J0.transpose() * ipc_hess.block<6, 6>(0, 6) * J1;
    mat12 off_T = off_diag.transpose();
    vec12 d0, d1;
    d0 = J0.transpose() * ee_grad.segment<6>(0);
    d1 = J1.transpose() * ee_grad.segment<6>(6);

    output_hessian_gradient(
#ifdef _SM_OUT_
        lut, sparse_hess,
        ii, jj, ci.mass > 0.0, cj.mass > 0.0,
#endif
#ifdef _TRIPLETS_
        ii, jj,
        triplets,
#endif
#ifdef _DIRECT_OUT_
        hess_0_ret, hess_1_ret, off_diag_ret,
#endif
        grad_0, grad_1,
        d0, d1,
        hess_0, hess_1, off_diag, off_T);

}

