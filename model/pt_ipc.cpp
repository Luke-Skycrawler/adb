#include "ipc.h"
#include "barrier.h"
#include "affine_body.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/friction/smooth_friction_mollifier.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include "time_integrator.h"
#ifndef TESTING
#include "settings.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif
#include "collision.h"
#include "time_integrator.h"
#include "ipc_extension.h"
using namespace utils;
using namespace Eigen;

scalar pt_uktk(
    AffineBody& ci, AffineBody& cj,
    q4& pt, i4& ij, const ::ipc::PointTriangleDistanceType& pt_type,
    Matrix<scalar, 2, 12>& Tk_T_ret, Vector<scalar, 2>& uk_ret, scalar d)

{

    Vector<scalar, 12> v_stack = pt_vstack(ci, cj, ij[1], ij[3]);

    auto lams = ::ipc::point_triangle_closest_point(pt[0], pt[1], pt[2], pt[3]);
    array<scalar, 3> tlams = { 1 - lams(0) - lams(1), lams(0), lams(1) };
    auto Pk = ::ipc::point_triangle_tangent_basis(pt[0], pt[1], pt[2], pt[3]);

    if (pt_type == ::ipc::PointTriangleDistanceType::P_T)
        ; // do nothing
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_T0) {
        Pk = ::ipc::point_point_tangent_basis(pt[0], pt[1]);
        tlams = { 1.0, 0.0, 0.0 };
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_T1) {
        Pk = ::ipc::point_point_tangent_basis(pt[0], pt[2]);
        tlams = { 0.0, 1.0, 0.0 };
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_T2) {
        Pk = ::ipc::point_point_tangent_basis(pt[0], pt[3]);
        tlams = { 0.0, 0.0, 1.0 };
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_E0) {
        auto elam = ::ipc::point_edge_closest_point(pt[0], pt[1], pt[2]);
        tlams = { 1.0f - elam, elam, 0.0f };
        Pk = ::ipc::point_edge_tangent_basis(pt[0], pt[1], pt[2]);
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_E1) {
        auto elam = ::ipc::point_edge_closest_point(pt[0], pt[2], pt[3]);
        tlams = { 0.0f, 1.0f - elam, elam };
        Pk = ::ipc::point_edge_tangent_basis(pt[0], pt[2], pt[3]);
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_E2) {
        auto elam = ::ipc::point_edge_closest_point(pt[0], pt[3], pt[1]);
        tlams = { elam, 0.0f, 1.0f - elam };
        Pk = ::ipc::point_edge_tangent_basis(pt[0], pt[3], pt[1]);
    }

    auto tp = (pt[1] * tlams[0] + pt[2] * tlams[1] + pt[3] * tlams[2]);
    auto closest = (pt[0] - tp).squaredNorm();
    assert(abs(d - closest) < 1e-12);
    const auto to_int = [](const ::ipc::PointTriangleDistanceType& pt_type) {
        if (pt_type == ::ipc::PointTriangleDistanceType::P_T)
            return 0;
        else if (pt_type == ::ipc::PointTriangleDistanceType::P_T0)
            return 1;
        else if (pt_type == ::ipc::PointTriangleDistanceType::P_T1)
            return 2;
        else if (pt_type == ::ipc::PointTriangleDistanceType::P_T2)
            return 3;
        else if (pt_type == ::ipc::PointTriangleDistanceType::P_E0)
            return 4;
        else if (pt_type == ::ipc::PointTriangleDistanceType::P_E1)
            return 5;
        else if (pt_type == ::ipc::PointTriangleDistanceType::P_E2)
            return 6;
        else
            assert(false);
            return -1;
    };
    if (!(abs(d - closest) < 1e-6)) {
        std::cerr << "pt error\n";
        std::cerr << "d: " << d << std::endl;
        std::cerr << "closest: " << closest << std::endl;
        std::cerr << "diff: " << abs(d - closest) << std::endl;
        std::cerr << "type" << to_int(pt_type) << std::endl;
        // exit(1);
    }
    Matrix<scalar, 3, 12> gamma;
    gamma.setZero(3, 12);
    for (int i = 0; i < 3; i++) {
        gamma(i, i) = -1.0;
        for (int j = 0; j < 3; j++)
            gamma(i, i + 3 + 3 * j) = tlams[j];
    }

    Tk_T_ret = Pk.transpose() * gamma;
    uk_ret = Tk_T_ret * v_stack;
    scalar contact_force = -barrier::barrier_derivative_d(d) * 2 * sqrt(d);
    return contact_force;
}


tuple<scalar, Vector<scalar, 2>, Matrix<scalar, 2, 12>> pt_uktk(AffineBody& ci, AffineBody& cj, q4& pt, i4& ij, const ::ipc::PointTriangleDistanceType& pt_type, scalar d)
{
    Vector<scalar, 2> uk;
    Matrix<scalar, 2, 12> Tk;
    scalar lam = pt_uktk(ci, cj, pt, ij, pt_type, Tk, uk, d);
    return { lam, uk, Tk };
}

void IPC::ipc_term(
    q4 pt, i4 ij, ipc::PointTriangleDistanceType pt_type, scalar dist,
#ifdef _SM_OUT_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<scalar>& sparse_hess,
#endif
#ifdef _TRIPLETS_

    vector<HessBlock>& triplets,
#endif
#ifdef _DIRECT_OUT_
    mat12& hess_p_ret, mat12& hess_t_ret, mat12& off_diag_ret,
#endif
    Vector<scalar, 12>& grad_p, Vector<scalar, 12>& grad_t
#ifdef _FRICTION_
    ,
    scalar& contact_lambda, Matrix<scalar, 2, 12>& Tk
#endif
)
{

    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];
    int ii = _i, jj = _j;
    auto &ci{ *globals.cubes[_i] }, &cj{ *globals.cubes[_j] };
    const auto& tidx{ cj.indices };
    Vector<scalar, 2> _uk;
    contact_lambda = pt_uktk(ci, cj, pt, ij, pt_type, Tk, _uk, dist);

    auto [ipc_hess, pt_grad] = ipc_hess_pt_12x12(pt, ij, pt_type, dist);

#ifdef _FRICTION_
        if (globals.pt_fric)
        friction(_uk, contact_lambda, Tk.transpose(), pt_grad, ipc_hess);
#endif

        auto p_tile = ci.vertices(v), t0_tile = cj.vertices(tidx[3 * f]), t1_tile = cj.vertices(tidx[3 * f + 1]), t2_tile = cj.vertices(tidx[3 * f + 2]);

    Matrix<scalar, 9, 12> Jt;
    Matrix<scalar, 3, 12> Jp;
    Jt.setZero(9, 12);
    Jp.setZero(3, 12);
    Jt.block<3, 12>(0, 0) = barrier::x_jacobian_q(t0_tile);
    Jt.block<3, 12>(3, 0) = barrier::x_jacobian_q(t1_tile);
    Jt.block<3, 12>(6, 0) = barrier::x_jacobian_q(t2_tile);
    Jp = barrier::x_jacobian_q(p_tile);
    mat12 hess_p = Jp.transpose() * ipc_hess.block<3, 3>(0, 0) * Jp;
    mat12 hess_t = Jt.transpose() * ipc_hess.block<9, 9>(3, 3) * Jt;

    mat12 off_diag = Jp.transpose() * ipc_hess.block<3, 9>(0, 3) * Jt;
    mat12 off_T = off_diag.transpose();
    auto dgp = Jp.transpose() * pt_grad.segment<3>(0);
    auto dgt = Jt.transpose() * pt_grad.segment<9>(3);

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
        hess_p_ret, hess_t_ret, off_diag_ret,
#endif
        grad_p, grad_t,
        dgp, dgt,
        hess_p, hess_t, off_diag, off_T);
}

