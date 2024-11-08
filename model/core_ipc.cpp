#include "ipc.h"
#include "affine_body.h"
#include "psd.h"
#include "barrier.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <tuple>
#include "ipc_extension.h"
#ifndef TESTING
#include "settings.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif

using namespace std;
using namespace Eigen;

tuple<mat12, vec12> IPC::ipc_hess_pt_12x12(
    array<vec3, 4> pt, array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, scalar dist)
{
    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];

    auto p = pt[0], t0 = pt[1], t1 = pt[2], t2 = pt[3];

    Vector<scalar, 12> pt_grad;
    Matrix<scalar, 12, 12> pt_hess;
    pt_grad = ipc::point_triangle_distance_gradient(p, t0, t1, t2, pt_type);
    pt_hess = ipc::point_triangle_distance_hessian(p, t0, t1, t2, pt_type);

    scalar B_ = barrier::barrier_derivative_d(dist);
    scalar B__ = barrier::barrier_second_derivative(dist);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    ipc_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;

    pt_grad *= B_;

    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);

    return { ipc_hess, pt_grad };
}
tuple<mat12, vec12, scalar> IPC::ipc_hess_ee_12x12(
    array<vec3, 4> ee, array<int, 4> ij,
    ipc::EdgeEdgeDistanceType ee_type, scalar dist)
{
    int _i = ij[0], _ei = ij[1], _j = ij[2], _ej = ij[3];


    auto ei0 = ee[0], ei1 = ee[1],
         ej0 = ee[2], ej1 = ee[3];
    vec12 ee_grad, p_grad;
    mat12 ee_hess, p_hess;
    scalar eps_x = globals.eps_x * (ei0 - ei1).squaredNorm() * (ej0 - ej1).squaredNorm();

    // 1e-3 * edge length square

    scalar p = ipc::edge_edge_mollifier(ei0, ei1, ej0, ej1, eps_x);
    p_grad = ipc::edge_edge_mollifier_gradient(ei0, ei1, ej0, ej1, eps_x);
    p_hess = ipc::edge_edge_mollifier_hessian(ei0, ei1, ej0, ej1, eps_x);

    ee_grad = ipc::edge_edge_distance_gradient(ei0, ei1, ej0, ej1, ee_type);
    ee_hess = ipc::edge_edge_distance_hessian(ei0, ei1, ej0, ej1, ee_type);

    scalar B_ = barrier::barrier_derivative_d(dist);
    scalar B__ = barrier::barrier_second_derivative(dist);
    scalar B = barrier::barrier_function(dist);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    ipc_hess = p_hess * B + B_ * (p_grad * ee_grad.transpose() + ee_grad * p_grad.transpose()) + p * (B__ * ee_grad * ee_grad.transpose() + B_ * ee_hess);

    ee_grad = p * ee_grad * B_ + p_grad * B;

    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);

    return { ipc_hess, ee_grad, p };
}
