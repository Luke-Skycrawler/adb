#include "cube.h"
#include "geometry.h"
#include "barrier.h"
#include <ipc/distance/point_triangle.hpp>
#include "../view/global_variables.h"
#include <array>
using namespace std;
void ipc_term(Matrix<double, 12, 12>& hess_p, Matrix<double, 12, 12>& hess_t, Vector<double, 12>& grad_p, Vector<double, 12>& grad_t, array<vec3, 4> pt, array<int, 4> ij
    // , vector<Cube> &cubes
)
{
    static auto vnp = Cube::vertices();
    static auto tidx = Cube::indices;

    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];

    auto p = pt[0], t0 = pt[1], t1 = pt[2], t2 = pt[3];
    Vector<double, 12> pt_grad;
    Matrix<double, 12, 12> pt_hess;
    ipc::point_triangle_distance_gradient(p, t0, t1, t2, pt_grad);
    ipc::point_triangle_distance_hessian(p, t0, t1, t2, pt_hess);

    double d = ipc::point_triangle_distance(p, t0, t1, t2);
    double B_ = barrier::barrier_derivative_d(d) * barrier::kappa;
    double B__ = barrier::barrier_second_derivative(d) * barrier::kappa;

    Matrix<double, 9, 12> Jt;
    Matrix<double, 3, 12> Jp;
    Matrix<double, 12, 12> off_diag;
    auto p_tile = vnp[v], t0_tile = vnp[tidx[3 * f]], t1_tile = vnp[tidx[3 * f + 1]], t2_tile = vnp[tidx[3 * f + 2]];
    Jt.block<3, 12>(0, 0) = barrier::x_jacobian_q(t0_tile);
    Jt.block<3, 12>(3, 0) = barrier::x_jacobian_q(t1_tile);
    Jt.block<3, 12>(9, 0) = barrier::x_jacobian_q(t2_tile);
    Jp = barrier::x_jacobian_q(p_tile);

    Matrix<double, 12, 12> ipc_hess = B_ * pt_hess + B__ * pt_grad * pt_grad.adjoint();
    // psd project

    int ii = _i, jj = _j;

    // H.block<12, 12> (ii * 12, ii * 12) += Jp.adjoint() * ipc_hess.block<3, 3>(0 ,0) * Jp;
    // H.block<12, 12> (jj * 12, jj * 12) += Jt.adjoint() * ipc_hess.block<9, 9>(3, 3) * Jt;
    // H.block<12, 12> (ii * 12, jj * 12) += Jp.adjoint() * ipc_hess.block<3, 9>(3, 0) * Jt;
    // H.block<12, 12> (jj * 12, ii * 12) += (Jp.adjoint() * ipc_hess.block<3, 9>(3, 0) * Jt).adjoint();

    hess_p += Jp.adjoint() * ipc_hess.block<3, 3>(0, 0) * Jp;
    hess_t += Jt.adjoint() * ipc_hess.block<9, 9>(3, 3) * Jt;
    off_diag = Jp.adjoint() * ipc_hess.block<3, 9>(3, 0) * Jt;
    globals.hess_triplets.push_back(HessBlock(ii, jj, off_diag));
    globals.hess_triplets.push_back(HessBlock(ii, jj, off_diag.adjoint()));
    // H.block<12, 12>(jj * 12, ii * 12) += (Jp.adjoint() * ipc_hess.block<3, 9>(3, 0) * Jt).adjoint();

    // g.segment<12>(ii * 12) += Jp.adjoint() * pt_grad.segment<3>(0) * B_;
    // g.segment<12>(jj * 12) += Jt.adjoint() * pt_grad.segment<9>(3) * B_;
    grad_p += Jp.adjoint() * pt_grad.segment<3>(0) * B_;
    grad_t += Jt.adjoint() * pt_grad.segment<9>(3) * B_;
}