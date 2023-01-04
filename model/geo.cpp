#include "cube.h"
#include "geometry.h"
#include "barrier.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include "../view/global_variables.h"
#include <array>
using namespace std;

MatrixXd PSD_projection(const MatrixXd& A12x12)
{

    SelfAdjointEigenSolver<MatrixXd> eig(A12x12);
    VectorXd lam = eig.eigenvalues();
    MatrixXd U = eig.eigenvectors();
    for (int i = 0; i < 12; i++) {
        lam(i) = max(lam(i), 0.0);
    }
    return U * lam.asDiagonal() * U.adjoint();
}
Matrix<double, 12, 12> project_to_psd(
    const Eigen::Matrix<double, 12, 12>& A)
{
    // https://math.stackexchange.com/q/2776803
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, 12, 12>>
        eigensolver(A);
    // Check if all eigen values are zero or positive.
    // The eigenvalues are sorted in increasing order.
    if (eigensolver.eigenvalues()[0] >= 0.0) {
        return A;
    }
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(eigensolver.eigenvalues());
    // Save a little time and only project the negative values
    for (int i = 0; i < A.rows(); i++) {
        if (D.diagonal()[i] < 0.0) {
            D.diagonal()[i] = 0.0;
        }
        else {
            break;
        }
    }
    return eigensolver.eigenvectors() * D
        * eigensolver.eigenvectors().transpose();
}
void ipc_term(
    array<vec3, 4> pt, array<int, 4> ij,
    vector<HessBlock> &triplets,
    Vector<double, 12>& grad_p, Vector<double, 12>& grad_t
    // Matrix<double, 12, 12>& hess_p, Matrix<double, 12, 12>& hess_t, 
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

    double dist = 0.0;

    dist = ipc::point_triangle_distance(p, t0, t1, t2);
    double B_ = barrier::barrier_derivative_d(dist);
    double B__ = barrier::barrier_second_derivative(dist);
    //spdlog::info("dist = {}, B = {}, B__ = {}", dist, B_, B__);

    Matrix<double, 9, 12> Jt;
    Matrix<double, 3, 12> Jp;
    Matrix<double, 12, 12> off_diag;
    auto p_tile = vnp[v], t0_tile = vnp[tidx[3 * f]], t1_tile = vnp[tidx[3 * f + 1]], t2_tile = vnp[tidx[3 * f + 2]];
    Jt.setZero(9, 12);
    Jp.setZero(3, 12);
    off_diag.setZero(12, 12);
    Jt.block<3, 12>(0, 0) = barrier::x_jacobian_q(t0_tile);
    Jt.block<3, 12>(3, 0) = barrier::x_jacobian_q(t1_tile);
    Jt.block<3, 12>(6, 0) = barrier::x_jacobian_q(t2_tile);
    Jp = barrier::x_jacobian_q(p_tile);

    Matrix<double, 12, 12> ipc_hess;
    ipc_hess.setZero(12, 12);
    // ipc_hess = PSD_projection(pt_hess  * B_) + pt_grad * pt_grad.adjoint() * B__;
    ipc_hess = project_to_psd(pt_hess * B_) + pt_grad * pt_grad.adjoint() * B__;
    // psd project
    // ipc_hess = PSD_projection(ipc_hess);

    int ii = _i, jj = _j;
    auto hess_p = Jp.adjoint() * ipc_hess.block<3, 3>(0, 0) * Jp;
    auto hess_t = Jt.adjoint() * ipc_hess.block<9, 9>(3, 3) * Jt;

    off_diag += Jp.adjoint() * ipc_hess.block<3, 9>(0, 3) * Jt;
    auto off_T = off_diag.adjoint();
    for (int i = 0; i < 12; i++ ){
        triplets.push_back(HessBlock(ii *12, jj * 12 + i, off_diag.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, ii * 12 + i, off_T.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(ii * 12, ii * 12 + i, hess_p.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, jj * 12 + i, hess_t.block<12, 1>(0, i)));
    }
    // globals.hess_triplets.push_back({ii * 12, jj * 12, off_diag});
    // globals.hess_triplets.push_back({jj * 12, ii * 12, off_T});
    
    grad_p += Jp.adjoint() * pt_grad.segment<3>(0) * B_;
    grad_t += Jt.adjoint() * pt_grad.segment<9>(3) * B_;
}