#include "cube.h"
// #include "geometry.h"
#include "barrier.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/friction/smooth_friction_mollifier.hpp>
#include "../view/global_variables.h"
#include "collision.h"
#include "time_integrator.h"

using namespace std;
using mat12 = Matrix<double, 12, 12>;
void put(double* values, int offset, int _stride, Matrix<double, 12, 12>& block)
{
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
#pragma omp atomic
            values[offset + _stride * j + i] += block(i, j);
        }
}
MatrixXd PSD_projection(const MatrixXd& A12x12)
{

    SelfAdjointEigenSolver<MatrixXd> eig(A12x12);
    VectorXd lam = eig.eigenvalues();
    MatrixXd U = eig.eigenvectors();
    for (int i = 0; i < 12; i++) {
        lam(i) = max(lam(i), 0.0);
    }
    return U * lam.asDiagonal() * U.transpose();
}
Eigen::MatrixXd project_to_psd(
    const Eigen::MatrixXd& A)
{
    // https://math.stackexchange.com/q/2776803
    Eigen::SelfAdjointEigenSolver<
        Eigen::MatrixXd>
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

double D_f0(double uk, double lam)
{
    static double mu = globals.mu, evh = globals.dt * 1e-2, h2 = globals.dt * globals.dt;
    double D_k = mu * lam * ipc::f0_SF(uk, evh);
    return D_k * h2;
}

void friction(
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk,
    Vector<double, 12>& g, Matrix<double, 12, 12>& H)
{
    static double mu = globals.mu;
    static const double evh = globals.dt * 1e-2, h2 = globals.dt * globals.dt;
    auto uk = _uk.norm();
    if (uk < 1e-10) return;
    auto f1 = ipc::f1_SF_over_x(uk, evh);
    Vector<double, 12> F_k = -mu * contact_lambda * Tk * f1 * _uk;
    // double D_k = mu * contact_lambda * ipc::f0_SF(uk, evh);
    double df1_term = ipc::df1_x_minus_f1_over_x3(uk, evh);
    Matrix2d M2x2 = (df1_term * _uk * _uk.transpose());
    M2x2+= f1 * Matrix2d::Identity(2, 2);
    M2x2 = project_to_psd(M2x2);
    Matrix<double, 12, 12> D_k_hessian = mu * contact_lambda * Tk * M2x2 * Tk.transpose();

    g -= F_k * h2;
    H += D_k_hessian * h2;
}

// void friction(
//     const Vector2d& _uk, double contact_lambda, const Matrix<double, 3, 2>& Tk,
//     vec3& g, mat3& H)
// {
//     static double mu = globals.mu;

//     static const double evh = globals.dt * 1e-2, h2 = globals.dt * globals.dt;
//     auto uk = _uk.norm();
//     if (uk < 1e-10) return;
//     auto f1 = ipc::f1_SF_over_x(uk, evh);
//     Vector<double, 3> F_k = -mu * contact_lambda * Tk * f1 * _uk;
//     double D_k = mu * contact_lambda * ipc::f0_SF(uk, evh);
//     double df1_term = ipc::df1_x_minus_f1_over_x3(uk, evh);
//     Matrix2d M2x2 = (df1_term * _uk * _uk.transpose() + f1 * Matrix2d::Identity(2, 2));
//     M2x2 = project_to_psd(M2x2);
//     Matrix<double, 3, 3> D_k_hessian = mu * contact_lambda * Tk * M2x2 * Tk.transpose();

//     g += F_k * h2;
//     H += D_k_hessian * h2;
// }

void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 3, 2>& Tk
#endif

)
{
    auto v_tile{ c.vertices(v) }, p{c.vt1(v)};

    c.grad += barrier::barrier_gradient_q(v_tile, p);
    c.hess += barrier::barrier_hessian_q(v_tile, p);
#ifdef _FRICTION_

    // auto g = barrier::barrier_gradient_x;
    // auto H = barrier::barrier_second_derivative(d) * g * g.transpose();
    // friction(_uk, contact_lambda, Tk, g, H);
    // auto J = barrier::x_jacobian_q(c.vertices(v));
    // c.grad += g * J;
    // c.hess += J.transpose() * H * J;
    
    auto J = barrier::x_jacobian_q(v_tile);
    friction(_uk, contact_lambda,
        J.transpose() * Tk, c.grad, c.hess);
#endif
};

void ipc_term(
    array<vec3, 4> pt, array<int, 4> ij,
#ifdef _SM_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_

    vector<HessBlock>& triplets,
#endif
    Vector<double, 12>& grad_p, Vector<double, 12>& grad_t
// Matrix<double, 12, 12>& hess_p, Matrix<double, 12, 12>& hess_t,
#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk
#endif
)
{
    // static const vec3* vnp = Cube::_vertices();
    // static unsigned* tidx = Cube::_indices;

    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];

    auto p = pt[0], t0 = pt[1], t1 = pt[2], t2 = pt[3];
    auto &ci{ *globals.cubes[_i] }, &cj{ *globals.cubes[_j] };
    auto* tidx = cj.indices;
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
    mat12 off_diag;
    auto p_tile = ci.vertices(v), t0_tile = cj.vertices(tidx[3 * f]), t1_tile = cj.vertices(tidx[3 * f + 1]), t2_tile = ci.vertices(tidx[3 * f + 2]);
    // auto p_tile = vnp[v], t0_tile = vnp[tidx[3 * f]], t1_tile = vnp[tidx[3 * f + 1]], t2_tile = vnp[tidx[3 * f + 2]];
    Jt.setZero(9, 12);
    Jp.setZero(3, 12);
    off_diag.setZero(12, 12);
    Jt.block<3, 12>(0, 0) = barrier::x_jacobian_q(t0_tile);
    Jt.block<3, 12>(3, 0) = barrier::x_jacobian_q(t1_tile);
    Jt.block<3, 12>(6, 0) = barrier::x_jacobian_q(t2_tile);
    Jp = barrier::x_jacobian_q(p_tile);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    // ipc_hess = PSD_projection(pt_hess  * B_) + pt_grad * pt_grad.transpose() * B__;
    if (globals.psd)
        ipc_hess = project_to_psd(pt_hess * B_) + pt_grad * pt_grad.transpose() * B__;
    else
        ipc_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;
    // psd project
    // ipc_hess = PSD_projection(ipc_hess);

    int ii = _i, jj = _j;
    mat12 hess_p = Jp.transpose() * ipc_hess.block<3, 3>(0, 0) * Jp;
    mat12 hess_t = Jt.transpose() * ipc_hess.block<9, 9>(3, 3) * Jt;

    off_diag = Jp.transpose() * ipc_hess.block<3, 9>(0, 3) * Jt;
    mat12 off_T = off_diag.transpose();

    pt_grad *= B_;
#ifdef _FRICTION_
    friction(_uk, contact_lambda, Tk, pt_grad, ipc_hess);
#endif

#ifdef _SM_
    auto outers = sparse_hess.outerIndexPtr();
    auto values = sparse_hess.valuePtr();

    auto stride_j = stride(jj, outers), stride_i = stride(ii, outers);
    auto oii = starting_offset(ii, ii, lut, outers), ojj = starting_offset(jj, jj, lut, outers), oij = starting_offset(ii, jj, lut, outers), oji = starting_offset(jj, ii, lut, outers);
#endif
#ifdef _TRIPLETS_

    for (int i = 0; i < 12; i++) {
        triplets.push_back(HessBlock(ii * 12, jj * 12 + i, off_diag.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, ii * 12 + i, off_T.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(ii * 12, ii * 12 + i, hess_p.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, jj * 12 + i, hess_t.block<12, 1>(0, i)));
    }
    // globals.hess_triplets.push_back({ii * 12, jj * 12, off_diag});
    // globals.hess_triplets.push_back({jj * 12, ii * 12, off_T});

#endif
    Vector<double, 12> dgp, dgt;
    dgp = Jp.transpose() * pt_grad.segment<3>(0);
    dgt = Jt.transpose() * pt_grad.segment<9>(3);
    {
        for (int i = 0; i < 12; i++) {
#pragma omp atomic
            grad_p(i) += dgp(i);
#pragma omp atomic
            grad_t(i) += dgt(i);
        }
        put(values, oij, stride_j, off_diag);
        put(values, oji, stride_i, off_T);
        put(values, oii, stride_i, hess_p);
        put(values, ojj, stride_j, hess_t);
    }
}

void ipc_term_ee(
    array<vec3, 4> ee, array<int, 4> ij,
#ifdef _SM_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    vector<HessBlock>& triplets,
#endif
    Vector<double, 12>& grad_0, Vector<double, 12>& grad_1
#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk
#endif
)
{
    static auto vnp = Cube::_vertices();

    int _i = ij[0], _ei = ij[1], _j = ij[2], _ej = ij[3];
    auto &ci(*globals.cubes[_i]), &cj(*globals.cubes[_j]);
    auto *eidxi = ci.edges, *eidxj = cj.edges;

    auto ei0 = ee[0], ei1 = ee[1],
         ej0 = ee[2], ej1 = ee[3];
    Vector<double, 12> ee_grad, p_grad;
    Matrix<double, 12, 12> ee_hess, p_hess;
    double eps_x = 1e-3;
    // 1e-3 * edge length square

    double p = ipc::edge_edge_mollifier(ei0, ei1, ej0, ej1, eps_x);
    ipc::edge_edge_mollifier_gradient(ei0, ei1, ej0, ej1, eps_x, p_grad);
    ipc::edge_edge_mollifier_hessian(ei0, ei1, ej0, ej1, eps_x, p_hess);

    ipc::edge_edge_distance_gradient(ei0, ei1, ej0, ej1, ee_grad);
    ipc::edge_edge_distance_hessian(ei0, ei1, ej0, ej1, ee_hess);

    double dist = 0.0;

    // dist = ipc::edge_edge_mollifier(ei0, ei1, ej0, ej1, eps_x);
    dist = ipc::edge_edge_distance(ei0, ei1, ej0, ej1);
    double B_ = barrier::barrier_derivative_d(dist);
    double B__ = barrier::barrier_second_derivative(dist);
    double B = barrier::barrier_function(dist);

    // spdlog::info("dist = {}, B = {}, B__ = {}", dist, B_, B__);

    Matrix<double, 6, 12> J0;
    Matrix<double, 6, 12> J1;
    auto ei0_tile = ci.vertices(eidxi[2 * _ei]), ei1_tile = ci.vertices(eidxi[2 * _ei + 1]),
         ej0_tile = cj.vertices(eidxj[2 * _ej]), ej1_tile = cj.vertices(eidxj[2 * _ej + 1]);
    // auto ei0_tile = vnp[eidx[2 * _ei]], ei1_tile = vnp[eidx[2 * _ei + 1]],
    //     ej0_tile = vnp[eidx[2 * _ej]], ej1_tile = vnp[eidx[2 * _ej + 1]];

    J0.block<3, 12>(0, 0) = barrier::x_jacobian_q(ei0_tile);
    J0.block<3, 12>(3, 0) = barrier::x_jacobian_q(ei1_tile);
    J1.block<3, 12>(0, 0) = barrier::x_jacobian_q(ej0_tile);
    J1.block<3, 12>(3, 0) = barrier::x_jacobian_q(ej1_tile);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    ipc_hess = p_hess * B + B_ * (p_grad * ee_grad.transpose() + ee_grad * p_grad.transpose()) + p * (B__ * ee_grad * ee_grad.transpose() + B_ * ee_hess);
    // ipc_hess = B__ * ee_grad * ee_grad.transpose() + project_to_psd(B_ * ee_hess);
    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);

    ee_grad = p * ee_grad * B_ + p_grad * B;

#ifdef _FRICTION_
    friction(_uk, contact_lambda, Tk, ee_grad, ipc_hess);
#endif

    int ii = _i, jj = _j;
    mat12 hess_0 = J0.transpose() * ipc_hess.block<6, 6>(0, 0) * J0;
    mat12 hess_1 = J1.transpose() * ipc_hess.block<6, 6>(6, 6) * J1;
    mat12 off_diag = J0.transpose() * ipc_hess.block<6, 6>(0, 6) * J1;
    mat12 off_T = off_diag.transpose();
#ifdef _SM_
    auto outers = sparse_hess.outerIndexPtr();
    auto values = sparse_hess.valuePtr();

    auto stride_j = stride(jj, outers), stride_i = stride(ii, outers);
    auto oii = starting_offset(ii, ii, lut, outers), ojj = starting_offset(jj, jj, lut, outers), oij = starting_offset(ii, jj, lut, outers), oji = starting_offset(jj, ii, lut, outers);

#endif
#ifdef _TRIPLETS_

#pragma omp critical
    for (int i = 0; i < 12; i++) {
        triplets.push_back(HessBlock(ii * 12, jj * 12 + i, off_diag.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, ii * 12 + i, off_T.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(ii * 12, ii * 12 + i, hess_0.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, jj * 12 + i, hess_1.block<12, 1>(0, i)));
    }

#endif
    Vector<double, 12> d0, d1;
    d0 = J0.transpose() * ee_grad.segment<6>(0);
    d1 = J1.transpose() * ee_grad.segment<6>(6);
    {
        for (int i = 0; i < 12; i++) {
#pragma omp atomic
            grad_0(i) += d0(i);
#pragma omp atomic
            grad_1(i) += d1(i);
        }
        put(values, oij, stride_j, off_diag);
        put(values, oji, stride_i, off_T);
        put(values, oii, stride_i, hess_0);
        put(values, ojj, stride_j, hess_1);
        // FIXME: atomic add
    }
}
