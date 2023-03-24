#include "cube.h"
// #include "geometry.h"
#include "barrier.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/friction/smooth_friction_mollifier.hpp>
#ifndef TESTING
#include "../view/global_variables.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif
#include "collision.h"
#include "time_integrator.h"
#include <omp.h>
using namespace std;
using mat12 = Matrix<double, 12, 12>;
void put(double* values, int offset, int _stride, const Matrix<double, 12, 12>& block)
{
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
//#pragma omp atomic
            values[offset + _stride * j + i] += block(i, j);
        }
}

void put2(double* values, int offset, int _stride, mat3 block[4][4])
{
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
            double db = block[i / 3][j / 3](i % 3, j % 3);
            int ofs = offset + _stride * j + i;
            // #pragma omp atomic
            values[ofs] += db;
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
    static double mu = globals.mu, evh = globals.dt * globals.evh, h2 = globals.dt * globals.dt;
    double D_k = mu * lam * ipc::f0_SF(uk, evh);
    return D_k * h2;
}

void friction(
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk,
    Vector<double, 12>& g, Matrix<double, 12, 12>& H)
{
    static double mu = globals.mu;
    static const double evh = globals.dt * globals.evh, h2 = globals.dt * globals.dt;
    double uk = sqrt(_uk(0) * _uk(0) + _uk(1) * _uk(1));
    //if (uk < 1e-10) return;

    auto f1 = ipc::f1_SF_over_x(uk, evh);
    
    Vector<double, 12> F_k = mu * contact_lambda * f1 * Tk  * _uk;
    Matrix<double, 12, 12> D_k_hessian;

    if (uk >= evh) {
        Vector2d ut{ -_uk(1), _uk(0) };
        D_k_hessian = mu * contact_lambda * f1 / (uk * uk) * Tk * ut * (ut.transpose() * Tk.transpose());
    }
    else if (uk <= 1e-10) {
        D_k_hessian = mu * contact_lambda * f1 * Tk * Tk.transpose();
    }
    else {
        double df1_term = ipc::df1_x_minus_f1_over_x3(uk, evh);
        Matrix2d M2x2 = (df1_term * _uk * _uk.transpose());
        M2x2 += f1 * Matrix2d::Identity(2, 2);
        M2x2 = project_to_psd(M2x2);
        D_k_hessian = mu * contact_lambda * Tk * M2x2 * Tk.transpose();
    }
    g += F_k * h2;
    H += D_k_hessian * h2;
}

#ifndef TESTING
void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 3, 2>& Tk
#endif

)
{
    if (c.mass < 0.0) return;
    auto v_tile{ c.vertices(v) }, p{ c.vt1(v) };
    c.grad += barrier::barrier_gradient_q(v_tile, p);
    c.hess += barrier::barrier_hessian_q(v_tile, p);
#ifdef _FRICTION_
    auto J = barrier::x_jacobian_q(v_tile);
    if (globals.vg_fric)
        friction(_uk, contact_lambda,
            J.transpose() * Tk, c.grad, c.hess);
#endif
};

void ipc_term(
    array<vec3, 4> pt, array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, double dist,
#ifdef _SM_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_

    vector<HessBlock>& triplets,
#endif
    Vector<double, 12>& grad_p, Vector<double, 12>& grad_t
#ifdef _FRICTION_
    ,double &contact_lambda, Matrix<double, 2, 12>& Tk
#endif
)
{

    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];

    auto p = pt[0], t0 = pt[1], t1 = pt[2], t2 = pt[3];
    auto &ci{ *globals.cubes[_i] }, &cj{ *globals.cubes[_j] };

    Vector2d _uk;
    contact_lambda = utils::pt_uktk(ci, cj, pt, ij, pt_type, Tk, _uk, dist, globals.dt);
    auto* tidx = cj.indices;
    Vector<double, 12> pt_grad;
    Matrix<double, 12, 12> pt_hess;
    ipc::point_triangle_distance_gradient(p, t0, t1, t2, pt_grad, pt_type);
    ipc::point_triangle_distance_hessian(p, t0, t1, t2, pt_hess, pt_type);

    double B_ = barrier::barrier_derivative_d(dist);
    double B__ = barrier::barrier_second_derivative(dist);
    // spdlog::info("dist = {}, B = {}, B__ = {}", dist, B_, B__);

    // Matrix<double, 9, 12> Jt;
    // Matrix<double, 3, 12> Jp;
    // mat12 off_diag;
    auto p_tile = ci.vertices(v), t0_tile = cj.vertices(tidx[3 * f]), t1_tile = cj.vertices(tidx[3 * f + 1]), t2_tile = cj.vertices(tidx[3 * f + 2]);
    // auto p_tile = vnp[v], t0_tile = vnp[tidx[3 * f]], t1_tile = vnp[tidx[3 * f + 1]], t2_tile = vnp[tidx[3 * f + 2]];
    // Jt.setZero(9, 12);
    // Jp.setZero(3, 12);
    // off_diag.setZero(12, 12);
    // Jt.block<3, 12>(0, 0) = barrier::x_jacobian_q(t0_tile);
    // Jt.block<3, 12>(3, 0) = barrier::x_jacobian_q(t1_tile);
    // Jt.block<3, 12>(6, 0) = barrier::x_jacobian_q(t2_tile);
    // Jp = barrier::x_jacobian_q(p_tile);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    // ipc_hess = PSD_projection(pt_hess  * B_) + pt_grad * pt_grad.transpose() * B__;
    ipc_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;

    pt_grad *= B_;

    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);
#ifdef _FRICTION_
    if (globals.pt_fric)
        friction(_uk, contact_lambda, Tk.transpose(), pt_grad, ipc_hess);
#endif

    int ii = _i, jj = _j;
    // mat12 hess_p = Jp.transpose() * ipc_hess.block<3, 3>(0, 0) * Jp;
    // mat12 hess_t = Jt.transpose() * ipc_hess.block<9, 9>(3, 3) * Jt;

    // mat12 off_diag = Jp.transpose() * ipc_hess.block<3, 9>(0, 3) * Jt;

    Vector4d kerp;
    kerp << 1.0, p_tile;
    Vector4d ker0;
    ker0 << 1.0, t0_tile;
    Vector4d ker1;
    ker1 << 1.0, t1_tile;
    Vector4d ker2;
    ker2 << 1.0, t2_tile;
    Matrix<double, 4, 3> kert;
    kert << ker0, ker1, ker2;

    Matrix4d blkp = kerp * kerp.transpose();
    mat12 hess_p, hess_t, off_diag;
    // mat12 hess_p, hess_t, hess__off;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            hess_p.block<3, 3>(i * 3, j * 3) = blkp(i, j) * ipc_hess.block<3, 3>(0, 0);

            mat3 hij;
            hij.setZero(3, 3);
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++) {
                    mat3 Akl = ipc_hess.block<3, 3>((k + 1) * 3, (l + 1) * 3);
                    hij += Akl * (kert(i, k) * kert(j, l));
                }
            hess_t.block<3, 3>(i * 3, j * 3) = hij;

            mat3 offd_ij;
            offd_ij.setZero(3, 3);
            for (int l = 0; l < 3; l++) {
                mat3 Akl = ipc_hess.block<3, 3>(0, (l + 1) * 3);
                offd_ij += Akl * (kerp(i) * kert(j, l));
            }
            off_diag.block<3, 3>(i * 3, j * 3) = offd_ij;
        }
    mat12 off_T = off_diag.transpose();


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
    vec3 seg = pt_grad.segment<3>(0);
    vec3 _0 = pt_grad.segment<3>(3), _1 = pt_grad.segment<3>(6), _2 = pt_grad.segment<3>(9);
    dgp << seg, seg * p_tile(0), seg * p_tile(1), seg * p_tile(2);
    dgt << _0 + _1 + _2,
        _0 * t0_tile(0) + _1 * t1_tile(0) + _2 * t2_tile(0),
        _0 * t0_tile(1) + _1 * t1_tile(1) + _2 * t2_tile(1),
        _0 * t0_tile(2) + _1 * t1_tile(2) + _2 * t2_tile(2);
    auto ptr = globals.writelock_cols.data();
    {
        if (cj.mass > 0.0) {
            omp_set_lock(ptr + jj);
            if (ci.mass > 0.0)
                put(values, oij, stride_j, off_diag);
            put(values, ojj, stride_j, hess_t);
            grad_t += dgt;
            omp_unset_lock(ptr + jj);
        }
        if (ci.mass > 0.0) {
            omp_set_lock(ptr + ii);
            if (cj.mass > 0.0)
                put(values, oji, stride_i, off_T);
            put(values, oii, stride_i, hess_p);
            grad_p += dgp;
            omp_unset_lock(ptr + ii);
        }
    }
}

void ipc_term_ee(
    array<vec3, 4> ee, array<int, 4> ij, ipc::EdgeEdgeDistanceType ee_type, double dist,
#ifdef _SM_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    vector<HessBlock>& triplets,
#endif
    Vector<double, 12>& grad_0, Vector<double, 12>& grad_1
#ifdef _FRICTION_
    , double& contact_lambda, Matrix<double, 2, 12>& Tk
#endif
)
{
    // static auto vnp = Cube::_vertices();

    int _i = ij[0], _ei = ij[1], _j = ij[2], _ej = ij[3];
    auto &ci(*globals.cubes[_i]), &cj(*globals.cubes[_j]);
    Vector2d _uk;
    contact_lambda = utils::ee_uktk(ci, cj, ee, ij, ee_type, Tk, _uk, dist, globals.dt);

    auto *eidxi = ci.edges, *eidxj = cj.edges;

    auto ei0 = ee[0], ei1 = ee[1],
         ej0 = ee[2], ej1 = ee[3];
    Vector<double, 12> ee_grad, p_grad;
    Matrix<double, 12, 12> ee_hess, p_hess;
    double eps_x = globals.eps_x * (ei0 - ei1).squaredNorm() * (ej0 - ej1).squaredNorm();

    // 1e-3 * edge length square

    double p = ipc::edge_edge_mollifier(ei0, ei1, ej0, ej1, eps_x);
    ipc::edge_edge_mollifier_gradient(ei0, ei1, ej0, ej1, eps_x, p_grad);
    ipc::edge_edge_mollifier_hessian(ei0, ei1, ej0, ej1, eps_x, p_hess);

    ipc::edge_edge_distance_gradient(ei0, ei1, ej0, ej1, ee_grad, ee_type);
    ipc::edge_edge_distance_hessian(ei0, ei1, ej0, ej1, ee_hess, ee_type);

    double B_ = barrier::barrier_derivative_d(dist);
    double B__ = barrier::barrier_second_derivative(dist);
    double B = barrier::barrier_function(dist);

    // spdlog::info("dist = {}, B = {}, B__ = {}", dist, B_, B__);

    // Matrix<double, 6, 12> J0;
    // Matrix<double, 6, 12> J1;
    auto ei0_tile = ci.vertices(eidxi[2 * _ei]), ei1_tile = ci.vertices(eidxi[2 * _ei + 1]),
         ej0_tile = cj.vertices(eidxj[2 * _ej]), ej1_tile = cj.vertices(eidxj[2 * _ej + 1]);
    // auto ei0_tile = vnp[eidx[2 * _ei]], ei1_tile = vnp[eidx[2 * _ei + 1]],
    //     ej0_tile = vnp[eidx[2 * _ej]], ej1_tile = vnp[eidx[2 * _ej + 1]];

    // J0.block<3, 12>(0, 0) = barrier::x_jacobian_q(ei0_tile);
    // J0.block<3, 12>(3, 0) = barrier::x_jacobian_q(ei1_tile);
    // J1.block<3, 12>(0, 0) = barrier::x_jacobian_q(ej0_tile);
    // J1.block<3, 12>(3, 0) = barrier::x_jacobian_q(ej1_tile);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    ipc_hess = p_hess * B + B_ * (p_grad * ee_grad.transpose() + ee_grad * p_grad.transpose()) + p * (B__ * ee_grad * ee_grad.transpose() + B_ * ee_hess);
    // ipc_hess = B__ * ee_grad * ee_grad.transpose() + project_to_psd(B_ * ee_hess);

#ifdef NO_MOLLIFIER
    ipc_hess = B__ * ee_grad * ee_grad.transpose() + B_ * ee_hess;

#else
    ee_grad = p * ee_grad * B_ + p_grad * B;
#endif


    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);

#ifdef _FRICTION_
    if (globals.ee_fric)
        friction(_uk, contact_lambda, Tk.transpose(), ee_grad, ipc_hess);
#endif

    int ii = _i, jj = _j;
    // mat12 hess_p = Jp.transpose() * ipc_hess.block<3, 3>(0, 0) * Jp;
    // mat12 hess_t = Jt.transpose() * ipc_hess.block<9, 9>(3, 3) * Jt;

    // mat12 off_diag = Jp.transpose() * ipc_hess.block<3, 9>(0, 3) * Jt;

    // Vector4d keri0;
    // keri0 << 1.0, ei0_tile;
    // Vector4d keri1;
    // keri1 << 1.0, ei1_tile;
    // Vector4d kerj0;
    // kerj0 << 1.0, ej0_tile;
    // Vector4d kerj1;
    // kerj1 << 1.0, ej1_tile;
    // Matrix<double, 4, 2> ker0, ker1;
    // ker0 << keri0, keri1;
    // ker1 << kerj0, kerj1;
    double ker0[4][2], ker1[4][2];

    for (int i = 0; i < 4; i++) {
        ker0[i][0] = i == 0 ? 1.0 : ei0_tile(i - 1);
        ker0[i][1] = i == 0 ? 1.0 : ei1_tile(i - 1);
        ker1[i][0] = i == 0 ? 1.0 : ej0_tile(i - 1);
        ker1[i][1] = i == 0 ? 1.0 : ej1_tile(i - 1);
    }

    // Matrix4d blk0 = ker0 * ker0.transpose();
    // Matrix4d blk1 = ker0 * ker1.transpose();

    mat3 hess_0[4][4], hess_1[4][4], off_diag[4][4], off_T[4][4];
    mat3 Akl[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) Akl[i][j] = ipc_hess.block<3, 3>(i * 3, j * 3);
    // mat12 hess_p, hess_t, hess__off;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            hess_0[i][j].setZero(3, 3);
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++) {
                    hess_0[i][j] += Akl[k][l] * (ker0[i][k] * ker0[j][l]);
                }

            hess_1[i][j].setZero(3, 3);
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++) {
                    hess_1[i][j] += Akl[k + 2][l + 2] * (ker1[i][k] * ker1[j][l]);
                }

            off_diag[i][j].setZero(3, 3);
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++) {
                    off_diag[i][j] += Akl[k][l + 2] * (ker0[i][k] * ker1[j][l]);
                }
            off_T[j][i] = off_diag[i][j].transpose();
        }
        // mat12 off_T = off_diag.transpose();

        // mat12 hess_0 = J0.transpose() * ipc_hess.block<6, 6>(0, 0) * J0;
        // mat12 hess_1 = J1.transpose() * ipc_hess.block<6, 6>(6, 6) * J1;
        // mat12 off_diag = J0.transpose() * ipc_hess.block<6, 6>(0, 6) * J1;
        // mat12 off_T = off_diag.transpose();
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
    // Vector<double, 12> d0, d1;
    // d0 = J0.transpose() * ee_grad.segment<6>(0);
    // d1 = J1.transpose() * ee_grad.segment<6>(6);

    Vector<double, 12> d0, d1;
    vec3 i0 = ee_grad.segment<3>(0), i1 = ee_grad.segment<3>(3);
    vec3 j0 = ee_grad.segment<3>(6), j1 = ee_grad.segment<3>(9);
    // d0 << i0 + i1,
    //     i0 * ei0_tile(0) + i1 * ei1_tile(0),
    //     i0 * ei0_tile(1) + i1 * ei1_tile(1),
    //     i0 * ei0_tile(2) + i1 * ei1_tile(2);
    for (int i = 0; i < 12; i++) {
        d0(i) = ker0[i / 3][0] * i0(i % 3)
            + ker0[i / 3][1] * i1(i % 3);

        d1(i) = ker1[i / 3][0] * j0(i % 3)
            + ker1[i / 3][1] * j1(i % 3);
    }
    // d1 << j0 + j1,
    //     j0 * ej0_tile(0) + j1 * ej1_tile(0),
    //     j0 * ej0_tile(1) + j1 * ej1_tile(1),
    //     j0 * ej0_tile(2) + j1 * ej1_tile(2);
    auto ptr = globals.writelock_cols.data();
    {
        if (cj.mass > 0.0) {
            omp_set_lock(ptr + jj);
            if (ci.mass > 0.0)
                put2(values, oij, stride_j, off_diag);
            put2(values, ojj, stride_j, hess_1);
            grad_1 += d1;
            omp_unset_lock(ptr + jj);
        }
        if (ci.mass > 0.0) {
            omp_set_lock(ptr + ii);
            if (cj.mass > 0.0)
                put2(values, oji, stride_i, off_T);
            put2(values, oii, stride_i, hess_0);
            grad_0 += d0;
            omp_unset_lock(ptr + ii);
        }
    }
}
double E_ground(const vec3& v)
{
    double e = 0.0;
    double d = vg_distance(v);
    d = d * d;
    if (d < barrier::d_hat) {
        e = barrier::barrier_function(d);
    }
    return e;
};
#endif