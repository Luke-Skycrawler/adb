#include "math.h"
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
#include <tuple>
// #define CUDA_PROJECT
#ifdef CUDA_PROJECT
#include "cuda_glue.h"

void pt_grad_hess12x12(vec3f* pt, float* pt_grad, float* pt_hess, bool psd, float* buf);
// buf is not optional
void ee_grad_hess12x12(vec3f* ee, float* ee_grad, float* ipc_hess, float* buf);
#endif
using namespace std;
using mat12 = Matrix<double, 12, 12>;
void put(double* values, int offset, int _stride, const Matrix<double, 12, 12>& block)
{
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
            // #pragma omp atomic
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
double D_f0(double uk, double lam)
{
    static double mu = globals.mu, evh = globals.dt * globals.evh, h2 = globals.dt * globals.dt;
    double D_k = mu * lam * ipc::f0_SF(uk, evh);
    return D_k;
}

void friction(
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk,
    Vector<double, 12>& g, Matrix<double, 12, 12>& H)
{
    static double mu = globals.mu;
    static const double evh = globals.dt * globals.evh, h2 = globals.dt * globals.dt;
    double uk = sqrt(_uk(0) * _uk(0) + _uk(1) * _uk(1));
    // if (uk < 1e-10) return;

    auto f1 = ipc::f1_SF_over_x(uk, evh);

    Vector<double, 12> F_k = mu * contact_lambda * f1 * Tk * _uk;
    Matrix<double, 12, 12> D_k_hessian;

    if (uk >= evh) {
        Vector2d ut{ -_uk(1), _uk(0) };
        D_k_hessian = mu * contact_lambda * f1 / (uk * uk) * Tk * ut * (ut.transpose() * Tk.transpose());
    }
    else if (uk <= globals.params_double["max_uk"]) {
        D_k_hessian = mu * contact_lambda * f1 * Tk * Tk.transpose();
    }
    else {
        double f2_term = -1.0 / (evh * evh);
        Matrix2d M2x2 = f2_term / uk * _uk * _uk.transpose();
        // Matrix2d M2x2 = (df1_term * _uk * _uk.transpose());
        M2x2 += f1 * Matrix2d::Identity(2, 2);
        M2x2 *= mu * contact_lambda;
        M2x2 = project_to_psd(M2x2);
        D_k_hessian = Tk * M2x2 * Tk.transpose();
    }
    g += F_k;
    H += D_k_hessian;
}
void output_hessian_gradient(
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
    int ii, int jj, bool ci_nonstatic, bool cj_nonstatic,
    vec12& grad_p, vec12& grad_t,
    const vec12& dgp, const vec12& dgt,
    const mat12& hess_p, const mat12& hess_t, const mat12& off_diag, const mat12& off_T)
{
    auto outers = sparse_hess.outerIndexPtr();
    auto values = sparse_hess.valuePtr();

    auto stride_j = stride(jj, outers), stride_i = stride(ii, outers);
    auto oii = starting_offset(ii, ii, lut, outers), ojj = starting_offset(jj, jj, lut, outers), oij = starting_offset(ii, jj, lut, outers), oji = starting_offset(jj, ii, lut, outers);
    auto ptr = globals.writelock_cols.data();

    if (cj_nonstatic) {
        omp_set_lock(ptr + jj);
        if (ci_nonstatic)
            put(values, oij, stride_j, off_diag);
        put(values, ojj, stride_j, hess_t);
        grad_t += dgt;
        omp_unset_lock(ptr + jj);
    }
    if (ci_nonstatic) {
        omp_set_lock(ptr + ii);
        if (cj_nonstatic)
            put(values, oji, stride_i, off_T);
        put(values, oii, stride_i, hess_p);
        grad_p += dgp;
        omp_unset_lock(ptr + ii);
    }
}
void output_hessian_gradient(
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
    int ii, int jj, bool ci_nonstatic, bool cj_nonstatic,
    vec12& grad_p, vec12& grad_t,
    const vec12& dgp, const vec12& dgt,
    mat3 hess_p[4][4], mat3 hess_t[4][4], mat3 off_diag[4][4], mat3 off_T[4][4])
{
    auto outers = sparse_hess.outerIndexPtr();
    auto values = sparse_hess.valuePtr();

    auto stride_j = stride(jj, outers), stride_i = stride(ii, outers);
    auto oii = starting_offset(ii, ii, lut, outers), ojj = starting_offset(jj, jj, lut, outers), oij = starting_offset(ii, jj, lut, outers), oji = starting_offset(jj, ii, lut, outers);
    auto ptr = globals.writelock_cols.data();

    if (cj_nonstatic) {
        omp_set_lock(ptr + jj);
        if (ci_nonstatic)
            put2(values, oij, stride_j, off_diag);
        put2(values, ojj, stride_j, hess_t);
        grad_t += dgt;
        omp_unset_lock(ptr + jj);
    }
    if (ci_nonstatic) {
        omp_set_lock(ptr + ii);
        if (cj_nonstatic)
            put2(values, oji, stride_i, off_T);
        put2(values, oii, stride_i, hess_p);
        grad_p += dgp;
        omp_unset_lock(ptr + ii);
    }
}

void output_hessian_gradient(

    vector<HessBlock>& triplets,
    int ii, int jj,
    vec12& grad_p, vec12& grad_t,
    const vec12& dgp, const vec12& dgt,
    const mat12& hess_p, const mat12& hess_t, const mat12& off_diag, const mat12& off_T)
{

    for (int i = 0; i < 12; i++) {
        triplets.push_back(HessBlock(ii * 12, jj * 12 + i, off_diag.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, ii * 12 + i, off_T.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(ii * 12, ii * 12 + i, hess_p.block<12, 1>(0, i)));
        triplets.push_back(HessBlock(jj * 12, jj * 12 + i, hess_t.block<12, 1>(0, i)));
    }
    // globals.hess_triplets.push_back({ii * 12, jj * 12, off_diag});
    // globals.hess_triplets.push_back({jj * 12, ii * 12, off_T});
}

void output_hessian_gradient(
    mat12& hess_p_ret, mat12& hess_t_ret, mat12& off_diag_ret,
    vec12& grad_p, vec12& grad_t,
    const vec12& dgp, const vec12& dgt,
    const mat12& hess_p, const mat12& hess_t, const mat12& off_diag, const mat12& off_T)
{
    {
        hess_p_ret = hess_p;
        hess_t_ret = hess_t;
        off_diag_ret = off_diag;
        grad_p = dgp;
        grad_t = dgt;
    }
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

tuple<mat12, vec12> ipc_hess_pt_12x12(
    array<vec3, 4> pt, array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, double dist)
{
    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];

    auto p = pt[0], t0 = pt[1], t1 = pt[2], t2 = pt[3];
    auto &ci{ *globals.cubes[_i] }, &cj{ *globals.cubes[_j] };

    Vector<double, 12> pt_grad;
    Matrix<double, 12, 12> pt_hess;
    ipc::point_triangle_distance_gradient(p, t0, t1, t2, pt_grad, pt_type);
    ipc::point_triangle_distance_hessian(p, t0, t1, t2, pt_hess, pt_type);

    double B_ = barrier::barrier_derivative_d(dist);
    double B__ = barrier::barrier_second_derivative(dist);

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    // ipc_hess = PSD_projection(pt_hess  * B_) + pt_grad * pt_grad.transpose() * B__;
    ipc_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;

    pt_grad *= B_;

    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);

#ifdef CUDA_PROJECT
    // overtime babysitting

    if (globals.params_int["cuda_ipc_term"]) {
        float g[12], h[144];
        vec3f ptf[4];
        for (int i = 0; i < 4; i++)
            ptf[i] = to_vec3f(pt[i]);
        float* buf = new float[144];

        pt_grad_hess12x12(ptf, g, h, true, buf);

        vec12 g_cuda = Map<Vector<float, 12>>(g).cast<double>();
        mat12 h_cuda = Map<Matrix<float, 12, 12>>(h).cast<double>();
        if (!g_cuda.isApprox(pt_grad, 1e-3)) {
            spdlog::error("ipc grad mismatch, norm diff = {}, norm ref = {}, margin = {}", (g_cuda - pt_grad).norm(), pt_grad.norm(), (g_cuda - pt_grad).norm() / pt_grad.norm());
        }
        if (!h_cuda.isApprox(ipc_hess, 1e-3)) {
            spdlog::error("ipc hess mismatch, norm diff= {}, norm ref = {}, margin = {}", (h_cuda - ipc_hess).norm(), ipc_hess.norm(), (h_cuda - ipc_hess).norm() / ipc_hess.norm());
        }
        auto distf = dev::point_triangle_distance(ptf[0], ptf[1], ptf[2], ptf[3]);
        auto B_f = dev::barrier_derivative_d(distf);
        auto B__f = dev::barrier_second_derivative(distf);
        auto Bf = dev::barrier_function(distf);
        auto B = barrier::barrier_function(dist);
        auto b0m = abs((Bf - B) / B);
        auto b1m = abs((B_f - B_) / B_);
        auto b2m = abs((B__f - B__) / B__);
        auto dm = abs((distf - dist) / dist);
        auto type = dev::point_triangle_distance_type(ptf[0], ptf[1], ptf[2], ptf[3]);
        auto pt_type_to_int = static_cast<int>(pt_type);
        if (pt_type_to_int != type) {
            spdlog::error("pt type mismatch, ref = {}, cuda = {}", pt_type_to_int, type);
        }
        if (b0m > 1e-3) {
            spdlog::error("B mismatch, margin = {}", b0m);
        }
        if (b1m > 1e-3) {
            spdlog::error("B. mismatch, margin = {}", b1m);
        }
        if (b2m > 1e-3) {
            spdlog::error("B.. mismatch, margin = {}, B.. = {}", b2m, B__);
        }
        if (dm > 1e-3) {
            spdlog::error("dist mismatch, margin = {}, dist = {}", dm, dist);
        }
        delete[] buf;
    }
#endif

    return { ipc_hess, pt_grad };
}

void ipc_term(
    array<vec3, 4> pt, array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, double dist,
#ifdef _SM_OUT_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_

    vector<HessBlock>& triplets,
#endif
#ifdef _DIRECT_OUT_
    mat12& hess_p_ret, mat12& hess_t_ret, mat12& off_diag_ret,
#endif
    Vector<double, 12>& grad_p, Vector<double, 12>& grad_t
#ifdef _FRICTION_
    ,
    double& contact_lambda, Matrix<double, 2, 12>& Tk
#endif
)
{

    int _i = ij[0], v = ij[1], _j = ij[2], f = ij[3];
    int ii = _i, jj = _j;
    auto &ci{ *globals.cubes[_i] }, &cj{ *globals.cubes[_j] };
    const auto& tidx{ cj.indices };
    Vector2d _uk;
    contact_lambda = utils::pt_uktk(ci, cj, pt, ij, pt_type, Tk, _uk, dist, globals.dt);

    auto [ipc_hess, pt_grad] = ipc_hess_pt_12x12(pt, ij, pt_type, dist);

#ifdef _FRICTION_
    if (globals.pt_fric)
        friction(_uk, contact_lambda, Tk.transpose(), pt_grad, ipc_hess);
#endif

    auto p_tile = ci.vertices(v), t0_tile = cj.vertices(tidx[3 * f]), t1_tile = cj.vertices(tidx[3 * f + 1]), t2_tile = cj.vertices(tidx[3 * f + 2]);

#define _NO_FANCY_
#ifdef _NO_FANCY_

    Matrix<double, 9, 12> Jt;
    Matrix<double, 3, 12> Jp;
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
#else

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

    Vector<double, 12> dgp, dgt;
    vec3 seg = pt_grad.segment<3>(0);
    vec3 _0 = pt_grad.segment<3>(3), _1 = pt_grad.segment<3>(6), _2 = pt_grad.segment<3>(9);
    dgp << seg, seg * p_tile(0), seg * p_tile(1), seg * p_tile(2);
    dgt << _0 + _1 + _2,
        _0 * t0_tile(0) + _1 * t1_tile(0) + _2 * t2_tile(0),
        _0 * t0_tile(1) + _1 * t1_tile(1) + _2 * t2_tile(1),
        _0 * t0_tile(2) + _1 * t1_tile(2) + _2 * t2_tile(2);
#endif

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
tuple<mat12, vec12, double> ipc_hess_ee_12x12(
    array<vec3, 4> ee, array<int, 4> ij,
    ipc::EdgeEdgeDistanceType ee_type, double dist)
{
    int _i = ij[0], _ei = ij[1], _j = ij[2], _ej = ij[3];
    auto &ci(*globals.cubes[_i]), &cj(*globals.cubes[_j]);

    const auto &eidxi = ci.edges, &eidxj = cj.edges;

    auto ei0 = ee[0], ei1 = ee[1],
         ej0 = ee[2], ej1 = ee[3];
    vec12 ee_grad, p_grad;
    mat12 ee_hess, p_hess;
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

    mat12 ipc_hess;
    ipc_hess.setZero(12, 12);
    ipc_hess = p_hess * B + B_ * (p_grad * ee_grad.transpose() + ee_grad * p_grad.transpose()) + p * (B__ * ee_grad * ee_grad.transpose() + B_ * ee_hess);
    // ipc_hess = B__ * ee_grad * ee_grad.transpose() + project_to_psd(B_ * ee_hess);

    ee_grad = p * ee_grad * B_ + p_grad * B;
    
    if (globals.psd)
        ipc_hess = project_to_psd(ipc_hess);

#ifdef CUDA_PROJECT
    if (globals.params_int["cuda_ipc_term"]) {
        float g[12], h[144];
        vec3f eef[4];
        for (int i = 0; i < 4; i++)
            eef[i] = to_vec3f(ee[i]);

        float* buf = new float[300];
        ee_grad_hess12x12(eef, g, h, buf);

        vec12 g_cuda = Map<Vector<float, 12>>(g).cast<double>();
        mat12 h_cuda = Map<Matrix<float, 12, 12>>(h).cast<double>();
        vec12 p_grad_cuda = Map<Vector<float, 12>>(buf).cast<double>();
        mat12 p_hess_cuda = Map<Matrix<float, 12, 12>>(buf + 12).cast<double>();

        if (!g_cuda.isApprox(ee_grad, 1e-3)) {
            spdlog::error("ee grad mismatch, norm diff = {}, norm ref = {}, margin = {}", (g_cuda - ee_grad).norm(), ee_grad.norm(), (g_cuda - ee_grad).norm() / ee_grad.norm());
        }
        if (!h_cuda.isApprox(ipc_hess, 1e-3)) {
            spdlog::error("ee hess mismatch, norm diff= {}, norm ref = {}, margin = {}", (h_cuda - ipc_hess).norm(), ipc_hess.norm(), (h_cuda - ipc_hess).norm() / ipc_hess.norm());
        }

        if (!p_grad_cuda.isApprox(p_grad, 1e-3)) {
            spdlog::error("p grad mismatch, norm diff = {}, norm ref = {}, margin = {}", (p_grad_cuda - p_grad).norm(), p_grad.norm(), (p_grad_cuda - p_grad).norm() / p_grad.norm());
        }
        if (!p_hess_cuda.isApprox(p_hess, 1e-3)) {
            spdlog::error("p hess mismatch, norm diff= {}, norm ref = {}, margin = {}", (p_hess_cuda - p_hess).norm(), p_hess.norm(), (p_hess_cuda - p_hess).norm() / p_hess.norm());
        }

        auto type = dev::edge_edge_distance_type(eef[0], eef[1], eef[2], eef[3]);
        auto ee_type_to_int = static_cast<int>(ee_type);

        auto distf = dev::edge_edge_distance(eef[0], eef[1], eef[2], eef[3], type);

        auto B_f = dev::barrier_derivative_d(distf);
        auto B__f = dev::barrier_second_derivative(distf);
        auto Bf = dev::barrier_function(distf);
        auto B = barrier::barrier_function(dist);
        auto b0m = abs((Bf - B) / B);
        auto b1m = abs((B_f - B_) / B_);
        auto b2m = abs((B__f - B__) / B__);
        auto dm = abs((distf - dist) / dist);
        if (type != ee_type_to_int) {
            spdlog::error("ee type mismatch, type = {}, type cuda = {}", ee_type_to_int, type);
        }
        if (dm > 1e-3) {
            spdlog::error("dist mismatch, margin = {}, dist = {}", dm, dist);
        }
        if (b0m > 1e-3) {
            spdlog::error("B mismatch, margin = {}", b0m);
        }
        if (b1m > 1e-3) {
            spdlog::error("B. mismatch, margin = {}", b1m);
        }
        if (b2m > 1e-3) {
            spdlog::error("B.. mismatch, margin = {}, B.. = {}", b2m, B__);
        }
        delete[] buf;
    }
#endif

    return { ipc_hess, ee_grad, p };
}
void ipc_term_ee(
    array<vec3, 4> ee, array<int, 4> ij, ipc::EdgeEdgeDistanceType ee_type, double dist,
#ifdef _SM_OUT_
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    vector<HessBlock>& triplets,
#endif
#ifdef _DIRECT_OUT_
    mat12& hess_0_ret, mat12& hess_1_ret, mat12& off_diag_ret,
#endif
    Vector<double, 12>& grad_0, Vector<double, 12>& grad_1
#ifdef _FRICTION_
    ,
    double& contact_lambda, Matrix<double, 2, 12>& Tk
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

    Vector2d _uk;
    contact_lambda = utils::ee_uktk(ci, cj, ee, ij, ee_type, Tk, _uk, dist, globals.dt, p);
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
#define _NO_FANCY_
#ifdef _NO_FANCY_
    Matrix<double, 6, 12> J0;
    Matrix<double, 6, 12> J1;

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
#else

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

#endif

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

    //     {
    // #ifdef _NO_FANCY_
    // #define PUT put
    // #else
    // #define PUT put2
    // #endif
    //         if (cj.mass > 0.0) {
    //             omp_set_lock(ptr + jj);
    //             if (ci.mass > 0.0) {
    //                 PUT(values, oij, stride_j, off_diag);
    //             }
    //             PUT(values, ojj, stride_j, hess_1);
    //             grad_1 += d1;
    //             omp_unset_lock(ptr + jj);
    //         }
    //         if (ci.mass > 0.0) {
    //             omp_set_lock(ptr + ii);
    //             if (cj.mass > 0.0)
    //                 PUT(values, oji, stride_i, off_T);
    //             PUT(values, oii, stride_i, hess_0);
    //             grad_0 += d0;
    //             omp_unset_lock(ptr + ii);
    //         }
    // #undef PUT
    //     }

    // #ifdef _SM_
    //     auto outers = sparse_hess.outerIndexPtr();
    //     auto values = sparse_hess.valuePtr();

    //     auto stride_j = stride(jj, outers), stride_i = stride(ii, outers);
    //     auto oii = starting_offset(ii, ii, lut, outers), ojj = starting_offset(jj, jj, lut, outers), oij = starting_offset(ii, jj, lut, outers), oji = starting_offset(jj, ii, lut, outers);
    //     auto ptr = globals.writelock_cols.data();

    // #endif
    // #ifdef _TRIPLETS_

    // #pragma omp critical
    //     for (int i = 0; i < 12; i++) {
    //         triplets.push_back(HessBlock(ii * 12, jj * 12 + i, off_diag.block<12, 1>(0, i)));
    //         triplets.push_back(HessBlock(jj * 12, ii * 12 + i, off_T.block<12, 1>(0, i)));
    //         triplets.push_back(HessBlock(ii * 12, ii * 12 + i, hess_0.block<12, 1>(0, i)));
    //         triplets.push_back(HessBlock(jj * 12, jj * 12 + i, hess_1.block<12, 1>(0, i)));
    //     }

    // #endif
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