#include "affine_body.h"
#include "sparse.h"
#include <omp.h>
#ifndef TESTING
#include "settings.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif
using namespace Eigen;
void put(scalar* values, int offset, int _stride, const Matrix<scalar, 12, 12>& block)
{
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
//#pragma omp atomic
            values[offset + _stride * j + i] += block(i, j);
        }
}

void put2(scalar* values, int offset, int _stride, mat3 block[4][4])
{
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
            scalar db = block[i / 3][j / 3](i % 3, j % 3);
            int ofs = offset + _stride * j + i;
            // #pragma omp atomic
            values[ofs] += db;
        }
}

void output_hessian_gradient(
    const std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<scalar>& sparse_hess,
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
    SparseMatrix<scalar>& sparse_hess,
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
