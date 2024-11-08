#pragma once 
#include "affine_body.h"

#include <vector>
#include <ipc/distance/distance_type.hpp>

#define _FRICTION_
#define _SM_
#ifdef _SM_
#define _SM_OUT_
#endif
// #define NO_MOLLIFIER
// #define _TRIPLETS_
// #define _PLUG_IN_LAN_

#ifdef _PLUG_IN_LAN_
#include "IpcFrictionConstraint.h"
#include "IpcCollisionConstraint.h"
#define _DIRECT_OUT_
#undef _SM_OUT_
#undef _TRIPLETS_
#endif

#ifdef _FRICTION_
void friction(
    const Eigen::Vector<scalar, 2>& _uk, scalar contact_lambda, const Eigen::Matrix<scalar, 12, 2>& Tk,
    Eigen::Vector<scalar, 12>& g, Eigen::Matrix<scalar, 12, 12>& H);

scalar D_f0(scalar uk, scalar lam);
#endif

struct IPC {
    void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
        ,
        const Eigen::Vector<scalar, 2>& _uk, scalar contact_lambda, const Eigen::Matrix<scalar, 3, 2>& Tk
#endif
    );

    void ipc_term(
        std::array<vec3, 4> pt, std::array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, scalar dist,
#ifdef _SM_OUT_
        const std::map<std::array<int, 2>, int>& lut,
        Eigen::SparseMatrix<scalar>& sparse_hess,
#endif
#ifdef _TRIPLETS_
        std::vector<HessBlock>& triplets,
#endif
#ifdef _DIRECT_OUT_
        mat12& hess_p_ret, mat12& hess_t_ret, mat12& off_diag_ret,
#endif
        vec12& grad_p, vec12& grad_t

#ifdef _FRICTION_
        ,
        scalar& contact_lambda, Eigen::Matrix<scalar, 2, 12>& Tk
#endif
    );
    void ipc_term_ee(
        std::array<vec3, 4> ee, std::array<int, 4> ij, ipc::EdgeEdgeDistanceType ee_type, scalar dist,

#ifdef _SM_OUT_
        const std::map<std::array<int, 2>, int>& lut,
        Eigen::SparseMatrix<scalar>& sparse_hess,
#endif
#ifdef _TRIPLETS_
        std::vector<HessBlock>& triplets,
#endif
#ifdef _DIRECT_OUT_
        mat12& hess_0_ret, mat12& hess_1_ret, mat12& off_diag_ret,
#endif
        vec12& grad_0, vec12& grad_1

#ifdef _FRICTION_
        ,
        scalar& contact_lambda, Eigen::Matrix<scalar, 2, 12>& Tk
#endif
    );

private:
    std::tuple<mat12, vec12> ipc_hess_pt_12x12(
        std::array<vec3, 4> pt, std::array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, scalar dist);
    std::tuple<mat12, vec12, scalar> ipc_hess_ee_12x12(
        std::array<vec3, 4> ee, std::array<int, 4> ij,
        ipc::EdgeEdgeDistanceType ee_type, scalar dist);

    void output_hessian_gradient(
        const std::map<std::array<int, 2>, int>& lut,
        Eigen::SparseMatrix<scalar>& sparse_hess,
        int ii, int jj, bool ci_nonstatic, bool cj_nonstatic,
        vec12& grad_p, vec12& grad_t,
        const vec12& dgp, const vec12& dgt,
        const mat12& hess_p, const mat12& hess_t, const mat12& off_diag, const mat12& off_T);
};