#include "affine_body.h"
#include "geometry.h"
#include <vector>
#include <array>
#include <map>
#include <ipc/distance/distance_type.hpp>
#define _FRICTION_
#define _SM_
#ifdef _SM_
#define _SM_OUT_
#endif
// #define NO_MOLLIFIER
// #define _TRIPLETS_
// #define _PLUG_IN_LAN_

#ifdef _FRICTION_
void friction(
    const Eigen::Vector<scalar, 2>& _uk, scalar contact_lambda, const Eigen::Matrix<scalar, 12, 2>& Tk,
    Eigen::Vector<scalar, 12>& g, Eigen::Matrix<scalar, 12, 12>& H);
#endif
#ifdef _PLUG_IN_LAN_
#include "IpcFrictionConstraint.h"
#include "IpcCollisionConstraint.h"
#define _DIRECT_OUT_
#undef _SM_OUT_
#undef _TRIPLETS_
#endif
#ifdef _TRIPLETS_
#include "../view/global_variables.h"
#endif
// x: vertex position in the static frame;
// p: affine body translation
// q: affine matrix
#ifdef _TIGHT_INCLUSION_ENABLE_

scalar vf_collision_detect(vec3& p_t0, vec3& p_t1, const AffineBody& cube, int id);
scalar vf_collision_detect(vec3& p_t0, vec3& p_t1, const Face &f_t0, const Face &f_t1);
scalar ee_collision_detect(const Edge& ei_t0, const Edge& ej_t0, const Edge& ei_t1, const Edge& ej_t1);
scalar ee_collision_detect(const AffineBody& ci, const AffineBody& cj, int eid_i, int eid_j);
#endif
scalar D_f0(scalar uk, scalar lam);
scalar collision_time(AffineBody& c, int i);
scalar pt_collision_time(
    const vec3& p0,
    const Face& t0,
    const vec3& p1,
    const Face& t1);
scalar ee_collision_time(
    const Edge& ei0,
    const Edge& ej0,
    const Edge& ei1,
    const Edge& ej1);

Eigen::Vector<scalar, 4> det_polynomial(const mat3& a, const mat3& b);
// only exported for testing

void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Eigen::Vector<scalar, 2>& _uk, scalar contact_lambda, const Eigen::Matrix<scalar, 3, 2>& Tk
#endif
);

// void output_hessian_gradient(
// #ifdef _SM_OUT_
//     const std::map<std::array<int, 2>, int>& lut,
//     SparseMatrix<scalar>& sparse_hess,
// #endif
// #ifdef _TRIPLETS_
//     vector<HessBlock>& triplets,
// #endif
// #ifdef _DIRECT_OUT_
//     mat12& hess_p_ret, mat12& hess_t_ret, mat12& off_diag_ret,
// #endif
//     vec12& grad_p, vec12& grad_t,
//     const vec12& dgp, const vec12& dgt,
//     const mat12& hess_p, const mat12& hess_t, const mat12& off_diag, const mat12& off_T);

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
    mat12 & hess_p_ret, mat12 & hess_t_ret, mat12 & off_diag_ret, 
#endif
    vec12& grad_p, vec12& grad_t

#ifdef _FRICTION_
    , scalar &contact_lambda, Eigen::Matrix<scalar, 2, 12>& Tk
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

void output_hessian_gradient(
    const std::map<std::array<int, 2>, int>& lut,
    Eigen::SparseMatrix<scalar>& sparse_hess,
    int ii, int jj, bool ci_nonstatic, bool cj_nonstatic,
    vec12& grad_p, vec12& grad_t,
    const vec12& dgp, const vec12& dgt,
    const mat12& hess_p, const mat12& hess_t, const mat12& off_diag, const mat12& off_T);
