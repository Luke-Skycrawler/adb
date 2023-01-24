#include "affine_body.h"
#include "geometry.h"
#include <vector>
#include <array>
#include <map>
#include <ipc/distance/distance_type.hpp>
#define _FRICTION_
#define _SM_
// #define NO_MOLLIFIER
// #define _TRIPLETS_
#ifdef _TRIPLETS_
#include "../view/global_variables.h"
#endif
// x: vertex position in the static frame;
// p: affine body translation
// q: affine matrix
double vf_collision_detect(vec3& p_t0, vec3& p_t1, const AffineBody& cube, int id);
double vf_collision_detect(vec3& p_t0, vec3& p_t1, const Face &f_t0, const Face &f_t1);
double ee_collision_detect(const Edge& ei_t0, const Edge& ej_t0, const Edge& ei_t1, const Edge& ej_t1);
double ee_collision_detect(const AffineBody& ci, const AffineBody& cj, int eid_i, int eid_j);

double D_f0(double uk, double lam);
double collision_time(AffineBody& c, int i);
double pt_collision_time(
    const vec3& p0,
    const Face& t0,
    const vec3& p1,
    const Face& t1);
double ee_collision_time(
    const Edge& ei0,
    const Edge& ej0,
    const Edge& ei1,
    const Edge& ej1);

Eigen::Vector4d det_polynomial(const mat3& a, const mat3& b);
// only exported for testing

void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Eigen::Vector2d& _uk, double contact_lambda, const Eigen::Matrix<double, 3, 2>& Tk
#endif
);
void ipc_term(
    std::array<vec3, 4> pt, std::array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, double dist, 
#ifdef _SM_
    const std::map<std::array<int, 2>, int>& lut,
    Eigen::SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    std::vector<HessBlock>& triplets,
#endif
    vec12& grad_p, vec12& grad_t

#ifdef _FRICTION_
    ,
    const Eigen::Vector2d& _uk, double contact_lambda, const Eigen::Matrix<double, 12, 2>& Tk
#endif
    // , Matrix<double, 12 , 12> &off_diag
    // , vector<Cube> &cubes
);
void ipc_term_ee(
    std::array<vec3, 4> ee, std::array<int, 4> ij, ipc::EdgeEdgeDistanceType ee_type, double dist, 

#ifdef _SM_
    const std::map<std::array<int, 2>, int>& lut,
    Eigen::SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    std::vector<HessBlock>& triplets,
#endif
    vec12& grad_0, vec12& grad_1

#ifdef _FRICTION_
    ,
    const Eigen::Vector2d& _uk, double contact_lambda, const Eigen::Matrix<double, 12, 2>& Tk
#endif
);
