#include "affine_body.h"
#include "geometry.h"
#include <vector>
#include <array>
#include <map>
#include "../view/global_variables.h"
#define _FRICTION_
#define _SM_
// #define _TRIPLETS_
// double vf_collision_time(const vec3& x, const vec3& p, const mat3& q, const vec3& p_next, const mat3& q_next);
double vf_collision_detect(vec3& p_t0, vec3& p_t1, const AffineBody& cube, int id);
// x: vertex position in the static frame;
// p: affine body translation
// q: affine matrix
double ee_collision_detect(const AffineBody& ci, const AffineBody& cj, int eid_i, int eid_j);

void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 3, 2>& Tk
#endif
);
void ipc_term(
    std::array<vec3, 4> pt, std::array<int, 4> ij,
#ifdef _SM_
    std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    std::vector<HessBlock>& triplets,
#endif
    Vector<double, 12>& grad_p, Vector<double, 12>& grad_t

#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk
#endif
    // , Matrix<double, 12 , 12> &off_diag
    // , vector<Cube> &cubes
);
void ipc_term_ee(
    std::array<vec3, 4> ee, std::array<int, 4> ij,

#ifdef _SM_
    std::map<std::array<int, 2>, int>& lut,
    SparseMatrix<double>& sparse_hess,
#endif
#ifdef _TRIPLETS_
    std::vector<HessBlock>& triplets,
#endif
    Vector<double, 12>& grad_0, Vector<double, 12>& grad_1

#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk
#endif
);

double D_f0(double uk, double lam);
