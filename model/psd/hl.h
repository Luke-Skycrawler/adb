#include "scalar_types.h"
#include <tuple>
#include <vector>

using mat9 = Eigen::Matrix<scalar, 9, 9>;
using vec9 = Eigen::Vector<scalar, 9>;
using mat9x12 = Eigen::Matrix<scalar, 9, 12>;
inline int idx(int i, int j)
{
    return i * 4 + j;
}

// edge-edge
mat3 C_ee(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);

vec2 beta_gamma_ee(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);
q4 dceedx_s(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);
mat9x12 dcdx_delta_ee(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);

// point-triangle
vec2 beta_gamma_pt(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);

mat3 C_vf(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);
q4 dcvfdx_s(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);

mat9x12 dcdx_delta_vf(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3);

vec9 gl(scalar l, const vec3& e2o);

mat9 Hl(const vec3& e0o, const vec3& e1o, const vec3& e2o, scalar l);

mat12 hessian_pt(const vec3& p, const vec3& t0, const vec3& t1, const vec3& t2, scalar l);

mat12 hessian_pt_eig(const vec3& p, const vec3& t0, const vec3& t1, const vec3& t2, scalar l, int keep = 0);
