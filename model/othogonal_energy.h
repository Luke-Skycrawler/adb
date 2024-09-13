#pragma once
#include "affine_body.h"
#include <array>
namespace othogonal_energy{

    using namespace Eigen;
    using q4 = std::array<vec3, 4>;

    //vec3 grad(mat3 &q, int i);
    //mat3 grad(mat3 &q);
    vec12 grad(const q4& q);
    //mat3 hessian(mat3 &q, int i, int j);
    mat12 hessian(const q4& q);
    scalar otho_energy(const Vector<scalar, -1>& q);
    //scalar otho_energy(mat3 &q);
}
