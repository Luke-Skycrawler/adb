#pragma once
#include <Eigen/Eigen>

namespace othogonal_energy{

    using namespace Eigen;
    using vec3 = Vector3f;
    using mat3 = Matrix3f;
    vec3 grad(mat3 &q, int i);
    mat3 grad(mat3 &q);
    mat3 hessian(mat3 &q, int i, int j);
}
