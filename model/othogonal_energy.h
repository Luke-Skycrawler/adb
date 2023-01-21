#pragma once
#include <Eigen/Eigen>
#include <array>
namespace othogonal_energy{

    using namespace Eigen;
    using vec3 = Vector3d;
    using mat3 = Matrix3d;
    using q4 = std::array<vec3, 4>;
    using vec12 = Eigen::Vector<double, 12>;
    using mat12 = Eigen::Matrix<double, 12, 12>;

    //vec3 grad(mat3 &q, int i);
    //mat3 grad(mat3 &q);
    vec12 grad(const q4& q);
    //mat3 hessian(mat3 &q, int i, int j);
    mat12 hessian(const q4& q);
    double otho_energy(const VectorXd& q);
    //double otho_energy(mat3 &q);
}
