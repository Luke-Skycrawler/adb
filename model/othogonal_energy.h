#pragma once
#include <Eigen/Eigen>

namespace othogonal_energy{

    using namespace Eigen;
    using vec3 = Vector3d;
    using mat3 = Matrix3d;
    
    //vec3 grad(mat3 &q, int i);
    //mat3 grad(mat3 &q);
    VectorXd grad(vec3 q[]);
    //mat3 hessian(mat3 &q, int i, int j);
    MatrixXd hessian(vec3 q[]);
    double otho_energy(const VectorXd& q);
    //double otho_energy(mat3 &q);
}
