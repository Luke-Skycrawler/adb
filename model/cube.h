#pragma once
#include <Eigen/Eigen>

using namespace Eigen;
using vec3 = Vector3f;
using mat3 = Matrix3f;

struct Cube {
    vec3 a_col[3];
    vec3 translation;
    vec3 q_dot[3];
    vec3 vc;
    vec3 dimensions;
    Cube(float scale = 1.0f) : vc(vc) {
        mat3 A;
        A.setIdentity(3, 3);
        for (int i = 0; i < 3; i++) {
            a_col[i] = A.col(i);
        }
        dimensions.setOnes(3, 1);
        dimensions *= scale;
    }
};