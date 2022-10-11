#pragma once
#include <Eigen/Eigen>

using namespace Eigen;
using vec3 = Vector3f;
using mat3 = Matrix3f;

struct Cube {
    mat3 A, q_dot, q_next;
    vec3 translation, vc, dimensions;   // not used 
    float mass;
    Cube(float scale = 1.0f) : vc(vc), mass(1.0f) {
        A.setIdentity(3, 3);
        dimensions.setOnes(3, 1);
        dimensions *= scale;
    }
};