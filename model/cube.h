#pragma once
#include <Eigen/Eigen>
#include <vector>
using namespace Eigen;
using vec3 = Vector3f;
using mat3 = Matrix3f;

struct Cube {
    mat3 A, q_dot, q_next;
    vec3 p, p_next, p_dot, dimensions, f, tau;   // not used 
    float mass, scale;
    static const vec3* vertices() {
        static const vec3 v[] ={
            vec3(-0.5f, -0.5f, -0.5f),
            vec3(-0.5f, -0.5f, 0.5f),
            vec3(-0.5f, 0.5f, -0.5f),
            vec3(-0.5f, 0.5f, 0.5f),
            vec3(0.5f, -0.5f, -0.5f),
            vec3(0.5f, -0.5f, 0.5f),
            vec3(0.5f, 0.5f, -0.5f),
            vec3(0.5f, 0.5f, 0.5f)
        };
        return v;
    }
    // FIXME: probably switch to a static function
    Cube(float scale = 1.0f): mass(1.0f), scale(scale), p(0.0f, 0.0f, 0.0f), p_dot(0.0f, 0.0f, 0.0f) {
        A.setIdentity(3, 3);
        dimensions.setOnes(3, 1);
        dimensions *= scale;
    }
    float vf_collision_detect(const vec3 &dp, const mat3 &dq);
};