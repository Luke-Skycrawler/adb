#pragma once
#include "cube.h"
#include <vector>

// void barrier_forces_p_q(vector<Cube> &cubes);
namespace barrier{
    const float d_hat = 0.04f, kappa = 1e8f;
    MatrixXf barrier_hessian_q(const vec3 &x, const vec3 &vertex);
    VectorXf distance_gradient_q(const vec3 &x, const vec3 &vertex);
    MatrixXf x_jacobian_q(const vec3 &x);
    vec3 barrier_gradient_x(const vec3 &vertex);
    VectorXf barrier_gradient_q(const vec3 &x, const vec3 &vertex);
    vec3 vf_distance_gradient_x(const vec3 &vertex);
    float vf_distance(const vec3 &vertex);
};
