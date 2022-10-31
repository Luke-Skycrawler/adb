#pragma once
#include "cube.h"
#include <vector>

// void barrier_forces_p_q(vector<Cube> &cubes);
namespace barrier
{
    const double d_hat = 0.01f, kappa = 1e8f;
    MatrixXd barrier_hessian_q(const vec3 &x, const vec3 &vertex);
    VectorXd distance_gradient_q(const vec3 &x, const vec3 &vertex);
    MatrixXd x_jacobian_q(const vec3 &x);
    vec3 barrier_gradient_x(const vec3 &vertex);
    VectorXd barrier_gradient_q(const vec3 &x, const vec3 &vertex);
    vec3 vf_distance_gradient_x(const vec3 &vertex);
    double vf_distance(const vec3 &vertex);
};
