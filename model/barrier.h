#pragma once
#include "affine_body.h"

namespace barrier {
extern scalar d_hat, kappa;
extern scalar d_sqrt;
Eigen::Matrix<scalar, -1, -1> barrier_hessian_q(const vec3& x, const vec3& vertex);
Eigen::Matrix<scalar, -1, -1> x_jacobian_q(const vec3& x);
vec3 barrier_gradient_x(const vec3& vertex);
Eigen::Vector<scalar, -1> barrier_gradient_q(const vec3& x, const vec3& vertex);
scalar barrier_second_derivative(scalar d);
scalar barrier_derivative_d(scalar x);
scalar barrier_function(scalar d);
};
