#pragma once
#include "cube.h"
#include <vector>
#include "geometry.h"

namespace barrier {
const double d_hat = 0.01, kappa = 1e9;
MatrixXd barrier_hessian_q(const vec3& x, const vec3& vertex);
MatrixXd x_jacobian_q(const vec3& x);
vec3 barrier_gradient_x(const vec3& vertex);
VectorXd barrier_gradient_q(const vec3& x, const vec3& vertex);
double barrier_second_derivative(double d);
double barrier_derivative_d(double x);
};
