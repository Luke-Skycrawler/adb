#pragma once
#include "cube.h"
#include <vector>
#include "geometry.h"

namespace barrier {
extern scalar d_hat, kappa;
extern scalar d_sqrt;
MatrixXd barrier_hessian_q(const vec3& x, const vec3& vertex);
MatrixXd x_jacobian_q(const vec3& x);
vec3 barrier_gradient_x(const vec3& vertex);
VectorXd barrier_gradient_q(const vec3& x, const vec3& vertex);
scalar barrier_second_derivative(scalar d);
scalar barrier_derivative_d(scalar x);
scalar barrier_function(scalar d);
};
