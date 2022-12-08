#include "barrier.h"
using namespace std;

namespace barrier
{

    double barrier_second_derivative(double d)
    {
        if (d >= d_hat)
            return 0.0f;
        return -kappa * (2 * log(d / d_hat) + (d - d_hat) / d + (d - d_hat) * (2 / d + d_hat / d / d));
    }

    double barrier_derivative_d(double x)
    {
        if (x >= d_hat)
            return 0.0f;
        return -(x - d_hat) * kappa * (2 * log(x / d_hat) + (x - d_hat) / x);
    }

    // nebla_q = nebla f J
    vec3 barrier_gradient_x(const vec3 &vertex)
    {
        return vf_distance_gradient_x(vertex) * barrier_derivative_d(vg_distance(vertex));
    }

    VectorXd barrier_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        // return distance_gradient_q(vertex) * barrier_derivative_d(vf_distance(vertex));
        return barrier_gradient_x(vertex).adjoint() * x_jacobian_q(tilex);
    }

    MatrixXd x_jacobian_q(const vec3 &tilex)
    {
        // x in static frame
        Matrix<double, 3, 12> J;
        J << Matrix3d::Identity(3, 3), Matrix3d::Identity(3, 3) * tilex(0), Matrix3d::Identity(3, 3) * tilex(1), Matrix3d::Identity(3, 3) * tilex(2);
        return J;
    }

    inline VectorXd distance_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        return vf_distance_gradient_x(vertex).adjoint() * x_jacobian_q(tilex);
    }

    MatrixXd barrier_hessian_q(const vec3 &tilex, const vec3 &vertex)
    {
        // arg: colliding vertex
        auto g(distance_gradient_q(tilex, vertex));
        // because distance hessian is 0, the first term of J^T B' hess_d J is 0
        return barrier_second_derivative(vg_distance(vertex)) * g * g.adjoint();
    }

    
};
