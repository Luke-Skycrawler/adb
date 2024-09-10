#include "barrier.h"
#include "geometry.h"

using namespace std;
using namespace Eigen;

scalar barrier::d_hat = 1e-4, barrier::kappa = 1e-3;
scalar barrier::d_sqrt = 1e-2;
namespace barrier
{
scalar barrier_function(scalar d)
{
    if (d >= d_hat) return 0.0;
    return kappa * -(d - d_hat) * (d - d_hat) * log(d / d_hat) / (d_hat * d_hat);
    }
    scalar barrier_second_derivative(scalar d)
    {
        if (d >= d_hat)
            return 0.0f;
        return -kappa * (2 * log(d / d_hat) + (d - d_hat) / d + (d - d_hat) * (2 / d + d_hat / d / d)) / (d_hat * d_hat);
    }

    scalar barrier_derivative_d(scalar x)
    {
        if (x >= d_hat)
            return 0.0f;
        return -(x - d_hat) * kappa * (2 * log(x / d_hat) + (x - d_hat) / x) / (d_hat * d_hat);
    }

    // nabla_q = nabla f J
    vec3 barrier_gradient_x(const vec3 &vertex)
    {
        scalar d = vg_distance(vertex);
        return vg_distance_gradient_x(vertex) * barrier_derivative_d(d * d) * 2 * d;
    }

    Vector<scalar, -1> barrier_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        return barrier_gradient_x(vertex).transpose() * x_jacobian_q(tilex);
    }

    Matrix<scalar, -1, -1> x_jacobian_q(const vec3 &tilex)
    {
        // x in static frame
        Matrix<scalar, 3, 12> J;
        //J.setZero(3, 12);
        //J.block<3, 3>(0, 0) = Matrix<scalar, 3, 3>::Identity(3, 3);
        //J.block<3, 3>(0, 3) = Matrix<scalar, 3, 3>::Identity(3, 3) * tilex(0);
        //J.block<3, 3>(0, 6) = Matrix<scalar, 3, 3>::Identity(3, 3) * tilex(1);
        //J.block<3, 3>(0, 9) = Matrix<scalar, 3, 3>::Identity(3, 3) * tilex(2);

        J << Matrix<scalar, 3, 3>::Identity(3, 3), Matrix<scalar, 3, 3>::Identity(3, 3) * tilex(0), Matrix<scalar, 3, 3>::Identity(3, 3) * tilex(1), Matrix<scalar, 3, 3>::Identity(3, 3) * tilex(2);
        return J;
    }

    inline Vector<scalar, -1> distance_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        scalar d = vg_distance(vertex);
        return vg_distance_gradient_x(vertex).transpose() * x_jacobian_q(tilex) * d * 2;
    }

    Matrix<scalar, -1, -1> barrier_hessian_q(const vec3 &tilex, const vec3 &vertex)
    {
        // arg: colliding vertex
        auto g(distance_gradient_q(tilex, vertex));
        // because distance hessian is 0, the first term of J^T B' hess_d J is 0
        scalar d = vg_distance(vertex);
        return barrier_second_derivative(d * d) * g * g.transpose();
    }

    
};
