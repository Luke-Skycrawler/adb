#include "barrier.h"
using namespace std;

namespace barrier
{

    // nebla_q = nebla f J
    vec3 vf_distance_gradient_x(const vec3 &vertex)
    {
        // edge-face detection, no need for face argument for now
        return vec3(0.0f, 1.0f, 0.0f);
    }

    float vf_distance(const vec3 &vertex)
    {
        // ground plane y = -0.5
        return vertex(1) + 0.5f;
    }

    float barrier_second_derivative(float d)
    {
        if (d >= d_hat)
            return 0.0f;
        return -kappa * (2 * log(d / d_hat) + (d - d_hat) / d + (d - d_hat) * (2 / d + d_hat / d / d));
    }

    float barrier_derivative_d(float x)
    {
        if (x >= d_hat)
            return 0.0f;
        return -(x - d_hat) * kappa * (2 * log(x / d_hat) + (x - d_hat) / x);
    }

    vec3 barrier_gradient_x(const vec3 &vertex)
    {
        return vf_distance_gradient_x(vertex) * barrier_derivative_d(vf_distance(vertex));
    }

    VectorXf barrier_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        // return distance_gradient_q(vertex) * barrier_derivative_d(vf_distance(vertex));
        return barrier_gradient_x(vertex).adjoint() * x_jacobian_q(tilex);
    }

    MatrixXf x_jacobian_q(const vec3 &tilex)
    {
        // x in static frame
        Matrix<float, 3, 12> J;
        J << Matrix3f::Identity(3, 3), Matrix3f::Identity(3, 3) * tilex(0), Matrix3f::Identity(3, 3) * tilex(1), Matrix3f::Identity(3, 3) * tilex(2);
        return J;
    }

    VectorXf distance_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        // VectorXf g(vf_distance_gradient_x(vertex).adjoint() * x_jacobian_q(tilex));
        // g.segment<3>(0).setZero();
        // return g;
        return vf_distance_gradient_x(vertex).adjoint() * x_jacobian_q(tilex);
    }

    MatrixXf barrier_hessian_q(const vec3 &tilex, const vec3 &vertex)
    {
        // arg: colliding vertex
        auto g(distance_gradient_q(tilex, vertex));
        return barrier_second_derivative(vf_distance(vertex)) * g * g.adjoint();
    }

};
