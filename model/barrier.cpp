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
    vec3 vf_distance_gradient_x(const vec3 &vertex)
    {
        // face = grond plane y = 0.5
        return vec3(0.0f, 1.0f, 0.0f);
    }

    /*void vf_distance_gradient_x(const vec3 &vertex, const Face &f, double g[12])
    {
        auto_gen::point_plane_distance_gradient(
            vertex(0),
            vertex(1),
            vertex(2),
            f.t0(0), 
            f.t0(1), 
            f.t0(2), 
            f.t1(0), 
            f.t1(1), 
            f.t1(2), 
            f.t2(0), 
            f.t2(1), 
            f.t2(2), 
            g
        ); 
    }*/

    double vf_distance(const vec3 &vertex)
    {
        // ground plane y = -0.5
        return vertex(1) + 0.5;
    }


    vec3 barrier_gradient_x(const vec3 &vertex)
    {
        return vf_distance_gradient_x(vertex) * barrier_derivative_d(vf_distance(vertex));
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

    VectorXd distance_gradient_q(const vec3 &tilex, const vec3 &vertex)
    {
        // VectorXd g(vf_distance_gradient_x(vertex).adjoint() * x_jacobian_q(tilex));
        // g.segment<3>(0).setZero();
        // return g;
        return vf_distance_gradient_x(vertex).adjoint() * x_jacobian_q(tilex);
    }

    MatrixXd barrier_hessian_q(const vec3 &tilex, const vec3 &vertex)
    {
        // arg: colliding vertex
        auto g(distance_gradient_q(tilex, vertex));
        return barrier_second_derivative(vf_distance(vertex)) * g * g.adjoint();
    }

    
};
