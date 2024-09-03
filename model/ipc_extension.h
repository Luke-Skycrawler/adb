#pragma once
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/friction/smooth_friction_mollifier.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>

namespace ipc {
    using VectorMax3f = VectorMax3<float>;
    using Vector12f = Vector<float, 12>;
    using Matrix12f = Eigen::Matrix<float, 12, 12>;

    Vector2<float> point_triangle_closest_point(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2);
    Eigen::Matrix<float, 3, 2> point_triangle_tangent_basis(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2);


    float point_edge_closest_point(
        const Eigen::Ref<const VectorMax3f>& p,
        const Eigen::Ref<const VectorMax3f>& e0,
        const Eigen::Ref<const VectorMax3f>& e1);
    MatrixMax<float, 3, 2> point_edge_tangent_basis(
        const Eigen::Ref<const VectorMax3f>& p,
        const Eigen::Ref<const VectorMax3f>& e0,
        const Eigen::Ref<const VectorMax3f>& e1);

    MatrixMax<float, 3, 2> point_point_tangent_basis(
        const Eigen::Ref<const VectorMax3f>& p0,
        const Eigen::Ref<const VectorMax3f>& p1);

    Vector2<float> edge_edge_closest_point(
        const Eigen::Ref<const Eigen::Vector3f>& ea0,
        const Eigen::Ref<const Eigen::Vector3f>& ea1,
        const Eigen::Ref<const Eigen::Vector3f>& eb0,
        const Eigen::Ref<const Eigen::Vector3f>& eb1);
    Eigen::Matrix<float, 3, 2> edge_edge_tangent_basis(
        const Eigen::Ref<const Eigen::Vector3f>& ea0,
        const Eigen::Ref<const Eigen::Vector3f>& ea1,
        const Eigen::Ref<const Eigen::Vector3f>& eb0,
        const Eigen::Ref<const Eigen::Vector3f>& eb1);


    PointTriangleDistanceType point_triangle_distance_type(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2);
        
    float point_triangle_distance(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2,
        PointTriangleDistanceType dtype = PointTriangleDistanceType::AUTO);
    Vector12f point_triangle_distance_gradient(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2,
        PointTriangleDistanceType dtype = PointTriangleDistanceType::AUTO);
    Matrix12f point_triangle_distance_hessian(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2,
        PointTriangleDistanceType dtype = PointTriangleDistanceType::AUTO);

EdgeEdgeDistanceType edge_edge_distance_type(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1);

float edge_edge_distance(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    EdgeEdgeDistanceType dtype = EdgeEdgeDistanceType::AUTO);

Vector12f edge_edge_distance_gradient(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    EdgeEdgeDistanceType dtype = EdgeEdgeDistanceType::AUTO);
Matrix12f edge_edge_distance_hessian(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    EdgeEdgeDistanceType dtype = EdgeEdgeDistanceType::AUTO);

float edge_edge_mollifier(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    const double eps_x);


Vector12f edge_edge_mollifier_gradient(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    const double eps_x);
   
Matrix12f edge_edge_mollifier_hessian(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    const double eps_x);

};