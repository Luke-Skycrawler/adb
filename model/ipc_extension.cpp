#include "ipc_extension.h"
namespace ipc {

    Vector2<float> point_triangle_closest_point(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2){
            return ipc::point_triangle_closest_point(p.cast<double>(), t0.cast<double>(), t1.cast<double>(), t2.cast<double>()).cast<float>();
        }
    Eigen::Matrix<float, 3, 2> point_triangle_tangent_basis(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2){
            return point_triangle_tangent_basis(p.cast<double>(), t0.cast<double>(), t1.cast<double>(), t2.cast<double>()).cast<float>();
        }


    float point_edge_closest_point(
        const Eigen::Ref<const VectorMax3f>& p,
        const Eigen::Ref<const VectorMax3f>& e0,
        const Eigen::Ref<const VectorMax3f>& e1){
            return ipc::point_edge_closest_point(p.cast<double>(), e0.cast<double>(), e1.cast<double>());
        }
    MatrixMax<float, 3, 2> point_edge_tangent_basis(
        const Eigen::Ref<const VectorMax3f>& p,
        const Eigen::Ref<const VectorMax3f>& e0,
        const Eigen::Ref<const VectorMax3f>& e1) {
            return ipc::point_edge_tangent_basis(p.cast<double>(), e0.cast<double>(), e1.cast<double>()).cast<float>();
        }

    MatrixMax<float, 3, 2> point_point_tangent_basis(
        const Eigen::Ref<const VectorMax3f>& p0,
        const Eigen::Ref<const VectorMax3f>& p1) {
            return ipc::point_point_tangent_basis(p0.cast<double>(), p1.cast<double>()).cast<float>();
        }

    Vector2<float> edge_edge_closest_point(
        const Eigen::Ref<const Eigen::Vector3f>& ea0,
        const Eigen::Ref<const Eigen::Vector3f>& ea1,
        const Eigen::Ref<const Eigen::Vector3f>& eb0,
        const Eigen::Ref<const Eigen::Vector3f>& eb1){
            return ipc::edge_edge_closest_point(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>()).cast<float>();
        }
    Eigen::Matrix<float, 3, 2> edge_edge_tangent_basis(
        const Eigen::Ref<const Eigen::Vector3f>& ea0,
        const Eigen::Ref<const Eigen::Vector3f>& ea1,
        const Eigen::Ref<const Eigen::Vector3f>& eb0,
        const Eigen::Ref<const Eigen::Vector3f>& eb1){
            return ipc::edge_edge_tangent_basis(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>()).cast<float>();
        }


    PointTriangleDistanceType point_triangle_distance_type(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2){
            return ipc::point_triangle_distance_type(p.cast<double>(), t0.cast<double>(), t1.cast<double>(), t2.cast<double>());
        }
        
    float point_triangle_distance(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2,
        PointTriangleDistanceType dtype) {
            return ipc::point_triangle_distance(p.cast<double>(), t0.cast<double>(), t1.cast<double>(), t2.cast<double>(), dtype);
        }
    Vector12f point_triangle_distance_gradient(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2,
        PointTriangleDistanceType dtype) {
            return ipc::point_triangle_distance_gradient(p.cast<double>(), t0.cast<double>(), t1.cast<double>(), t2.cast<double>(), dtype).cast<float>();
        }
    Matrix12f point_triangle_distance_hessian(
        const Eigen::Ref<const Eigen::Vector3f>& p,
        const Eigen::Ref<const Eigen::Vector3f>& t0,
        const Eigen::Ref<const Eigen::Vector3f>& t1,
        const Eigen::Ref<const Eigen::Vector3f>& t2,
        PointTriangleDistanceType dtype) {
            return ipc::point_triangle_distance_hessian(p.cast<double>(), t0.cast<double>(), t1.cast<double>(), t2.cast<double>(), dtype).cast<float>();
        }

Vector12f edge_edge_distance_gradient(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    EdgeEdgeDistanceType dtype) {
        return ipc::edge_edge_distance_gradient(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>(), dtype).cast<float>();
    }
Matrix12f edge_edge_distance_hessian(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    EdgeEdgeDistanceType dtype) {
        return ipc::edge_edge_distance_hessian(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>(), dtype).cast<float>();
    }

float edge_edge_mollifier(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    const double eps_x) {
        return ipc::edge_edge_mollifier(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>(), eps_x);
    }


Vector12f edge_edge_mollifier_gradient(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    const double eps_x) {
        return ipc::edge_edge_mollifier_gradient(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>(), eps_x).cast<float>();
    }
   
Matrix12f edge_edge_mollifier_hessian(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    const double eps_x) {
        return ipc::edge_edge_mollifier_hessian(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>(), eps_x).cast<float>();
    }
EdgeEdgeDistanceType edge_edge_distance_type(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1) {
        return ipc::edge_edge_distance_type(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>());
    }
float edge_edge_distance(
    const Eigen::Ref<const Eigen::Vector3f>& ea0,
    const Eigen::Ref<const Eigen::Vector3f>& ea1,
    const Eigen::Ref<const Eigen::Vector3f>& eb0,
    const Eigen::Ref<const Eigen::Vector3f>& eb1,
    EdgeEdgeDistanceType dtype) {
        return ipc::edge_edge_distance(ea0.cast<double>(), ea1.cast<double>(), eb0.cast<double>(), eb1.cast<double>(), dtype);
    }


};