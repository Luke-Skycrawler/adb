#pragma once
#include <Eigen/Dense>
#ifndef CUDA_SOURCE
#define func
#endif
using scalar = double;
using mat3 = Eigen::Matrix<scalar, 3, 3>;
using vec3 = Eigen::Vector<scalar, 3>;
using q4 = std::array<vec3, 4>;
// using i4 = std::array<int, 4>;
// using q4 = Eigen::Matrix<scalar, 4, 3>;
using i4 = Eigen::Vector4i;
using vec12 = Eigen::Vector<scalar, 12>;
using mat12 = Eigen::Matrix<scalar, 12, 12>;

struct Face {
    vec3 t0, t1, t2;
    func Face(const vec3& t0, const vec3& t1, const vec3& t2)
        : t0(t0), t1(t1), t2(t2) {}
    // Face(const AffineBody& c, int triangle_id, bool use_line_search_increment = false, bool batch = false);
    inline func vec3 normal() const
    {
        return (t0 - t1).cross(t0 - t2);
    }

    inline func vec3 unit_normal() const
    {
        auto n = (t0 - t1).cross(t0 - t2);
        return n / sqrt(n.dot(n));
    }
    func Face() {}
};

struct Edge {
    vec3 e0, e1;
    func Edge(const vec3& e0, const vec3& e1)
        : e0(e0), e1(e1) {}
    func Edge() {}
    // Edge(const AffineBody& c, int eid, bool use_line_search_increment = false, bool batch = false);
};