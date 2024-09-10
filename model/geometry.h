#pragma once
#include "affine_body.h"
#include <ipc/distance/point_triangle.hpp>
#include <tuple>

struct Face {
    vec3 t0, t1, t2;
    Face(const vec3& t0, const vec3& t1, const vec3& t2)
        : t0(t0), t1(t1), t2(t2) {}
    Face(const AffineBody& c, unsigned triangle_id, bool use_line_search_increment = false, bool batch = false);
    vec3 normal() const;
    vec3 unit_normal() const;
};

struct Edge {
    vec3 e0, e1;
    Edge(const vec3& e0, const vec3& e1)
        : e0(e0), e1(e1) {}
    Edge(const AffineBody& c, unsigned eid, bool use_line_search_increment = false, bool batch = false);
};
scalar vf_distance(const vec3& v, const Face& f, ipc::PointTriangleDistanceType &pt_type);

std::tuple<scalar, ipc::PointTriangleDistanceType> vf_distance(const vec3& v, const Face& f);

scalar ee_distance(const Edge& ei, const Edge& ej);
vec3 vg_distance_gradient_x(const vec3& vertex);
scalar vg_distance(const vec3& vertex);
scalar E_ground(const vec3& v);
bool inside(const Face &f, const vec3 &p);
Face linerp(const Face &t_t1, const Face &t_t0, scalar t);
Edge linerp(const Edge& e_t1, const Edge& e_t0, scalar t);

mat3 rotation(scalar a, scalar b, scalar c);

inline mat3 cross_matrix(const vec3& a)
{
    mat3 ret;
    ret << 0, -a[2], a[1],
        a[2], 0, -a[0],
        -a[1], a[0], 0;
    return ret;
}
inline std::array<vec3, 4> skew(const vec3& a)
{
    return {
        -vec3(0, -a[2], a[1]),
        -vec3(a[2], 0, -a[0]),
        -vec3(-a[1], a[0], 0)
    };
}