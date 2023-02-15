#pragma once
#include "affine_body.h"
#include <ipc/distance/point_triangle.hpp>


double vf_distance(const vec3& v, const Face& f, ipc::PointTriangleDistanceType &pt_type);
double ee_distance(const Edge& ei, const Edge& ej);
vec3 vg_distance_gradient_x(const vec3& vertex);
double vg_distance(const vec3& vertex);
double E_ground(const vec3& v);
bool inside(const Face &f, const vec3 &p);
Face linerp(const Face &t_t1, const Face &t_t0, double t);
Edge linerp(const Edge& e_t1, const Edge& e_t0, double t);

mat3 rotation(double a, double b, double c);

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