#pragma once
#include "cube.h"

struct Face {
    vec3 t0, t1, t2;
    Face(const vec3& t0, const vec3& t1, const vec3& t2)
        : t0(t0), t1(t1), t2(t2) {}
    Face(const Cube& c, int triangle_id, bool use_line_search_increment = false);
    vec3 normal() const;
    vec3 unit_normal() const;
};

using Edge = std::pair<vec3, vec3>;
double vf_distance(const vec3& vertex, const Cube& c, int id);

double vf_distance(const vec3& v, const Face& f);

void ee_distance(const Edge &ei, const Edge &ej);

// Symbolically generated derivatives;
namespace autogen {
    void point_plane_distance_gradient(
        double v01,
        double v02,
        double v03,
        double v11,
        double v12,
        double v13,
        double v21,
        double v22,
        double v23,
        double v31,
        double v32,
        double v33,
        double g[12]);

    void point_plane_distance_hessian(
        double v01,
        double v02,
        double v03,
        double v11,
        double v12,
        double v13,
        double v21,
        double v22,
        double v23,
        double v31,
        double v32,
        double v33,
        double H[144]);
    
    void line_line_distance_gradient(
        double v01,
        double v02,
        double v03,
        double v11,
        double v12,
        double v13,
        double v21,
        double v22,
        double v23,
        double v31,
        double v32,
        double v33,
        double g[12]);

void line_line_distance_hessian(
        double v01,
        double v02,
        double v03,
        double v11,
        double v12,
        double v13,
        double v21,
        double v22,
        double v23,
        double v31,
        double v32,
        double v33,
        double H[144]);
} // namespace autogen

