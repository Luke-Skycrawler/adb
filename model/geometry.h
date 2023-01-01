#pragma once
#include "cube.h"


double vf_distance(const vec3& vertex, const Cube& c, int id);
double vf_distance(const vec3& v, const Face& f);
double ee_distance(const Edge& ei, const Edge& ej);
vec3 vg_distance_gradient_x(const vec3& vertex);
double vg_distance(const vec3& vertex);

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
