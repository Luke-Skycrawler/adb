#pragma once
#include "affine_body.h"
#include <ipc/distance/point_triangle.hpp>


double vf_distance(const vec3& v, const Face& f, ipc::PointTriangleDistanceType &pt_type);
double ee_distance(const Edge& ei, const Edge& ej);
vec3 vg_distance_gradient_x(const vec3& vertex);
double vg_distance(const vec3& vertex);
double E_ground(const vec3& v);
