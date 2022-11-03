#pragma once
#include "cube.h"

struct Face {
    vec3 t0, t1, t2;
    Face(const vec3& t0, const vec3& t1, const vec3& t2)
        : t0(t0), t1(t1), t2(t2) {}
    Face(const Cube& c, int triangle_id, bool use_line_search_increment = false);
    vec3 normal();
    vec3 unit_normal();
};

double vf_distance(const vec3& vertex, const Cube& c, int id);
