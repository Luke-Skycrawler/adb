#pragma once
#include "affine_body.h"
#include "geometry.h"

// x: vertex position in the static frame;
// p: affine body translation
// q: affine matrix
#ifdef _TIGHT_INCLUSION_ENABLE_

scalar vf_collision_detect(vec3& p_t0, vec3& p_t1, const AffineBody& cube, int id);
scalar vf_collision_detect(vec3& p_t0, vec3& p_t1, const Face &f_t0, const Face &f_t1);
scalar ee_collision_detect(const Edge& ei_t0, const Edge& ej_t0, const Edge& ei_t1, const Edge& ej_t1);
scalar ee_collision_detect(const AffineBody& ci, const AffineBody& cj, int eid_i, int eid_j);
#endif
scalar collision_time(AffineBody& c, int i);
scalar pt_collision_time(
    const vec3& p0,
    const Face& t0,
    const vec3& p1,
    const Face& t1);
scalar ee_collision_time(
    const Edge& ei0,
    const Edge& ej0,
    const Edge& ei1,
    const Edge& ej1);

Eigen::Vector<scalar, 4> det_polynomial(const mat3& a, const mat3& b);
// only exported for testing

