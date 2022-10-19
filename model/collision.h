#include "cube.h"

float vf_collision_time(const vec3 &x, const vec3 &p, const mat3 &q, const vec3 &p_next, const mat3 &q_next);
// x: vertex position in the static frame;
// p: affine body translation 
// q: affine matrix

