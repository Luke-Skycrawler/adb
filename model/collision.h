#include "cube.h"
#include "geometry.h"
#include <vector>
double vf_collision_time(const vec3& x, const vec3& p, const mat3& q, const vec3& p_next, const mat3& q_next);
double vf_collision_detect(vec3& p_t0, vec3& p_t1, const Cube& cube, int id);
// x: vertex position in the static frame;
// p: affine body translation
// q: affine matrix
// std::vector<std::pair<vec3, Face>> vf_colliding_set(const Cube & c1, const Cube &c2);
int vf_colliding_response(Cube& c1, Cube& c2);

int ee_colliding_response(Cube& ci, Cube& cj);
double ee_collision_detect(const Cube& ci, const Cube& cj, int eid_i, int eid_j);