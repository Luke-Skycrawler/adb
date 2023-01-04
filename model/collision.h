#include "cube.h"
#include "geometry.h"
#include <vector>
#include <array>
#include "../view/global_variables.h"
//double vf_collision_time(const vec3& x, const vec3& p, const mat3& q, const vec3& p_next, const mat3& q_next);
double vf_collision_detect(vec3& p_t0, vec3& p_t1, const Cube& cube, int id);
// x: vertex position in the static frame;
// p: affine body translation
// q: affine matrix
// std::vector<std::pair<vec3, Face>> vf_colliding_set(const Cube & c1, const Cube &c2);
// int vf_colliding_response(Cube& c1, Cube& c2);

// int ee_colliding_response(Cube& ci, Cube& cj);
//int vf_colliding_response(int c1, int c2);

//int ee_colliding_response(int ci, int cj);
double ee_collision_detect(const Cube& ci, const Cube& cj, int eid_i, int eid_j);

void ipc_term(
    std::array<vec3, 4> pt, std::array<int, 4> ij,
    std::vector<HessBlock> &triplets,
    Vector<double, 12>& grad_p, Vector<double, 12>& grad_t
    // , Matrix<double, 12 , 12> &off_diag
    // , vector<Cube> &cubes
);
void ipc_term_ee(Matrix<double, 12, 12>& hess_0, Matrix<double, 12, 12>& hess_1, Vector<double, 12>& grad_0, Vector<double, 12>& grad_1, std::array<vec3, 4> ee, std::array<int, 4> ij);