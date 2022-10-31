#include "collision.h"
#include "barrier.h"
#define TIGHT_INCLUSION_DOUBLE
#include <tight_inclusion/ccd.hpp>
#include <tight_inclusion/interval_root_finder.hpp>

using namespace barrier;


double vf_collision_time(const vec3 &x, const vec3 &p, const mat3 &q, const vec3 &p_next, const mat3 &q_next){
    // one-way collision
    // assert that (p, q) is collision-free and (p_next, q_next) is the state after penetration 
    vec3 initial_guess(q * x + p);
    double t = 0.0f;
    double distance = barrier::vf_distance(initial_guess);
    vec3 dd_dx(barrier::vf_distance_gradient_x(initial_guess));
    // distance derivative of x

    vec3 x_t = (q_next - q) * x + (p_next - p);
    t -= distance / (dd_dx.dot(x_t));
    // distance function should be linear, 
    // leading to an exact solution 

    return t;
}

double Cube::vf_collision_detect(const vec3 &dp, const mat3 &dq) {
    // TODO: ensure the updated q_next is collision-free
    double min_toi = 1.0f;
    for (int i = 0; i < 8; i++) {
        const vec3 v0(vertices()[i]);
        const vec3 &v((q_next - dq) * v0 + (p_next - dp));
        double d = vf_distance(v); 
        if (d < 0){
            double t = vf_collision_time(v0, p, A, p_next -dp, q_next-dq);
            if (t<min_toi){
                min_toi = t;
            }
        }
    }
    return min_toi;
}

double vf_collision_detect(vec3 &v0, vec3 &v1, const Cube& c, int id){
    auto q(c.q_next - c.dq);
    auto p(c.p_next - c.dp);

    int a = Cube::indices[id * 3 + 0], 
        b = Cube::indices[id * 3 + 1], 
        _c = Cube::indices[id * 3 + 2]; 

    const vec3 &v0a(c.vertices()[a]),
        &v0b(c.vertices()[b]),
        &v0c(c.vertices()[_c]);
    
    vec3 va(q * v0a + p),
        vb(q * v0b + p),
        vc(q * v0c + p);

    vec3 va1(c.q_next * v0a + c.p_next),
        vb1(c.q_next * v0b + c.p_next),
        vc1(c.q_next * v0c + c.p_next);
    
    auto err = Eigen::Array3d(-1, -1, -1);
    double minimum_seperation = 1e-3, tolerance = 1e-6, t_max = 1;
    ticcd::Scalar toi = 1.0, output_tolerance;
    int max_itr = 1e6;
    std::vector<ticcd::Vector3> bounding_box;
    for (int i = 0; i < 8; i++) {
        bounding_box.push_back(c.vertices()[i] * 20.0);
    }
    const ticcd::Array3 err_vf(ticcd::get_numerical_error(bounding_box, true, false));

    bool _b = ticcd::vertexFaceCCD(
           v0, va, vb, vc,
           v1, va1, vb1, vc1,
           err_vf, minimum_seperation, toi, tolerance,
           t_max, max_itr, output_tolerance);
    return toi;
}