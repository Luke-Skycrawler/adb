#include "collision.h"
#include "barrier.h"
#include <tight_inclusion/ccd.hpp>
using namespace barrier;


float vf_collision_time(const vec3 &x, const vec3 &p, const mat3 &q, const vec3 &p_next, const mat3 &q_next){
    // one-way collision
    // assert that (p, q) is collision-free and (p_next, q_next) is the state after penetration 
    vec3 initial_guess(q * x + p);
    float t = 0.0f;
    float distance = barrier::vf_distance(initial_guess);
    vec3 dd_dx(barrier::vf_distance_gradient_x(initial_guess));
    // distance derivative of x

    vec3 x_t = (q_next - q) * x + (p_next - p);
    t -= distance / (dd_dx.dot(x_t));
    // distance function should be linear, 
    // leading to an exact solution 

    return t;
}

float Cube::vf_collision_detect(const vec3 &dp, const mat3 &dq) {
    // TODO: ensure the updated q_next is collision-free
    float min_toi = 1.0f;
    for (int i = 0; i < 8; i++) {
        const vec3 v0(vertices()[i]);
        const vec3 &v((q_next - dq) * v0 + (p_next - dp));
        float d = vf_distance(v); 
        if (d < 0){
            float t = vf_collision_time(v0, p, A, p_next -dp, q_next-dq);
            if (t<min_toi){
                min_toi = t;
            }
        }
    }
    return min_toi;
}