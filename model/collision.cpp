#include "collision.h"
#include "barrier.h"

using namespace barrier;


float vf_collision_time(const vec3 &x, const vec3 &p, const mat3 &q, const vec3 &p_next, const vec3 &q_next){
    // one-way collision
    // assert that (p, q) is collision-free and (p_next, q_next) is the state after penetration 
    vec3 initial_guess(x * q + p);
    float t = 0.0f;
    float distance = barrier::vf_distance(initial_guess);
    vec3 dd_dx(barrier::vf_distance_gradient_x(initial_guess));
    // distance derivative of x

    vec3 x_t((q_next - q) * x + (p_next - p));
    t -= distance / (dd_dx.dot(x_t));
    // distance function should be linear, 
    // leading to an exact solution 
    
    return t;
}