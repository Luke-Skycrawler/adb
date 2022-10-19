#include "collision.h"
#include "barrier.h"

using namespace barrier;


float vf_collision_time(const vec3 &x, const vec3 &p, const mat3 &q, const vec3 &p_next, const mat3 &q_next){
    // one-way collision
    // assert that (p, q) is collision-free and (p_next, q_next) is the state after penetration 
    vec3 initial_guess(q * x + p);
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

float Cube::vf_collision_detect(const vec3 &dp, const mat3 &dq) {
    // TODO: ensure the updated q_next is collision-free
    for (int i = 0; i < 8; i++) {
        const vec3 &v((q_next - dq) * vertices()[i] + (p_next - dp));
        float d = vf_distance(v); 
        if (d < 0){
            float t = vf_collision_time(vertices()[i], p, A, p_next, q_next);
            toi[i] = t;
        }
        else {
            toi[i] = 1.0f;
        }
    }
    return *min_element(toi.begin(), toi.end());
}