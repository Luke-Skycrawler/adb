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
    Vector3d _v0 = v0.cast<double>(),
        _v1 = v1.cast<double>(),
        _va = va.cast<double>(),
        _vb = vb.cast<double>(),
        _vc = vc.cast<double>(),
        _va1 = va1.cast<double>(),
        _vb1 = vb1.cast<double>(),
        _vc1 = vc1.cast<double>();

    int max_itr = 1e6;
    //bool _b = ticcd::vertexFaceCCD(
    //        _v0, _va, _vb, _vc,
    //        _v1, _va1, _vb1, _vc1,
    //        err, minimum_seperation, toi, tolerance,
    //        t_max, max_itr, output_tolerance);
    return toi;
}