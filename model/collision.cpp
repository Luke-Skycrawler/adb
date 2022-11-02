#include "collision.h"
#include "barrier.h"
#define TIGHT_INCLUSION_DOUBLE
#include <tight_inclusion/ccd.hpp>
#include <tight_inclusion/interval_root_finder.hpp>
#include <iostream>
using namespace barrier;
using namespace std;

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

double vf_collision_detect(vec3 &p_t0, vec3 &p_t1, const Cube& c, int id){
    auto q(c.q_next - c.dq);
    auto p(c.p_next - c.dp);

    int a = Cube::indices[id * 3 + 0], 
        b = Cube::indices[id * 3 + 1], 
        _c = Cube::indices[id * 3 + 2]; 

    const vec3 &v0a(c.vertices()[a]),
        &v0b(c.vertices()[b]),
        &v0c(c.vertices()[_c]);
    
    vec3 t0_t1(q * v0a + p),
        t1_t1(q * v0b + p),
        t2_t1(q * v0c + p);

    vec3 t0_t0(c.q_next * v0a + c.p_next),
        t1_t0(c.q_next * v0b + c.p_next),
        t2_t0(c.q_next * v0c + c.p_next);
    
    ticcd::Scalar toi = 1.0, output_tolerance;
    std::vector<ticcd::Vector3> bounding_box;
    for (int i = 0; i < 8; i++) {
        bounding_box.push_back(c.vertices()[i] * 20.0);
    }
    const ticcd::Array3 err_vf(ticcd::get_numerical_error(bounding_box, true, false));
    double min_distance = 1e-4, tmax = 1, adjusted_tolerance = 1e-6;
    long max_iterations = 1e2;

    bool is_impacting = ticcd::vertexFaceCCD(
        p_t0, t0_t0, t1_t0, t2_t0, p_t1, t0_t1, t1_t1, t2_t1,
        Eigen::Array3d::Constant(-1), // rounding error (auto)
        min_distance,                 // minimum separation distance
        toi,                          // time of impact
        adjusted_tolerance,           // delta
        tmax,                         // maximum time to check
        max_iterations,               // maximum number of iterations
        output_tolerance,             // delta_actual
        true);
    if (toi < 1.0) {
        cout << "lowest level collision detected" << endl;
    }
    return toi;
}