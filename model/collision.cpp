#include "collision.h"
#include "barrier.h"
#define TIGHT_INCLUSION_DOUBLE
#include <tight_inclusion/ccd.hpp>
#include <tight_inclusion/interval_root_finder.hpp>
#include <iostream>
#include <spdlog/spdlog.h>
#include "../view/global_variables.h"

using namespace barrier;
using namespace std;

double vf_collision_detect(vec3& p_t0, vec3& p_t1, const AffineBody& c, int id)
{
    Face f_t0(c, id, false), f_t1(c, id, true);
    ticcd::Scalar toi = 1.0, output_tolerance;
    std::vector<ticcd::Vector3> bounding_box;
   
    double min_distance = 1e-6, tmax = 1, adjusted_tolerance = 1e-6;
    long max_iterations = 1e6;

    bool is_impacting = ticcd::vertexFaceCCD(
        p_t0, f_t0.t0, f_t0.t1, f_t0.t2, p_t1, f_t1.t0, f_t1.t1, f_t1.t2,
        Eigen::Array3d::Constant(-1), // rounding error (auto)
        min_distance, // minimum separation distance
        toi, // time of impact
        adjusted_tolerance, // delta
        tmax, // maximum time to check
        max_iterations, // maximum number of iterations
        output_tolerance, // delta_actual
        true);
    if (toi < 1.0) {
        spdlog::warn("vf collision detected at toi = {}", toi);
    }
    return toi;
}

double ee_collision_detect(const AffineBody& ci, const AffineBody& cj, int eid_i, int eid_j)
{
    ticcd::Scalar toi = 1.0, output_tolerance;
    double min_distance = 1e-6, tmax = 1, adjusted_tolerance = 1e-6;
    long max_iterations = 1e6;
    Edge ei_t1(ci, eid_i, true), ej_t1(cj, eid_j, true), ei_t0(ci, eid_i), ej_t0(cj, eid_j);

    bool is_impacting = ticcd::edgeEdgeCCD(
        ei_t0.e0, ei_t0.e1, ej_t0.e0, ej_t0.e1, ei_t1.e0, ei_t1.e1, ej_t1.e0, ej_t1.e1,
        Eigen::Array3d::Constant(-1), // rounding error (auto)
        min_distance, // minimum separation distance
        toi, // time of impact
        adjusted_tolerance, // delta
        tmax, // maximum time to check
        max_iterations, // maximum number of iterations
        output_tolerance, // delta_actual
        true);
    if (toi < 1.0) {
        spdlog::warn("ee collision detected at toi = {}", toi);
    }
    else
        toi = 1.0;
    return toi;
}

double collision_time(AffineBody& c, int i)
{
    double toi = 1.0;
    const vec3 v_t2(c.vt2(i));
    const vec3 v_t1(c.vt1(i));

    double d2 = vg_distance(v_t2);
    double d1 = vg_distance(v_t1);
    assert(d1 > 0);
    if (d2 < 0) {

        double t = d1 / (d1 - d2);
        auto vtoi = v_t2 * (t * 0.8) + v_t1 * (1 - t * 0.8);
        double dtoi = vg_distance(vtoi);
        spdlog::error("dtoi = {}, d0 = {}, d1 = {}, toi = {}", dtoi, d1, d2, t);

        assert(dtoi > 0.0);
        assert(t > 0.0 && t < 1.0);
        toi = min(toi, t);
    }
    return toi;
};
