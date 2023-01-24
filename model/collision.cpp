#include "collision.h"
#include "barrier.h"
#define TIGHT_INCLUSION_DOUBLE
#include <tight_inclusion/ccd.hpp>
#include <tight_inclusion/interval_root_finder.hpp>
#include <iostream>
#include <spdlog/spdlog.h>

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
        if(dtoi < 0.0)
            spdlog::error("dtoi = {}, d0 = {}, d1 = {}, toi = {}", dtoi, d1, d2, t);

        assert(dtoi > 0.0);
        assert(t > 0.0 && t < 1.0);
        toi = min(toi, t);
    }
    return toi;
};

void cubic_binomial(const double a[3], const double b[3], double polynomial[4])
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                double c11 = i ? a[0] : b[0];
                double c22 = j ? a[1] : b[1];
                double c33 = k ? a[2] : b[2];
                // int I = (i<< 2) + (j << 1) + k;
                double t = c11 * c22 * c33;
                int J = i + k + j;
                polynomial[J] += t;
            }
}

Vector4d det_polynomial(const mat3& a, const mat3& b)
{
    double pos_polynomial[4]{ 0.0 }, neg_polynomial[4]{ 0.0 };
    double c11c22c33[2][3]{
        { a(0, 0), a(1, 1), a(2, 2) },
        { b(0, 0), b(1, 1), b(2, 2) }
    },
        c12c23c31[2][3]{
            { a(0, 1), a(1, 2), a(2, 0) },
            { b(0, 1), b(1, 2), b(2, 0) }
        },
        c13c21c32[2][3]{
            { a(0, 2), a(1, 0), a(2, 1) },
            { b(0, 2), b(1, 0), b(2, 1) }
        };
    double c11c23c32[2][3]{
        { a(0, 0), a(1, 2), a(2, 1) }, { b(0, 0), b(1, 2), b(2, 1) }
    },
        c12c21c33[2][3]{
            { a(0, 1), a(1, 0), a(2, 2) }, { b(0, 1), b(1, 0), b(2, 2) }
        },
        c13c22c31[2][3]{
            { a(0, 2), a(1, 1), a(2, 0) }, { b(0, 2), b(1, 1), b(2, 0) }
        };
    cubic_binomial(
        c11c22c33[0],
        c11c22c33[1],
        pos_polynomial);
    cubic_binomial(
        c12c23c31[0],
        c12c23c31[1],
        pos_polynomial);
    cubic_binomial(
        c13c21c32[0],
        c13c21c32[1],
        pos_polynomial);
    cubic_binomial(
        c11c23c32[0],
        c11c23c32[1],
        neg_polynomial);
    cubic_binomial(
        c12c21c33[0],
        c12c21c33[1],
        neg_polynomial);
    cubic_binomial(
        c13c22c31[0],
        c13c22c31[1],
        neg_polynomial);
    Vector4d ret;
    for (int i = 0; i < 4; i++) ret(i) = pos_polynomial[i] - neg_polynomial[i];
    return ret;
}
#include "../cyCodeBase/cyPolynomial.h"
double pt_collision_time(
    const vec3& p0,
    const Face& t0,
    const vec3& p1,
    const Face& t1)
{
    mat3 a1, a2, a3, a4;
    mat3 b1, b2, b3, b4;

    b1 << t0.t0, t0.t1, t0.t2;
    b2 << p0, t0.t1, t0.t2;
    b3 << p0, t0.t0, t0.t2;
    b4 << p0, t0.t0, t0.t1;

    a1 << t1.t0, t1.t1, t1.t2;
    a2 << p1, t1.t1, t1.t2;
    a3 << p1, t1.t0, t1.t2;
    a4 << p1, t1.t0, t1.t1;

    a1 -= b1;
    a2 -= b2;
    a3 -= b3;
    a4 -= b4;

    Vector4d t = det_polynomial(a1, b1) - det_polynomial(a2, b2) + det_polynomial(a3, b3) - det_polynomial(a4, b4);
    double root = 1.0;
    bool found = cy::CubicFirstRoot(root, t.data(), 0.0, 1.0);
    return found ? root : 1.0;
}