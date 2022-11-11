#include "geometry.h"
#include <Eigen/Geometry>
#include <cmath>
using namespace std;
Edge::Edge(const Cube& c, int id, bool b)
{
    int _0 = Cube::edges[id * 2];
    int _1 = Cube::edges[id * 2 + 1];
    e0 = c.vi(_0, b);
    e1 = c.vi(_1, b);
}

vec3 Face::normal() const
{
    return (t0 - t1).cross(t0 - t2);
}

vec3 Face::unit_normal() const
{
    auto n = (t0 - t1).cross(t0 - t2);
    return n / sqrt(n.dot(n));
}

Face::Face(const Cube& c, int id, bool b)
{

    int _a = Cube::indices[id * 3 + 0],
        _b = Cube::indices[id * 3 + 1],
        _c = Cube::indices[id * 3 + 2];

    t0 = c.vi(_a, b);
    t1 = c.vi(_b, b);
    t2 = c.vi(_c, b);
}

double vf_distance(const vec3& v, const Cube& c, int id)
{
    Face f(c, id);
    return vf_distance(v, f);
}

double vf_distance(const vec3& vertex)
{
    // ground plane y = -0.5
    return vertex(1) + 0.5;
}
vec3 vf_distance_gradient_x(const vec3& vertex)
{
    // face = grond plane y = 0.5
    return vec3(0.0f, 1.0f, 0.0f);
}

double vf_distance(const vec3& _v, const Face& f)
{
    double d = f.unit_normal().dot(_v - f.t0);
    double area = (f.t1 - f.t0).cross(f.t2 - f.t0).norm();
    vec3 v = _v - d * f.unit_normal();
    d = abs(d);
    double a2 = ((f.t0 - v).cross(f.t1 - v).norm() + (f.t1 - v).cross(f.t2 - v).norm() + (f.t2 - v).cross(f.t0 - v).norm());
    if (abs(a2 - area) > 1e-3) {
        // projection outside of triangle
        double ab = (f.t1 - f.t0).norm();
        double bc = (f.t2 - f.t1).norm();
        double ca = (f.t0 - f.t2).norm();

        double d_ab = (v - f.t0).cross(f.t0 - f.t1).norm() / ab;
        double d_bc = (v - f.t1).cross(f.t1 - f.t2).norm() / bc;
        double d_ac = (v - f.t2).cross(f.t2 - f.t0).norm() / ca;

        double d_a = (v - f.t0).norm();
        double d_b = (v - f.t1).norm();
        double d_c = (v - f.t2).norm();

        d_ab = abs(d_a * d_a - d_b * d_b) >= ab * ab ? min(d_a, d_b) : d_ab;
        d_bc = abs(d_b * d_b - d_c * d_c) >= bc * bc ? min(d_c, d_b) : d_bc;
        d_ac = abs(d_a * d_a - d_c * d_c) >= ca * ca ? min(d_a, d_c) : d_ac;

        double d_projected = min(d_ab, min(d_bc, d_ac));
        // double d_projected = min(min(min(min(min(d_ab, d_bc), d_ac), d_a), d_b), d_c);
        d = sqrt(d * d + d_projected * d_projected);
    }
    return d;
}

double ee_distance(const Edge& ei, const Edge& ej)
{
    double d = 0.0;
    auto ei0 = ei.e0, ei1 = ei.e1, ej0 = ej.e0, ej1 = ej.e1;
    auto n = (ei1 - ei0).cross(ej1 - ej0);
    if (n.norm() < 1e-6) {
        // degenerate case
        auto t = ei1 - ei0;
        d = t.cross(ej0 - ei0).norm() / t.dot(t);
    }
    d = abs(n.dot(ej0 - ei0)) / n.dot(n);
    return d;
}
