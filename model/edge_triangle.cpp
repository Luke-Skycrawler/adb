#include "edge_triangle.h"
#include <Eigen/Geometry>
#include <cmath>
using namespace std;
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
    auto& q = b ? (c.q_next - c.dq) : c.q_next;
    auto& p = b ? (c.p_next - c.dp) : c.p_next;

    int _a = Cube::indices[id * 3 + 0],
        _b = Cube::indices[id * 3 + 1],
        _c = Cube::indices[id * 3 + 2];

    const vec3 &v0a(c.vertices()[_a]),
        &v0b(c.vertices()[_b]),
        &v0c(c.vertices()[_c]);

    t0 = q * v0a + p;
    t1 = q * v0b + p;
    t2 = q * v0c + p;
}

double vf_distance(const vec3& v, const Cube& c, int id)
{
    Face f(c, id);
    return vf_distance(v, f);
}

double vf_distance(const vec3& _v, const Face &f)
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
        //double d_projected = min(min(min(min(min(d_ab, d_bc), d_ac), d_a), d_b), d_c);
        d = sqrt(d * d + d_projected * d_projected);
    }
    return d;
}

