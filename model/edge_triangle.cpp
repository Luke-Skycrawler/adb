#include "edge_triangle.h"
#include <Eigen/Geometry>
#include <cmath>
vec3 Face::normal()
{
    return (t0 - t1).cross(t0 - t2);
}

vec3 Face::unit_normal()
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
    double d = abs(f.unit_normal().dot(v - f.t0));
    return d;
}

double vf_distance(const vec3& v, const Face &f)
{
    double d = abs(f.unit_normal().dot(v - f.t0));
    return d;
}

