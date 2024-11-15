#include "geometry.h"
#include <Eigen/Geometry>
#include "ipc_extension.h"
// #include <cmath>

using namespace std;
Edge AffineBody::edge(int id, bool b, bool batch) const
{
    Edge ret;
    int _0 = edges[id * 2], _1 = edges[id * 2 + 1];
    if (batch) {
        ret.e0 = v_transformed[_0];
        ret.e1 = v_transformed[_1];
    }
    else if (b) {
        ret.e0 = vt2(_0);
        ret.e1 = vt2(_1);
    }
    else {
        mat3 a;
        vec3 b = q[0];
        a << q[1], q[2], q[3];
        ret.e0 = a * vertices(_0) + b;
        ret.e1 = a * vertices(_1) + b;
    }
    return ret;
}

Face AffineBody::face(int id, bool b, bool batch) const
{
    Face ret;
    int _a = indices[id * 3 + 0],
        _b = indices[id * 3 + 1],
        _c = indices[id * 3 + 2];

    if (batch) {
        ret.t0 = v_transformed[_a];
        ret.t1 = v_transformed[_b];
        ret.t2 = v_transformed[_c];
    }
    else if (b) {
        ret.t0 = vt2(_a);
        ret.t1 = vt2(_b);
        ret.t2 = vt2(_c);
    }
    else {
        mat3 a;
        vec3 b = q[0];
        a << q[1], q[2], q[3];
        ret.t0 = a * vertices(_a) + b;
        ret.t1 = a * vertices(_b) + b;
        ret.t2 = a * vertices(_c) + b;
    }
    return ret;
}

scalar vg_distance(const vec3& vertex) // not squared
{
    // ground plane y = -0.5
    scalar d = vertex(1) + 0.5;
    return d;
}
vec3 vg_distance_gradient_x(const vec3& vertex)
{
    // face = grond plane y = 0.5
    return vec3(0.0f, 1.0f, 0.0f);
}

inline scalar ab(const vec3& a, const vec3& b)
{
    return (a - b).norm();
}
inline scalar ab_sqr(const vec3& v0, const vec3& v1) {
    return (v0 - v1).dot(v0 - v1);
}

inline scalar h(const vec3 &a, const vec3 &b, const vec3 &c) {
    // height of triangle
    // ab as base
    return (b-a).cross(c-a).norm() / ab(a, b);
}

inline scalar area(const vec3 &a, const vec3 &b, const vec3 &c) {
    return (c -a).cross(b- a).norm();
}
inline bool is_obtuse_triangle(const vec3 &a, const vec3 &b, const vec3 &c) {
    // determines if projection of c is in the span of ab
    return abs(ab_sqr(c,a) - ab_sqr(c, b)) > ab_sqr(a, b);
}
inline scalar ev_distance(const Edge& e, const vec3& v)
{
    scalar d = h(e.e0, e.e1, v);
    bool s = is_obtuse_triangle(e.e0, e.e1, v);
    if (s) {
        d = min(ab(v, e.e0), ab(v, e.e1));
    }
    return d;
}
using namespace ipc;

tuple<scalar, PointTriangleDistanceType> vf_distance(const vec3& v, const Face& f) {

#define STEAL_IPC
#ifndef STEAL_IPC
    PointTriangleDistanceType pt_type;
    scalar d = vf_distance(v, f, pt_type);
#else
    auto pt_type = ipc::point_triangle_distance_type(v, f.t0, f.t1, f.t2);
    scalar d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2, pt_type);
#endif
    return {d, pt_type};
}

scalar vf_distance(const vec3& _v, const Face& f, PointTriangleDistanceType &pt_type)
{
    auto n = f.unit_normal();
    scalar d = n.dot(_v - f.t0);
    scalar a1 = area(f.t1, f.t0, f.t2);
    vec3 v = _v - d * n;
    d = d * d;
    // scalar a2 = ((f.t0 - v).cross(f.t1 - v).norm() + (f.t1 - v).cross(f.t2 - v).norm() + (f.t2 - v).cross(f.t0 - v).norm());
    scalar a2 = area(f.t0, f.t1, v) + area(f.t1, f.t2, v) + area(f.t2, f.t0, v);
    if (a2 > a1 + 1e-8) {
        // projection outside of triangle

        scalar d_ab = h(f.t0, f.t1, v);
        scalar d_bc = h(f.t1, f.t2, v);
        scalar d_ac = h(f.t0, f.t2, v);

        scalar d_a = ab(v, f.t0);
        scalar d_b = ab(v, f.t1);
        scalar d_c = ab(v, f.t2);


        scalar dab = is_obtuse_triangle(f.t0, f.t1, v) ? min(d_a, d_b): d_ab;
        scalar dbc = is_obtuse_triangle(f.t2, f.t1, v) ? min(d_c, d_b): d_bc;
        scalar dac = is_obtuse_triangle(f.t0, f.t2, v) ? min(d_a, d_c): d_ac;

        scalar d_projected = std::min({dab, dbc, dac});
        d += d_projected * d_projected;

        if (d_projected == d_ab) pt_type = PointTriangleDistanceType::P_E0;
        else if (d_projected == d_bc) pt_type = PointTriangleDistanceType::P_E1;
        else if (d_projected == d_ac) pt_type = PointTriangleDistanceType::P_E2;
        else if (d_projected == d_a) pt_type = PointTriangleDistanceType::P_T0;
        else if (d_projected == d_b) pt_type = PointTriangleDistanceType::P_T1;
        else pt_type = PointTriangleDistanceType::P_T2;
    }
    else pt_type = PointTriangleDistanceType::P_T;
    return d;
}

bool inside(const Face &f, const vec3 &v){
    scalar a1 = area(f.t1, f.t0, f.t2);
    scalar a2 = area(f.t0, f.t1, v) + area(f.t1, f.t2, v) + area(f.t2, f.t0, v);
    return a2 <= a1 + 1e-8;
}

// PointTriangleDistanceType pt_distance_type(const vec3& _v, const Face& f)
// {
//     auto n = f.unit_normal();
//     scalar d = n.dot(_v - f.t0);

//     vec3 v = _v - d * n;

//     if (inside(f, v)) return PointTriangleDistanceType::P_T;

//     scalar d_ab = h(f.t0, f.t1, v);
//     scalar d_bc = h(f.t1, f.t2, v);
//     scalar d_ac = h(f.t0, f.t2, v);

//     scalar d_a = ab(v, f.t0);
//     scalar d_b = ab(v, f.t1);
//     scalar d_c = ab(v, f.t2);

//     scalar _d_ab = is_obtuse_triangle(f.t0, f.t1, v) ? min(d_a, d_b) : d_ab;
//     scalar _d_bc = is_obtuse_triangle(f.t2, f.t1, v) ? min(d_c, d_b) : d_bc;
//     scalar _d_ac = is_obtuse_triangle(f.t0, f.t2, v) ? min(d_a, d_c) : d_ac;

//     scalar d_projected = min(_d_ab, min(_d_bc, _d_ac));
//     if (d_projected == d_ab) return PointTriangleDistanceType::P_E0;
//     if (d_projected == d_bc) return PointTriangleDistanceType::P_E1;
//     if (d_projected == d_ac) return PointTriangleDistanceType::P_E2;
//     if (d_projected == d_a) return PointTriangleDistanceType::P_T0;
//     if (d_projected == d_b) return PointTriangleDistanceType::P_T1;
//     return PointTriangleDistanceType::P_T2;
// }

scalar ee_distance(const Edge& ei, const Edge& ej)
{
    scalar d = 0.0;
    auto ei0 = ei.e0, ei1 = ei.e1, ej0 = ej.e0, ej1 = ej.e1;
    auto n = (ei1 - ei0).cross(ej1 - ej0);

    if (n.norm() < 1e-12) {
        // degenerate case
        d = h(ei1, ei0, ej0);
        if (!is_obtuse_triangle(ei1, ei0, ej0) || !is_obtuse_triangle(ei1, ei0, ej1)
            || !is_obtuse_triangle(ej0, ej1, ei1) || !is_obtuse_triangle(ej0, ej1, ei0)) {
            return d;
        }
        const auto& vei(ab(ei0, ej0) < ab(ei1,ej0) ? ei0: ei1);
        const auto& vej(ab(ej0, ei0) < ab(ej1, ei0) ? ej0 : ej1);
        d = sqrt(d * d + ab_sqr(vei, vej));
        return d;
    }


    d = n.dot(ej0 - ei0) / n.norm();
    ej0 -= d * n / n.norm();
    ej1 -= d * n / n.norm();
    d = abs(d);

    auto t00 = (ei1 - ei0).cross(ej0 - ei0);
    auto t01 = (ei1 - ei0).cross(ej1 - ei0);

    auto t10 = (ej1 - ej0).cross(ei0 - ej0);
    auto t11 = (ej1 - ej0).cross(ei1 - ej0);

    bool s0 = t00.dot(t01) < 0;
    bool s1 = t10.dot(t11) < 0;
    
    if (s0 && s1) {
        return d;
    }
    scalar _d;
    if (s0 || s1) {
        if (s0) {
            bool s = h(ej0, ej1, ei0) > h(ej0, ej1, ei1);
            const vec3 &v(s ? ei1 : ei0);
            _d = ev_distance(ej, v);
        }
        else {
            bool s = h(ei0, ei1, ej0) > h(ei0, ei1, ej1);
            const vec3& v(s ? ej1 : ej0);
            _d = ev_distance(ei, v);
        }
        
    }
    else {
        _d = min(min(min(ab(ei0, ej0), ab(ei1, ej0)), ab(ei0, ej1)), ab(ei1, ej1));
    }
    //d = sqrt(d * d + _d * _d);
    d = d * d + _d * _d;
    return d;
}

mat3 rotation(scalar a, scalar b, scalar c)
{
    auto s1 = sin(a);
    auto s2 = sin(b);
    auto s3 = sin(c);

    auto c1 = cos(a);
    auto c2 = cos(b);
    auto c3 = cos(c);

    mat3 R;
    R << c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2,
        c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3,
        -s2, c2 * s3, c2 * c3;
    // R = R.transpose();
    return R;
}
