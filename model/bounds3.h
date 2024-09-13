#pragma once
#include "affine_body.h"
#include "bvh/bvh.h"

using lu = bounds3;

lu compute_aabb(const AffineBody& c);
lu affine(const lu& aabb, q4& q);
lu affine(lu aabb, AffineBody& c, int vtn);


inline lu compute_aabb(const Edge& e)
{
    vec3 l, u;
    auto e0 {e.e0.array()}, e1 {e.e1.array()};
    l = e0.min(e1);
    u = e0.max(e1);
    return { l, u };
}


inline lu compute_aabb(const Edge& e, scalar d_hat_sqrt)
{
    vec3 l, u;
    auto e0 {e.e0.array()}, e1 {e.e1.array()};
    l = e0.min(e1) - d_hat_sqrt;
    u = e0.max(e1) + d_hat_sqrt;
    return { l, u };
}

inline lu compute_aabb(const Face& f)
{
    vec3 l, u;
    auto t0 {f.t0.array()}, 
        t1 {f.t1.array()}, 
        t2 {f.t2.array()};
    l = t0.min(t1).min(t2);
    u = t0.max(t1).max(t2);
    return { l, u };
}


inline bool intersects(const lu& a, const lu& b)
{
    const auto overlaps = [&](int i) -> bool {
        return a.lower[i] <= b.upper[i] && a.upper[i] >= b.lower[i];
    };
    return overlaps(0) && overlaps(1) && overlaps(2);
}

inline bool intersection(const lu& a, const lu& b, lu& ret)
{
    vec3 l, u;
    l = a.lower.array().max(b.lower.array());
    u = a.upper.array().min(b.upper.array());
    bool intersects = (l.array() <= u.array()).all();
    ret = { l, u };
    return intersects;
}

inline lu merge(const lu& a, const lu& b)
{
    vec3 l, u;
    l = a.lower.array().min(b.lower.array());
    u = a.upper.array().max(b.upper.array());
    return { l, u };
}
inline lu compute_aabb(const vec3& p0, const vec3& p1)
{
    vec3 l, u;
    auto e0{ p0.array() }, e1{ p1.array() };
    l = e0.min(e1);
    u = e0.max(e1);
    return { l, u };
}
inline lu compute_aabb(const Edge& e1, const Edge& e2)
{
    vec3 l, u;
    auto e10{ e1.e0.array() }, e11{ e1.e1.array() };
    auto e20{ e2.e0.array() }, e21{ e2.e1.array() };
    l = e10.min(e11).min(e20).min(e21);
    u = e10.max(e11).max(e20).max(e21);
    return { l, u };
}
inline lu compute_aabb(const Face& f1, const Face& f2)
{
    return merge(compute_aabb(f1), compute_aabb(f2));
}

inline lu dialate(lu &aabb, scalar d) {
    return {aabb.lower - vec3(d, d, d), aabb.upper + vec3(d, d, d)};
}