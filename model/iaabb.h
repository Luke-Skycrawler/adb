#pragma once
#include "affine_body.h"
#include "geometry.h"
#include "barrier.h"
#ifdef TESTING
#include "../iAABB/pch.h"

#endif
using lu = std::array<vec3, 2>;

struct PList {
    std::vector<int> vi, vj, fi, fj, ei, ej;
};

struct Intersection {
    int i, j;
    lu cull;
    PList* plist;
    inline bool operator==(Intersection &b) const {
        return i == b.i && j == b.j;
    }
    inline bool operator<(Intersection& b) const {
        return i < b.i || (i == b.i && j < b.j);
    }
};

struct BoundingBox {
    int body;
    double p;
    bool true_for_l_false_for_u;
};

inline lu merge(const lu& a, const lu& b)
{
    vec3 l, u;
    l = a[0].array().min(b[0].array());
    u = a[1].array().max(b[1].array());
    return { l, u };
}
lu compute_aabb(const AffineBody& c);
inline lu compute_aabb(const Edge& e)
{
    vec3 l, u;
    auto e0 {e.e0.array()}, e1 {e.e1.array()};
    l = e0.min(e1);
    u = e0.max(e1);
    return { l, u };
}
inline lu compute_aabb(const Edge& e, double d_hat_sqrt)
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

lu affine(const lu& aabb, q4& q);
lu affine(lu aabb, AffineBody& c, int vtn);


void intersect_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu> &aabbs,
    std::vector<Intersection> &ret,
    int vtn);
void intersect_sort(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu> &aabbs,
    std::vector<Intersection> &ret,
    int vtn);
#ifdef TESTING
struct double_int{
    double d;
    int i;
};
#endif
double primitive_brute_force(
    int n_cubes,
    std::vector<Intersection>& overlaps, // assert sorted
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int vtn,

#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#ifndef _BODY_WISE_
    Globals &globals,
#endif
#endif
    ///////////////////////////return values////////////////////// 
    // if vtn == 3 returns ccd toi 
    // if vtn == 2 ccd returns collision set by following params
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx
);

double iaabb_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    int vtn,
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#ifndef _BODY_WISE_
    Globals &globals,
#endif
#endif
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx
);

inline bool intersects(const lu& a, const lu& b)
{
    const auto overlaps = [&](int i) -> bool {
        return a[0][i] <= b[1][i] && a[1][i] >= b[0][i];
    };
    return overlaps(0) && overlaps(1) && overlaps(2);
}

double primitive_brute_force_thrust(
    int n_cubes,
    std::vector<Intersection>& overlaps, // assert sorted
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int vtn,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx);





