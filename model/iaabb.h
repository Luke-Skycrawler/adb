#pragma once
#include "affine_body.h"
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
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx,
    std::vector<Eigen::Matrix<double, 2, 12>>& pt_tk,
    std::vector<Eigen::Matrix<double, 2, 12>>& ee_tk,
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#ifndef _BODY_WISE_
    Globals &globals,
#endif
#endif
    bool gen_basis = false);

double iaabb_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu>& aabbs,
    int vtn,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx,
    std::vector<Eigen::Matrix<double, 2, 12>>& pt_tk,
    std::vector<Eigen::Matrix<double, 2, 12>>& ee_tk,
#ifdef TESTING
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
#ifndef _BODY_WISE_
    Globals &globals,
#endif
#endif

    bool gen_basis = false);