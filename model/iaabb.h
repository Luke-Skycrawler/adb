#pragma once
#include "affine_body.h"
#include "geometry.h"
#include "bvh/bvh.h"
#include "bounds3.h"
#ifdef TESTING
#include "../iAABB/pch.h"

#endif
// using lu = std::array<vec3, 2>;

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
    scalar p;
    bool true_for_l_false_for_u;
};


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
    scalar d;
    int i;
};
#endif
scalar primitive_brute_force(
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

scalar iaabb_brute_force(
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
