#pragma once
#include "affine_body.h"
#include "geometry.h"
#include "bvh/bvh.h"
#include "bounds3.h"
#include <omp.h>
// using lu = std::array<vec3, 2>;

struct PList {
    std::vector<int> vi, vj, fi, fj, ei, ej;
};

struct Intersection {
    int i, j;
    lu cull;
    PList* plist;
    inline bool operator==(const Intersection &b) const {
        return i == b.i && j == b.j;
    }
    inline bool operator<(const Intersection& b) const {
        return i < b.i || (i == b.i && j < b.j);
    }
};

struct BoundingBox {
    int body;
    scalar p;
    bool true_for_l_false_for_u;
};

struct IAABB {
    std::vector<std::array<int, 2>> edges, points, triangles;
    int n_points, n_triangles, n_edges;

    int g_cnt = 0;
    int p_cnt = 0;

    bool ground = true;
    std::vector<BoundingBox> bounds[3];
    std::vector<lu> affine_bb;
    std::vector<int> buckets;
    std::vector<PList> lists;
    std::vector<std::vector<std::array<int, 2>>> vidx_thread_local;
    std::vector<vec3> vt1_buffer;
    std::vector<int> vertex_starting_index;

    IAABB(std::vector<std::unique_ptr<AffineBody>>& cubes, bool ground = true);
    void intersect_brute_force(
        int n_cubes,
        const std::vector<std::unique_ptr<AffineBody>>& cubes,
        const std::vector<lu>& aabbs,
        std::vector<Intersection>& ret,
        int vtn);
    void intersect_sort(
        int n_cubes,
        const std::vector<std::unique_ptr<AffineBody>>& cubes,
        const std::vector<lu>& aabbs,
        std::vector<Intersection>& ret,
        int vtn);

    scalar primitive_brute_force(
        int n_cubes,
        std::vector<Intersection>& overlaps, // assert sorted
        const std::vector<std::unique_ptr<AffineBody>>& cubes,
        int vtn,

        ///////////////////////////return values//////////////////////
        // if vtn == 3 returns ccd toi
        // if vtn == 2 ccd returns collision set by following params
        std::vector<q4>& pts,
        std::vector<i4>& idx,
        std::vector<q4>& ees,
        std::vector<i4>& eidx,
        std::vector<std::array<int, 2>>& vidx);

    scalar iaabb_brute_force(
        int n_cubes,
        const std::vector<std::unique_ptr<AffineBody>>& cubes,
        const std::vector<lu>& aabbs,
        int vtn,
        std::vector<q4>& pts,
        std::vector<i4>& idx,
        std::vector<q4>& ees,
        std::vector<i4>& eidx,
        std::vector<std::array<int, 2>>& vidx);
};
