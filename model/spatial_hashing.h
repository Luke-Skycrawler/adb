#pragma once
#include <vector>
#include <memory>
#include <atomic>
#include <unordered_set>
#include "affine_body.h"
#define SPATIAL_HASHING_H

using element_type = int short;
union Primitive {
    struct {
        element_type pid, body;
    } pbody;
    int as_uint;
    Primitive(int u)
        : as_uint(u) {}
    Primitive()
        : as_uint(0) {}
    Primitive(element_type p, element_type b)
        : pbody({ p, b }) {}
    inline bool operator==(const Primitive& b)
    {
            return as_uint == b.as_uint;
    }
    inline bool operator<(const Primitive& b) {
        return as_uint < b.as_uint;
    }
};
using BodyGroup = std::map<int, std::unique_ptr<std::vector<int>>>;
using vec3i = Eigen::Vector<int, 3>;
using hi = int;
struct spatial_hashing {
    spatial_hashing(int vec3_compressed_bits, int n_buffer,
        scalar MIN_XYZ, scalar MAX_XYZ, scalar dx, int set_size);
    ~spatial_hashing() {
        delete[] count;
        delete[] hashtable;
        delete[] overflow;
        delete[] bitmap_l1;
        delete[] bitmap_l2;
        delete[] sets;
    }
    std::unordered_set<int>* sets;
    std::vector<Primitive> *collisions;
    const int vec3_compressed_bits, n_entries, n_overflow_buffer, n_l1_bitmap, n_l2_bitmap, n_buffer, set_size;
    element_type * count_non_atomic;
    std::atomic<element_type>* count, count_overflow;
    Primitive *overflow, *hashtable;

    bool *bitmap_l1, *bitmap_l2;

    const scalar MIN_XYZ, MAX_XYZ, dx;

    // using hi = int long long;
    // hashing index type
    hi hash(const vec3i& grid_index);
    vec3i tovec3i(const vec3& f);

    void query_interval(const vec3i& l, const vec3i& u, element_type body_exl, std::vector<Primitive>& ret);
    void register_interval(const vec3i& l, const vec3i& u, const Primitive& t);

    void remove_all_entries();
    void register_edge(const vec3& a, const vec3& b, element_type body, element_type pid);
    void query_edge(const vec3& a, const vec3& b, element_type group_exl, scalar dhat,
        std::vector<Primitive> &ret);
    void register_vertex(const vec3& a, element_type body, element_type pid);
    void query_triangle(const vec3& a, const vec3& b, const vec3& c, element_type group_exl, scalar dhat,
        std::vector<Primitive> &ret);
    void register_edge_trajectory(
        const vec3& a0, const vec3& b0,
        const vec3& a1, const vec3& b1,
        element_type body, element_type pid);
        
    void query_edge_trajectory(
        const vec3& a0, const vec3& b0,
        const vec3& a1, const vec3& b1,
        element_type group_exl,
        std::vector<Primitive> &ret);

    void query_triangle_trajectory(
        const vec3& a0, const vec3& b0, const vec3& c0,
        const vec3& a1, const vec3& b1, const vec3& c1,
        element_type group_exl,
        std::vector<Primitive> &ret);
};