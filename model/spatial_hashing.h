#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <atomic>
#define SPATIAL_HASHING_H
using mat3 = Eigen::Matrix3d;
using vec3 = Eigen::Vector3d;

using element_type = unsigned short;
struct Primitive {
    element_type pid, body;
};
using BodyGroup = std::map<unsigned, std::unique_ptr<std::vector<unsigned>>>;
using vec3i = Eigen::Vector<int, 3>;
using hi = unsigned;
struct spatial_hashing {

    spatial_hashing(int vec3_compressed_bits, int n_buffer,
        double MIN_XYZ, double MAX_XYZ, double dx);
    ~spatial_hashing() {
        delete[] count;
        delete[] hashtable;
        delete[] overflow;
        delete[] bitmap_l1;
        delete[] bitmap_l2;
    }
    const int vec3_compressed_bits, n_entries, n_overflow_buffer, n_l1_bitmap, n_l2_bitmap, n_buffer;
    std::atomic<element_type>* count, count_overflow;
    Primitive *overflow, *hashtable;

    bool *bitmap_l1, *bitmap_l2;

    const double MIN_XYZ, MAX_XYZ, dx;

    // using hi = unsigned long long;
    // hashing index type
    hi hash(const vec3i& grid_index);
    vec3i tovec3i(const vec3& f);

    std::vector<Primitive> query_interval(const vec3i& l, const vec3i& u, element_type body_exl);
    void register_interval(const vec3i& l, const vec3i& u, const Primitive& t);

    void remove_all_entries();
    void register_edge(const vec3& a, const vec3& b, element_type body, element_type pid);
    std::vector<Primitive> query_edge(const vec3& a, const vec3& b, element_type group_exl, double dhat = 1e-2);
    void register_vertex(const vec3& a, element_type body, element_type pid);
    std::vector<Primitive> query_triangle(const vec3& a, const vec3& b, const vec3& c, element_type group_exl, double dhat = 1e-2);
};