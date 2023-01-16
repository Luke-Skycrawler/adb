#pragma once
#include <vector>
#include "affine_body.h"
#include <memory>
#define SPATIAL_HASHING_H
struct Primitive {
    unsigned pid, body;
};
using BodyGroup = std::map<unsigned, std::unique_ptr<std::vector<unsigned>>>;
namespace spatial_hashing {
using vec3i = Eigen::Vector<int, 3>;
using hi = unsigned long long;
// hashing index type
vec3i tovec3i(const vec3& f);
hi hash(const vec3i& grid_index);
void remove_all_entries();
void register_edge(const vec3& a, const vec3& b, unsigned body, unsigned pid);
std::vector<Primitive> query_edge(const vec3& a, const vec3& b, int group_exl, double dhat = 1e-2);
void register_vertex(const vec3& a, unsigned body, unsigned pid);
std::vector<Primitive> query_triangle(const vec3& a, const vec3& b, const vec3& c, int group_exl, double dhat = 1e-2);
};