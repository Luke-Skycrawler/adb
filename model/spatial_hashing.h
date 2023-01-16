#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <memory>
#define SPATIAL_HASHING_H
using mat3 = Eigen::Matrix3d;
using vec3 = Eigen::Vector3d;
struct Primitive {
    unsigned pid, body;
};
using BodyGroup = std::map<unsigned, std::unique_ptr<std::vector<unsigned>>>;
namespace spatial_hashing {
using vec3i = Eigen::Vector<int, 3>;
using hi = unsigned long long;
// hashing index type
void remove_all_entries();
void register_edge(const vec3& a, const vec3& b, unsigned body, unsigned pid);
std::vector<Primitive> query_edge(const vec3& a, const vec3& b, int group_exl, double dhat = 1e-2);
void register_vertex(const vec3& a, unsigned body, unsigned pid);
std::vector<Primitive> query_triangle(const vec3& a, const vec3& b, const vec3& c, int group_exl, double dhat = 1e-2);

};