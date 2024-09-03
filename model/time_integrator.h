#pragma once
#include "cube.h"
#include "othogonal_energy.h"
#include <vector>
#include <memory>
#include <array>
#include <map>
#include <string>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <tuple>

void implicit_euler(std::vector<std::unique_ptr<AffineBody>>& cubes, scalar dt);
void gen_collision_set(
    bool vt2, int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx);

scalar step_size_upper_bound(Eigen::Vector<scalar, -1>& dq, std::vector<std::unique_ptr<AffineBody>>& cubes,
    int n_cubes, int n_pt, int n_ee, int n_g,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx);

void player_load(
    std::string& path,
    int timestep,
    const std::vector<std::unique_ptr<AffineBody>>& cubes);
void player_save(
    std::string& path,
    int timestep,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    bool init);

inline int starting_offset(int i, int j, const std::map<std::array<int, 2>, int>& lut, int* outers)
{
    auto it = lut.find({ i, j });
    int k = it->second;
    return k * 12 + outers[j * 12];
}

inline int stride(int j, int* outers)
{
    return outers[j * 12 + 1] - outers[j * 12];
    // full 12x12 matrix, no overflow issue
}

namespace utils {

void gen_empty_sm(
    int n_cubes,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx,
    SparseMatrix<scalar>& sparse_hess,
    std::map<std::array<int, 2>, int>& lut);

void clear(SparseMatrix<scalar>& sm);

vec12 pt_vstack(AffineBody& ci, AffineBody& cj, int v, int f);
vec12 ee_vstack(AffineBody& ci, AffineBody& cj, int ei, int ej);
vec12 cat(const q4& q);
vec12 grad_residue_per_body(AffineBody& c, scalar dt);
mat12 hess_inertia_per_body(AffineBody& c, scalar dt);
scalar norm_M(const vec12& x, const AffineBody& c);
scalar norm_1(Vector<scalar, -1>& dq, int n_cubes);
void damping_dense(Matrix<scalar, -1, -1>& big_hess, scalar dt, int n_cubes);

void damping_sparse(SparseMatrix<scalar>& sparse_hess, scalar dt, int n_cubes);
void build_from_triplets(SparseMatrix<scalar>& sparse_hess_trip, Matrix<scalar, -1, -1>& big_hess, int hess_dim, int n_cubes);
scalar E(const vec12& q, const vec12& q_tiled, const AffineBody& c, scalar dt);

std::vector<std::array<unsigned, 2>> gen_edge_list(
    std::vector<std::unique_ptr<AffineBody>>& cubes, int n_cubes);
std::vector<std::array<unsigned, 2>> gen_point_list(
    std::vector<std::unique_ptr<AffineBody>>& cubes, int n_cubes);
std::vector<std::array<unsigned, 2>> gen_triangle_list(
    std::vector<std::unique_ptr<AffineBody>>& cubes, int n_cubes);

}
void dump_states(
    const std::vector<std::unique_ptr<AffineBody>>& cubes
);
