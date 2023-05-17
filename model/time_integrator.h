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

void implicit_euler(std::vector<std::unique_ptr<AffineBody>>& cubes, double dt);
void gen_collision_set(
    bool vt2, int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx);

double step_size_upper_bound(Eigen::VectorXd& dq, std::vector<std::unique_ptr<AffineBody>>& cubes,
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
    SparseMatrix<double>& sparse_hess,
    std::map<std::array<int, 2>, int>& lut);

void clear(SparseMatrix<double>& sm);

vec12 pt_vstack(AffineBody& ci, AffineBody& cj, int v, int f);
vec12 ee_vstack(AffineBody& ci, AffineBody& cj, int ei, int ej);
vec12 cat(const q4& q);
vec12 grad_residue_per_body(AffineBody& c, double dt);
mat12 hess_inertia_per_body(AffineBody& c, double dt);
double norm_M(const vec12& x, const AffineBody& c);
double norm_1(VectorXd& dq, int n_cubes);
void damping_dense(MatrixXd& big_hess, double dt, int n_cubes);

void damping_sparse(SparseMatrix<double>& sparse_hess, double dt, int n_cubes);
void build_from_triplets(SparseMatrix<double>& sparse_hess_trip, MatrixXd& big_hess, int hess_dim, int n_cubes);
double E(const vec12& q, const vec12& q_tiled, const AffineBody& c, double dt);

std::vector<std::array<unsigned, 2>> gen_edge_list(
    std::vector<std::unique_ptr<AffineBody>>& cubes, int n_cubes);
std::vector<std::array<unsigned, 2>> gen_point_list(
    std::vector<std::unique_ptr<AffineBody>>& cubes, int n_cubes);
std::vector<std::array<unsigned, 2>> gen_triangle_list(
    std::vector<std::unique_ptr<AffineBody>>& cubes, int n_cubes);

double ee_uktk(
    AffineBody& ci, AffineBody& cj,
    std::array<vec3, 4>& ee, std::array<int, 4>& ij, const ::ipc::EdgeEdgeDistanceType& ee_type,
    Matrix<double, 2, 12>& Tk_T_ret, Vector2d& uk_ret, double d, double dt, double mollifier);

double pt_uktk(
    AffineBody& ci, AffineBody& cj,
    std::array<vec3, 4>& pt, std::array<int, 4>& ij, const ::ipc::PointTriangleDistanceType& pt_type,
    Matrix<double, 2, 12>& Tk_T_ret, Vector2d& uk_ret, double d, double dt);

std::tuple<double, Eigen::Vector2d, Eigen::Matrix<double, 2, 12>> ee_uktk(
    AffineBody& ci, AffineBody& cj,
    std::array<vec3, 4>& ee, std::array<int, 4>& ij, const ::ipc::EdgeEdgeDistanceType& ee_type,
    double d, double dt, double mollifier);

std::tuple<double, Eigen::Vector2d, Eigen::Matrix<double, 2, 12>> pt_uktk(AffineBody& ci, AffineBody& cj, std::array<vec3, 4>& pt, std::array<int, 4>& ij, const ::ipc::PointTriangleDistanceType& pt_type, double d, double dt);
}
void dump_states(
    const std::vector<std::unique_ptr<AffineBody>>& cubes
);
