#include "cube.h"
#include "othogonal_energy.h"
#include <vector>
#include <memory>
#include <array>
#include <map>

void implicit_euler(std::vector<std::unique_ptr<AffineBody>>& cubes, double dt);
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
namespace utils{

void gen_empty_sm(
    int n_cubes,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx,
    SparseMatrix<double>& sparse_hess,
    std::map<std::array<int, 2>, int>& lut
    );

void clear(SparseMatrix<double>& sm);


vec12 pt_vstack(AffineBody& ci, AffineBody& cj, int v, int f);
vec12 ee_vstack(AffineBody& ci, AffineBody& cj, int ei, int ej);
vec12 cat(const q4 &q);
vec12 grad_residue_per_body(AffineBody& c, double dt);
mat12 hess_inertia_per_body(AffineBody& c, double dt);
double norm_M(const vec12& x, const AffineBody& c);
 double norm_1 (VectorXd& dq, int n_cubes);
void damping_dense(MatrixXd& big_hess, double dt, int n_cubes);

void damping_sparse(SparseMatrix<double>& sparse_hess, double dt, int n_cubes);
void build_from_triplets(SparseMatrix<double>& sparse_hess_trip, MatrixXd& big_hess, int hess_dim, int n_cubes);


}
