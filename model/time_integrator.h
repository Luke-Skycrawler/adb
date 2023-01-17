#include "cube.h"
#include "othogonal_energy.h"
#include <vector>
#include <memory>
#include <array>
void implicit_euler(std::vector<std::unique_ptr<AffineBody>>& cubes, double dt);
void gen_empty_sm(
    int n_cubes,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx,
    SparseMatrix<double>& sparse_hess);

void clear(SparseMatrix<double>& sm);