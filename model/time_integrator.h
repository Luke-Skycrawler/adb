#include "cube.h"
#include "othogonal_energy.h"
#include <vector>
#include <memory>
#include <array>
#include <map>

void implicit_euler(std::vector<std::unique_ptr<AffineBody>>& cubes, double dt);
void gen_empty_sm(
    int n_cubes,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx,
    SparseMatrix<double>& sparse_hess,
    std::map<std::array<int, 2>, int>& lut
    );

void clear(SparseMatrix<double>& sm);
inline int starting_offset(int i, int j, const std::map<std::array<int, 2>, int>& lut, int* outers)
{
    auto it = lut.find({i, j});
    int k = it->second;
    return k * 12 + outers[j * 12];
}

inline int stride(int j, int* outers)
{
    return outers[j * 12 + 1] - outers[j * 12];
    // full 12x12 matrix, no overflow issue
}

