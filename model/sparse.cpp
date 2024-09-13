#include "time_integrator.h"
#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <spdlog/spdlog.h>
using namespace std;
using namespace Eigen;
using namespace utils;

namespace utils {

void gen_empty_sm(
    int n_cubes,
    vector<array<int, 4>>& idx,
    vector<array<int, 4>>& eidx,
    SparseMatrix<scalar>& sparse_hess,
    map<array<int, 2>, int>& lut
    )
{
    set<array<int, 2>> cset;
    const auto insert = [&](int a, int b) {
        cset.insert({ a, b });
        cset.insert({ b, a });
    };
    for (auto& t : idx)
        insert(t[0], t[2]);
    for (int i = 0; i < n_cubes; i++) cset.insert({ i, i });

    for (auto& t : eidx) insert(t[0], t[2]);
    auto old = cset.begin();
    auto old_col = (*old)[0];
    for (auto it = cset.begin();; it++) {
        if (it == cset.end() || (*it)[0] != old_col) {
            for (int c = 0; c < 12; c++) {

                auto cc = c + old_col * 12;
                sparse_hess.startVec(cc);
                int k = 0;
                for (auto kt = old; kt != it; ++kt) {
                    lut[{ (*kt)[1], (*kt)[0] }] = k++;
                    auto row = (*kt)[1];
                    for (int r = 0; r < 12; r++) {
                        auto rr = row * 12 + r;
                        sparse_hess.insertBack(rr, cc) = 0.0;
                    }
                }
            }
            if (it == cset.end()) break;
            old = it;
            old_col = (*it)[0];
        }
    }
    sparse_hess.makeCompressed();
    spdlog::info("\nsparse matrix : #non-zero blocks = {}", cset.size());
    // cout << sparse_hess;
}

void clear(SparseMatrix<scalar>& sm)
{
    for (int j = 0; j < sm.outerSize(); ++j)
        for (SparseMatrix<scalar>::InnerIterator it(sm, j); it; ++it) {
            auto r = it.row();
            it.valueRef() = 0.0;
        }
}

};
