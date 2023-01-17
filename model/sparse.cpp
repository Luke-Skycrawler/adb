#include <Eigen/Eigen>
#include "time_integrator.h"
#include <vector>
#include <array>
#include <set>
using namespace std;
using namespace Eigen;

void gen_empty_sm(
    int n_cubes,
    vector<array<int, 4>>& idx,
    vector<array<int, 4>>& eidx,
    SparseMatrix<double>& sparse_hess)
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

    for (auto it = cset.begin(); it != cset.end(); ++it) {
        auto col = (*it)[0];
        for (auto jt = ++it; jt != cset.end(); ++jt) {
            auto j = (*jt)[0];
            if (j != col || next(jt, 1) == cset.end()) {
                for (int c = 0; c < 12; c++) {
                    auto cc = c + col * 12;
                    sparse_hess.startVec(cc);
                    for (auto kt = it; kt != jt; ++kt) {
                        auto row = (*kt)[1];
                        for (int r = 0; r < 12; r++) {
                            auto rr = row * 12 + r;
                            sparse_hess.insertBack(rr, cc) = 0.0;
                        }
                    }
                }
            }
        }
    }
    sparse_hess.makeCompressed();
    exit(0);
}

void clear(SparseMatrix<double>& sm)
{
    for (int j = 0; j < sm.outerSize(); ++j)
        for (SparseMatrix<double>::InnerIterator it(sm, j); it; ++it)
            it.valueRef() = 0.0;
}