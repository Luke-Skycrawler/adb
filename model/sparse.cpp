#include <Eigen/Eigen>
#include "time_integrator.h"
#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <spdlog/spdlog.h>
#include <assert.h>
using namespace std;
using namespace Eigen;
using namespace utils;

#include "cuda_globals.cuh"

static const bool strict = false;
void build_csr(int n_cubes, const thrust::device_vector<i2>& lut, CsrSparseMatrix& sparse_matrix);
void make_lut_test_glue(map<array<int, 2>, int>& lut, thrust::host_vector<i2>& host_lut)
{
    host_lut.resize(lut.size());
    int i = 0;
    for (auto& p : lut) {
        host_lut[i++] = { p.first[0], p.first[1] };
    }
}
void gen_empty_sm_glue(

    int n_cubes,
    vector<array<int, 4>>& idx,
    vector<array<int, 4>>& eidx,
    SparseMatrix<double>& sparse_hess,
    map<array<int, 2>, int>& lut)
{
    // TESTED
    // cuda version
    gen_empty_sm(n_cubes, idx, eidx, sparse_hess, lut);
    thrust::host_vector<i2> host_lut;
    make_lut_test_glue(lut, host_lut);

    host_cuda_globals.lut = host_lut;
    // make_lut(host_lut.size(), PTR(host_lut));
    build_csr(n_cubes, host_cuda_globals.lut, host_cuda_globals.hess);
    auto& hess = host_cuda_globals.hess;
    assert(sparse_hess.nonZeros() == hess.nnz); // "error: nnz: " ;// sparse_hess.nonZeros() ;// " " ;// hess.nnz;
    assert(sparse_hess.rows() == hess.rows); // "error: rows: " ;// sparse_hess.rows() ;// " " ;// hess.rows;
    assert(sparse_hess.cols() == hess.cols); // "error: cols: " ;// sparse_hess.cols() ;// " " ;// hess.cols;

    // thrust::host_vector<float> host_values(hess.values.begin(), hess.values.end());
    // thrust::host_vector<int> host_inner(hess.inner.begin(), hess.inner.end()),
    //     host_outer_start(hess.outer_start.begin(), hess.outer_start.end());
    const auto from_thrust = [](thrust::device_vector<int> &a) ->vector<int> {
        thrust::host_vector<int> host(a.begin(), a.end());
        vector<int > ret; 
        ret.resize(host.size());
        for (int i = 0; i < host.size(); i ++) {
            ret[i] = host[i];
        }
        return ret;
    } ;
    vector<int> outer = from_thrust(hess.outer_start);
    vector<int> inner = from_thrust(hess.inner);
    for (int i = 0; i < sparse_hess.cols(); i++) {
        auto ref = sparse_hess.outerIndexPtr()[i], act = outer[i];
        if (!strict) {
            if (ref != act)
                spdlog::warn("outer, ref, value = {}, {}", ref, act);
        }
        else
            assert(ref == act); 
    }
    for (int i = 0; i < sparse_hess.nonZeros(); i++) {
        auto ref = sparse_hess.innerIndexPtr()[i], act = inner[i];

        if (!strict) {
            if (ref != act)
                spdlog ::warn("ref, value = {}, {}", ref, act);

        }

        else 
            assert(ref == act) ;
    }
}
namespace utils {

void gen_empty_sm(
    int n_cubes,
    vector<array<int, 4>>& idx,
    vector<array<int, 4>>& eidx,
    SparseMatrix<double>& sparse_hess,
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
}

void clear(SparseMatrix<double>& sm)
{
    for (int j = 0; j < sm.outerSize(); ++j)
        for (SparseMatrix<double>::InnerIterator it(sm, j); it; ++it) {
            auto r = it.row();
            it.valueRef() = 0.0;
        }
}

};
