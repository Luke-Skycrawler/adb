#include <Eigen/Eigen>
#include "time_integrator.h"
#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <spdlog/spdlog.h>
#include <assert.h>
#ifdef CUDA_PROJECT
#include "cuda_globals.cuh"
#endif
using namespace std;
using namespace Eigen;
using namespace utils;


#ifdef CUDA_PROJECT
void map_to_thrust_vector(map<array<int, 2>, int>& lut, thrust::host_vector<i2>& host_lut)
{
    host_lut.resize(lut.size());
    int i = 0;
    for (auto& p : lut) {
        host_lut[i++] = { p.first[0], p.first[1] };
    }
}


void compare (SparseMatrix<double> sparse_hess, CsrSparseMatrix& hess) {
    const bool strict = false;

    assert(sparse_hess.nonZeros() == hess.nnz); // "error: nnz: " ;// sparse_hess.nonZeros() ;// " " ;// hess.nnz;
    assert(sparse_hess.rows() == hess.rows); // "error: rows: " ;// sparse_hess.rows() ;// " " ;// hess.rows;
    assert(sparse_hess.cols() == hess.cols); // "error: cols: " ;// sparse_hess.cols() ;// " " ;// hess.cols;
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
        auto value_ref = sparse_hess.valuePtr()[i];
        
        //float value_act = hess.values[i];
        if (!strict) {
            if (ref != act)
                spdlog ::warn("ref, value = {}, {}", ref, act);
            // if (abs(value_ref -value_act) > 1e-1)
            //     spdlog::warn("value_ref, value_act = {}, {}", value_ref, value_act);
        }

        else
            assert(ref == act);// && abs(value_ref - value_act) < 1e-12);
    }
}

void to_csr(Eigen::SparseMatrix<double>& hess, CsrSparseMatrix& ret)
{
    ret.rows = hess.rows();
    int cols = ret.cols = hess.cols();
    int nnz = ret.nnz = hess.nonZeros();

    auto tmp_outer = thrust::host_vector<int>{ hess.outerIndexPtr(), hess.outerIndexPtr() + cols };
    tmp_outer.push_back(nnz);
    ret.outer_start = tmp_outer;
    ret.inner = thrust::device_vector<int>{ hess.innerIndexPtr(), hess.innerIndexPtr() + nnz };
    vector<float> values;
    values.resize(nnz);
    for (int i = 0; i < nnz; i++) {
        values[i] = hess.valuePtr()[i];
    }
     ret.values = thrust::device_vector<float>{ values.data(), values.data() + nnz };
}

void cuda_solve_glue(Eigen::VectorXd &_dq, Eigen::SparseMatrix<double>& sparse_hess, Eigen::VectorXd &r)
{
    auto& hess{ host_cuda_globals.hess };
    if (host_cuda_globals.params["solve_with_cuda_matrix"]) {
        
    }
    else {
        to_csr(sparse_hess, hess);
        vector<float> r_vec;
        r_vec.resize(sparse_hess.cols());
        for (int i = 0; i < sparse_hess.cols(); i++) {
            r_vec[i] = r[i];
        }
        // compare(sparse_hess, hess);
        cudaMemcpy(host_cuda_globals.b, r_vec.data(), r_vec.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

    }
    auto dq{ host_cuda_globals.dq };
    gpuCholSolver(hess, dq,  host_cuda_globals.b);
    vector<float> dq_vec;
    dq_vec.resize(sparse_hess.cols());
    cudaMemcpy(dq_vec.data(), dq, dq_vec.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < hess.cols; i++) {
        _dq[i] = -dq_vec[i];
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
    map_to_thrust_vector(lut, host_lut);

    auto cuda_gen = from_thrust(host_cuda_globals.lut);
    auto host_gen = from_thrust(host_lut);
    vector<i2> cuda_minus_host, host_minus_cuda;

    //sort(cuda_gen.begin(), cuda_gen.end());
    sort(host_gen.begin(), host_gen.end());
    
    set_difference(cuda_gen.begin(), cuda_gen.end(), host_gen.begin(), host_gen.end(), std::back_inserter(cuda_minus_host));
    set_difference(host_gen.begin(), host_gen.end(), cuda_gen.begin(), cuda_gen.end(), std::back_inserter(host_minus_cuda));

    if (host_minus_cuda.size()) {
        spdlog::error("host_minus_cuda.size() = {}", host_minus_cuda.size());
    }
    // if (cuda_minus_host.size()) {
    //     spdlog::info("cuda_minus_host.size() = {}", cuda_minus_host.size());
    //     for (auto t: cuda_minus_host) {
    //         printf("(%d, %d)", t[0], t[1]);
    //     }
    // }
    if (host_cuda_globals.params["test_placeholder"]) {
        
        thrust::device_vector<i2> dev_lut = host_lut;
        build_csr(n_cubes, dev_lut, host_cuda_globals.hess);
        auto& hess = host_cuda_globals.hess;

        compare(sparse_hess, hess);
    }
    else {
        // generate the sparse matrix according to cuda generated lut
        build_csr(n_cubes, host_cuda_globals.lut, host_cuda_globals.hess);
    }
}
#endif

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
