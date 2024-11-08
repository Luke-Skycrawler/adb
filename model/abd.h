#pragma once
#include "time_integrator.h"
#define _GLOBAL_VARIABLE_DEFINE_ONLY_
#include "collision.h"
#include "settings.h"
#include "geometry.h"
#include "ipc.h"

struct ABD {
    std::vector<std::unique_ptr<AffineBody>>& cubes;
    void implicit_euler(scalar dt);
    GlobalVariableMainCPP& globals;
    ABD(std::vector<std::unique_ptr<AffineBody>>& cubes, GlobalVariableMainCPP& globals)
        : cubes(cubes), globals(globals), n_cubes(cubes.size()), hess_dim(n_cubes * 12), ipc_assembler(n_cubes), tol(globals.params_double["tol"]), ts(globals.ts), sparse_hess(hess_dim, hess_dim) {}

    scalar line_search(const Eigen::Vector<scalar, -1>& dq, const Eigen::Vector<scalar, -1>& grad, Eigen::Vector<scalar, -1>& q0, scalar& E0, scalar& E1,
        int n_cubes, int n_pt, int n_ee, int n_g,
        std::vector<std::array<vec3, 4>>& pts,
        std::vector<std::array<int, 4>>& idx,
        std::vector<std::array<vec3, 4>>& ees,
        std::vector<std::array<int, 4>>& eidx,
        std::vector<std::array<int, 2>>& vidx,
        const std::vector<Eigen::Matrix<scalar, 2, 12>>& pt_tk,
        const std::vector<Eigen::Matrix<scalar, 2, 12>>& ee_tk,
        const std::vector<scalar>& pt_contact_forces,
        const std::vector<scalar>& ee_contact_forces,
        const std::vector<scalar>& g_contact_forces,
        const std::vector<std::unique_ptr<AffineBody>>& cubes,
        scalar dt);

    void line_search(scalar dt);
    void solve(scalar dt);
    void ccd();
    void ipc();

    static const int __IPC__ = 0;
    static const int __SOLVER__ = 1;
    static const int __CCD__ = 2;
    static const int __LINE_SEARCH__ = 3;

    const int n_cubes, hess_dim;
    IAABB culling;
    IPC ipc_assembler;
    bool term_cond;
    int& ts;
    scalar tol;
    Eigen::Vector<scalar, -1> lastdq;
    int iter = 0;

    Eigen::Vector<scalar, -1> r, q0_cat, dq;
    scalar sup_dq = 0.0;
    Eigen::Matrix<scalar, -1, -1> big_hess;

    scalar toi = 1.0, factor = 1.0;
    scalar alpha = 1.0;

    // collision set
    std::vector<std::array<vec3, 4>> pts;
    std::vector<std::array<int, 4>> idx;

    std::vector<std::array<vec3, 4>> ees;
    std::vector<std::array<int, 4>> eidx;

    std::vector<std::array<int, 2>> vidx;

    std::vector<Eigen::Matrix<scalar, 2, 12>> pt_tk;
    std::vector<Eigen::Matrix<scalar, 2, 12>> ee_tk;
    Eigen::Vector<scalar, 4> times;
    int n_pt, n_ee, n_g;
    std::vector<scalar> pt_contact_forces, ee_contact_forces, g_contact_forces;

#ifdef _TRIPLETS_
    std::vector<HessBlock> hess_triplets;
#endif
#ifdef _SM_
    // look-up table for sparse matrix
    map<std::array<int, 2>, int> lut;
    Eigen::SparseMatrix<scalar> sparse_hess;
#endif
};
