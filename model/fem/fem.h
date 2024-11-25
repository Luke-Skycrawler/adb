#pragma once
#include "../scalar_types.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
struct TOBJLoader {
    Eigen::Matrix<scalar, -1, -1> V;
    Eigen::Matrix<int, -1, -1> F, T;
    std::vector<vec3> v_deformed;     
    TOBJLoader(const std::string &filename);    
    int n_nodes, n_tets; 
    void import_tobj(const std::string& filename);
};

struct FEMTet: Tet {
    mat3 Bm;
    mat3 dPK(const mat3 &F, const mat3 &dF);
    Eigen::Vector4i index; 
    scalar W;
};

struct TetFEM: TOBJLoader {
    Eigen::SparseMatrix<scalar, Eigen::ColMajor> K, M; 
    Eigen::Vector<scalar, -1> M_diag, dx, b;
    TetFEM(const std::string &filename); 
    void eigs(Eigen::Vector<scalar, -1> &lambdas, Eigen::Matrix<scalar, -1, -1> &Q);
    const scalar mu = 2e6, lambda = 125;

    void step(scalar dt);
    std::vector<FEMTet> tets;
    std::vector<vec3> velocity_0, x_0;
    // velocity and position of last time step 

    void solve();
    
    private:

    void precompute_Dm();
    void tet_kernel_sifakis();
    void define_KM();
    void tet_kernel();
    vec3 center_of_tet(int e);
    vec3 bf_tet(int e, int i, const vec3 &x) const;
    scalar volume(int e) const;
    vec3 normal(int e, int i) const;
    void force_residues();
    mat3 PK1(const mat3 &F) const;
    std::vector<Eigen::Triplet<scalar>> triplets;
    
};