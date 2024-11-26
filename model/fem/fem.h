#pragma once
#include "../scalar_types.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <memory>
struct Vertex
{
  vec3 x, f, v, v_n;
  scalar M_inv;

  Vertex(const vec3 &x, scalar M_inv = 1.0f) : x(x), f(0.0f, 0.0f, 0.0f), v(0.0f, 0.0f, 0.0f), M_inv(M_inv), v_n(0.0f, 0.0f, 0.0f) {}
  Vertex() : x(0.0f, 0.f, 0.f), f(0.0f, 0.f, 0.f), v(0.f, 0.f, 0.0f), M_inv(1.0f), v_n(0.0f, 0.f, 0.0f) {}
};
struct Tetrahedron
{
  Vertex &i, &j, &k, &l;
  mat3 Bm;
  mat3 piola_tensor(mat3 &F);
  mat3 differential_piola(mat3 &F, mat3 &dF);
  int index[4];
  scalar W;

  Tetrahedron(std::vector<Vertex> &v, int i, int j, int k, int l) : i(v[i]), j(v[j]), k(v[k]), l(v[l])
  {
    precomputation();
    rem_index(i, j, k, l);
  }
  Tetrahedron(const Tetrahedron &a) : i(a.i), j(a.j), k(a.k), l(a.l)
  {
    precomputation();
    for (int i = 0; i < 4; i++)
    {
      index[i] = a.index[i];
    }
  }
  inline void rem_index(int i, int j, int k, int l)
  {
    index[0] = i;
    index[1] = j;
    index[2] = k;
    index[3] = l;
  }
  void precomputation();

  void compute_elastic_forces();
};

struct TOBJLoader
{
  Eigen::Matrix<scalar, -1, -1> V;
  Eigen::Matrix<int, -1, -1> F, T;
  std::vector<vec3> v_deformed;
  TOBJLoader(const std::string &filename);
  int n_nodes, n_tets;
  void import_tobj(const std::string &filename);
  int n;
  std::vector<Tetrahedron> tets;
  std::vector<Vertex> vtxs;
};

struct FEMObject : TOBJLoader {
    Eigen::Vector<scalar, -1> dx, b;
    Eigen::SparseMatrix<scalar> A;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<scalar>> solver;

    int sys_offset;
    FEMObject(const std::string& filename)
        : TOBJLoader(filename), dx(3 * n_nodes), b(3 * n_nodes), A(3 * n_nodes, 3 * n_nodes)
    {
    }

    void add_dx_dv(scalar dt);
    // void derive_and_add_dv();

    // fill A and b;
    void build_sparse(scalar dt);
    void compute_f();
    std::vector<Eigen::Triplet<scalar>> triplets;
    void stiffness_kernel();
    inline void solve()
    {
        // Compute the ordering permutation vector from the structural pattern of A
        // solver.analyzePattern(A);
        // // Compute the numerical factorization
        // solver.factorize(A);

        // Use the factors to solve the linear system
        solver.compute(A);
        dx = solver.solve(b);
    }
    void step(scalar dt);
};

struct FEMSimulator {
    std::vector<std::unique_ptr<FEMObject>> objects;
    Eigen::Vector<scalar, -1> dx, b;
    Eigen::SparseMatrix<scalar> A;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<scalar>> solver;
    int n_dofs;

    void step(scalar dt);
    std::vector<Eigen::Triplet<scalar>> triplets;
    FEMSimulator(const std::string& config_file);

private:
    void combine_triplets();
    inline void solve()
    {
        solver.compute(A);
        dx = solver.solve(b);
    }

    void add_dx_dv(scalar dt);
};

// struct FEMTet: Tet {
//     mat3 Bm;
//     mat3 dPK(const mat3 &F, const mat3 &dF);
//     Eigen::Vector4i index;
//     scalar W;
// };

// struct TetFEM: TOBJLoader {
//     Eigen::SparseMatrix<scalar, Eigen::ColMajor> K, M; 
//     Eigen::Vector<scalar, -1> M_diag, dx, b;
//     TetFEM(const std::string &filename); 
//     void eigs(Eigen::Vector<scalar, -1> &lambdas, Eigen::Matrix<scalar, -1, -1> &Q);
//     const scalar mu = 2e6, lambda = 125;

//     void step(scalar dt);
//     std::vector<FEMTet> tets;
//     std::vector<vec3> velocity_0, x_0;
//     // velocity and position of last time step 

//     void solve();
    
//     private:

//     void precompute_Dm();
//     void tet_kernel_sifakis();
//     void define_KM();
//     void tet_kernel();
//     vec3 center_of_tet(int e);
//     vec3 bf_tet(int e, int i, const vec3 &x) const;
//     scalar volume(int e) const;
//     vec3 normal(int e, int i) const;
//     void force_residues();
//     mat3 PK1(const mat3 &F) const;
//     std::vector<Eigen::Triplet<scalar>> triplets;
    
// };