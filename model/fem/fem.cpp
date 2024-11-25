#include "fem.h"
#include <igl/massmatrix.h>
#include <Eigen/Sparse>
#include <spdlog/spdlog.h>
#include <fstream>
#include <igl/boundary_facets.h>
using namespace Eigen;
using namespace std;

static const scalar mu = 2e6, lam = 125.0;
static const vec3 gravity(0.0, -9.8, 0.0);
static const scalar tol = 1e-6;

// mat3 FEMTet::dPK(const mat3& F, const mat3& dF)
// {
//     mat3 F_inv = F.inverse();
//     mat3 F_inv_T = F_inv.transpose();
//     mat3 B = F_inv * dF;
//     scalar det_F = F.determinant();

//     return mu * dF + (mu - lam * log(det_F)) * F_inv_T * dF.transpose() * F_inv_T + (lam * B.trace()) * F_inv_T;
// }
// mat3 TetFEM::PK1(const mat3& F) const
// {
//     mat3 F_inv_T = F.inverse().transpose();
//     return mu * (F - F_inv_T) + lam * log(F.determinant()) / 2.0 * F_inv_T;
// }
// void TetFEM::precompute_Dm()
// {
//     for(int e = 0; e < n_tets; e++) {
//         vec3 x0 = V.row(T(e, 0));
//         vec3 x1 = V.row(T(e, 1));
//         vec3 x2 = V.row(T(e, 2));
//         vec3 x3 = V.row(T(e, 3));

//         mat3 Dm;
//         Dm << x0 - x3, x1 - x3, x2 - x3;
//         mat3 inv_Dm = Dm.inverse();
//         tets[e].Bm = inv_Dm;
//         tets[e].W = abs(Dm.determinant()) / 6.0;
//     }
// }
// void TetFEM::force_residues()
// {
//     for(int i = 0; i < n_nodes; i++) {
//         b.segment<3>(i * 3) = M_diag(i) * gravity;
//     }
//     for(int e = 0; e < n_tets; e++) {
//         mat3 Ds;
//         vec3 t0 = v_deformed[T(e, 0)];
//         vec3 t1 = v_deformed[T(e, 1)];
//         vec3 t2 = v_deformed[T(e, 2)];
//         vec3 t3 = v_deformed[T(e, 3)];

//         Ds << t0 - t3, t1 - t3, t2 - t3;
//         mat3 F = Ds * tets[e].Bm;
//         mat3 P = PK1(F);
//         mat3 H = -tets[e].W * P * tets[e].Bm.transpose();
//         int i = T(e, 0);
//         int j = T(e, 1);
//         int k = T(e, 2);
//         int l = T(e, 3);
//         b.segment<3>(i * 3) += H.col(0);
//         b.segment<3>(j * 3) += H.col(1);
//         b.segment<3>(k * 3) += H.col(2);
//         b.segment<3>(l * 3) -= H.col(0) + H.col(1) + H.col(2);
//     }
// }
// TetFEM::TetFEM(const string& filename)
//     : TOBJLoader(filename)
// {
//     x_0.resize(n_nodes);
//     velocity_0.resize(n_nodes);
//     tets.resize(n_tets);
//     b.resize(n_nodes * 3);
//     M_diag.resize(n_nodes * 3);
//     dx.resize(n_nodes * 3);
//     b.setZero();
//     M_diag.setZero();
//     dx.setZero();
//     K.resize(n_nodes * 3, n_nodes * 3);
//     for (int i = 0; i < n_nodes; i ++) {
//         x_0[i] = v_deformed[i]; 
//         velocity_0[i].setZero();
//     }
//     define_KM();
// }

// void TetFEM::eigs(Vector<scalar, -1>& lambdas, Matrix<scalar, -1, -1>& Q)
// {
// }
// void TetFEM::tet_kernel_sifakis()
// {
//     for(int e = 0; e < n_tets; e++) {
//         for(int _j = 0; _j < 4; _j++) {
//             vec3 t0 = v_deformed[T(e, 0)];
//             vec3 t1 = v_deformed[T(e, 1)];
//             vec3 t2 = v_deformed[T(e, 2)];
//             vec3 t3 = v_deformed[T(e, 3)];

//             mat3 Ds;
//             Ds << t0 - t3, t1 - t3, t2 - t3;

//             mat3 F = Ds * tets[e].Bm;
//             for(int _i = 0; _i < 4; _i++) {
//                 for(int k = 0; k < 3; k++) {
//                     mat3 dDs;
//                     dDs.setZero();
//                     if(_j < 3) {
//                         dDs(k, _j) = -1.0;
//                     }
//                     else {
//                         dDs(k, 0) = 1.0;
//                         dDs(k, 1) = 1.0;
//                         dDs(k, 2) = 1.0;
//                     }
//                     mat3 dF = dDs * tets[e].Bm;
//                     mat3 dP = dF * tets[e].dPK(F, dF);
//                     mat3 dH = -tets[e].W * dP * tets[e].Bm.transpose();
//                     int i = T(e, _i);
//                     int j = T(e, _j);

//                     vec3 df;
//                     if(_i == 3) {
//                         df = -vec3(dH(0, 0) + dH(0, 1) + dH(0, 2), dH(1, 0) + dH(1, 1) + dH(1, 2), dH(2, 0) + dH(2, 1) + dH(2, 2));
//                     }
//                     else {
//                         df = vec3(dH(0, _i), dH(1, _i), dH(2, _i));
//                     }
//                     for(int l = 0; l < 3; l++) {
//                         triplets.push_back({ i * 3 + l, j * 3 + k, df(l) });
//                     }
//                 }
//             }
//         }
//     }

// }

// void TetFEM::define_KM()
// {
//     // compute K and M
//     precompute_Dm();
//     cout << tets[0].Bm;
//     tet_kernel_sifakis();
//     K.setFromTriplets(triplets.begin(), triplets.end());
//     cout << K.block(0, 0, 3, 3);
//     SparseMatrix<scalar> M1;
//     igl::massmatrix(V, T, igl::MASSMATRIX_TYPE_BARYCENTRIC, M1);
//     auto M1_diag = M1.diagonal();
//     M_diag.resize(M1_diag.size() * 3);
//     for(int i = 0; i < n_nodes; i++) {
//         M_diag(i * 3) = M1_diag(i);
//         M_diag(i * 3 + 1) = M1_diag(i);
//         M_diag(i * 3 + 2) = M1_diag(i);
//     }
//     exit(0);
//     //M = Eigen::SparseMatrix<scalar>(M_diag.asDiagonal());
// }

// void TetFEM::tet_kernel()
// {

//     for(int e = 0; e < n_tets; e++) {
//         vec3 x = center_of_tet(e);
//         for(int _i = 0; _i < 4; _i++) {
//             int i = T(e, _i);
//             vec3 dbidx = bf_tet(e, _i, x);
//             for(int _j = 0; _j < 4; _j++) {
//                 int j = T(e, _j);
//                 vec3 dbjdx = bf_tet(e, _j, x);
//                 for(int k = 0; k < 3; k++) {
//                     mat3 grad_v = vec3::Unit(k) * dbidx.transpose();
//                     mat3 eps = (grad_v + grad_v.transpose()) / 2;
//                     vec3 c = eps.trace() * lambda * dbjdx + 2 * mu * eps * dbjdx;
//                     for(int l = 0; l < 3; l++) {
//                         triplets.push_back({ i * 3 + k, j * 3 + l, c(l) * volume(e) });
//                     }
//                 }
//             }
//         }
//     }
// }

// vec3 TetFEM::center_of_tet(int e)
// {
//     vec3 x(0.0, 0.0, 0.0);
//     for(int i = 0; i < 4; i++) {
//         x += V.row(T(e, i));
//     }
//     return x / 4;
// }

// vec3 TetFEM::bf_tet(int e, int _i, const vec3& x) const
// {
//     vec3 n = normal(e, _i);
//     vec3 x0 = V.row(T(e, _i));
//     scalar k = 0.75 / ((x0 - x).dot(n));
//     return k * n;
// }

// vec3 TetFEM::normal(int e, int _i) const
// {
//     Vector4i vi = T.row(e);
//     int v = T(e, _i);
//     vec3 x = V.row(v);
//     vi[_i] = vi[3];

//     vec3 x0 = V.row(vi[0]);
//     vec3 x1 = V.row(vi[1]);
//     vec3 x2 = V.row(vi[2]);

//     vec3 n = (x1 - x0).cross(x2 - x0).normalized();

//     if(n.dot(x - x0) < 0.0) {
//         n = -n;
//     }
//     return n;
// }
// scalar TetFEM::volume(int e) const
// {
//     vec3 x0 = V.row(T(e, 0));
//     vec3 x1 = V.row(T(e, 1));
//     vec3 x2 = V.row(T(e, 2));
//     vec3 x3 = V.row(T(e, 3));

//     return (1.0 / 6.0) * (x1 - x0).cross(x2 - x0).dot(x3 - x0);
// }

// void TetFEM::step(scalar dt)
// {

//     bool term_cond = false;
//     int iter = 0;
//     do {
//         cout << "newton iter" << iter++ << endl;
//         triplets.resize(0);
//         K.setZero();
//         tet_kernel_sifakis();
//         // get K
//         force_residues();
//         // get b


//         for (int i = 0; i < n_nodes; i ++) {
//             for (int k = 0; k < 3; k ++) {
                
//                 triplets.push_back({ i * 3 + k, i * 3 + k, M_diag(i * 3 + k) / (dt * dt) });
//             }
//             auto m = M_diag(i * 3);
//             b.segment<3>(i * 3) += m / dt * (velocity_0[i] - (v_deformed[i] - x_0[i]) / dt);
//         }
//         // M terms
//         K.setFromTriplets(triplets.begin(), triplets.end());        

//         solve();
//         for (int i = 0 ; i <n_nodes; i ++) {
//             v_deformed[i] += dx.segment<3>(i * 3);   
//         }
//         term_cond = b.norm() < tol;
//     } while(!term_cond);

//     for (int i = 0; i < n_nodes; i++) {
//         velocity_0[i] = (v_deformed[i] - x_0[i]) / dt;
//         x_0[i] = v_deformed[i];
//     }
// }

// void TetFEM::solve() {
//     SimplicialLDLT<SparseMatrix<scalar, ColMajor>> ldlt_solver;
//     ldlt_solver.compute(K);
//     dx = ldlt_solver.solve(b);
//     if(isnan(dx.norm())) {
//         spdlog::error("solver nan");
//         exit(1);
//     }
// }




void Rod::compute_f()
{
  for (auto &v : vtxs)
  {
    v.f = gravity;
  }
  for (auto &e : tets)
  {
    e.compute_elastic_forces();
    // e.compute_barrier_forces(vtxs);
  }
}

void Rod::step(scalar dt)
{
    bool term_cond = false;
    int iter = 0;
    do {
        // FIXME: specify a tolerance and max iteration count
        // newton method
        cout << "newton iter" << iter++ << endl;
        A.setZero();
        b.setZero();
        build_sparse(dt);

        compute_f();
        for(int i = 0; i < n; i++) {
            auto& vtx = vtxs[i];
            vec3 dv = vtx.v_n - vtx.v;
            for(int d = 0; d < 3; d++) {
                if(vtxs[i].M_inv == 0.0f) {
                    b(i * 3 + d) = 0.0f;
                }
                else
                    b(i * 3 + d) = (1.0f / dt) * dv[d] + vtx.f[d];
            }
        }

        solve();
        add_dx_dv(dt);
        term_cond = dx.norm() < tol;
    } while(!term_cond);
    for(int i = 0; i < n_nodes; i++) {
        vtxs[i].v_n = vtxs[i].v;
        v_deformed[i] = vtxs[i].x;
    }
}

void Rod::add_dx_dv(scalar dt)
{
  for (int i = 0; i < n; i++)
  {
    int I = 3 * i;
    vec3 tmp(
        dx.coeff(I + 0),
        dx.coeff(I + 1),
        dx.coeff(I + 2));
    vtxs[i].x += tmp;
    vtxs[i].v += tmp / dt;
  }
}

void Rod::stiffness_kernel()
{
  A.setZero();
  for (auto &e : tets)
  {
    for (int _j = 0; _j < 4; _j++)
    {
      mat3 Ds;
      Ds << e.i.x - e.l.x, e.j.x - e.l.x, e.k.x - e.l.x;
      mat3 F = Ds * e.Bm;
      int J = e.index[_j];
      if (vtxs[J].M_inv == 0.0f)
        continue;
      for (int k = 0; k < 3; k++)
      {
        // like ti.static
        vec3 d_v[4], df[4];
        for (int _ = 0; _ < 4; _++)
        {
          d_v[_].setZero();
          df[_].setZero();
        }
        d_v[_j][k] = -1.0f;
        mat3 d_Ds;
        d_Ds << d_v[0] - d_v[3], d_v[1] - d_v[3], d_v[2] - d_v[3];
        mat3 dF(d_Ds * e.Bm);
        mat3 dP(e.differential_piola(F, dF));
        mat3 dH(-e.W * dP * e.Bm.transpose());
        df[0] = dH.col(0);
        df[1] = dH.col(1);
        df[2] = dH.col(2);
        df[3] = -(dH.col(0) + dH.col(1) + dH.col(2));

        /* put vec3 to
        row = [3i, 3i+2]
        col = 3j + k
        */
        int col = 3 * J + k;
        for (int _i = 0; _i < 4; _i++)
        {
          int I = e.index[_i];
          if (vtxs[I].M_inv == 0.0f)
            continue;

          for (int l = 0; l < 3; l++)
          {
            triplets.push_back({3 * I + l, col, df[_i][l]});
            // A.coeffRef(3 * I + l, col) += df[_i][l];
          }
        }
      }
    }
  }
}
void Rod::build_sparse(scalar dt)
{
  A.setZero();
  triplets.clear();
  stiffness_kernel();
  for (int i = 0; i < 3 * n; i++)
  {
    triplets.push_back({i, i, 1.0f / (dt * dt)});
  }
  A.setFromTriplets(triplets.begin(), triplets.end());
}

void Tetrahedron::precomputation()
{
  mat3 Dm;
  Dm << i.x - l.x, j.x - l.x, k.x - l.x;
  Bm = Dm.inverse();
  W = abs(1.0f / 6 * Dm.determinant());
}

void Tetrahedron::compute_elastic_forces()
{
  mat3 Ds;
  Ds << i.x - l.x, j.x - l.x, k.x - l.x;
  mat3 F = Ds * Bm;
  mat3 P = piola_tensor(F);
  mat3 H = -W * P * Bm.transpose();
  i.f += H.col(0);
  j.f += H.col(1);
  k.f += H.col(2);
  l.f += -(H.col(0) + H.col(1) + H.col(2));
}

static const scalar d_hat = 0.08, r_boundary = 0.6, kappa = 1000.0f;
inline scalar barrier_second_dirivative(scalar x){
  // absolute value
  return kappa * ((x - d_hat) * (d_hat / (x * x) + 3 / x) + 2 * log(x / d_hat)); 
}

inline vec3 barrier_gradient(scalar x){
  // returns the absolute value of the gradient
  return kappa * (vec3((x - d_hat) * ((x-d_hat)/x + 2 * log(x / d_hat)), 0.0f, 0.0f));
}
inline void barrier(SparseMatrix<scalar> &K, int i, scalar df){
  int I = 3 * i;
  K.coeffRef(I, I) += df;
}

mat3 Tetrahedron::piola_tensor(mat3 &F)
{
  // linear elasticity
  // return (F + transpose(F) - mat3(1.0f) * 2.0f) * mu + mat3(1.0f) * lambda * (F[0][0] + F[1][1] + F[2][2] - 3.0f);
  // neo-hookean
  auto F_inv_T = F.inverse().transpose();
  return mu * (F - F_inv_T) + lam * log(F.determinant()) / 2.0f * F_inv_T;
}

mat3 Tetrahedron::differential_piola(mat3 &F, mat3 &dF)
{
  auto F_inv_T = F.inverse().transpose();
  mat3 B = F.inverse() * dF;
  scalar tr = B.trace();
  return mu * dF + (mu - lam * log(F.determinant())) * F_inv_T * dF.transpose() * F_inv_T + lam * tr * F_inv_T;
}

void TOBJLoader::import_tobj(const string &filename)
{
  ifstream file(filename);
  string line;
  vector<vec3> vertices;
  vector<array<int, 4>> tets;
  while (getline(file, line))
  {
    istringstream iss(line);
    string type;
    iss >> type;
    if (type == "v")
    {
      scalar v0, v1, v2;
      iss >> v0 >> v1 >> v2;
      vertices.push_back(vec3(v0, v1, v2));
    }
    else if (type == "t")
    {
      int t0, t1, t2, t3;
      iss >> t0 >> t1 >> t2 >> t3;
      tets.push_back({t0, t1, t2, t3});
    }
  }
  V.resize(vertices.size(), 3);
  T.resize(tets.size(), 4);
  for (int i = 0; i < vertices.size(); i++)
  {
    V.row(i) = vertices[i];
  }
  for (int i = 0; i < tets.size(); i++)
  {
    T.row(i) = Eigen::Vector4i{tets[i][0], tets[i][1], tets[i][2], tets[i][3]};
  }
}

TOBJLoader::TOBJLoader(const string &filename)
{
  import_tobj(filename);
  n_nodes = V.rows();
  n_tets = T.rows();

  v_deformed.resize(n_nodes);
  for (int i = 0; i < n_nodes; i++)
  {
    v_deformed[i] = V.row(i);
  }
  igl::boundary_facets(T, F);
  cout << filename << "loaded, " << n_nodes << " nodes, " << n_tets << " tets\n";

  n = n_nodes;
  for (int i = 0; i < n_nodes; i++)
  {
    scalar M_inv = i < 25 ? 0.0f : 1.0f;
    vtxs.push_back({vec3(v_deformed[i][0], v_deformed[i][1], v_deformed[i][2]), M_inv});
  }
  for (int e = 0; e < n_tets; e++)
  {
    tets.push_back({vtxs, T(e, 0), T(e, 1), T(e, 2), T(e, 3)});
  }
}
