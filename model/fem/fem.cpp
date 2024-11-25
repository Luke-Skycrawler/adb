#include "fem.h"
#include <igl/massmatrix.h>
#include <Eigen/Sparse>
#include <spdlog/spdlog.h>

using namespace Eigen;
using namespace std;

static const scalar mu = 2e6, lam = 125.0;
static const vec3 gravity(0.0, -9.8, 0.0);
static const scalar tol = 1e-6;

mat3 FEMTet::dPK(const mat3& F, const mat3& dF)
{
    mat3 F_inv = F.inverse();
    mat3 F_inv_T = F_inv.transpose();
    mat3 B = F_inv * dF;
    scalar det_F = F.determinant();

    return mu * dF + (mu - lam * log(det_F)) * F_inv_T * dF.transpose() * F_inv_T + (lam * B.trace()) * F_inv_T;
}
mat3 TetFEM::PK1(const mat3& F) const
{
    mat3 F_inv_T = F.inverse().transpose();
    return mu * (F - F_inv_T) + lam * log(F.determinant()) / 2.0 * F_inv_T;
}
void TetFEM::precompute_Dm()
{
    for(int e = 0; e < n_tets; e++) {
        vec3 x0 = V.row(T(e, 0));
        vec3 x1 = V.row(T(e, 1));
        vec3 x2 = V.row(T(e, 2));
        vec3 x3 = V.row(T(e, 3));

        mat3 Dm;
        Dm << x0 - x3, x1 - x3, x2 - x3;
        mat3 inv_Dm = Dm.inverse();
        tets[e].Bm = inv_Dm;
        tets[e].W = abs(Dm.determinant()) / 6.0;
    }
}
void TetFEM::force_residues()
{
    for(int i = 0; i < n_nodes; i++) {
        b.segment<3>(i * 3) = M_diag(i) * gravity;
    }
    for(int e = 0; e < n_tets; e++) {
        mat3 Ds;
        vec3 t0 = v_deformed[T(e, 0)];
        vec3 t1 = v_deformed[T(e, 1)];
        vec3 t2 = v_deformed[T(e, 2)];
        vec3 t3 = v_deformed[T(e, 3)];

        Ds << t0 - t3, t1 - t3, t2 - t3;
        mat3 F = Ds * tets[e].Bm;
        mat3 P = PK1(F);
        mat3 H = -tets[e].W * P * tets[e].Bm.transpose();
        int i = T(e, 0);
        int j = T(e, 1);
        int k = T(e, 2);
        int l = T(e, 3);
        b.segment<3>(i * 3) += H.col(0);
        b.segment<3>(j * 3) += H.col(1);
        b.segment<3>(k * 3) += H.col(2);
        b.segment<3>(l * 3) -= H.col(0) + H.col(1) + H.col(2);
    }
}
TetFEM::TetFEM(const string& filename)
    : TOBJLoader(filename)
{
    x_0.resize(n_nodes);
    velocity_0.resize(n_nodes);
    tets.resize(n_tets);
    b.resize(n_nodes * 3);
    M_diag.resize(n_nodes * 3);
    dx.resize(n_nodes * 3);
    b.setZero();
    M_diag.setZero();
    dx.setZero();
    K.resize(n_nodes * 3, n_nodes * 3);
    for (int i = 0; i < n_nodes; i ++) {
        x_0[i] = v_deformed[i]; 
        velocity_0[i].setZero();
    }
    define_KM();
}

void TetFEM::eigs(Vector<scalar, -1>& lambdas, Matrix<scalar, -1, -1>& Q)
{
}
void TetFEM::tet_kernel_sifakis()
{
    for(int e = 0; e < n_tets; e++) {
        for(int _j = 0; _j < 4; _j++) {
            vec3 t0 = v_deformed[T(e, 0)];
            vec3 t1 = v_deformed[T(e, 1)];
            vec3 t2 = v_deformed[T(e, 2)];
            vec3 t3 = v_deformed[T(e, 3)];

            mat3 Ds;
            Ds << t0 - t3, t1 - t3, t2 - t3;

            mat3 F = Ds * tets[e].Bm;
            for(int _i = 0; _i < 4; _i++) {
                for(int k = 0; k < 3; k++) {
                    mat3 dDs;
                    dDs.setZero();
                    if(_j < 3) {
                        dDs(k, _j) = -1.0;
                    }
                    else {
                        dDs(k, 0) = 1.0;
                        dDs(k, 1) = 1.0;
                        dDs(k, 2) = 1.0;
                    }
                    mat3 dF = dDs * tets[e].Bm;
                    mat3 dP = dF * tets[e].dPK(F, dF);
                    mat3 dH = -tets[e].W * dP * tets[e].Bm.transpose();
                    int i = T(e, _i);
                    int j = T(e, _j);

                    vec3 df;
                    if(_i == 3) {
                        df = -vec3(dH(0, 0) + dH(0, 1) + dH(0, 2), dH(1, 0) + dH(1, 1) + dH(1, 2), dH(2, 0) + dH(2, 1) + dH(2, 2));
                    }
                    else {
                        df = vec3(dH(0, _i), dH(1, _i), dH(2, _i));
                    }
                    for(int l = 0; l < 3; l++) {
                        triplets.push_back({ i * 3 + l, j * 3 + k, df(l) });
                    }
                }
            }
        }
    }

}

void TetFEM::define_KM()
{
    // compute K and M
    precompute_Dm();
    cout << tets[0].Bm;
    tet_kernel_sifakis();
    K.setFromTriplets(triplets.begin(), triplets.end());
    cout << K.block(0, 0, 3, 3);
    SparseMatrix<scalar> M1;
    igl::massmatrix(V, T, igl::MASSMATRIX_TYPE_BARYCENTRIC, M1);
    auto M1_diag = M1.diagonal();
    M_diag.resize(M1_diag.size() * 3);
    for(int i = 0; i < n_nodes; i++) {
        M_diag(i * 3) = M1_diag(i);
        M_diag(i * 3 + 1) = M1_diag(i);
        M_diag(i * 3 + 2) = M1_diag(i);
    }
    exit(0);
    //M = Eigen::SparseMatrix<scalar>(M_diag.asDiagonal());
}

void TetFEM::tet_kernel()
{

    for(int e = 0; e < n_tets; e++) {
        vec3 x = center_of_tet(e);
        for(int _i = 0; _i < 4; _i++) {
            int i = T(e, _i);
            vec3 dbidx = bf_tet(e, _i, x);
            for(int _j = 0; _j < 4; _j++) {
                int j = T(e, _j);
                vec3 dbjdx = bf_tet(e, _j, x);
                for(int k = 0; k < 3; k++) {
                    mat3 grad_v = vec3::Unit(k) * dbidx.transpose();
                    mat3 eps = (grad_v + grad_v.transpose()) / 2;
                    vec3 c = eps.trace() * lambda * dbjdx + 2 * mu * eps * dbjdx;
                    for(int l = 0; l < 3; l++) {
                        triplets.push_back({ i * 3 + k, j * 3 + l, c(l) * volume(e) });
                    }
                }
            }
        }
    }
}

vec3 TetFEM::center_of_tet(int e)
{
    vec3 x(0.0, 0.0, 0.0);
    for(int i = 0; i < 4; i++) {
        x += V.row(T(e, i));
    }
    return x / 4;
}

vec3 TetFEM::bf_tet(int e, int _i, const vec3& x) const
{
    vec3 n = normal(e, _i);
    vec3 x0 = V.row(T(e, _i));
    scalar k = 0.75 / ((x0 - x).dot(n));
    return k * n;
}

vec3 TetFEM::normal(int e, int _i) const
{
    Vector4i vi = T.row(e);
    int v = T(e, _i);
    vec3 x = V.row(v);
    vi[_i] = vi[3];

    vec3 x0 = V.row(vi[0]);
    vec3 x1 = V.row(vi[1]);
    vec3 x2 = V.row(vi[2]);

    vec3 n = (x1 - x0).cross(x2 - x0).normalized();

    if(n.dot(x - x0) < 0.0) {
        n = -n;
    }
    return n;
}
scalar TetFEM::volume(int e) const
{
    vec3 x0 = V.row(T(e, 0));
    vec3 x1 = V.row(T(e, 1));
    vec3 x2 = V.row(T(e, 2));
    vec3 x3 = V.row(T(e, 3));

    return (1.0 / 6.0) * (x1 - x0).cross(x2 - x0).dot(x3 - x0);
}

void TetFEM::step(scalar dt)
{

    bool term_cond = false;
    do {
        triplets.resize(0);
        K.setZero();
        tet_kernel_sifakis();
        // get K
        force_residues();
        // get b


        for (int i = 0; i < n_nodes; i ++) {
            for (int k = 0; k < 3; k ++) {
                
                triplets.push_back({ i * 3 + k, i * 3 + k, M_diag(i * 3 + k) / (dt * dt) });
            }
            auto m = M_diag(i * 3);
            b.segment<3>(i * 3) += m / dt * (velocity_0[i] - (v_deformed[i] - x_0[i]) / dt);
        }
        // M terms
        K.setFromTriplets(triplets.begin(), triplets.end());        

        solve();
        for (int i = 0 ; i <n_nodes; i ++) {
            v_deformed[i] += dx.segment<3>(i * 3);   
        }
        term_cond = b.norm() < tol;
    } while(!term_cond);

    for (int i = 0; i < n_nodes; i++) {
        velocity_0[i] = (v_deformed[i] - x_0[i]) / dt;
        x_0[i] = v_deformed[i];
    }
}

void TetFEM::solve() {
    SimplicialLDLT<SparseMatrix<scalar, ColMajor>> ldlt_solver;
    ldlt_solver.compute(K);
    dx = ldlt_solver.solve(b);
    if(isnan(dx.norm())) {
        spdlog::error("solver nan");
        exit(1);
    }
}