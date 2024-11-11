#include "time_integrator.h"
#include "settings.h"
#include <algorithm>
#include "othogonal_energy.h"
#include "barrier.h"

#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
using namespace std;
using namespace Eigen;
using namespace utils;
namespace utils {

vec12 pt_vstack(AffineBody& ci, AffineBody& cj, int v, int f)
{
    auto _0 = cj.indices[f * 3 + 0];
    auto _1 = cj.indices[f * 3 + 1];
    auto _2 = cj.indices[f * 3 + 2];
    vec12 v_stack;
    v_stack << ci.v_transformed[v] - ci.vt0(v),
        cj.v_transformed[_0] - cj.vt0(_0),
        cj.v_transformed[_1] - cj.vt0(_1),
        cj.v_transformed[_2] - cj.vt0(_2);
    return v_stack;
}
vec12 ee_vstack(AffineBody& ci, AffineBody& cj, int ei, int ej)
{
    vec12 v_stack;

    int i0 = ci.edges[ei * 2 + 0];
    int i1 = ci.edges[ei * 2 + 1];
    int j0 = cj.edges[ej * 2 + 0];
    int j1 = cj.edges[ej * 2 + 1];

    v_stack << ci.v_transformed[i0] - ci.vt0(i0),
        ci.v_transformed[i1] - ci.vt0(i1),
        cj.v_transformed[j0] - cj.vt0(j0),
        cj.v_transformed[j1] - cj.vt0(j1);
    return v_stack;
}

#ifndef TESTING
vec12 cat(const q4& q)
{
    vec12 ret;
    for (int i = 0; i < 4; i++) {
        ret.segment<3>(i * 3) = q[i];
    }
    return ret;
}

vec12 grad_residue_per_body(AffineBody& c, scalar dt)
{
    if (c.mass < 0.0) 
    return vec12::Zero(12);
    vec12 grad = othogonal_energy::grad(c.q);
    const auto M = [&](const vec12& dq) -> vec12 {
        auto ret = dq;
        ret.head(3) *= c.mass;
        ret.tail(9) *= c.Ic;
        return ret;
    };
    return dt * dt * grad + M(cat(c.q) - c.q_tile(dt, globals.gravity));
}

mat12 hess_inertia_per_body(AffineBody& c, scalar dt)
{
    mat12 H;
    H.setZero(12, 12);
    if (c.mass < 0.0) return mat12::Identity(12, 12);
    for (int i = 0; i < 3; i++) H(i, i) = c.mass;
    for (int i = 3; i < 12; i++) H(i, i) = c.Ic;
    auto hess_otho = othogonal_energy::hessian(c.q);
    return H + hess_otho * dt * dt;
}

scalar norm_M(const vec12& x, const AffineBody& c)
{
    // assert shape of x
    auto p = x.head(3);
    auto q = x.tail(9);
    return p.dot(p) * c.mass + q.dot(q) * c.Ic;
}

scalar norm_1(Vector<scalar, -1>& dq, int n_cubes)
{
    scalar norm = 0.0;
    for (int i = 0; i < n_cubes; i++) {
        auto dqs = dq.segment<12>(i * 12);
        norm = max(norm, dqs.array().abs().sum());
    }
    return norm;
}

void damping_dense(Matrix<scalar, -1, -1>& big_hess, scalar dt, int n_cubes)
{
    Matrix<scalar, -1, -1> D = globals.beta * big_hess;
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < 3; j++) { D(i * 12 + j, i * 12 + j) += globals.cubes[i]->mass; }
        for (int j = 3; j < 12; j++) { D(i * 12 + j, i * 12 + j) += globals.cubes[i]->Ic; }
    }
    big_hess += D / dt;
}

void damping_sparse(SparseMatrix<scalar>& sparse_hess, scalar dt, int n_cubes)
{
    SparseMatrix<scalar>& D = sparse_hess;
    sparse_hess *= (1 + globals.beta);
    scalar dab = globals.alpha - globals.beta;
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < 3; j++) { D.coeffRef(i * 12 + j, i * 12 + j) += globals.cubes[i]->mass * dab; }
        for (int j = 3; j < 12; j++) { D.coeffRef(i * 12 + j, i * 12 + j) += globals.cubes[i]->Ic * dab; }
    }
}

void build_from_triplets(SparseMatrix<scalar>& sparse_hess_trip, Matrix<scalar, -1, -1>& big_hess, int hess_dim, int n_cubes, vector<HessBlock> hess_triplets)
{

    vector<int> starting_point;
    starting_point.resize(n_cubes * 12);
    static const auto merge_triplets = [&](vector<HessBlock>& triplets) {
        sort(triplets.begin(), triplets.end(), [&](const HessBlock& a, const HessBlock& b) -> bool {
            return a.j < b.j || (a.j == b.j && a.i < b.i);
        });
        int n = triplets.size();
        for (int i = 0; i < n; i++) {
            if (triplets[i].i == -1) continue;
            for (int j = i + 1; j < n; j++)
                if (triplets[i].i == triplets[j].i && triplets[i].j == triplets[j].j) {
                    // disable triplet j
                    triplets[j].i = -1;
                    triplets[i].block += triplets[j].block;
                }
                else {
                    if (triplets[i].j != triplets[j].j)
                        starting_point[triplets[j].j] = j;
                    break;
                }
        }
        starting_point[0] = 0;
    };

    // vector<Triplet<scalar>> bht;

    // static const auto insert = [&](vector<Triplet<scalar>>& bht, const Matrix<scalar, 12, 12>& m, int r, int c) {
    //     for (int i = 0; i < 12; i++)
    //         for (int j = 0; j < 12; j++) {
    //             bht.push_back({ r + i, c + j, m(i, j) });
    //         }
    // };
    static const auto insert2 = [&](SparseMatrix<scalar>& sm, int tid) {
        auto& triplet = hess_triplets[tid];
        bool new_col = tid == 0 || hess_triplets[tid - 1].j != triplet.j;

        int c = triplet.j, r = triplet.i;
        if (new_col)
            sm.startVec(c);
        for (int i = 0; i < 12; i++) {
            sm.insertBack(i + r, c) = triplet.block(i, 0);
        }
    };

    if (globals.sparse) {
        int n_ele = (n_cubes + hess_triplets.size()) * 12 * 12;
        // bht.resize(n_ele);
        // sparse_hess.reserve(n_ele);
    }

    for (int k = 0; k < n_cubes; k++) {
        // if (globals.dense)
        //     big_hess.block<12, 12>(k * 12, k * 12) += cubes[k].hess;

        // if(globals.sparse)
        //     insert(bht, cubes[k].hess, k * 12, k * 12);
        auto& c = *globals.cubes[k];
        for (int i = 0; i < 12; i++)
            hess_triplets.push_back({ k * 12, k * 12 + i, c.hess.block<12, 1>(0, i) });
        // hess_triplets.push_back({k * 12, k * 12, cubes[k].hess});
    }

    merge_triplets(hess_triplets);

    const int nt = hess_triplets.size();

    for (int k = 0; k < nt; k++) {
        auto& triplet(hess_triplets[k]);
        if (triplet.i == -1) continue;
        if (globals.dense)
            big_hess.block<12, 1>(triplet.i, triplet.j) = triplet.block;
        // big_hess.block<12, 12>(triplet.i, triplet.j) = triplet.block;
        if (globals.sparse)
            // insert(bht, triplet.block, triplet.i * 12, triplet.j * 12);
            insert2(sparse_hess_trip, k);
    }
}
scalar E(const vec12& q, const vec12& q_tiled, const AffineBody& c, scalar dt)
{
    if(c.mass < 0.0) return 0.0;
    return othogonal_energy::otho_energy(q) * dt * dt + 0.5 * norm_M(q - q_tiled, c);
}
vector<array<int, 2>> gen_edge_list(
    vector<unique_ptr<AffineBody>>& cubes, int n_cubes)
{
    vector<array<int, 2>> ret;
    int sz = 0;
    for (int i = 0; i < n_cubes; i++) sz += cubes[i]->n_edges;
    ret.reserve(sz);
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < cubes[i]->n_edges; j++)
            ret.push_back({ i, j });
    }
    return ret;
}
vector<array<int, 2>> gen_point_list(
    vector<unique_ptr<AffineBody>>& cubes, int n_cubes)
{
    vector<array<int, 2>> ret;
    int sz = 0;
    for (int i = 0; i < n_cubes; i++) sz += cubes[i]->n_vertices;
    ret.reserve(sz);
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < cubes[i]->n_vertices; j++)
            ret.push_back({ i, j });
    }
    return ret;
}
vector<array<int, 2>> gen_triangle_list(
    vector<unique_ptr<AffineBody>>& cubes, int n_cubes)
{
    vector<array<int, 2>> ret;
    int sz = 0;
    for (int i = 0; i < n_cubes; i++) sz += cubes[i]->n_faces;
    ret.reserve(sz);
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < cubes[i]->n_faces; j++) {
            ret.push_back({ i, j });
        }
    }
    return ret;
}
#endif
};
#ifndef TESTING
vec12 AffineBody::q_tile(scalar dt, const vec3& f) const
{
    auto _q = cat(q0);
    auto _dqdt = cat(dqdt);
    _q = _q + dt * _dqdt;
    _q.head(3) += dt * dt * f;
    return _q;
}

#include <iostream>
#include <fstream>
#include <filesystem>
void player_load(
    std::string& path,
    int timestep,
    const std::vector<std::unique_ptr<AffineBody>>& cubes)
{
    string filename = path + "/" + to_string(timestep);
    ifstream in(filename, ios::in | ios::binary);
    int n_cubes = cubes.size();

    for (int i = 0; i < n_cubes; i++) {
        auto& c{ *cubes[i] };
        for (int j = 0; j < 4; j++)
            in.read((char*)c.q0[j].data(), 3 * sizeof(scalar));
        for (int j = 0; j < 4; j++)
            in.read((char*)c.dqdt[j].data(), 3 * sizeof(scalar));
        c.p = c.q0[0];
        c.A << c.q0[1], c.q0[2], c.q0[3];
    }
    in.close();
}

void dump_states(
    const std::vector<std::unique_ptr<AffineBody>>& cubes
){
    string filename = "trace/ub_failure";
    ofstream out(filename, ios::out | ios::binary | ios::trunc);
    int n_cubes = cubes.size();
    for (int i = 0; i < n_cubes; i++) {
        auto& c{ *cubes[i] };
        for (int j = 0; j < 4; j++)
            out.write((char*)c.q[j].data(), 3 * sizeof(scalar));
        for (int j = 0; j < 4; j++)
            out.write((char*)c.dq.segment<3>(j * 3).data(), 3 * sizeof(scalar));
    }
    out.close();
}
void player_save(
    std::string& path,
    int timestep,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    bool init)
{

    if (init) {

        // TODO: save config.json, scene.json along with binary files

            filesystem::path folder(path);
            filesystem::path config("../config.json");
            filesystem::path scene(globals.scene);

            if (filesystem::exists(path)) {
                    filesystem::remove_all(path);
        }

        filesystem::create_directory(path);

        filesystem::copy(config, folder);
        filesystem::copy(scene, folder);
    }

    string filename = path + "/" + to_string(timestep);
    ofstream out(filename, ios::out | ios::binary | ios::trunc);
    int n_cubes = cubes.size();
    for (int i = 0; i < n_cubes; i++) {
        auto& c{ *cubes[i] };
        for (int j = 0; j < 4; j++)
            out.write((char*)c.q0[j].data(), 3 * sizeof(scalar));
        for (int j = 0; j < 4; j++)
            out.write((char*)c.dqdt[j].data(), 3 * sizeof(scalar));
    }
    out.close();
}
#endif