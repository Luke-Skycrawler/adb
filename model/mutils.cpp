 #include "time_integrator.h"
#include "../view/global_variables.h"
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

vec12 cat(const q4& q)
{
    vec12 ret;
    for (int i = 0; i < 4; i++) {
        ret.segment<3>(i * 3) = q[i];
    }
    return ret;
}

vec12 grad_residue_per_body(AffineBody& c, double dt)
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

mat12 hess_inertia_per_body(AffineBody& c, double dt)
{
    mat12 H;
    H.setZero(12, 12);
    if (c.mass < 0.0) return mat12::Identity(12, 12);
    for (int i = 0; i < 3; i++) H(i, i) = c.mass;
    for (int i = 3; i < 12; i++) H(i, i) = c.Ic;
    auto hess_otho = othogonal_energy::hessian(c.q);
    return H + hess_otho * dt * dt;
}

double norm_M(const vec12& x, const AffineBody& c)
{
    // assert shape of x
    auto p = x.head(3);
    auto q = x.tail(9);
    return p.dot(p) * c.mass + q.dot(q) * c.Ic;
}

double norm_1(VectorXd& dq, int n_cubes)
{
    double norm = 0.0;
    for (int i = 0; i < n_cubes; i++) {
        auto dqs = dq.segment<12>(i * 12);
        norm = max(norm, dqs.array().abs().sum());
    }
    return norm;
}

void damping_dense(MatrixXd& big_hess, double dt, int n_cubes)
{
    MatrixXd D = globals.beta * big_hess;
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < 3; j++) { D(i * 12 + j, i * 12 + j) += globals.cubes[i]->mass; }
        for (int j = 3; j < 12; j++) { D(i * 12 + j, i * 12 + j) += globals.cubes[i]->Ic; }
    }
    big_hess += D / dt;
}

void damping_sparse(SparseMatrix<double>& sparse_hess, double dt, int n_cubes)
{
    SparseMatrix<double>& D = sparse_hess;
    sparse_hess *= (1 + globals.beta);
    double dab = globals.alpha - globals.beta;
    for (int i = 0; i < n_cubes; i++) {
        for (int j = 0; j < 3; j++) { D.coeffRef(i * 12 + j, i * 12 + j) += globals.cubes[i]->mass * dab; }
        for (int j = 3; j < 12; j++) { D.coeffRef(i * 12 + j, i * 12 + j) += globals.cubes[i]->Ic * dab; }
    }
}

void build_from_triplets(SparseMatrix<double>& sparse_hess_trip, MatrixXd& big_hess, int hess_dim, int n_cubes)
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

    // vector<Triplet<double>> bht;

    // static const auto insert = [&](vector<Triplet<double>>& bht, const Matrix<double, 12, 12>& m, int r, int c) {
    //     for (int i = 0; i < 12; i++)
    //         for (int j = 0; j < 12; j++) {
    //             bht.push_back({ r + i, c + j, m(i, j) });
    //         }
    // };
    static const auto insert2 = [&](SparseMatrix<double>& sm, int tid) {
        auto& triplet = globals.hess_triplets[tid];
        bool new_col = tid == 0 || globals.hess_triplets[tid - 1].j != triplet.j;

        int c = triplet.j, r = triplet.i;
        if (new_col)
            sm.startVec(c);
        for (int i = 0; i < 12; i++) {
            sm.insertBack(i + r, c) = triplet.block(i, 0);
        }
    };

    if (globals.sparse) {
        int n_ele = (n_cubes + globals.hess_triplets.size()) * 12 * 12;
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
            globals.hess_triplets.push_back({ k * 12, k * 12 + i, c.hess.block<12, 1>(0, i) });
        // globals.hess_triplets.push_back({k * 12, k * 12, cubes[k].hess});
    }

    merge_triplets(globals.hess_triplets);

    const int nt = globals.hess_triplets.size();

    for (int k = 0; k < nt; k++) {
        auto& triplet(globals.hess_triplets[k]);
        if (triplet.i == -1) continue;
        if (globals.dense)
            big_hess.block<12, 1>(triplet.i, triplet.j) = triplet.block;
        // big_hess.block<12, 12>(triplet.i, triplet.j) = triplet.block;
        if (globals.sparse)
            // insert(bht, triplet.block, triplet.i * 12, triplet.j * 12);
            insert2(sparse_hess_trip, k);
    }
}
double E(const vec12& q, const vec12& q_tiled, const AffineBody& c, double dt)
{
    if(c.mass < 0.0) return 0.0;
    return othogonal_energy::otho_energy(q) * dt * dt + 0.5 * norm_M(q - q_tiled, c);
}
vector<array<unsigned, 2>> gen_edge_list(
    vector<unique_ptr<AffineBody>>& cubes, int n_cubes)
{
    vector<array<unsigned, 2>> ret;
    int sz = 0;
    for (int i = 0; i < n_cubes; i++) sz += cubes[i]->n_edges;
    ret.reserve(sz);
    for (unsigned i = 0; i < n_cubes; i++) {
        for (unsigned j = 0; j < cubes[i]->n_edges; j++)
            ret.push_back({ i, j });
    }
    return ret;
}
vector<array<unsigned, 2>> gen_point_list(
    vector<unique_ptr<AffineBody>>& cubes, int n_cubes)
{
    vector<array<unsigned, 2>> ret;
    int sz = 0;
    for (int i = 0; i < n_cubes; i++) sz += cubes[i]->n_vertices;
    ret.reserve(sz);
    for (unsigned i = 0; i < n_cubes; i++) {
        for (unsigned j = 0; j < cubes[i]->n_vertices; j++)
            ret.push_back({ i, j });
    }
    return ret;
}
vector<array<unsigned, 2>> gen_triangle_list(
    vector<unique_ptr<AffineBody>>& cubes, int n_cubes)
{
    vector<array<unsigned, 2>> ret;
    int sz = 0;
    for (int i = 0; i < n_cubes; i++) sz += cubes[i]->n_faces;
    ret.reserve(sz);
    for (unsigned i = 0; i < n_cubes; i++) {
        for (unsigned j = 0; j < cubes[i]->n_faces; j++) {
            ret.push_back({ i, j });
        }
    }
    return ret;
}

double pt_uktk(
    AffineBody& ci, AffineBody& cj,
    array<vec3, 4>& pt, array<int, 4>& ij, ::ipc::PointTriangleDistanceType& pt_type,
    Matrix<double, 2, 12>& Tk_T_ret, Vector2d& uk_ret, double d, double dt)

{

    Vector<double, 12> v_stack = pt_vstack(ci, cj, ij[1], ij[3]);

    auto lams = ::ipc::point_triangle_closest_point(pt[0], pt[1], pt[2], pt[3]);
    array<double, 3> tlams = { 1 - lams(0) - lams(1), lams(0), lams(1) };

    if (pt_type == ::ipc::PointTriangleDistanceType::P_T)
        ; // do nothing
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_T0)
        tlams = { 1.0, 0.0, 0.0 };
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_T1)
        tlams = { 0.0, 1.0, 0.0 };
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_T2)
        tlams = { 0.0, 0.0, 1.0 };
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_E0) {
        auto elam = ::ipc::point_edge_closest_point(pt[0], pt[1], pt[2]);
        tlams = { 1.0 - elam, elam, 0.0 };
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_E1) {
        auto elam = ::ipc::point_edge_closest_point(pt[0], pt[2], pt[3]);
        tlams = { 0.0, 1.0 - elam, elam };
    }
    else if (pt_type == ::ipc::PointTriangleDistanceType::P_E2) {
        auto elam = ::ipc::point_edge_closest_point(pt[0], pt[3], pt[1]);
        tlams = { elam, 0.0, 1.0 - elam };
    }

    auto tp = (pt[1] * tlams[0] + pt[2] * tlams[1] + pt[3] * tlams[2]);
    auto closest = (pt[0] - tp).squaredNorm();
    assert(abs(d - closest) < 1e-8);
    auto Pk = ::ipc::point_triangle_tangent_basis(pt[0], pt[1], pt[2], pt[3]);
    Matrix<double, 3, 12> gamma;
    gamma.setZero(3, 12);
    for (int i = 0; i < 3; i++) {
        gamma(i, i) = -1.0;
        for (int j = 0; j < 3; j++)
            gamma(i, i + 3 + 3 * j) = tlams[j];
    }

    Tk_T_ret = Pk.transpose() * gamma;
    uk_ret = Tk_T_ret * v_stack;
    double contact_force = -barrier::barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
    return contact_force;
}
double ee_uktk(
    AffineBody& ci, AffineBody& cj,
    array<vec3, 4>& ee, array<int, 4>& ij, ::ipc::EdgeEdgeDistanceType& ee_type,
    Matrix<double, 2, 12>& Tk_T_ret, Vector2d& uk_ret, double d, double dt)
{
    auto v_stack = ee_vstack(ci, cj, ij[1], ij[3]);
    auto ei0 = ee[0], ei1 = ee[1], ej0 = ee[2], ej1 = ee[3];
    auto rei = ei0 - ei1, rej = ej0 - ej1;
    auto cnorm = rei.cross(rej).squaredNorm();
    auto sin2 = cnorm / rei.squaredNorm() / rej.squaredNorm();
    Matrix<double, 3, 2> degeneracy;
    degeneracy.col(0) = rei.normalized();
    degeneracy.col(1) = (ej0 - ei0).cross(rei).normalized();
    bool par = sin2 < 1e-8;

    auto lams = ::ipc::edge_edge_closest_point(ei0, ei1, ej0, ej1);

    if (ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB)
        ;

    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA0_EB0)
        lams = { 0.0, 0.0 };
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA0_EB1)
        lams = { 0.0, 1.0 };
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA1_EB0)
        lams = { 1.0, 0.0 };
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA1_EB1)
        lams = { 1.0, 1.0 };
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB0) {
        auto pe = ::ipc::point_edge_closest_point(ej0, ei0, ei1);
        lams = { pe, 0.0 };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA_EB1) {
        auto pe = ::ipc::point_edge_closest_point(ej1, ei0, ei1);
        lams = { pe, 1.0 };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA0_EB) {
        auto pe = ::ipc::point_edge_closest_point(ei0, ej0, ej1);
        lams = { 0.0, pe };
    }
    else if (ee_type == ::ipc::EdgeEdgeDistanceType::EA1_EB) {
        auto pe = ::ipc::point_edge_closest_point(ei1, ej0, ej1);
        lams = { 1.0, pe };
    }
    const auto clip = [&](double& a, double l, double u) {
        a = max(min(u, a), l);
    };
    clip(lams(0), 0.0, 1.0);
    clip(lams(1), 0.0, 1.0);
    array<double, 4> lambdas = { 1 - lams(0), lams(0), 1 - lams(1), lams(1) };
    if (par) {
        // ignore this friction, already handled in point-triangle pair
        lambdas = { 0.0, 0.0, 0.0, 0.0 };
        // uk will be set to zero
    }
    auto pei = ei0 * lambdas[0] + ei1 * lambdas[1];
    auto pej = ej0 * lambdas[2] + ej1 * lambdas[3];
    auto closest = (pei - pej).squaredNorm();
    assert(par || abs(d - closest) < 1e-8);

    auto Pk = par ? degeneracy : ::ipc::edge_edge_tangent_basis(ei0, ei1, ej0, ej1);
    Matrix<double, 3, 12> gamma;
    gamma.setZero(3, 12);
    for (int i = 0; i < 3; i++) {
        gamma(i, i) = -lambdas[0];
        gamma(i, i + 3) = -lambdas[1];
        gamma(i, i + 6) = lambdas[2];
        gamma(i, i + 9) = lambdas[3];
    }
    Matrix<double, 2, 12> Tk_T = Pk.transpose() * gamma;

    // Matrix<double, 12, 24> jacobian;
    // auto _i0 = ci.edges[_ei * 2], _i1 = ci.edges[_ei * 2 + 1],
    //      _j0 = cj.edges[_ej * 2], _j1 = cj.edges[_ej * 2 + 1];

    // jacobian.setZero(12, 24);
    // jacobian.block<3, 12>(0, 0) = x_jacobian_q(ci.vertices(_i0));
    // jacobian.block<3, 12>(3, 0) = x_jacobian_q(cj.vertices(_i1));
    // jacobian.block<3, 12>(6, 12) = x_jacobian_q(cj.vertices(_j0));
    // jacobian.block<3, 12>(9, 12) = x_jacobian_q(cj.vertices(_j1));

    // auto Tq_k = Tk_T * jacobian;

    // auto contact_force_lam = barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
    Vector2d uk = Tk_T * v_stack;
    auto contact_force = -barrier::barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
    Tk_T_ret = Tk_T;
    uk_ret = uk;

    return contact_force;
}
};

vec12 AffineBody::q_tile(double dt, const vec3& f) const
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
            in.read((char*)c.q0[j].data(), 3 * sizeof(double));
        for (int j = 0; j < 4; j++)
            in.read((char*)c.dqdt[j].data(), 3 * sizeof(double));
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
            out.write((char*)c.q[j].data(), 3 * sizeof(double));
        for (int j = 0; j < 4; j++)
            out.write((char*)c.dq.segment<3>(j * 3).data(), 3 * sizeof(double));
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
            out.write((char*)c.q0[j].data(), 3 * sizeof(double));
        for (int j = 0; j < 4; j++)
            out.write((char*)c.dqdt[j].data(), 3 * sizeof(double));
    }
    out.close();
}
