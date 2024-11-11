#include "pch.h"
#include "../model/barrier.h"
#include "../model/cube.h"
#include "../model/iaabb.h"
#include "../model/geometry.h"
#include "../model/spatial_hashing.h"
#include "../model/collision.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#define _USE_MATH_DEFINES
//#define _FAILED_
//#define _LOAD_
#include <psd.h>
#include <random>

#include <iostream>
#include <fstream>
#include <filesystem>

inline void Cube::draw(Shader& shader) const {}
bool predefined = false;
int *Cube::_edges = nullptr, *Cube::_indices = nullptr;


using namespace std;
using namespace Eigen;
void failure_load(
    const std::vector<std::unique_ptr<AffineBody>>& cubes)
{
    string filename = "../view/trace/ub_failure";
    ifstream in(filename, ios::in | ios::binary);
    int n_cubes = cubes.size();

    for (int i = 0; i < n_cubes; i++) {
        auto& c{ *cubes[i] };
        for (int j = 0; j < 4; j++)
            in.read((char*)c.q[j].data(), 3 * sizeof(double));
        for (int j = 0; j < 4; j++){
            vec3 tmp;
            in.read((char*)tmp.data(), 3 * sizeof(double));
            c.dq.segment<3>(3 * j) = tmp;
        }
        c.p = c.q[0];
        c.A << c.q[1], c.q[2], c.q[3];
    }
    in.close();

}
void player_load(
    int timestep,
    const std::vector<std::unique_ptr<AffineBody>>& cubes)
{
    string filename = "../view/trace/" + to_string(timestep);
    ifstream in(filename, ios::in | ios::binary);
    int n_cubes = cubes.size();

    for (int i = 0; i < n_cubes; i++) {
        auto& c{ *cubes[i] };
        for (int j = 0; j < 4; j++)
            in.read((char*)c.q[j].data(), 3 * sizeof(double));
        for (int j = 0; j < 4; j++)
            in.read((char*)c.dqdt[j].data(), 3 * sizeof(double));
        c.p = c.q[0];
        c.A << c.q[1], c.q[2], c.q[3];
    }
    in.close();
}

Globals globals{
    {}, {}, {}, 
    0.5, // mu
    1e-3, // evh
    1e-2, // dt
    false
};
class iAABBTest : public ::testing::Test {
public:
    std::vector<std::unique_ptr<AffineBody>> cubes;
    static const double space_range[2];
    vector<lu> aabbs;
    int n_cubes;
    default_random_engine gen;
    static uniform_real_distribution<double> dist;
    vector<array<double, 6>> args;
    void diff(vector<i4>& a, vector<i4>& b);
    void diff(vector<Intersection>& a, vector<Intersection>& b);

protected:
    void SetUp() override
    {
        #ifdef _LOAD_
        n_cubes = 3;
        #else
        n_cubes = predefined ? 6 : 50;
        #endif

        Cube::gen_indices();
        aabbs.resize(n_cubes);
        args.reserve(n_cubes);
        for (int i = 0; i < n_cubes; i++) {
            unique_ptr<AffineBody> a;
            a = make_unique<Cube>();
            double aa, bb, cc;
            double p0, p1, p2;

            double aa_t2, bb_t2, cc_t2;
            double p0_t2, p1_t2, p2_t2;

            if (predefined) {
#ifdef _FAILED_
                aa = ags[6 * i];
                bb = ags[6 * i + 1];
                cc = ags[6 * i + 2];
                p0 = ags[6 * i + 3];
                p1 = ags[6 * i + 4];
                p2 = ags[6 * i + 5];
#else
                aa = 0.0, bb = 0.0, cc = 0.0;
                p0 = p1 = p2 = 0.8 * i;
                p0  = i * (1.0 + barrier::d_sqrt / 2);
#endif
            }
            else {
                aa = dist(gen) * M_PI * 2, bb = dist(gen) * M_PI * 2, cc = dist(gen) * M_PI * 2;
                p0 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
                p1 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
                p2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];

                aa_t2 = dist(gen) * M_PI * 2, bb_t2 = dist(gen) * M_PI * 2, cc_t2 = dist(gen) * M_PI * 2;
                p0_t2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
                p1_t2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
                p2_t2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            }
            mat3 r = rotation(aa, bb, cc);
            mat3 r_t2 = rotation(aa_t2, bb_t2, cc_t2);

            for (int i = 0; i < 3; i++) {
                a->q[i + 1] = r.col(i);
                a -> dq.segment<3>(3 * (i + 1)) = r.col(i);
            }
            a->dq.segment<3>(0) = vec3(p0_t2, p1_t2, p2_t2);
            a->q[0] = vec3(p0, p1, p2);
            a->q0 = a->q;

            auto b = compute_aabb(*a);
            aabbs[i] = b;
            cubes.push_back(move(a));
            args.push_back({aa, bb, cc, p0, p1, p2});
        }
        #ifdef _FAILED_
        player_load(40, cubes);
        #endif
        #ifdef _LOAD_
        failure_load(cubes);
        #endif
        globals.points.resize(0);
        globals.edges.resize(0);
        globals.triangles.resize(0);
        for (int I = 0; I < n_cubes; I++) {
            auto& c{ *cubes[I] };

            for (int i = 0; i < c.n_vertices; i++)
                globals.points.push_back({ I, i });
            for (int i = 0; i < c.n_edges; i++)
                globals.edges.push_back({ I, i });
            for (int i = 0; i < c.n_faces; i++)
                globals.triangles.push_back({ I, i });
        }
    }
};
uniform_real_distribution<double> iAABBTest ::dist(0.0, 1.0);
const double iAABBTest::space_range[2]{ -3.0, 3.0 };


#include "finitediff.hpp"
void friction(
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk,
    Vector<double, 12>& g, Matrix<double, 12, 12>& H);

double D_f0(double uk, double lam);

namespace utils {

vec12 pt_vstack(AffineBody& ci, AffineBody& cj, int v, int f);
vec12 ee_vstack(AffineBody& ci, AffineBody& cj, int ei, int ej);

double ee_uktk(
    AffineBody& ci, AffineBody& cj,
    q4& ee, i4& ij, const ::ipc::EdgeEdgeDistanceType& ee_type,
    Matrix<double, 2, 12>& Tk_T_ret, Vector2d& uk_ret, double d, double dt);

double pt_uktk(
    AffineBody& ci, AffineBody& cj,
    q4& pt, i4& ij, const ::ipc::PointTriangleDistanceType& pt_type,
    Matrix<double, 2, 12>& Tk_T_ret, Vector2d& uk_ret, double d, double dt);

};

void iAABBTest::diff(vector<i4>& a, vector<i4>& b)
{
    EXPECT_EQ(a.size(), b.size()) << "size mismatch";
    cout << a.size() << " "
         << b.size() << "\n";
    vector<i4> adb, bda;
    set_difference(a.begin(), a.end(), b.begin(), b.end(), back_inserter(adb));
    set_difference(b.begin(), b.end(), a.begin(), a.end(), back_inserter(bda));
    EXPECT_EQ(adb.size(), 0) << "should include following: ";

    set<int> cubes_set;
    if (adb.size()) for (auto &a: adb) {
        cout << "( " << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << ")\n";
        cubes_set.insert(a[0]);
        cubes_set.insert(a[2]);
    }

    EXPECT_EQ(bda.size(), 0) << "included extras: ";
    if (bda.size()) for (auto &a: bda) {
        cout << "( " << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << ")\n";
    }

    cout << "aftermath: \n";
    cout << "{\n";
    for (auto& a : cubes_set) {
        auto& ai{ args[a] };
        for (int i = 0; i < 6; i++) cout << ai[i] << ", ";
        cout << "\n";
    }
    cout << "};\n" << cubes_set.size();

}


void iAABBTest::diff(vector<Intersection>& a, vector<Intersection>& b)
{
    EXPECT_EQ(a.size(), b.size()) << "size mismatch";
    cout << a.size() << " "
         << b.size() << "\n";
    vector<Intersection> adb, bda;
    set_difference(a.begin(), a.end(), b.begin(), b.end(), back_inserter(adb));
    set_difference(b.begin(), b.end(), a.begin(), a.end(), back_inserter(bda));
    EXPECT_EQ(adb.size(), 0) << "should include following: ";

    if (adb.size())
        for (auto& a : adb) {
            cout << "( " << a.i << " " << a.j << ")\n";
        }

    EXPECT_EQ(bda.size(), 0) << "included extras: ";
    if (bda.size())
        for (auto& a : bda) {
            cout << "( " << a.i << " " << a.j << ")\n";
        }

}

array<double, 2> brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
    std::vector<i4>& idx,
    std::vector<i4>& eidx
    )
{
    double toi_pt = 1.0, toi_ee = 1.0;
    for (int i = 0; i < n_cubes; i++)

        for (int j = 0; j < n_cubes; j++) {
            if (i == j) continue;
            auto &ci{ *cubes[i] }, &cj{ *cubes[j] };
            for (int v = 0; v < ci.n_vertices; v++) {
                auto p1{ ci.vt1(v) };
                auto p2{ ci.vt2(v)};
                for (int f = 0; f < cj.n_faces; f++) {
                    Face t1{ cj, f }, t2{ cj, f, true };
                    double t = pt_collision_time(p1, t1, p2, t2);
                    if (t < 1.0) {
                        pt_tois.push_back({t, int(pt_tois.size())});
                        idx.push_back({i, int(v), j, int(f)});
                    }
                    toi_pt = min(toi_pt, t);
                }
            }
        }
    for (int i = 0; i < n_cubes; i++)
        for (int j = i + 1; j < n_cubes; j++) {
            auto &ci{ *cubes[i] }, &cj{ *cubes[j] };
            for (int ei = 0; ei < ci.n_edges; ei++) {
                Edge ei1{ ci, ei }, ei2{ ci, ei, true };
                for (int ej = 0; ej < cj.n_edges; ej++) {
                    Edge ej1{ cj, ej }, ej2{ cj, ej, true };
                    double t = ee_collision_time(ei1, ej1, ei2, ej2);
                    if (t < 1.0) {
                        ee_tois.push_back({t, int(ee_tois.size())});
                        eidx.push_back({i, int(ei), j, int(ej)});
                    }
                    toi_ee = min(toi_ee, t);
                }
            }
        }
    return { toi_pt, toi_ee };
}
#ifdef ALL_TESTS
TEST_F(iAABBTest, sort_against_bf)
{
    vector<Intersection> overlaps_bf, overlaps_sort;
    intersect_brute_force(n_cubes, cubes, aabbs, overlaps_bf, 1);
    intersect_sort(n_cubes, cubes, aabbs, overlaps_sort, 1);
    const auto les = [](const Intersection& a, const Intersection& b) -> bool {
        return a.i < b.i || (a.i == b.i && a.j < b.j);
    };
    std::sort(overlaps_bf.begin(), overlaps_bf.end(), les);
    std::sort(overlaps_sort.begin(), overlaps_sort.end(), les);
    cout << "size: bf = " << overlaps_bf.size() << " sort = " << overlaps_sort.size() << "\n";
    diff(overlaps_bf, overlaps_sort);
}

TEST_F(iAABBTest, pipelined)
{
    vector<Intersection> overlaps_sort;
    vector<q4> pts;
    vector<i4> idx;
    vector<q4> ees;
    vector<i4> eidx;
    vector<array<int, 2>> vidx;
    vector<Matrix<double, 2, 12>> pt_tk;
    vector<Matrix<double, 2, 12>> ee_tk;

    vector<q4> pts_ref;
    vector<i4> idx_ref;
    vector<q4> ees_ref;
    vector<i4> eidx_ref;
    vector<array<int, 2>> vidx_ref;
    vector<Matrix<double, 2, 12>> pt_tk_ref;
    vector<Matrix<double, 2, 12>> ee_tk_ref;
    vector<double_int> foo, bar;
    // intersect_sort(n_cubes, cubes, aabbs, overlaps_sort, 1);
    //primitive_brute_force(n_cubes, overlaps_sort, cubes, 1, foo, bar, globals, pts, idx, ees, eidx, vidx);
    iaabb_brute_force(n_cubes, cubes, aabbs, 1, foo, bar, globals, pts, idx, ees, eidx, vidx);
    int nsqr = n_cubes * n_cubes;
    for (int I = 0; I < nsqr; I++) {
        int i = I / n_cubes, j = I % n_cubes;
        auto &ci(*cubes[i]), &cj(*cubes[j]);
        if (i == j) continue;
        for (int v = 0; v < ci.n_vertices; v++)
            for (int f = 0; f < cj.n_faces; f++) {
                Face _f(cj, f);
                vec3 p = ci.vt1(v);
                auto fu = _f.t0.cwiseMax(_f.t1).cwiseMax(_f.t2).array() + barrier ::d_sqrt;
                auto fl = _f.t0.cwiseMin(_f.t1).cwiseMin(_f.t2).array() - barrier::d_sqrt;
                if ((p.array() <= fu.array()).all() && (p.array() >= fl.array()).all()) {
                }
                else
                    continue;

                auto pt_type = ipc::point_triangle_distance_type(p, _f.t0, _f.t1, _f.t2);

                double d = ipc::point_triangle_distance(p, _f.t0, _f.t1, _f.t2, pt_type);
                if (d < barrier::d_hat) {
                    q4 pt = { p, _f.t0, _f.t1, _f.t2 };
                    i4 ij = { i, v, j, f };

                    {
                        pts_ref.push_back(pt);
                        idx_ref.push_back(ij);
                    }
                }
            }
    }
    for (int i = 0; i < n_cubes; i++)
        for (int j = i + 1; j < n_cubes; j++) {
            auto &ci(*cubes[i]), &cj(*cubes[j]);
            for (int _ei = 0; _ei < ci.n_edges; _ei++)
                for (int _ej = 0; _ej < cj.n_edges; _ej++) {
                    Edge ei(ci, _ei), ej(cj, _ej);
                    auto iu = ei.e0.cwiseMax(ei.e1).array() + barrier ::d_sqrt / 2;
                    auto il = ei.e0.cwiseMin(ei.e1).array() - barrier ::d_sqrt / 2;

                    auto ju = ej.e0.cwiseMax(ej.e1).array() + barrier ::d_sqrt / 2;
                    auto jl = ej.e0.cwiseMin(ej.e1).array() - barrier ::d_sqrt / 2;
                    if ((iu.array() >= jl.array()).all() && (ju.array() >= il.array()).all()) {}
                    else
                        continue;
                    double d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
                    if (d < barrier::d_hat) {
                        q4 ee = { ei.e0, ei.e1, ej.e0, ej.e1 };
                        i4 ij = { i, _ei, j, _ej };

#pragma omp critical
                        {
                            ees_ref.push_back(ee);
                            eidx_ref.push_back(ij);
                        }
                    }
                }
        }
    sort(idx.begin(), idx.end());
    sort(idx_ref.begin(), idx_ref.end());
    sort(eidx.begin(), eidx.end());
    sort(eidx_ref.begin(), eidx_ref.end());

    diff(idx_ref, idx); 
    diff(eidx_ref, eidx);

    iaabb_brute_force(n_cubes, cubes, aabbs, 1, foo, bar, globals, pts, idx, ees, eidx, vidx);
    sort(idx.begin(), idx.end());
    sort(eidx.begin(), eidx.end());

    diff(idx_ref, idx); 
    diff(eidx_ref, eidx);

}

TEST_F(iAABBTest, upper_bound_against_sh)
{
    vector<q4> pts;
    vector<i4> idx, idx_iaabb;
    vector<q4> ees;
    vector<i4> eidx, eidx_iaabb;
    vector<array<int, 2>> vidx;
    vector<Matrix<double, 2, 12>> pt_tk;
    vector<Matrix<double, 2, 12>> ee_tk;
    vector<double_int> pt_tois, ee_tois, pt_toi_iaabb, ee_toi_iaabb;
    double t = iaabb_brute_force(n_cubes, cubes, aabbs, 3, pt_toi_iaabb, ee_toi_iaabb, globals,  pts, idx_iaabb, ees, eidx_iaabb, vidx
        );
    auto t_ref = brute_force(n_cubes, cubes, pt_tois, ee_tois, idx, eidx);
    EXPECT_EQ(min(t_ref[0], t_ref[1]), t);
    sort(idx.begin(), idx.end());
    sort(eidx.begin(), eidx.end());
    sort(idx_iaabb.begin(), idx_iaabb.end());
    sort(eidx_iaabb.begin(), eidx_iaabb.end());
    
    diff(idx, idx_iaabb);
    diff(eidx, eidx_iaabb);
}
#endif
TEST_F(iAABBTest, finite_diff)
{
    vector<q4> pts;
    vector<i4> idx;
    vector<q4> ees;
    vector<i4> eidx;
    vector<array<int, 2>> vidx;
    vector<Matrix<double, 2, 12>> pt_tk;
    vector<Matrix<double, 2, 12>> ee_tk;
    vector<double_int> foo, bar;


    iaabb_brute_force(n_cubes, cubes, aabbs, 1, foo, bar, globals, pts, idx, ees, eidx, vidx);
    for (int i = 0; i < n_cubes; i++){
        auto &c {*cubes[i]};
        c.dq.setZero(12);
        c.project_vt2();
    }
    int gf = 0, hf = 0, gb = 0, hb = 0;
    for (int _i = 0; _i < idx.size(); _i++) {
        auto& pt{ pts[_i] };
        auto& ij{ idx[_i] };
        auto &ci{ *cubes[ij[0]] }, &cj{ *cubes[ij[2]] };
        auto du = utils::pt_vstack(ci, cj, ij[1], ij[3]);

        const auto &[d, pt_type] = vf_distance(ci.vt1(ij[1]), Face{ cj, int(ij[3]) });
        Vector2d _uk;
        Matrix<double, 2, 12> Tk;
        double lam = utils::pt_uktk(ci, cj, pt, ij, pt_type, Tk, _uk, d, globals.dt);

        const auto f = [&](const VectorXd& x) -> double{
            Vector2d uk = Tk * x;
            double u = uk.norm();
            auto ret = D_f0(u, lam);
            return ret;
        };

        const auto b = [&](const VectorXd& x) -> double {
            vec3 pt[4];
            for (int i = 0; i < 4; i++) pt[i] = x.segment<3>(i * 3);
            auto [d, pt_type] = vf_distance(pt[0], Face{ pt[1], pt[2], pt[3] });
            return barrier::barrier_function(d);
        };

        Vector<double, 12> g, pt_grad;
        Matrix<double, 12, 12> H, pt_hess;
        g.setZero(12);
        pt_grad.setZero(12);
        H.setZero(12, 12);
        pt_hess.setZero(12, 12);

        friction(_uk, lam, Tk.transpose(), g, H);

        ipc::point_triangle_distance_gradient(pt[0], pt[1], pt[2], pt[3], pt_grad, pt_type);
        ipc::point_triangle_distance_hessian(pt[0], pt[1], pt[2], pt[3], pt_hess, pt_type);
        double B_ = barrier::barrier_derivative_d(d);
        double B__ = barrier::barrier_second_derivative(d);
        pt_hess = pt_hess * B_ + pt_grad * pt_grad.transpose() * B__;
        pt_grad *= B_;
        VectorXd fgrad, bgrad;
        MatrixXd fhess, bhess;
        fd::finite_gradient(du, f, fgrad);
        fd::finite_gradient(du, b, bgrad);
        fd::finite_hessian(du, f, fhess);
        fd::finite_hessian(du, f, bhess);


        bool fgpass = fd::compare_gradient(g, fgrad, 1e-3, "fgrad miss");
        bool fhpass = fd::compare_hessian(H, fhess, 1e-3, "fhess miss"); 
        // << "idx = " << H << "\n"
        //                                                  << fhess << "grad failed count = " << hf++;

        bool bgpass= fd::compare_gradient(pt_grad, bgrad, 1e-3, "b grad miss");
        //  << "idx = " << pt_grad.transpose() << "\n"
        //                                             << bgrad.transpose() << "grad failed count = " << gf ++;

        bool bhpass = fd::compare_hessian(pt_hess, bhess, 1e-3, "b hess miss");
        //  << "idx = " << pt_hess << "\n"
        //                                                  << bhess << "grad failed count = " << hf++;
        if (fgpass) {
            gf ++;
            cout << fgrad.transpose() <<"\n" << g.transpose() <<"\n\n";
        }
        
        if (bgpass) {
            gb ++;
            cout << bgrad.transpose() << "\n" << pt_grad.transpose() << "\n\n";
        }
        if (fhpass) hf ++;
        if (bhpass) hb ++;

        EXPECT_TRUE(fgpass && bgpass && fhpass && bhpass); 
    }
    spdlog::info("all pts {}, passed = f, gH {}, {}, B, gH {}, {}", idx.size(), gf, hf, gb, hb);
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}