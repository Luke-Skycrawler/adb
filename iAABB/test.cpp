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
#include <math.h>
#include <random>

#include <iostream>
#include <fstream>
#include <filesystem>

inline void Cube::draw(Shader& shader) const {}
bool predefined = false;
unsigned *Cube::_edges = nullptr, *Cube::_indices = nullptr;


using namespace std;
using namespace Eigen;

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

class iAABBTest : public ::testing::Test {
public:
    std::vector<std::unique_ptr<AffineBody>> cubes;
    static const double space_range[2];
    vector<lu> aabbs;
    int n_cubes;
    default_random_engine gen;
    static uniform_real_distribution<double> dist;
    vector<array<double, 6>> args;
    void diff(vector<array<int, 4>>& a, vector<array<int, 4>>& b);
    void diff(vector<Intersection>& a, vector<Intersection>& b);

protected:
    void SetUp() override
    {
        #ifdef _LOAD_
        n_cubes = 3;
        #else
        n_cubes = predefined ? 6 : 10;
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
            auto b = compute_aabb(*a);
            aabbs[i] = b;
            cubes.push_back(move(a));
            args.push_back({aa, bb, cc, p0, p1, p2});
        }
        player_load(40, cubes);
    }
};
uniform_real_distribution<double> iAABBTest ::dist(0.0, 1.0);
const double iAABBTest::space_range[2]{ -3.0, 3.0 };
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


void iAABBTest::diff(vector<array<int, 4>>& a, vector<array<int, 4>>& b)
{
    EXPECT_EQ(a.size(), b.size()) << "size mismatch";
    cout << a.size() << " "
         << b.size() << "\n";
    vector<array<int, 4>> adb, bda;
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
TEST_F(iAABBTest, pipelined)
{
    vector<Intersection> overlaps_sort;
    vector<array<vec3, 4>> pts;
    vector<array<int, 4>> idx;
    vector<array<vec3, 4>> ees;
    vector<array<int, 4>> eidx;
    vector<array<int, 2>> vidx;
    vector<Matrix<double, 2, 12>> pt_tk;
    vector<Matrix<double, 2, 12>> ee_tk;

    vector<array<vec3, 4>> pts_ref;
    vector<array<int, 4>> idx_ref;
    vector<array<vec3, 4>> ees_ref;
    vector<array<int, 4>> eidx_ref;
    vector<array<int, 2>> vidx_ref;
    vector<Matrix<double, 2, 12>> pt_tk_ref;
    vector<Matrix<double, 2, 12>> ee_tk_ref;
    vector<double_int> foo, bar;
    intersect_sort(n_cubes, cubes, aabbs, overlaps_sort, 1);
    primitive_brute_force(n_cubes, overlaps_sort, cubes, 1, pts, idx, ees, eidx, vidx, pt_tk, ee_tk, foo, bar, false);
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
                    array<vec3, 4> pt = { p, _f.t0, _f.t1, _f.t2 };
                    array<int, 4> ij = { i, v, j, f };

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
                        array<vec3, 4> ee = { ei.e0, ei.e1, ej.e0, ej.e1 };
                        array<int, 4> ij = { i, _ei, j, _ej };

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
}

array<double, 2> brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    std::vector<double_int>& pt_tois, std::vector<double_int>& ee_tois,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<int, 4>>& eidx
    )
{
    double toi_pt = 1.0, toi_ee = 1.0;
    for (int i = 0; i < n_cubes; i++)

        for (int j = 0; j < n_cubes; j++) {
            if (i == j) continue;
            auto &ci{ *cubes[i] }, &cj{ *cubes[j] };
            for (unsigned v = 0; v < ci.n_vertices; v++) {
                auto p1{ ci.vt1(v) };
                auto p2{ ci.vt2(v)};
                for (unsigned f = 0; f < cj.n_vertices; f++) {
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
            for (unsigned ei = 0; ei < ci.n_edges; ei++) {
                Edge ei1{ ci, ei }, ei2{ ci, ei, true };
                for (unsigned ej = 0; ej < cj.n_edges; ej++) {
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
TEST_F(iAABBTest, upper_bound_against_sh)
{
    vector<array<vec3, 4>> pts;
    vector<array<int, 4>> idx, idx_iaabb;
    vector<array<vec3, 4>> ees;
    vector<array<int, 4>> eidx, eidx_iaabb;
    vector<array<int, 2>> vidx;
    vector<Matrix<double, 2, 12>> pt_tk;
    vector<Matrix<double, 2, 12>> ee_tk;
    vector<double_int> pt_tois, ee_tois, pt_toi_iaabb, ee_toi_iaabb;
    double t = iaabb_brute_force(n_cubes, cubes, aabbs, 3, pts, idx_iaabb, ees, eidx_iaabb, vidx, pt_tk, ee_tk, pt_toi_iaabb, ee_toi_iaabb, false);
    auto t_ref = brute_force(n_cubes, cubes, pt_tois, ee_tois, idx, eidx);
    EXPECT_EQ(min(t_ref[0], t_ref[1]), t);
    sort(idx.begin(), idx.end());
    sort(eidx.begin(), eidx.end());
    sort(idx_iaabb.begin(), idx_iaabb.end());
    sort(eidx_iaabb.begin(), eidx_iaabb.end());
    
    diff(idx, idx_iaabb);
    diff(eidx, eidx_iaabb);
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}