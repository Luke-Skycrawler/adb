#include "pch.h"
#include "../model/barrier.h"
#include "../model/cube.h"
#include "../model/iaabb.h"
#include "../model/geometry.h"
#include "../model/spatial_hashing.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#define _USE_MATH_DEFINES
//#define _FAILED_
#include <math.h>
#include <random>

inline void Cube::draw(Shader& shader) const {}
bool predefined = true;
unsigned *Cube::_edges = nullptr, *Cube::_indices = nullptr;
void gen_collision_set(
    bool vt2, int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    std::vector<std::array<vec3, 4>>& ees,
    std::vector<std::array<int, 4>>& eidx,
    std::vector<std::array<int, 2>>& vidx,
    std::vector<Matrix<double, 2, 12>>& pt_tk,
    std::vector<Matrix<double, 2, 12>>& ee_tk,
    bool gen_basis = false);

using namespace std;
using namespace Eigen;

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
        n_cubes = predefined ? 6 : 100;

        Cube::gen_indices();
        aabbs.resize(n_cubes);
        args.reserve(n_cubes);
        for (int i = 0; i < n_cubes; i++) {
            unique_ptr<AffineBody> a;
            a = make_unique<Cube>();
            double aa, bb, cc;
            double p0, p1, p2;

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
            }
            mat3 r = rotation(aa, bb, cc);

            for (int i = 0; i < 3; i++) a->q[i + 1] = r.col(i);
            
            a->q[0] = vec3(p0, p1, p2);
            auto b = compute_aabb(*a);
            aabbs[i] = b;
            cubes.push_back(move(a));
            args.push_back({aa, bb, cc, p0, p1, p2});
        }
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
    // for (int i = 0; i < n_cubes; i++) {
    //     cout << aabbs[i][0].transpose() << " , " << aabbs[i][1].transpose() << " , ";
    // }
    cout << "size: bf = " << overlaps_bf.size() << " sort = " << overlaps_sort.size() << "\n";
    //EXPECT_EQ(overlaps_bf.size(), overlaps_sort.size())
    //    << "size mismatch"
    //    << "\n";
    //for (int i = 0; i < min.size(); i++) {
    //    auto a{ overlaps_bf[i] }, b{ overlaps_sort[i] };
    //    EXPECT_TRUE(a.i == b.i && a.j == b.j) << "brute force: (" << a.i << ", " << a.j << "), sort: (" << b.i << "," << b.j << "\n";
    //}
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
    intersect_sort(n_cubes, cubes, aabbs, overlaps_sort, 1);
    primitive_brute_force(n_cubes, overlaps_sort, cubes, 1, pts, idx, ees, eidx, vidx, pt_tk, ee_tk, false);
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}