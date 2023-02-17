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
bool predefined = false;
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

protected:
    void SetUp() override
    {
        n_cubes = predefined ? 4 : 100;
#ifdef _FAILED_
        n_cubes = 19;
        predefined = true;
        // double args[] = {
        //     213.417, 126.994, 145.515, -0.861929, 2.7898, -2.07337,
        //     12.3542, 352.773, 179.476, 2.86201, -0.820881, 1.07712,
        //     259.378, 355.817, 257.796, 2.47546, 0.032991, 0.349613,
        //     11.9767, 123.782, 319.581, -2.02277, 2.26418, -1.73819
        // };
        double ags[] = {                                                                                                                       0.321802, 0.228967, 2.56813, -0.252065, -0.0745864, 1.76385,                                                            4.36828, 4.27143, 2.46502, 0.369345, -1.75159, 0.164229,                                                                2.53972, 2.21647, 3.72482, -0.861929, 2.7898, -2.07337,                                                                 5.42037, 3.89784, 0.751137, -0.168259, -0.958682, 0.179052,                                                             2.40227, 4.79043, 0.254287, -1.48226, 0.028626, 1.93563,                                                                5.57775, 2.16041, 0.209033, -2.02277, 2.26418, -1.73819,                                                                5.72125, 3.07008, 4.74877, -2.35163, -0.638461, 2.22112,                                                                3.39379, 0.11593, 5.65602, -1.83303, 2.31599, -0.352659,                                                                5.22267, 2.04181, 4.12039, -0.47886, 1.69145, -2.32084,                                                                 6.24256, 1.14086, 4.79364, -1.40448, -1.53517, -2.39557,                                                                4.81829, 2.05961, 0.810947, -1.24824, 2.08266, -1.61541,                                                                2.23272, 4.52371, 0.411203, 0.0831551, -0.45613, 1.13881,                                                               3.06872, 3.23282, 1.5395, -1.00197, 0.0141743, 0.576348,                                                                5.6985, 0.480689, 4.48559, 0.70323, -1.86549, -0.618109,                                                                1.74461, 5.3916, 5.71102, 0.703675, -1.91509, 0.0901423,                                                                3.3286, 3.12821, 2.10988, -1.48306, -1.59795, -1.96102,                                                                 3.74466, 4.5672, 5.51395, 1.24296, -1.59141, -0.132026,                                                                 4.83784, 3.08977, 3.06735, -0.425241, -0.631634, 1.65872,                                                               1.35484, 3.3945, 4.5429, -1.29784, 0.257173, 1.53555                                                                   };
#endif
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
                // p0 = p1 = p2 = 0.8 * i;
                p0 = p1 = p2 = i * (1.0 + barrier::d_sqrt / 2);
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
    EXPECT_EQ(overlaps_bf.size(), overlaps_sort.size())
        << "size mismatch"
        << "\n";
    for (int i = 0; i < overlaps_bf.size(); i++) {
        auto a{ overlaps_bf[i] }, b{ overlaps_sort[i] };
        EXPECT_TRUE(a.i == b.i && a.j == b.j) << "brute force: (" << a.i << ", " << a.j << "), sort: (" << b.i << "," << b.j << "\n";
    }
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