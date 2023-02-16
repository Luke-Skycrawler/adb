#include "pch.h"
#include "../model/barrier.h"
#include "../model/cube.h"
#include "../model/iaabb.h"
#include "../model/geometry.h"
#include "../model/spatial_hashing.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <random>

inline void Cube::draw(Shader& shader) const {}
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

protected:
    void SetUp() override
    {
        n_cubes = 50;
        Cube::gen_indices();
        aabbs.resize(n_cubes);
        for (int i = 0; i < n_cubes; i++) {
            unique_ptr<AffineBody> a;
            a = make_unique<Cube>();
            double aa = dist(gen) * M_PI * 2, bb = dist(gen) * M_PI * 2, cc = dist(gen) * M_PI * 2;
            mat3 r = rotation(aa, bb, cc);
            double p0, p1, p2;
            p0 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            p1 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            p2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            for (int i = 0; i < 3; i++) a->q[i + 1] = r.col(i);
            a->q[0] = vec3(p0, p1, p2);
            auto b = compute_aabb(*a);
            aabbs[i] = b;
            cubes.push_back(move(a));
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
    cout << "size: bf = " << pts_ref.size() << " sort = " << pts.size() << "\n";
    EXPECT_EQ(pts_ref.size(), pts.size())
        << "size mismatch"
        << "\n";
    // const auto les = [] (const array<int, 4> &a, const array<int, 4> &b) {
    //     return a.
    // };
    // sort(idx.begin(), idx.end());
    // sort(idx_ref.begin(), idx_ref.end());
    for (int i = 0; i < idx_ref.size(); i++) {
        auto a{ idx_ref[i] }, b{ idx[i] };
        EXPECT_TRUE(a[0] == b[0] && a[2] == b[2]) << "brute force: (" << a[0] << ", " << a[2] << "), sort: (" << b[0] << "," << b[2] << ")\n";
    }
}
//TEST(body_level, compute_intersection_against_brute_force)
//{
//
//    int n_cubes = 100;
//    const double space_range[2]{ -3.0, 3.0 };
//
//    std::vector<std::unique_ptr<AffineBody>> cubes;
//    vector <lu> aabbs; 
//    aabbs.resize(n_cubes);
//
//    default_random_engine gen;
//    uniform_real_distribution<double> dist(0.0, 1.0);
//    for (int i = 0; i < n_cubes; i++)
//    {
//        unique_ptr<AffineBody> a;
//        a = make_unique<Cube>();
//        double aa = dist(gen) * M_PI * 2, bb = dist(gen) * M_PI * 2, cc = dist(gen) * M_PI * 2;
//        mat3 r = rotation(aa, bb, cc);
//        double p0, p1, p2;
//        p0 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
//        p1 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
//        p2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
//        for (int i = 0; i < 3; i++) a->q[i + 1] = r.col(i);
//        a->q[0] = vec3(p0, p1, p2);
//        auto b = compute_aabb(*a);
//        aabbs[i] = b;
//        cubes.push_back(move(a));
//    }
//    vector<Intersection> overlaps_bf, overlaps_sort;
//    intersect_brute_force(n_cubes, cubes, aabbs, overlaps_bf, 1);
//    intersect_sort(n_cubes, cubes, aabbs, overlaps_sort, 1);
//    const auto les = [](const Intersection &a, const Intersection&b) ->bool {
//        return a.i < b.i || (a.i == b.i && a.j < b.j);
//    };
//    std::sort(overlaps_bf.begin(), overlaps_bf.end(), les);
//    std::sort(overlaps_sort.begin(), overlaps_sort.end(), les);
//    //for (int i = 0; i < n_cubes; i++) {
//    //    cout << aabbs[i][0].transpose() << " , " << aabbs[i][1].transpose() << " , ";
//    //}
//    cout << "size: bf = " << overlaps_bf.size() << " sort = " << overlaps_sort.size() << "\n";
//    EXPECT_EQ(overlaps_bf.size(), overlaps_sort.size())
//        << "size mismatch"
//        << "\n";
//    for (int i = 0; i < overlaps_bf.size(); i++) {
//        auto a{ overlaps_bf[i] }, b{ overlaps_sort[i] };
//        EXPECT_TRUE(a.i == b.i && a.j == b.j) << "brute force: (" << a.i << ", " << a.j << "), sort: (" << b.i << "," << b.j << "\n";
//    }
//}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}