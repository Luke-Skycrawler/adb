#include "pch.h"
#include "../model/cube.h"
#include "../model/iaabb.h"
#include "../model/geometry.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <random>
inline void Cube::draw(Shader& shader) const {}
unsigned *Cube::_edges = nullptr, *Cube::_indices = nullptr;

using namespace std;
using namespace Eigen;
TEST(body_level, compute_intersection_against_brute_force)
{

    int n_cubes = 100;
    const double space_range[2]{ -3.0, 3.0 };

    std::vector<std::unique_ptr<AffineBody>> cubes;
    vector <lu> aabbs; 
    aabbs.resize(n_cubes);

    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n_cubes; i++)
    {
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
        aabbs.push_back(b);
        cubes.push_back(move(a));
    }
    vector<Intersection> overlaps_bf, overlaps_sort;
    intersect_brute_force(n_cubes, cubes, aabbs, overlaps_bf, 1);
    intersect_sort(n_cubes, cubes, aabbs, overlaps_sort, 1);
    const auto les = [](const Intersection &a, const Intersection&b) ->bool {
        return a.i < b.i || (a.i == b.i && a.j < b.j);
    };
    std::sort(overlaps_bf.begin(), overlaps_bf.end(), les);
    std::sort(overlaps_sort.begin(), overlaps_sort.end(), les);
    EXPECT_EQ(overlaps_bf.size(), overlaps_sort.size())
        << "size mismatch"
        << "\n";
    for (int i = 0; i < overlaps_bf.size(); i++) {
        auto a{ overlaps_bf[i] }, b{ overlaps_sort[i] };
        EXPECT_TRUE(a.i == b.i && a.j == b.j) << "brute force: (" << a.i << ", " << a.j << "), sort: (" << b.i << "," << b.j << "\n";
    }
    
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}