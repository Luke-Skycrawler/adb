#include "pch.h"
#include "../model/cuda_header.cuh"
#include <cmath>
#include "../model/affine_body.h"
#include "../model/collision.h"
#include <random>
#include <spdlog/spdlog.h>

using namespace std;
using namespace Eigen;
TEST(set_triangle_moving_point, deliberate_collide)
{
   static const int n_pts = 1000;
   array<vec3, 5> pts[n_pts];
   double tois[n_pts];
   default_random_engine gen;
   uniform_real_distribution<double> dist(0.0, 1.0);
   for (int i = 0; i < n_pts; i++) {
       double toi = dist(gen);
       for (int j = 0; j < 12; j++) {
           pts[i][j / 3](j % 3) = dist(gen);
           pts[i][4] = pts[i][2] / toi - pts[i][0] * (1 - toi) / toi;
       }
       tois[i] = toi;
   }
   for (int i = 0; i < n_pts; i++) {
       auto& pt{ pts[i] };
       Face f{ pt[1], pt[2], pt[3] };
       double toi = pt_collision_time(pts[i][0], f, pts[i][4], f);
       EXPECT_TRUE(abs(toi - tois[i]) < 1e-6) << "toi = " << toi << " truth = " << tois[i] << "\n"
                                              << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
   }
}

TEST(det_poly, Eigen_ref)
{
    static const int n_pts = 80;
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int k = 0; k < n_pts; k++) {
        mat3 a, b;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                a(i, j) = dist(gen);
                b(i, j) = dist(gen);
            }
        Vector4d p = det_polynomial(a, b);
        const auto eval = [](double t, Vector4d& poly) -> double {
            return poly(0) + t * (poly(1) + t * (poly(2) + t * poly(3)));
        };
        double t0 = eval(0.0, p);
        double t1 = eval(1.0, p);
        double det1 = (a + b).determinant();
        double det0 = b.determinant();
        double w = dist(gen);
        double dett = (a * w + b).determinant();
        double tt = eval(w, p);
        EXPECT_TRUE(abs(t0 - det0) < 1e-6) << "a = " << a << " \ndet(a) = " << det0 << " computed = " << t0;
        EXPECT_TRUE(abs(t1 - det1) < 1e-6) << "b = " << b << " \ndet(b) = " << det1 << " computed = " << t1;
        EXPECT_TRUE(abs(tt - dett) < 1e-6) << "t = " << w << " \ndet = " << dett << " computed = " << tt;
    }
}

// TEST(random, tight_inclusion_ref)
// {
//     static const int n_pts = 1000;
//     array<vec3, 8> pts[n_pts];
//     double tois[n_pts];
//     default_random_engine gen;
//     uniform_real_distribution<double> dist(0.0, 1.0);
//     for (int i = 0; i < n_pts; i++) {
//         for (int j = 0; j < 24; j++) {
//             pts[i][j / 3](j % 3) = dist(gen);
//         }
//     }
//     for (int i = 0; i < n_pts; i++) {
//         auto& pt{ pts[i] };
//         Face f0{ pt[1], pt[2], pt[3] }, f1{ pt[5], pt[6], pt[7] };
//         double ticcdt = vf_collision_detect(pt[0], pt[4], f0, f1);
//         ticcdt = min(ticcdt, 1.0);
//         double selft = pt_collision_time(pt[0], f0, pt[4], f1);
//         EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
//                                                 << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
//     }
// }

// TEST(random_ee, tight_inclusion_ref) {
//     static const int n_pts = 1000;
//     array<vec3, 8> pts[n_pts];
//     double tois[n_pts];
//     default_random_engine gen;
//     uniform_real_distribution<double> dist(0.0, 1.0);
//     for (int i = 0; i < n_pts; i++) {
//         for (int j = 0; j < 24; j++) {
//             pts[i][j / 3](j % 3) = dist(gen);
//         }
//     }
//     for (int i = 0; i < n_pts; i++) {
//         auto& pt{ pts[i] };
//         Edge ei0{ pt[0], pt[1]}, ej0{pt[2], pt[3] }, ei1{ pt[4], pt[5]}, ej1{pt[6], pt[7] };
//         double ticcdt = ee_collision_detect(ei0,ej0, ei1, ej1);
//         ticcdt = min(ticcdt, 1.0);
//         double selft = ee_collision_time(ei0, ej0, ei1, ej1);
//         EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
//                                                 << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
//     }
// }
#include "../model/cuda_header.cuh"
__device__ __host__ float pt_collision_time(
    const vec3f &p0,
    const Facef &t0, 
    const vec3f &p1,
    const Facef &t1
);

__device__ __host__ float ee_collision_time(
    const Edgef &ei0, 
    const Edgef &ej0,
    const Edgef &ei1,
    const Edgef &ej1
);


inline vec3f to_vec3f(const vec3& a)
{
    return make_float3(a[0], a[1], a[2]);
}
TEST(random_eef, tight_inclusion_ref) {
    static const int n_pts = 1000;
    array<vec3, 8> pts[n_pts];
    double tois[n_pts];
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < 24; j++) {
            pts[i][j / 3](j % 3) = dist(gen);
        }
    }
    for (int i = 0; i < n_pts; i++) {
        auto& pt{ pts[i] };
        Edgef ei0{ to_vec3f(pt[0]), to_vec3f( pt[1])}, ej0{to_vec3f(pt[2]), to_vec3f(pt[3]) }, 
            ei1{ to_vec3f(pt[4]), to_vec3f(pt[5])}, ej1{to_vec3f(pt[6]), to_vec3f(pt[7]) };

        Edge ei0d{ pt[0], pt[1] }, ej0d{ pt[2], pt[3] }, ei1d{ pt[4], pt[5] }, ej1d{ pt[6], pt[7] };
        
        double ticcdt = ee_collision_time(ei0d,ej0d, ei1d, ej1d);
        ticcdt = min(ticcdt, 1.0);
        float selft = ee_collision_time(ei0, ej0, ei1, ej1);
        EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
                                                << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
    }
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}