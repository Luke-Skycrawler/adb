#include "pch.h"
#include "../model/affine_body.h"
#include "../model/collision.h"
#include <random>
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
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}