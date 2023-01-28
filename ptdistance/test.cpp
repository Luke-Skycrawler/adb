#include "pch.h"
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ipc/distance/point_triangle.hpp>
#include "../model/affine_body.h"
#include "../model/geometry.h"
#include <random>
#include <array>
using namespace std;

TEST(ipctkref, random) {
    static const int n_pts = 1000;
    array<vec3, 4> pts[n_pts];
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n_pts; i++)
        for (int j = 0; j < 12; j++) {
            pts[i][j / 3](j % 3) = dist(gen);
        }
    for (int i = 0; i < n_pts; i++) {
        auto& pt{ pts[i] };
        Face f{ pt[1], pt[2], pt[3] };
        double dipc = ipc::point_triangle_distance(pt[0], pt[1], pt[2], pt[3]);
        ipc::PointTriangleDistanceType pt_type;

        double dself = vf_distance(pt[0], f, pt_type);
        EXPECT_TRUE(abs(dipc - dself) < 1e-6) << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
       ;
    }

}
//TEST(distance_type, random)
//{
//    static const int n_pts = 1000;
//    array<vec3, 4> pts[n_pts];
//    default_random_engine gen;
//    uniform_real_distribution<double> dist(0.0, 1.0);
//    for (int i = 0; i < n_pts; i++)
//       for (int j = 0; j < 12; j++) {
//            pts[i][j / 3](j % 3) = dist(gen);
//       }
//    for (int i = 0; i < n_pts; i++) {
//       auto& pt{ pts[i] };
//       Face f{ pt[1], pt[2], pt[3] };
//       auto tipc = ipc::point_triangle_distance_type(pt[0], pt[1], pt[2], pt[3]);
//       ipc::PointTriangleDistanceType pt_type;
//       auto tself = pt_distance_type(pt[0], f, pt_type);
//       int tipc_int = static_cast<int>(tipc);
//       EXPECT_TRUE(tipc == tself) << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
//       ;
//    }
//}
#include <chrono>
#include <spdlog/spdlog.h>
#define DURATION_TO_DOUBLE(X) duration_cast<duration<double>>(high_resolution_clock::now() - (X)).count() * 1000

using namespace std::chrono;

TEST(ipctkref, time_cosumption) {
    static const int n_pts = 1000;
    array<vec3, 4> pts[n_pts];
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n_pts; i++)
       for (int j = 0; j < 12; j++) {
            pts[i][j / 3](j % 3) = dist(gen);
       }

    auto ipctkstart = high_resolution_clock::now();
    auto selfstart = high_resolution_clock::now();
    for (int i = 0; i < n_pts; i++) {
       auto& pt{ pts[i] };
       ipc::PointTriangleDistanceType pt_type;
       Face f{ pt[1], pt[2], pt[3] };
       double dself = vf_distance(pt[0], f, pt_type);
    }
    double selfs = DURATION_TO_DOUBLE(selfstart);
    for (int i = 0; i < n_pts; i++) {
       auto& pt{ pts[i] };
       Face f{ pt[1], pt[2], pt[3] };
       double dipc = ipc::point_triangle_distance(pt[0], pt[1], pt[2], pt[3]);
    }
    double ipcs = DURATION_TO_DOUBLE(ipctkstart);
    
    spdlog::info("time: ipc = {:0.6f} sec. self = {:0.6f} sec", ipcs, selfs);
    
}
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}