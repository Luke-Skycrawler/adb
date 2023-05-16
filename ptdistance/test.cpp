#include "pch.h"
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "../model/affine_body.h"
#include "../model/geometry.h"
#include <random>
#include <array>
#include "../model/cuda_header.cuh"
#include "../model/collision.h"
//#include "../model/cuda_glue.h"
using namespace std;


inline vec3f to_vec3f(const vec3& a)
{
    return make_float3(a[0], a[1], a[2]);
}
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


int edge_edge_distance_type(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1)
{

    
    // EA0_EB0, ///< The edges are closest at vertex 0 of edge A and 0 of edge B.
    // EA0_EB1, ///< The edges are closest at vertex 0 of edge A and 1 of edge B.
    // EA1_EB0, ///< The edges are closest at vertex 1 of edge A and 0 of edge B.
    // EA1_EB1, ///< The edges are closest at vertex 1 of edge A and 1 of edge B.
    // /// The edges are closest at the interior of edge A and vertex 0 of edge B.
    // EA_EB0,
    // /// The edges are closest at the interior of edge A and vertex 1 of edge B.
    // EA_EB1,
    // /// The edges are closest at vertex 0 of edge A and the interior of edge B.
    // EA0_EB,
    // /// The edges are closest at vertex 1 of edge A and the interior of edge B.
    // EA1_EB,
    // EA_EB, ///< The edges are closest at an interior point of edge A and B.
    // AUTO   ///< Automatically determine the closest pair.

    auto u = ea1 - ea0;
    auto v = eb1 - eb0;
    auto w = ea0 - eb0;
    auto a = dot(u, u);
    auto b = dot(u, v);
    auto c = dot(v, v);
    auto d = dot(u, w);
    auto e = dot(v, w);
    auto D = a * c - b * b;
    auto tD = D;

    int default_case = 8;

    float sN = (b * e - c * d), tN;
    if (sN <= 0.0f) {
        tN = e;
        tD = c;
        default_case = 6;
    } else if (sN >= D) {
        tN = e + b;
        tD = c;
        default_case = 7;
    } else {
        tN = a * e - b * d;
        auto tmp = cross(u, v);
        if (tN > 0.0 && tN < tD && dot(tmp, tmp) < 1e-20f * a * c) {
            // avoid nearly parallel edge-edge
            if (sN  < D / 2) {
                tN = e;
                tD = c;
                default_case = 6;
            } else {
                tN = e + b;
                tD = c;
                default_case = 7;
            }
        }
    }

    if (tN<= 0.0f) {
        if (-d <= 0.0f) {
            return 0;
        }
        else if (-d >= a) {
            return 2;
        }
        else {
            return 4;
        }
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0f) {
            return 1;
        } else if ((-d + b) >= a) {
            return 3;
        } else {
            return 5;
        }
    }
    return default_case;
}

TEST(ipctkref, random_ee) {
    static const int n_ees = 1000;
    array<vec3, 4> ees[n_ees];
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n_ees; i++)
        for (int j = 0; j < 12; j++) {
            ees[i][j / 3](j % 3) = dist(gen);
        }
    for (int i = 0; i < n_ees; i++) {
        auto& ee{ ees[i] };
        // Edge ei{ ee[0], ee[1]}, ej { ee[2],  ee[3] };
        auto tipc = ipc::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
        auto iipc = static_cast<int>(tipc);
        vec3f eef[4];

        for (int j = 0; j < 4; j++)
            eef[j] = to_vec3f(ee[j]);
        int tself = edge_edge_distance_type(eef[0], eef[1], eef[2], eef[3]);
        EXPECT_TRUE(iipc == tself) << "ipc type = " << iipc << ", self = " << tself <<"\n";
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

__device__ __host__ float pt_collision_time(
    const vec3f& p0,
    const Facef& t0,
    const vec3f& p1,
    const Facef& t1);

__device__ __host__ float ee_collision_time(
    const Edgef& ei0,
    const Edgef& ej0,
    const Edgef& ei1,
    const Edgef& ej1);

TEST(random_eef, cy_ref)
{
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
        Edgef ei0{ to_vec3f(pt[0]), to_vec3f(pt[1]) }, ej0{ to_vec3f(pt[2]), to_vec3f(pt[3]) },
            ei1{ to_vec3f(pt[4]), to_vec3f(pt[5]) }, ej1{ to_vec3f(pt[6]), to_vec3f(pt[7]) };

        Edge ei0d{ pt[0], pt[1] }, ej0d{ pt[2], pt[3] }, ei1d{ pt[4], pt[5] }, ej1d{ pt[6], pt[7] };

        double ticcdt = ee_collision_time(ei0d, ej0d, ei1d, ej1d);
        ticcdt = min(ticcdt, 1.0);
        float selft = ee_collision_time(ei0, ej0, ei1, ej1);
        EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
                                                << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
    }
}
inline Facef to_facef(const Face& f)
{
    return {
        to_vec3f(f.t0),
        to_vec3f(f.t1),
        to_vec3f(f.t2)
    };
}

TEST(random_ptf, cy_ref)
{
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
        Face f0{ pt[1], pt[2], pt[3] }, f1{ pt[5], pt[6], pt[7] };
        double ticcdt = pt_collision_time(pt[0], f0, pt[4], f1);
        auto selft = pt_collision_time(to_vec3f(pt[0]), to_facef(f0), to_vec3f(pt[4]), to_facef(f1));
        EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
                                                << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
    }
}
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}