#include "pch.h"
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/smooth_friction_mollifier.hpp>

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
    }

}


// int edge_edge_distance_type(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1)
// {

    
//     // EA0_EB0, ///< The edges are closest at vertex 0 of edge A and 0 of edge B.
//     // EA0_EB1, ///< The edges are closest at vertex 0 of edge A and 1 of edge B.
//     // EA1_EB0, ///< The edges are closest at vertex 1 of edge A and 0 of edge B.
//     // EA1_EB1, ///< The edges are closest at vertex 1 of edge A and 1 of edge B.
//     // /// The edges are closest at the interior of edge A and vertex 0 of edge B.
//     // EA_EB0,
//     // /// The edges are closest at the interior of edge A and vertex 1 of edge B.
//     // EA_EB1,
//     // /// The edges are closest at vertex 0 of edge A and the interior of edge B.
//     // EA0_EB,
//     // /// The edges are closest at vertex 1 of edge A and the interior of edge B.
//     // EA1_EB,
//     // EA_EB, ///< The edges are closest at an interior point of edge A and B.
//     // AUTO   ///< Automatically determine the closest pair.

//     auto u = ea1 - ea0;
//     auto v = eb1 - eb0;
//     auto w = ea0 - eb0;
//     auto a = dot(u, u);
//     auto b = dot(u, v);
//     auto c = dot(v, v);
//     auto d = dot(u, w);
//     auto e = dot(v, w);
//     auto D = a * c - b * b;
//     auto tD = D;

//     int default_case = 8;

//     float sN = (b * e - c * d), tN;
//     if (sN <= 0.0f) {
//         tN = e;
//         tD = c;
//         default_case = 6;
//     } else if (sN >= D) {
//         tN = e + b;
//         tD = c;
//         default_case = 7;
//     } else {
//         tN = a * e - b * d;
//         auto tmp = cross(u, v);
//         if (tN > 0.0 && tN < tD && dot(tmp, tmp) < 1e-20f * a * c) {
//             // avoid nearly parallel edge-edge
//             if (sN  < D / 2) {
//                 tN = e;
//                 tD = c;
//                 default_case = 6;
//             } else {
//                 tN = e + b;
//                 tD = c;
//                 default_case = 7;
//             }
//         }
//     }

//     if (tN<= 0.0f) {
//         if (-d <= 0.0f) {
//             return 0;
//         }
//         else if (-d >= a) {
//             return 2;
//         }
//         else {
//             return 4;
//         }
//     } else if (tN >= tD) {
//         if ((-d + b) <= 0.0f) {
//             return 1;
//         } else if ((-d + b) >= a) {
//             return 3;
//         } else {
//             return 5;
//         }
//     }
//     return default_case;
// }
void pt_grad_hess12x12(vec3f* pt, float* pt_grad, float* pt_hess, bool psd, float* buf);
// buf is not optional
void ee_grad_hess12x12(vec3f* ee, float* ee_grad, float* ipc_hess, float* buf);
using namespace Eigen;
TEST(ipctkref, random_ee) {
    static const int n_ees = 100;
    array<vec3, 4> ees[n_ees];
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n_ees; i++)
        for (int j = 0; j < 12; j++) {
            ees[i][j / 3](j % 3) = dist(gen);
        }

    bool encountered_types[9] {false}; 
    for (int i = 0; i < n_ees; i++) {
        auto& ee{ ees[i] };
        // Edge ei{ ee[0], ee[1]}, ej { ee[2],  ee[3] };
        auto tipc = ipc::edge_edge_distance_type(ee[0], ee[1], ee[2], ee[3]);
        auto iipc = static_cast<int>(tipc);
        vec3f eef[4];

        for (int j = 0; j < 4; j++)
            eef[j] = to_vec3f(ee[j]);
        int tself = dev::edge_edge_distance_type(eef[0], eef[1], eef[2], eef[3]);
        EXPECT_TRUE(iipc == tself) << "ipc type = " << iipc << ", self = " << tself <<"\n";
        auto dist_ipc = ipc::edge_edge_distance(ee[0], ee[1], ee[2], ee[3], tipc);
        auto dist_self = dev::edge_edge_distance(eef[0], eef[1], eef[2], eef[3], tself);
        EXPECT_TRUE(abs(dist_ipc - dist_self) < 1e-6) << "ee distance error";

        vec12 grad_ipc;
        ipc::edge_edge_distance_gradient(ee[0], ee[1], ee[2], ee[3], grad_ipc, tipc);
        float gradf[12], buf[300];
        dev::edge_edge_distance_gradient(eef[0], eef[1], eef[2], eef[3], gradf, tself, buf);
        auto grad_self = Map<Vector<float , 12>>(gradf).cast<double>();
        EXPECT_TRUE(grad_self.isApprox(grad_ipc, 1e-3)) << "type " << tself <<  "ee grad error, diff norm = " << (grad_ipc - grad_self).norm() << ", margin = " << (grad_ipc - grad_self).norm() / grad_ipc.norm();
        
        mat12 hess_ipc;
        ipc::edge_edge_distance_hessian(ee[0], ee[1], ee[2], ee[3], hess_ipc, tipc);
        float hessf[144];
        dev::edge_edge_distance_hessian(eef[0], eef[1], eef[2], eef[3], hessf, tself, buf);
        auto hess_self = Map<Matrix<float , 12, 12>>(hessf).cast<double>();
        EXPECT_TRUE(hess_self.isApprox(hess_ipc, 1e-3)) << "type" << tself << " ee hess error, diff norm = " << (hess_ipc - hess_self).norm() << ", margin = " << (hess_ipc - hess_self).norm() / hess_ipc.norm();

        float closest, ux, uy;
        ee_uktk(eef, closest, tself, nullptr, ux, uy);
        EXPECT_TRUE(abs(closest - dist_self) < 1e-6) << "pt distance error, closest = " << closest << ", dist = " << dist_self << " type = " << tself;
        if (tself == 8) {
            auto lams = ipc::edge_edge_closest_point(ee[0], ee[1], ee[2], ee[3]);
            float a, b;
            edge_edge_closest_point(eef[0], eef[1], eef[2], eef[3], a, b);
            EXPECT_TRUE(abs(a - lams[0]) < 1e-6) << "closest point error, a = " << a << ", lams[0] = " << lams[0];
            EXPECT_TRUE(abs(b - lams[1]) < 1e-6) << "closest point error, b = " << b << ", lams[1] = " << lams[1];
        }

        encountered_types[tself] =true;
    }
    cout << "encountered types: ";
    for (int i = 0; i < 9; i ++) if (encountered_types[i]) cout << i << " ";
    cout << "\n";
}

TEST(ipctkref, random_pt) {
    static const int n_pts = 100;
    array<vec3, 4> pts[n_pts];
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n_pts; i++)
        for (int j = 0; j < 12; j++) {
            pts[i][j / 3](j % 3) = dist(gen);
        }

    bool encountered_types[7] {false}; 
    for (int i = 0; i < n_pts; i++) {
        auto& pt{ pts[i] };
        auto tipc = ipc::point_triangle_distance_type(pt[0], pt[1], pt[2], pt[3]);
        auto iipc = static_cast<int>(tipc);
        vec3f ptf[4];

        for (int j = 0; j < 4; j++)
            ptf[j] = to_vec3f(pt[j]);
        int tself = dev::point_triangle_distance_type(ptf[0], ptf[1], ptf[2], ptf[3]);
        EXPECT_TRUE(iipc == tself) << "ipc type = " << iipc << ", self = " << tself <<"\n";
        auto dist_ipc = ipc::point_triangle_distance(pt[0], pt[1], pt[2], pt[3], tipc);
        auto dist_self = dev::point_triangle_distance(ptf[0], ptf[1], ptf[2], ptf[3], tself);
        EXPECT_TRUE(abs(dist_ipc - dist_self) < 1e-6) << "pt distance error";

        vec12 grad_ipc;
        ipc::point_triangle_distance_gradient(pt[0], pt[1], pt[2], pt[3], grad_ipc, tipc);
        float gradf[12], buf[300];
        dev::point_triangle_distance_gradient(ptf[0], ptf[1], ptf[2], ptf[3], gradf, tself, buf);
        auto grad_self = Map<Vector<float , 12>>(gradf).cast<double>();
        EXPECT_TRUE(grad_self.isApprox(grad_ipc, 1e-3)) << "type " << tself <<  "pt grad error, diff norm = " << (grad_ipc - grad_self).norm() << ", margin = " << (grad_ipc - grad_self).norm() / grad_ipc.norm();
        
        mat12 hess_ipc;
        ipc::point_triangle_distance_hessian(pt[0], pt[1], pt[2], pt[3], hess_ipc, tipc);
        float hessf[144];
        dev::point_triangle_distance_hessian(ptf[0], ptf[1], ptf[2], ptf[3], hessf, tself, buf);
        auto hess_self = Map<Matrix<float , 12, 12>>(hessf).cast<double>();
        EXPECT_TRUE(hess_self.isApprox(hess_ipc, 1e-3)) << "type" << tself << " pt hess error, diff norm = " << (hess_ipc - hess_self).norm() << ", margin = " << (hess_ipc - hess_self).norm() / hess_ipc.norm();

        float closest, ux, uy;
        pt_uktk(ptf, closest, tself, nullptr, ux, uy);
        EXPECT_TRUE(abs(closest - dist_self) < 1e-6) << "ee distance error, closest = " << closest << ", dist = " << dist_self << " type = " << tself;
        encountered_types[tself] =true;
    }
    cout << "encountered types: ";
    for (int i = 0; i < 7; i ++) if (encountered_types[i]) cout << i << " ";
    cout << "\n";
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

// TEST(random_eef, cy_ref)
// {
//     static const int n_ees = 1000000;
//     array<vec3, 8>* ees = new array<vec3, 8>[n_ees];
//     double tois[n_ees];
//     default_random_engine gen;
//     uniform_real_distribution<double> dist(0.0, 1.0);
//     for (int i = 0; i < n_ees; i++) {
//        for (int j = 0; j < 24; j++) {
//             ees[i][j / 3](j % 3) = dist(gen);
//        }
//     }
//     for (int i = 0; i < n_ees; i++) {
//        auto& ee{ ees[i] };
//        Edgef ei0{ to_vec3f(ee[0]), to_vec3f(ee[1]) }, ej0{ to_vec3f(ee[2]), to_vec3f(ee[3]) },
//            ei1{ to_vec3f(ee[4]), to_vec3f(ee[5]) }, ej1{ to_vec3f(ee[6]), to_vec3f(ee[7]) };

//        Edge ei0d{ ee[0], ee[1] }, ej0d{ ee[2], ee[3] }, ei1d{ ee[4], ee[5] }, ej1d{ ee[6], ee[7] };

//        double ticcdt = ee_collision_time(ei0d, ej0d, ei1d, ej1d);
//        ticcdt = min(ticcdt, 1.0);
//        float selft = ee_collision_time(ei0, ej0, ei1, ej1);
//        EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
//                                                << ee[0].transpose() << " " << ee[1].transpose() << " " << ee[2].transpose() << " " << ee[3].transpose() << " "
//                                                << ee[4].transpose() << " " << ee[5].transpose() << " " << ee[6].transpose() << " " << ee[7].transpose();
//     }
// }

TEST(ee_failed_case, cy_ref)
{
    array<vec3, 8> ee{
        vec3{
            0.76998698001534638, 0.52191623024087419, 0.77911846846106003 },
        vec3{
            0.83760369176268235, 0.38750874151707543, 0.83927468129196192 },
        vec3{
            0.80349972107488199, 0.48671638982127269, 0.042644820842929754 },
        vec3{
            0.15611553748487378, 0.10728702102678885, 0.43864950524604462 },
        vec3{
            0.40246360472469922, 0.34612986513016075, 0.013736339106809598 },
        vec3{
            0.64244239849620144, 0.7296534193195211, 0.70907958767250301 },
        vec3{
            0.48479663060750033, 0.74796128036213472, 0.57174560980604283 },
        vec3{
            0.77195343480467049, 0.10616174525034776, 0.495019684136278 }
    };
    Edgef ei0{ to_vec3f(ee[0]), to_vec3f(ee[1]) }, ej0{ to_vec3f(ee[2]), to_vec3f(ee[3]) },
        ei1{ to_vec3f(ee[4]), to_vec3f(ee[5]) }, ej1{ to_vec3f(ee[6]), to_vec3f(ee[7]) };

    Edge ei0d{ ee[0], ee[1] }, ej0d{ ee[2], ee[3] }, ei1d{ ee[4], ee[5] }, ej1d{ ee[6], ee[7] };

    double ticcdt = ee_collision_time(ei0d, ej0d, ei1d, ej1d);
    ticcdt = min(ticcdt, 1.0);
    float selft = ee_collision_time(ei0, ej0, ei1, ej1);
    EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
                                            << ee[0].transpose() << " " << ee[1].transpose() << " " << ee[2].transpose() << " " << ee[3].transpose();
}
inline Facef to_facef(const Face& f)
{
    return {
        to_vec3f(f.t0),
        to_vec3f(f.t1),
        to_vec3f(f.t2)
    };
}

// TEST(random_ptf, cy_ref)
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
//         double ticcdt = pt_collision_time(pt[0], f0, pt[4], f1);
//         auto selft = pt_collision_time(to_vec3f(pt[0]), to_facef(f0), to_vec3f(pt[4]), to_facef(f1));
//         EXPECT_TRUE(abs(ticcdt - selft) < 1e-4) << "computed = " << selft << " truth = " << ticcdt << "\n"
//                                                 << pt[0].transpose() << " " << pt[1].transpose() << " " << pt[2].transpose() << " " << pt[3].transpose();
//     }
// }

Matrix2d project_to_psd(const Matrix2d& A);

void friction(
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 12, 2>& Tk,
    Vector<double, 12>& g, Matrix<double, 12, 12>& H,
    Matrix<double, 2, 2>& M_ret)
{
    static double mu = 0.2;
    static const double evh = 1e-5, h2 = 1e-4;
    double uk = sqrt(_uk(0) * _uk(0) + _uk(1) * _uk(1));
    // if (uk < 1e-10) return;

    auto f1 = ipc::f1_SF_over_x(uk, evh);

    Vector<double, 12> F_k = mu * contact_lambda * f1 * Tk * _uk;
    Matrix<double, 12, 12> D_k_hessian;

    if (uk >= evh) {
        Vector2d ut{ -_uk(1), _uk(0) };
        D_k_hessian = mu * contact_lambda * f1 / (uk * uk) * Tk * ut * (ut.transpose() * Tk.transpose());
    }
    else if (uk <= 0) {
        D_k_hessian = mu * contact_lambda * f1 * Tk * Tk.transpose();
    }
    else {
        double f2_term = -1.0 / (evh * evh);
        Matrix2d M2x2 = f2_term / uk * _uk * _uk.transpose();
        // Matrix2d M2x2 = (df1_term * _uk * _uk.transpose());
        M2x2 += f1 * Matrix2d::Identity(2, 2);
        M2x2 *= mu * contact_lambda;
        auto M2x2_psd = project_to_psd(M2x2);
        EXPECT_TRUE(M2x2.determinant() >= 0.0);
        EXPECT_TRUE(M2x2_psd.isApprox(M2x2)) << "M2x2 = \n" << M2x2 << "\nM2x2_psd = \n" << M2x2_psd;
        D_k_hessian = Tk * M2x2 * Tk.transpose();
        
        M_ret=  M2x2;
    }
    g += F_k;
    H += D_k_hessian;
}


void friction(
    float2 u,
    float lam,
#ifdef TESTING
    float3 Tk[8],
#else
    float _Tk[8],
#endif
    int pt_1_ee_0,
    float* g, float* H);
TEST(random_uk, friction) {
    static const int n_u = 1;
    default_random_engine gen;
    static const double evh = 1e-5;
    uniform_real_distribution<double> dist(0.0, 1.0);
    Vector2d u[n_u];
    for (int i = 0; i < n_u; i ++) {
        u[i] = Vector2d{ dist(gen),
            dist(gen) };
        auto len = dist(gen);
        u[i] = u[i].normalized() * len * evh;
    }
    Matrix<double, 3, 2> tk;
    tk << 1, 0, 0, 0, 1, 0;
    Matrix<double, 12, 2> Tk;
    for (int i = 0; i < 4; i++) {
        Tk.block<3, 2>(3 * i, 0) = tk;
    }
    for (int i = 0; i < n_u; i ++) {
        Matrix<double, 2, 2> M;
        vec12 foo = Vector<double, 12>::Zero();
        mat12 bar = Matrix<double, 12, 12>::Zero();
        Vector<float, 12> g;
        Matrix<float, 12, 12> H;
        g.setZero(12);
        H.setZero(12, 12);
        float3 float8[8] {
            {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0},
            {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}
        };
        friction(u[i], 1.0f, Tk, foo, bar, M);
        friction(make_float2(u[i](0), u[i](1)), 1.0f, float8, 1, g.data(), H.data());
        EXPECT_TRUE(foo.isApprox(g.cast<double>()), 1e-3) << "diff norm = " << ((foo - g.cast<double>())).norm() << " norm ref = " << foo.norm(); 
        EXPECT_TRUE(bar.isApprox(H.cast<double>()), 1e-3) << "diff norm = " << ((bar - H.cast<double>())).norm() << " norm ref = " << bar.norm();
    }

}
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}