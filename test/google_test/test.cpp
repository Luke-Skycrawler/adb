#define GOOGLE_TEST
#include "pch.h"
#include "gtest/gtest.h"
#include "../../model/geo.cpp"
#include "../../model/geometry.h"
#include "../../model/geometry.cpp"
#include "../../model/time_integrator.h"
//#include "../../view/env.h"
#include "../../model/othogonal_energy.h"
#include "../../model/othogonal_energy.cpp"
#include "../../model/barrier.h"
#include "../../model/barrier.cpp"
#include "spdlog/spdlog.h"
#include <vector>
#include "../../test_cases/spinning_cube.cpp"

double Cube::vg_collision_time()
{
    double toi = 1.0;
    for (int i = 0; i < n_vertices; i++) {
        // const vec3 tile_v(vertices()[i]);
        const vec3 v_t2(vt2(i));
        const vec3 v_t1(vt1(i));

        double d2 = vg_distance(v_t2);
        double d1 = vg_distance(v_t1);
        assert(d1 > 0);
        if (d2 < 0) {

            double t = d1 / (d1 - d2);
            auto vtoi = v_t2 * (t * 0.8) + v_t1 * (1 - t * 0.8);
            double dtoi = vg_distance(vtoi);
            spdlog::warn("dtoi = {}, d0 = {}, d1 = {}, toi = {}", dtoi, d1, d2, toi);

            assert(dtoi > 0.0);
            assert(t > 0.0 && t < 1.0);
            toi = min(toi, t);
        }
    }
    return toi;
}

void Cube::prepare_q_array()
{
    q[0] = p_next;
    q0[0] = p;
    dqdt[0] = p_dot;
    for (int i = 1; i < 4; i++) {
        q[i] = q_next.col(i - 1);
        q0[i] = A.col(i - 1);
        dqdt[i] = q_dot.col(i - 1);
    }
}

// #include "../../model/collision.cpp"
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std;
using namespace barrier;
using namespace Eigen;
int Cube::indices[] = { 0 };
int Cube::edges[] = { 0 };
double vf_collision_detect(vec3& p_t0, vec3& p_t1, const Cube& c, int id){
    return 1.0;
}
double Cube::vf_collision_detect(const vec3& dp, const mat3& dq){
  return 1.0;
}


TEST(vf_distance_test, basic_random) {
  std::vector<vec3> vs;
  std::vector<Face> fs;
  for (auto &v : vs){
    for (auto &f : fs){
      double d = vf_distance(v, f);
      EXPECT_EQ(d, 0.0);
      //vf_distance_gradient_x()
    }
  }
  
}
TEST(ee_distance_test, basic_random){
  std::vector<Edge> e1s;
  std::vector<Edge> e2s;
  for (auto &ei: e1s)
    for(auto &ej: e2s) {
      double d = ee_distance(ei, ej);
      EXPECT_EQ(d, 0.0);
    }  
}

//TEST(numerical_grad, Eo_gradient_q) {
//  double dx = 1e-4;
//  for (int _i = 0; _i < 4; _i++, dx /= 2.0){
//    for(int i = 0; i < 12; i ++){
//      double d1 = touch(g, i, dx);
//      d = 
//
//      
//    }
//  }
//}



/*
TEST_F(Implicit_Euler_Test, numerical_grad_Eo){
  for (auto &c:globals.cubes) {
    auto g_computed = othogonal_energy::grad(c.q_next);
    for (int i =0 ; i < 3; i++) 
      for (int j = 0; j < 3; j ++){
        double dx = 1e-4;
        auto q0 = c.q_next;
        auto q1 = q0;
        mat3 grad;
        double e1, e0;
        double g_old = 0.0, g = 0.0;
        e1 = othogonal_energy::otho_energy(q0);
        do {
          g_old = g;
          q1(i, j) += dx;
          e0 = othogonal_energy::otho_energy(q1);
          g = (e1 - e0)  / dx;
          dx /= 2;
          q1 = q0;
          spdlog::info("g0, g1 = {} {}\ndx = {}", g_old, g, dx);
        }
        while (dx  > 1e-8 && abs(g_old - g) > 1e-4);
        spdlog::info("final g0, g1 = {} {}\ndx = {}", g_old, g, dx);
        grad(i, j) = g;
        double t = abs(g - g_computed(i, j));
        EXPECT_TRUE(t < 1e-4) << "inconsistant with numerical grad at ("<< i <<"," << j  << "), \ng= " << g <<" , computed = " << g_computed(i, j);
      }
  }
}
TEST_F(Implicit_Euler_Test, non_increasing_across_iter){
  for (int i = 0; i < 100; i++){
    implicit_euler(globals.cubes, globals.dt);    
  }
}
*/

const double dist = 0.09;
// double d_hat = 10.0;

inline double barrier_gradient(const double d, const double dHat2)
{
    double grad = 0.0;
    if (d < dHat2) {
        double t2 = d - dHat2;
        grad = barrier::kappa * (t2 * std::log(d / dHat2) * -2.0 - (t2 * t2) / d) / (dHat2 * dHat2);
    }
    return grad;
}

inline double barrier_hessian(const double d, const double dHat2)
{
    double hess = 0.0;
    if (d < dHat2) {
        double t2 = d - dHat2;
        hess = barrier::kappa * ((std::log(d / dHat2) * -2.0 - t2 * 4.0 / d) + 1.0 / (d * d) * (t2 * t2)) / (dHat2 * dHat2);
    }
    return hess;
}

double w[] = {
    1.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 1.0, 0.0, 0.0
};

int dim = 3;
double barrier_hessian(double d)
{
    if (d >= d_hat)
        return 0.0f;
    return -barrier::kappa * (2 * log(d / d_hat) + (d - d_hat) / d + (d - d_hat) * (2 / d + d_hat / d / d)) / (d_hat * d_hat);
}

double barrier_gradient(double x)
{
    if (x >= d_hat)
        return 0.0f;
    return -(x - d_hat) * barrier::kappa * (2 * log(x / d_hat) + (x - d_hat) / x) / (d_hat * d_hat);
}

void gradient(const vec3& p, const vec3& t0, const vec3& t1, const vec3& t2, VectorXd& grad)
{
    Vector<double, 12> PT_grad;
    //double dE = barrier_gradient(dist * dist, d_hat);
    double d = ipc::point_triangle_distance(p, t0, t1, t2);
    double dE = barrier_gradient(d);
    ipc::point_triangle_distance_gradient(p, t0, t1, t2, PT_grad);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            grad.data()[3 * i + j] = w[i] * dE * PT_grad.data()[j];
            grad.data()[12 + 3 * i + j] = (w[4 + i] * PT_grad.data()[3 + j] + w[8 + i] * PT_grad.data()[6 + j] + w[12 + i] * PT_grad.data()[9 + j]) * dE;
        }
    }
}

void hessian(const vec3& p, const vec3& t0, const vec3& t1, const vec3& t2, MatrixXd& hess)
{
    Eigen::Matrix<double, 12, 12> PT_hess;
    Vector<double, 12> PT_grad;
    ipc::point_triangle_distance_hessian(p, t0, t1, t2, PT_hess);
    ipc::point_triangle_distance_gradient(p, t0, t1, t2, PT_grad);
    double d = ipc::point_triangle_distance(p, t0, t1, t2);
    double dE = barrier_gradient(d);
    double dE2 = barrier_hessian(d);

    for (int r = 0; r < 12; r++)
        for (int c = r; c < 12; c++) {
            double val = dE * PT_hess.data()[12 * c + r] + dE2 * PT_grad.data()[r] * PT_grad.data()[c];
            PT_hess.data()[12 * c + r] = val;
            PT_hess.data()[12 * r + c] = val;
        }

    int rowId, colId;
    double aa_wr, aa_wc, bb_wr0, bb_wc0, bb_wr1, bb_wc1, bb_wr2, bb_wc2, ab_wr, ab_wc0, ab_wc1, ab_wc2, val;
    for (int r = 0; r < 4; r++) {
        aa_wr = w[r];
        bb_wr0 = w[4 + r];
        bb_wr1 = w[8 + r];
        bb_wr2 = w[12 + r];
        for (int c = r; c < 4; c++) {
            aa_wc = w[c];
            bb_wc0 = w[4 + c];
            bb_wc1 = w[8 + c];
            bb_wc2 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = (r == c ? rr : 0); cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;
                    val = aa_wr * aa_wc * PT_hess.data()[12 * cc + rr];
                    hess.data()[24 * colId + rowId] = val;
                    hess.data()[24 * rowId + colId] = val;

                    val = bb_wc0 * (bb_wr0 * PT_hess.data()[12 * (3 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (3 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (3 + cc) + 9 + rr]) + bb_wc1 * (bb_wr0 * PT_hess.data()[12 * (6 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (6 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (6 + cc) + 9 + rr]) + bb_wc2 * (bb_wr0 * PT_hess.data()[12 * (9 + cc) + 3 + rr] + bb_wr1 * PT_hess.data()[12 * (9 + cc) + 6 + rr] + bb_wr2 * PT_hess.data()[12 * (9 + cc) + 9 + rr]);
                    hess.data()[24 * (12 + colId) + (12 + rowId)] = val;
                    hess.data()[24 * (12 + rowId) + (12 + colId)] = val;
                }
        }
    }

    for (int r = 0; r < 4; r++) {
        ab_wr = w[r];

        for (int c = 0; c < 4; c++) {
            ab_wc0 = w[4 + c];
            ab_wc1 = w[8 + c];
            ab_wc2 = w[12 + c];
            for (int rr = 0; rr < 3; rr++)
                for (int cc = 0; cc < 3; cc++) {
                    rowId = 3 * r + rr;
                    colId = 3 * c + cc;
                    val = ab_wr * (ab_wc0 * PT_hess.data()[12 * (3 + cc) + rr] + ab_wc1 * PT_hess.data()[12 * (6 + cc) + rr] + ab_wc2 * PT_hess.data()[12 * (9 + cc) + rr]);
                    hess.data()[24 * (12 + colId) + rowId] = val;
                    hess.data()[24 * rowId + (12 + colId)] = val;
                }
        }
    }
}



class Implicit_Euler_Test : public ::testing::Test {
protected:
#include "../../view/global_variables.h"

    void SetUp() override
    {
        Cube::gen_indices();
        reset();
    }
    void reset()
    {
        std::ifstream f("../../config.json");
        json data = json::parse(f);
        double dt = data["dt"];
#ifdef _TEST_CASE_2_CUBES
        globals.cubes = cube_blocks(2);
#else
#ifdef _TEST_CASE_FROM_FILE

#else
        globals.cubes.push_back(*spinning_cube());
#endif
#endif
        globals.dt = data["dt"];
        globals.max_iter = data["max_iter"];
    }
    GlobalVariableMainCPP globals;
    //#include "../../model/time_integrator.cpp"
    // void TearDown() override {}
};

#include <random>
TEST_F(Implicit_Euler_Test, grad_random_distance)
{
    double ds[10];
    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < 10; i++) {
        double d = distribution(gen) * d_hat;
        double b1 = barrier_gradient(d);
        double b2 = barrier_gradient(d, d_hat);
        EXPECT_TRUE(abs(b1 - b2) < 1);
    }
}
TEST_F(Implicit_Euler_Test, hess_random_distance)
{
    double ds[10];
    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < 10; i++) {
        double d = distribution(gen) * d_hat;
        double b1 = barrier_hessian(d);
        double b2 = barrier_hessian(d, d_hat);
        EXPECT_TRUE(abs(b1 - b2) < 1);
    }
}
TEST_F(Implicit_Euler_Test, point_triangle)
{
    double lam1, lam2, lam3;

    double lams[8][3] = {
        { 0.0, 0.0, 1.0 },
        { 0.0, 1.0, 0.0 },
        { 1.0, 0.0, 0.0 },
        { 0.5, 0.5, 0.0 },
        { 1 / 3., 1 / 3., 1 / 3. },
        { 1., 1., -1. },
        { 1., -1., 1. },
        { -1., 1., 1. }
    };
    vec3 p(0.0, 0.0, 0.0),
        t0(0.0, dist, 0.0), t1(0.0, dist, 1.0), t2(1.0, dist, 0.0),
        p0(0.0, 0.0, 0.0), p2(1.0, 0.0, 0.0), p1(0.0, 0.0, 1.0);

    for (int i = 0; i < 8; i++) {
        p = lams[i][0] * p0 + lams[i][1] * p1 + lams[i][2] * p2;

        MatrixXd hess1;
        Matrix<double, 12, 12> hess21, hess22;
        VectorXd grad1;
        Vector<double, 12> grad21, grad22;
        hess1.setZero(24, 24);
        // hess2.setZero(24, 24);

        grad1.setZero(24);
        grad21.setZero(12);
        grad22.setZero(12);
        // grad2.setZero(24);

        gradient(p, t0, t1, t2, grad1);
        hessian(p, t0, t1, t2, hess1);
        hess21.setZero(12, 12);
        hess22.setZero(12, 12);

        array<vec3, 4> pt = { p, t0, t1, t2 };
        array<int, 4> ij = { 0, 0, 1, 1 };
        
        Matrix<double, 12, 12> off_diag;  
        off_diag.setZero(12, 12);

        ipc_term(hess21, hess22, grad21, grad22, pt, ij, off_diag);
        bool hess_eq = ((hess1 - hess21).array().abs() <= hess1.array().abs() * 1e-2).all();
        bool grad_eq = ((grad1 - grad22).array().abs() <= grad1.array().abs() * 1e-2).all();
        for (int k = 0; k < 12; k++)
            for (int j = 0; j < 12; j++) {
                //EXPECT_EQ(hess1(k, j), hess21(k, j)) << "i = " << k << ", j = " << j;
                //EXPECT_EQ(hess1(k + 12, j + 12), hess22(k, j)) << "i = " << k << ", j = " << j;
                EXPECT_EQ(hess1(k, j + 12), off_diag(k, j)) << "i = " << k << ", j = " << j;
            }
        
            EXPECT_EQ(grad1(0), grad21(0));
        EXPECT_EQ(grad1(1), grad21(1));
        EXPECT_EQ(grad1(2), grad21(2));

            EXPECT_EQ(grad1(12), grad22(0));
        EXPECT_EQ(grad1(12 + 1), grad22(1));
        EXPECT_EQ(grad1(12 + 2), grad22(2));
        
        
        //EXPECT_TRUE(grad_eq);
        //EXPECT_TRUE(hess_eq);
    }
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}