#include "pch.h"
#include "gtest/gtest.h"
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
// #include "../../model/collision.cpp"

#define GOOGLE_TEST
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

TEST(numerical_grad, hessian){

}

#include "../../view/global_variables.h"
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Implicit_Euler_Test : public ::testing::Test {
 protected:
  void SetUp() override {
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
  #include "../../model/time_integrator.cpp"
  // void TearDown() override {}

};

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
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}