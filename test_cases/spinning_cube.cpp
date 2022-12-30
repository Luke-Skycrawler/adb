#include "tests.h"
#include <nlohmann/json.hpp>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
using namespace std;
using json = nlohmann::json;

inline mat3 cross_matrix(const vec3& a)
{
    mat3 ret;
    ret << 0, -a[2], a[1],
        a[2], 0, -a[0],
        -a[1], a[0], 0;
    return ret;
}
static const vec3 omega(0.0, 0.0, 20.0);
Cube* spinning_cube()
{
    auto R = cross_matrix(omega);
    auto* cube = new Cube();
    cube->q_dot = R;
    cube->p = vec3(0.0, 1.0, 0.0);
    // cube->p = vec3(0.0, 0.2, 0.0);
    // cube->p_dot = vec3(0.0, -200.0, 0.0);
    cube->prepare_q_array();
    return cube;
}

vector<Cube> cube_blocks(int n)
{
    vector<Cube> cubes;
    for (int i = 0; i < n; i++) {
        Cube* cube = new Cube();
        auto R = cross_matrix(omega);
        cube->q_dot = R;

        cube->p = vec3(i * 0.5, 1.0 * (i * 1.5 + 0.1), i * 0.5);
        cube->p_dot = vec3(0.0, i * -5.0, 0.0);
        cube->prepare_q_array();
        cubes.push_back(*cube);
    }
    return cubes;
}

vector<Cube> customize(string file){
    vector<Cube> cubes;
    std::fstream f(file);
    json data = json::parse(f);
    for (auto &it: data) {
        Cube a;
        array<double, 3> omega = {0.0, 0.0, 20.0};
        omega = it["omega"];
        auto p = it["p"];
        auto p_dot = it["p_dot"];
        double theta = it["theta"] / 180.0 * M_PI;


        a.p_dot = vec3(p_dot[0], p_dot[1], p_dot[2]);
        a.p = vec3(p[0], p[1], p[2]);
        a.q_dot = cross_matrix(vec3(omega[0], omega[1], omega[2]));
        double s = sin(theta), c = cos(theta);
        
        a.A(0, 0) = c;
        a.A(1, 1) = c;
        a.A(1, 0) = s;
        a.A(0, 1) = -s;
        a.prepare_q_array();
        cubes.push_back(a);
    }
    return cubes;
}