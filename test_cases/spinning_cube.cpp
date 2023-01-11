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
inline array<vec3, 4> skew(const vec3& a)
{
    return {
        -vec3(0, -a[2], a[1]),
        -vec3(a[2], 0, -a[0]),
        -vec3(-a[1], a[0], 0)
    };
}
static const vec3 omega(0.0, 0.0, 20.0);
unique_ptr<Cube> spinning_cube()
{
    auto _cube = make_unique<Cube>();
    auto& cube = *_cube;
    auto a = skew(omega);
    for (int i = 0; i < 3; i++) cube.dqdt[i + 1] = a[i];
    cube.q[0] = vec3(0.0, 1.0, 0.0);
    cube.q0 = cube.q;
    return _cube;
}

void cube_blocks(int n)
{
    for (int i = 0; i < n; i++) {
        auto _cube = make_unique<Cube>();
        auto& cube = *_cube;
        auto R = skew(omega);
        for (int j = 0; j < 3; j++) cube.dqdt[j + 1] = R[j];
        cube.dqdt[0] = vec3(0.0, i * -5.0, 0.0);

        cube.q[0] = vec3(i * 0.5, 1.0 * (i * 1.5 + 0.1), i * 0.5);
        cube.q0 = cube.q;
        globals.cubes.push_back(move(_cube));
    }
}

void customize(string file)
{
    vector<Cube> cubes;
    std::fstream f(file);
    json data = json::parse(f);
    for (auto &it: data) {
        auto _a = make_unique<Cube>();
        auto& a = *_a;
        array<double, 3> omega = {0.0, 0.0, 20.0};
        omega = it["omega"];
        auto p = it["p"];
        auto p_dot = it["p_dot"];
        double theta = it["theta"] / 180.0 * M_PI;


        a.dqdt[0] = vec3(p_dot[0], p_dot[1], p_dot[2]);
        a.q[0] = vec3(p[0], p[1], p[2]);
        auto R = skew(vec3(omega[0], omega[1], omega[2]));
        for (int i = 0; i < 3; i++) a.dqdt[i + 1] = R[i];
        double s = sin(theta), c = cos(theta);
        a.q[1](0) = c;
        a.q[2](1) = c;
        a.q[2](0) = -s;
        a.q[1](1) = s;
        a.q0 = a.q;
        globals.cubes.push_back(move(_a));
    }
}