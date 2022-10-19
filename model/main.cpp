#include "../test_cases/tests.h"
#include "time_integrator.h"
#include <iostream>
using namespace std;

static const float dt = 1e-3;
int main(void) {
    vec3 omega(10.0f, 0.0f, 0.0f);
    auto cube = spinning_cube(omega);
    vector<Cube> cubes;
    cubes.push_back(*cube);
    int timestep = 0;
    while (1){
        implicit_euler(cubes);
        if (timestep++ >= 10) break;
    }
    return 0;
}