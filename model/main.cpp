#include "../test_cases/tests.h"
#include "time_integrator.h"

using namespace std;

static const dt = 1e-3;
int main(void) {
    vec3 omega(1.0f, 0.0f, 0.0f);
    auto cube = spinning_cube(omega);
    vector<Cube> cubes;
    cubes.push_back(cube);
    while (1){
        implicit_euler(dt, cubes);
        break;
    }
    return 0;
}