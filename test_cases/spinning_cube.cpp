#include "tests.h"
using namespace std;
inline mat3 cross_matrix(const vec3& a)
{
    mat3 ret;
    ret << 0, -a[2], a[1],
        a[2], 0, -a[0],
        -a[1], a[0], 0;
    return ret;
}
static const vec3 omega(0.0, 0.0, 200.0);
Cube* spinning_cube()
{
    auto R = cross_matrix(omega);
    auto* cube = new Cube();
    cube->q_dot = R;
    cube->p = vec3(0.0, 1.0, 0.0);
    // cube->p = vec3(0.0, 0.2, 0.0);
    cube->p_dot = vec3(0.0, -200.0, 0.0);
    return cube;
}

vector<Cube> cube_blocks(int n)
{
    vector<Cube> cubes;
    for (int i = 0; i < n; i++) {
        Cube* cube = new Cube();
        auto R = cross_matrix(omega);
        cube->q_dot = R;

        cube->p = vec3(0.0, 1.0 * (i * 3 + 1), 0.0);
        cube->p_dot = vec3(0.0, -20.0, 0.0);
        cubes.push_back(*cube);
    }
    return cubes;
}
