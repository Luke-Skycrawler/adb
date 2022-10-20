#include "tests.h"

inline mat3 cross_matrix(const vec3 &a)
{
    mat3 ret;
    ret << 0, -a[2], a[1],
        a[2], 0, -a[0],
        -a[1], a[0], 0;
    return ret;
}
static const vec3 omega(0.0f, 0.0f, 2.0f);
Cube *spinning_cube()
{
    auto R = cross_matrix(omega);
    auto *cube = new Cube();
    cube->q_dot = R;
    // cube->p = vec3(0.0f, 1.0f, 0.0f);
    cube->p = vec3(0.0f, 0.2f, 0.0f);
    // cube->p_dot = vec3(0.0f, -200.0f, 0.0f);
    return cube;
}
