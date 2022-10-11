#include "tests.h"

inline mat3 cross_matrix(vec3 a){
    mat3 ret;
    ret <<  0, -a[2], a[1],
            a[2], 0, -a[0],
            -a[1], a[0], 0;
    return ret;
}
Cube *spinning_cube(vec3 &omega){
    auto R = cross_matrix(omega);
    auto *cube = new Cube();
    cube->q_dot = R;
    return cube;
}
