#include "tests.h"

inline mat3 cross_matrix(vec3 a){
    mat3 ret;
    ret <<  0, -a[2], a[1],
            a[2], 0, -a[0],
            -a[1], a[0], 0;
    return ret;
}
Cube &spinning_cube(vec3 &omega){
    auto A = cross_matrix(omega);
    auto cube = Cube();
    for (int i = 0; i< 3; i++){
        cube.q_dot[i] = A.col(i);
    }
    return cube;
}
