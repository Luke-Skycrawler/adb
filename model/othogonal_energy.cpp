#include "othogonal_energy.h"

using namespace othogonal_energy;

static const float kappa = 1e4;

vec3 &gradient(mat3 &q, int i){
    auto ai = q.col(i);
    auto ret = 4 * kappa * ai * (ai.dot(ai) - 1);
    for (int j=0;j<3;j++){
        if (i == j) continue;
        auto aj = q.col(j);
        ret += 4 * kappa * (aj.dot(ai)) * aj; 
    }
    return ret; 
}
mat3 &gradient(mat3 &q){
    mat3 ret;
    for (int i =0;i<3;i++){
        auto gi = gradient(q, i);
        ret.col(i) = gi;
    }
    return ret;
}

mat3 &hessian(mat3 &q, int i, int j){
    mat3 ret;
    auto ai = q.col(i);
    if (i == j) {
        ret = 2 * kappa * (4 * ai * ai.adjoint() + 2 * (ai.dot(ai) - 1) * Matrix3f::Identity);
        // adjoint = transpose
        for (int j = 0; j < 3; j ++){
            if (j==i) continue;
            auto aj = q.col(j);
            ret += 2 * aj * aj.adjoint();
        }
    }
    else {
        auto aj = q.col(j);
        ret = aj.dot(ai) * Matrix3f::Identity + aj * ai.adjoint();
    }
    return ret;
}

