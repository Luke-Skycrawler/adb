#include "othogonal_energy.h"

using namespace othogonal_energy;

namespace othogonal_energy {

    static const double kappa = 1e6;
    vec3 grad(mat3& q, int i) {
        auto ai = q.col(i);
        vec3 ret = 4 * kappa * ai * (ai.dot(ai) - 1);
        for (int j = 0; j < 3; j++) {
            if (i == j) continue;
            auto aj = q.col(j);
            ret += 4 * kappa * (aj.dot(ai)) * aj;
        }
        return ret;
    }
    mat3 grad(mat3& q) {
        mat3 ret;
        for (int i = 0; i < 3; i++) {
            auto gi = grad(q, i);
            ret.col(i) = gi;
        }
        return ret;
    }

    mat3 hessian(mat3& q, int i, int j) {
        mat3 ret;
        auto ai = q.col(i);
        if (i == j) {
            ret = 2 * kappa * (4 * ai * ai.adjoint() + 2 * (ai.dot(ai) - 1) * Matrix3d::Identity(3, 3));
            // adjoint = transpose
            for (int j = 0; j < 3; j++) {
                if (j == i) continue;
                auto aj = q.col(j);
                ret += 4 * kappa * aj * aj.adjoint();
            }
        }
        else {
            auto aj = q.col(j);
            ret = 4 * kappa * (Matrix3d::Identity(3, 3) * aj.dot(ai) + aj * ai.adjoint());
        }
        return ret;
    }

    double otho_energy(mat3 &q) {
        mat3 qqtmi = q * q.adjoint() - Matrix3d::Identity(3,3);
        return qqtmi.squaredNorm() * kappa;
    }
};
