#include "othogonal_energy.h"
#include "marcros_settings.h"

using namespace othogonal_energy;

namespace othogonal_energy {

    static const double kappa = 1e9;

#ifdef TEST_PYTHON
    inline double kronecker(int i, int j)
    {
        return i == j ? 1.0 : 0.0;
    }
    mat3 grad(mat3& q)
    {
        mat3 ret;
        for (int i = 0; i < 3; i++) {
            vec3 g(0.0, 0.0, 0.0);
            for (int j = 0; j < 3; j++) {
                auto qi = q.col(i);
                auto qj = q.col(j);
                g += (qi.dot(qj) - kronecker(i, j)) * qj;
            }
            ret.col(i) = 4 * kappa * g;
        }
        return ret;
    }

    mat3 hessian(mat3& q, int i, int j)
    {
        auto qi = q.col(i), qj = q.col(j);
        
        mat3 h = (qi.dot(qj) - kronecker(i, j)) * Matrix3d::Identity(3,3);
        for (int k = 0; k < 3; k ++){
            auto qk = q.col(k);
            h += (kronecker(k, j) * qi + kronecker(i, j) * qk) * qk.adjoint();
        }
        return 4 * kappa * h;
    }
#else
    vec3 grad(mat3& q, int i)
    {
        auto ai = q.col(i);
        vec3 ret = 4 * kappa * ai * (ai.dot(ai) - 1);
        for (int j = 0; j < 3; j++) {
            if (i == j) continue;
            auto aj = q.col(j);
            ret += 4 * kappa * (aj.dot(ai)) * aj;
        }
        return ret;
    }
    mat3 grad(mat3& q)
    {
        mat3 ret;
        for (int i = 0; i < 3; i++) {
            auto gi = grad(q, i);
            ret.col(i) = gi;
        }
        return ret;
    }
    // #endif
    mat3 hessian(mat3& q, int i, int j)
    {
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

#endif

    double otho_energy(mat3 &q) {
        mat3 qqtmi = q.adjoint() * q - Matrix3d::Identity(3,3);
        return qqtmi.squaredNorm() * kappa;
    }
};
