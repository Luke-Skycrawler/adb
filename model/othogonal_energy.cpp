// #include <cmath>
#include "othogonal_energy.h"

using namespace othogonal_energy;
#define TEST_PYTHON
namespace othogonal_energy {

    static const scalar kappa = 1e9;

#ifdef TEST_PYTHON
    inline scalar kronecker(int i, int j)
    {
        return i == j ? 1.0 : 0.0;
    }

    vec12 grad(const q4& q)
    {
        vec12 ret;
        ret.setZero(12);
        for (int i = 1; i < 4; i++) {
            vec3 g(0.0, 0.0, 0.0);
            for (int j = 1; j < 4; j++) {
                g += (q[i].dot(q[j]) - kronecker(i, j)) * q[j];
            }
            ret.segment<3>(i * 3) = 4 * kappa * g;
        }
        return ret;
    }

    mat12 hessian(const q4& q)
    {
        mat12 H;
        H.setZero(12, 12);
        for (int i= 1; i < 4;i++)for(int j = 1; j < 4;j ++) {
            mat3 h;
            h.setZero(3,3);
            if (i == j) {
                h = 2 * q[i] * q[i].transpose() + (q[i].dot(q[i]) -1) * Matrix<scalar, 3, 3>::Identity(3, 3);
                for(int k = 1; k < 4; k++)if(k!= i) h += q[k] * q[k].transpose();
            }
            else {
                h = Matrix<scalar, 3, 3>::Identity(3, 3) * q[j].dot(q[i]) + q[j] * q[i].transpose();
            }
            H.block<3, 3>(3 * i, 3 * j) = h * 4 * kappa;
        }
        return H;
    }

    scalar otho_energy(const Vector<scalar, -1>& x)
    {
        scalar E = 0;
        vec3 q[4];
        for (int i = 1; i < 4; i++) {
            q[i] = x.segment<3>(i * 3);
        }
        for (int i = 1; i < 4; i++)
            for (int j = 1; j < 4; j++) {
                scalar e = pow(q[i].dot(q[j]) - 1.0 * (i == j), 2);
                E += e;
            }
        return E * kappa;
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
            ret = 2 * kappa * (4 * ai * ai.transpose() + 2 * (ai.dot(ai) - 1) * Matrix<scalar, 3, 3>::Identity(3, 3));
            // adjoint = transpose
            for (int j = 0; j < 3; j++) {
                if (j == i) continue;
                auto aj = q.col(j);
                ret += 4 * kappa * aj * aj.transpose();
            }
        }
        else {
            auto aj = q.col(j);
            ret = 4 * kappa * (Matrix<scalar, 3, 3>::Identity(3, 3) * aj.dot(ai) + aj * ai.transpose());
        }
        return ret;
    }


    /*scalar otho_energy(const mat3 &q) {
        mat3 qqtmi = q.transpose() * q - Matrix<scalar, 3, 3>::Identity(3,3);
        return qqtmi.squaredNorm() * kappa;
    }*/
    #endif
};
