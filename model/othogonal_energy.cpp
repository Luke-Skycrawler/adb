// #include <cmath>
#include "othogonal_energy.h"

using namespace othogonal_energy;
#define TEST_PYTHON
namespace othogonal_energy {

    static const scalar kappa = 1e9;

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
                vec3 qi = q.col(i), qj = q.col(j);
                g += (qi.dot(qj) - kronecker(i, j)) * qj;
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
            vec3 qi = q.col(i), qj = q.col(j);
            h.setZero(3,3);
            if (i == j) {
                h = 2 * qi * qi.transpose() + (qi.dot(qi) -1) * Matrix<scalar, 3, 3>::Identity(3, 3);
                for(int k = 1; k < 4; k++)if(k!= i) {
                    auto &qk = q.col(k);
                    h += qk * qk.transpose();
                }
            }
            else {
                h = Matrix<scalar, 3, 3>::Identity(3, 3) * qj.dot(qi) + qj * qi.transpose();
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

};
