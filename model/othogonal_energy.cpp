#include "othogonal_energy.h"
#include <cmath>
#include "marcros_settings.h"

using namespace othogonal_energy;
namespace othogonal_energy {

    static const double kappa = 1e9;

#ifdef TEST_PYTHON
    inline double kronecker(int i, int j)
    {
        return i == j ? 1.0 : 0.0;
    }
    
    VectorXd grad(vec3 q[])
    {
        Vector<double, 12> ret;
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

    //MatrixXd hessian(vec3 q[])
    //{
    //    Matrix<double, 12, 12> H;
    //    H.setZero(12, 12);
    //    for (int i = 1; i < 4; i++) {
    //        auto qi = q[i];
    //        for (int j = 1; j < 4; j++) {
    //            auto qj = q[j];
    //            mat3 h = (qi.dot(qj) - kronecker(i, j)) * Matrix3d::Identity(3, 3);
    //            for (int k = 1; k < 4; k++) {
    //                auto qk = q[k];
    //                h += (kronecker(k, j) * qi + kronecker(i, j) * qk) * qk.adjoint();
    //            }
    //            H.block<3,3>(i * 3, j * 3) = 4 * kappa * h;
    //        }
    //    }
    //    return H;
    //}
    MatrixXd hessian(vec3 q[]) {
        Matrix<double, 12, 12> H;
        H.setZero(12, 12);
        for (int i= 1; i < 4;i++)for(int j = 1; j < 4;j ++) {
            mat3 h;
            h.setZero(3,3);
            if (i == j) {
                h = 2 * q[i] * q[i].transpose() + (q[i].dot(q[i]) -1) * Matrix3d::Identity(3, 3);
                for(int k = 1; k < 4; k++)if(k!= i) h += q[k] * q[k].transpose();
            }
            else {
                h = Matrix3d::Identity(3, 3) * q[j].dot(q[i]) + q[j] * q[i].transpose();
            }
            H.block<3, 3>(3 * i, 3 * j) = h * 4 * kappa;
        }
        return H;
    }

    double otho_energy(const VectorXd& x)
    {
        double E = 0;
        vec3 q[4];
        for (int i = 1; i < 4; i++) {
            q[i] = x.segment<3>(i * 3);
        }
        for (int i = 1; i < 4; i++)
            for (int j = 1; j < 4; j++) {
                double e = pow(q[i].dot(q[j]) - 1.0 * (i == j), 2);
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


    /*double otho_energy(const mat3 &q) {
        mat3 qqtmi = q.adjoint() * q - Matrix3d::Identity(3,3);
        return qqtmi.squaredNorm() * kappa;
    }*/
    #endif
};
