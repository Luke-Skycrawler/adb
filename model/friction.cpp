#include <affine_body.h>
#include "psd.h"
#include <ipc/friction/smooth_friction_mollifier.hpp>

#ifndef TESTING
#include "settings.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif
using namespace std;
using namespace Eigen;


void friction(
    const Vector<scalar, 2>& _uk, scalar contact_lambda, const Matrix<scalar, 12, 2>& Tk,
    Vector<scalar, 12>& g, Matrix<scalar, 12, 12>& H)
{
    static scalar mu = globals.mu;
    static const scalar evh = globals.dt * globals.evh, h2 = globals.dt * globals.dt;
    scalar uk = sqrt(_uk(0) * _uk(0) + _uk(1) * _uk(1));
    //if (uk < 1e-10) return;

    auto f1 = ipc::f1_SF_over_x(uk, evh);
    
    Vector<scalar, 12> F_k = mu * contact_lambda * f1 * Tk  * _uk;
    Matrix<scalar, 12, 12> D_k_hessian;

    if (uk >= evh) {
        Vector<scalar, 2> ut{ -_uk(1), _uk(0) };
        D_k_hessian = mu * contact_lambda * f1 / (uk * uk) * Tk * ut * (ut.transpose() * Tk.transpose());
    }
    else if (uk <= globals.params_double["max_uk"]) {
        D_k_hessian = mu * contact_lambda * f1 * Tk * Tk.transpose();
    }
    else {
        scalar f2_term = -1.0 / (evh * evh);
        Matrix<scalar, 2, 2> M2x2 = f2_term / uk * _uk * _uk.transpose();
        // Matrix<scalar, 2, 2> M2x2 = (df1_term * _uk * _uk.transpose());
        M2x2 += f1 * Matrix<scalar, 2, 2>::Identity(2, 2);
        M2x2 *= mu * contact_lambda;
        M2x2 = project_to_psd(M2x2);
        D_k_hessian = Tk * M2x2 * Tk.transpose();
    }
    g += F_k;
    H += D_k_hessian;
}

scalar D_f0(scalar uk, scalar lam)
{
    static scalar mu = globals.mu, evh = globals.dt * globals.evh, h2 = globals.dt * globals.dt;
    scalar D_k = mu * lam * ipc::f0_SF(uk, evh);
    return D_k;
}
