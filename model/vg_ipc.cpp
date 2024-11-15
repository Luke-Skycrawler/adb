#include "affine_body.h"
#include "ipc.h"
#include "collision.h"
#include "barrier.h"
#ifndef TESTING
#include "settings.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif
using namespace Eigen;
void IPC::ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Vector<scalar, 2>& _uk, scalar contact_lambda, const Matrix<scalar, 3, 2>& Tk
#endif

)
{
    if (c.mass < 0.0) return;
    auto v_tile{ c.vertices(v) }, p{ c.vt1(v) };
    c.grad += barrier::barrier_gradient_q(v_tile, p);
    c.hess += barrier::barrier_hessian_q(v_tile, p);
#ifdef _FRICTION_
    auto J = barrier::x_jacobian_q(v_tile);
    if (globals.vg_fric)
        friction(_uk, contact_lambda,
            J.transpose() * Tk, c.grad, c.hess);
#endif
};
