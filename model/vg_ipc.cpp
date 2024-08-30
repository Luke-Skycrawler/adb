#include "affine_body.h"
#include "collision.h"
#include "barrier.h"
#ifndef TESTING
#include "../view/global_variables.h"
#else 
#include "../iAABB/pch.h"
extern Globals globals;
#endif

void ipc_term_vg(AffineBody& c, int v
#ifdef _FRICTION_
    ,
    const Vector2d& _uk, double contact_lambda, const Matrix<double, 3, 2>& Tk
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
