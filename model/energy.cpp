#include "geometry.h"
#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "ipc.h"
#include "settings.h"
#include <assert.h>
#include <array>
#include "timer.h"
#include <ipc/distance/edge_edge_mollifier.hpp>
#include "ipc_extension.h"

using namespace std;
using namespace barrier;
using namespace Eigen;
using namespace utils;


scalar E_barrier_plus_inert(
    const Vector<scalar, -1>& q_plus_dq, const Vector<scalar, -1>& dq, int n_cubes, 
    const vector<i4>& idx,
    const vector<i4>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt)
{
    scalar e = 0.0;
    int n_pt = idx.size(), n_ee = eidx.size(), n_g = vidx.size();
// inertia energy
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : e)
    for (int i = 0; i < n_cubes; i++) {
        auto& c(*cubes[i]);
        c.dq = dq.segment<12>(i * 12);

        auto q_tiled = c.q_tile(dt, globals.gravity);
        auto _q = q_plus_dq.segment<12>(12 * i);
        scalar e_inert = E(_q, q_tiled, c, dt);
        e += e_inert;
        // used for pt_vstack. vt1 and vt2 should not use buffer
        c.project_vt2();
    }
    if (globals.pt)
// point-triangle energy
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) 
        for (int k = 0; k < n_pt; k++) {
            auto& ij = idx[k];
            Face f(cubes[ij[2]]->face(ij[3], true, true));
            vec3 v(cubes[ij[0]]->v_transformed[ij[1]]);
            auto [d, pt_type] = vf_distance(v, f);
            e += barrier::barrier_function(d);
            #ifdef _PLUG_IN_LAN_
            // auto &ptf {};
            #endif
        }

    // ee ipc energy
    if (globals.ee)
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) 
        for (int k = 0; k < n_ee; k++) {
            auto& ij(eidx[k]);
            Edge ei(cubes[ij[0]]->edge(ij[1], true, true)), ej(cubes[ij[2]]->edge(ij[3], true, true));
            scalar eps_x = (ei.e0 - ei.e1).squaredNorm() * (ej.e0 - ej.e1).squaredNorm() * globals.eps_x;
            scalar p = ipc::edge_edge_mollifier(ei.e0, ei.e1, ej.e0, ej.e1, eps_x);
            scalar d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
#ifdef NO_MOLLIFIER
            e += barrier_function(d);
#else
            e += barrier_function(d) * p;
#endif
        }

    // vertex-ground ipc energy
    if (globals.ground)
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e)
        for (int k = 0; k < n_g; k++) {
            auto& v{vidx[k]};
            vec3 vt2 = cubes[v[0]]->v_transformed[v[1]];
            scalar e_ground = E_ground(vt2);
            e += e_ground;
        }
    return e;
}

scalar E_global(const Vector<scalar, -1>& q_plus_dq, const Vector<scalar, -1>& dq, int n_cubes, int n_pt, int n_ee, int n_g,
    const vector<i4>& idx,
    const vector<i4>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<scalar, 2, 12>>& pt_tk,
    const vector<Matrix<scalar, 2, 12>>& ee_tk,
    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt, scalar& _ef, bool _vt2)
{
    scalar e = 0.0;
    scalar ef = 0.0;
// inertia energy
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : e)
    for (int i = 0; i < n_cubes; i++) {
        auto& c(*cubes[i]);
        c.dq = dq.segment<12>(i * 12);

        auto q_tiled = c.q_tile(dt, globals.gravity);
        auto _q = q_plus_dq.segment<12>(12 * i);
        scalar e_inert = E(_q, q_tiled, c, dt);
        // if (_vt2)
        //     c.project_vt2();
        // else
        // c.project_vt1();
        c.project_vt2();
        // used for pt_vstack. vt1 and vt2 should not use buffer
        e += e_inert;
    }
    if (globals.pt)
// point-triangle energy
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) reduction(+ \
                                                                   : ef)
        for (int k = 0; k < n_pt; k++) {
            auto& ij = idx[k];
            Face f(cubes[ij[2]]->face(ij[3], _vt2, _vt2));
            vec3 v(cubes[ij[0]]->v_transformed[ij[1]]);
            if (!_vt2) v = cubes[ij[0]]->vt1(ij[1]);
            // q4 a = {v, f.t0, f.t1, f.t2};
            auto [d, pt_type] = vf_distance(v, f);
            // scalar d = ipc::point_triangle_distance(v, f.t0, f.t1, f.t2);
            e += barrier::barrier_function(d);
#ifdef _FRICTION_
            auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
            auto v_stack = pt_vstack(*cubes[ij[0]], *cubes[ij[2]], ij[1], ij[3]);
            auto uk = _vt2 ? 0.0 : (pt_tk[k] * v_stack).norm();
            if (globals.pt_fric)
                ef += D_f0(uk, contact_force);
#endif
        }

    // ee ipc energy
    if (globals.ee)
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) reduction(+ \
                                                                   : ef)
        for (int k = 0; k < n_ee; k++) {
            auto& ij(eidx[k]);
            Edge ei(cubes[ij[0]]->edge(ij[1], _vt2, _vt2)), ej(cubes[ij[2]]->edge(ij[3], _vt2, _vt2));
            scalar eps_x = (ei.e0 - ei.e1).squaredNorm() * (ej.e0 - ej.e1).squaredNorm() * globals.eps_x;
            scalar p = ipc::edge_edge_mollifier(ei.e0, ei.e1, ej.e0, ej.e1, eps_x);
            scalar d = ipc::edge_edge_distance(ei.e0, ei.e1, ej.e0, ej.e1);
#ifdef NO_MOLLIFIER
            e += barrier_function(d);
#else
            e += barrier_function(d) * p;
#endif
#ifdef _FRICTION_
            auto contact_force = -barrier_derivative_d(d) / (dt * dt) * 2 * sqrt(d);
            auto v_stack = ee_vstack(*cubes[ij[0]], *cubes[ij[2]], ij[1], ij[3]);
            auto uk = _vt2 ? 0.0 : (ee_tk[k] * v_stack).norm();
            if (globals.ee_fric)
                ef += D_f0(uk, contact_force);
#endif
        }

    // vertex-ground ipc energy
    if (globals.ground)
#pragma omp parallel for schedule(static) reduction(+                \
                                                    : e) reduction(+ \
                                                                   : ef)
        for (int k = 0; k < n_g; k++) {
            auto& v{
                vidx[k]
            };
            vec3 vt2 = cubes[v[0]]->v_transformed[v[1]];
            vec3 vd = _vt2 ? vt2 : cubes[v[0]]->vt1(v[1]);
            scalar e_ground = E_ground(vd);
            e += e_ground;
#ifdef _FRICTION_
            auto& c{ *cubes[v[0]] };
            auto vt0 = c.vt0(v[1]);
            scalar d = vg_distance(vd);
            if (d * d < barrier::d_hat) {
                auto contact_force = -barrier_derivative_d(d * d) / (dt * dt) * 2 * d;
                vec3 _uk = vt2 - vt0;
                scalar uk = sqrt(_uk(0) * _uk(0) + _uk(2) * _uk(2));
                if (globals.vg_fric)
                    ef += D_f0(uk, contact_force);
            }
#endif
        }
    _ef = ef;
    return e;
}

scalar E_fric(
    const Vector<scalar, -1>& dq, int n_cubes, 
    int n_pt, int n_ee, int n_g,
    const vector<i4>& idx,
    const vector<i4>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<scalar, 2, 12>>& pt_tk,
    const vector<Matrix<scalar, 2, 12>>& ee_tk,
    const vector<scalar>& pt_contact_forces,
    const vector<scalar>& ee_contact_forces,
    const vector<scalar>& g_contact_forces,

    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt
)
{
    scalar ef = 0.0;


    // prepare v_stack
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_cubes; i++) {
        auto& c(*cubes[i]);
        c.dq = dq.segment<12>(i * 12);
        c.project_vt2();
    }

    if (globals.pt && globals.pt_fric) {
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : ef)
        for (int k = 0; k < n_pt; k++) {
            auto& ij{ idx[k] };
            auto &ci{ *cubes[ij[0]] }, &cj{ *cubes[ij[2]] };
            auto v_stack = pt_vstack(ci, cj, ij[1], ij[3]);
            scalar uk = (pt_tk[k] * v_stack).norm();
            ef += D_f0(uk, pt_contact_forces[k]);
        }
    }

    if (globals.ee && globals.ee_fric) {
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : ef)
        for (int k = 0; k < n_ee; k++) {
            auto& ij{ eidx[k] };
            auto &ci{ *cubes[ij[0]] }, &cj{ *cubes[ij[2]] };
            auto v_stack = ee_vstack(ci, cj, ij[1], ij[3]);
            scalar uk = (ee_tk[k] * v_stack).norm();
            ef += D_f0(uk, ee_contact_forces[k]);
        }
    }

    if (globals.ground && globals.vg_fric) {
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : ef)
        for (int k = 0; k < n_g; k++) {
            auto& ij{ vidx[k] };
            auto& c{ *cubes[ij[0]] };
            vec3 _uk = c.v_transformed[ij[1]] - c.vt0(ij[1]);
            scalar uk = sqrt(_uk(0) * _uk(0) + _uk(2) * _uk(2));
            ef += D_f0(uk, g_contact_forces[k]);
        }
    }
    return ef;
}


scalar E_ground(const vec3& v)
{
    scalar e = 0.0;
    scalar d = vg_distance(v);
    d = d * d;
    if (d < barrier::d_hat) {
        e = barrier::barrier_function(d);
    }
    return e;
};