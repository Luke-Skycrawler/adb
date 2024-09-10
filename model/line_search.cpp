#include "time_integrator.h"
#include "settings.h"
#include <array>

using namespace Eigen;

scalar E_global(const Vector<scalar, -1>& q_plus_dq, const Vector<scalar, -1>& dq, int n_cubes, int n_pt, int n_ee, int n_g,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<scalar, 2, 12>>& pt_tk,
    const vector<Matrix<scalar, 2, 12>>& ee_tk,
    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt, scalar& _ef, bool _vt2);


scalar E_barrier_plus_inert(const Vector<scalar, -1>& q_plus_dq, const Vector<scalar, -1>& dq, int n_cubes,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt);

scalar E_fric(
    const Vector<scalar, -1>& dq, int n_cubes, 
    int n_pt, int n_ee, int n_g,
    const vector<array<int, 4>>& idx,
    const vector<array<int, 4>>& eidx,
    const vector<array<int, 2>>& vidx,
    const vector<Matrix<scalar, 2, 12>>& pt_tk,
    const vector<Matrix<scalar, 2, 12>>& ee_tk,
    const vector<scalar>& pt_contact_forces,
    const vector<scalar>& ee_contact_forces,
    const vector<scalar>& g_contact_forces,
    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt);

scalar line_search(const Vector<scalar, -1>& dq, const Vector<scalar, -1>& grad, Vector<scalar, -1>& q0, scalar& E0, scalar& E1,
    int n_cubes, int n_pt, int n_ee, int n_g,
    vector<array<vec3, 4>>& pts,
    vector<array<int, 4>>& idx,
    vector<array<vec3, 4>>& ees,
    vector<array<int, 4>>& eidx,
    vector<array<int, 2>>& vidx,
    const vector<Matrix<scalar, 2, 12>>& pt_tk,
    const vector<Matrix<scalar, 2, 12>>& ee_tk,
    const vector<scalar> &pt_contact_forces,
    const vector<scalar> &ee_contact_forces,
    const vector<scalar> &g_contact_forces,
    const vector<unique_ptr<AffineBody>>& cubes,
    scalar dt)
{
    static scalar tol
    {
        globals.params_double["tol"]
    };
    static const scalar c1 = globals.params_double["c1"];
    scalar alpha = 1.0;
    bool wolfe = false;
    scalar ef0 = 0.0;
    E0 = E_barrier_plus_inert(q0, 0.0 * dq, n_cubes, idx, eidx, vidx, cubes, dt) 
    + E_fric(0.0 * dq, n_cubes, n_pt, n_ee, n_g,
        idx, eidx, vidx,
        pt_tk, ee_tk, 
        pt_contact_forces, ee_contact_forces, g_contact_forces, 
        cubes, dt    
    );
    scalar qdg = dq.dot(grad);
    Vector<scalar, -1> q1;
    static vector<array<vec3, 4>> pts_new, pts_iaab;
    static vector<array<int, 4>> idx_new, idx_iaab;

    static vector<array<vec3, 4>> ees_new, ees_iaab;
    static vector<array<int, 4>> eidx_new, eidx_iaab;

    static vector<array<int, 2>> vidx_new, vidx_iaab;

    auto dq_norm = dq.norm();
    do {
        q1 = q0 + dq * alpha;
        auto dqk = dq * alpha;
        for (int i = 0; i < n_cubes; i++) {
            auto& c(*cubes[i]);
            c.dq = dqk.segment<12>(i * 12);
        }

        if (globals.iaabb % 2)
            iaabb_brute_force(n_cubes, cubes, globals.aabbs, 2,
#ifdef IAABB_COMPARING
                pts_iaab,
                idx_iaab,
                ees_iaab,
                eidx_iaab,
                vidx_iaab);
#else
                pts_new,
                idx_new,
                ees_new,
                eidx_new,
                vidx_new);
        else
#endif
        gen_collision_set(true, n_cubes, cubes,
            pts_new,
            idx_new,
            ees_new,
            eidx_new,
            vidx_new);
        scalar ef1 = 0.0, E2 = 0.0, ef2 = 0.0;
        scalar E3 = E_barrier_plus_inert(q1, dqk, n_cubes, idx_new, eidx_new, vidx_new, cubes, dt);
        scalar ef = E_fric(dqk, n_cubes, n_pt, n_ee, n_g, idx, eidx, vidx, pt_tk, ee_tk, pt_contact_forces, ee_contact_forces, g_contact_forces, cubes,  dt);
        E1 = E3 + ef;
        wolfe = E1 <= E0 + c1 * alpha * qdg;
        // spdlog::info("wanted descend = {}, E1 - E0 = {}, E1 = {}, E0 = {}, alpha = {}", c1 * alpha * qdg, E1 - E0, E1, E0, alpha);
        alpha /= 2;
        if (!(!wolfe && grad.norm() > 1e-3)) break;
        if (dq_norm * alpha * 2 < tol) {
            // smaller than Newton iter convergence condition, clip it
            if (globals.params_int["clip"]) {
                alpha = 0.0;
                break;
            }
            else {
                // continue to loop until & line search
            }
        }
    } while (true);
    pts = pts_new;
    idx = idx_new;
    ees = ees_new;
    eidx = eidx_new;
    vidx = vidx_new;

    return alpha * 2;
}
