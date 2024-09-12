#define _PLUG_IN_LAN_
#ifdef _PLUG_IN_LAN_
#include "affine_body.h"
#include "collision.h"
#include "../view/global_variables.h"
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/friction/closest_point.hpp>
#include <ipc/friction/tangent_basis.hpp>
#include "IpcFrictionConstraint.h"
#include <tuple>
using namespace Eigen;
using namespace std;

tuple<vec12, vec12, mat12, mat12, mat12> pt_ipc_friction_constraint(AffineBody &ci, AffineBody &cj, Face &f, vec3 &p, ipc::PointTriangleDistanceType pt_type){

    Vector4i cid_p{ 0, 0, 0, 0 };
    Vector4i cid_t0{ 4, 4, 4, 4 };
    Vector4i cid_t1{ 4, 4, 4, 4 };
    Vector4i cid_t2{ 4, 4, 4, 4 };

    auto pr{ ci.vertices(ij[1]) },
        t0r{ cj.vertices(cj.indices[ij[3] * 3]) },
        t1r{ cj.vertices(cj.indices[ij[3] * 3 + 1]) },
        t2r{ cj.vertices(cj.indices[ij[3] * 3 + 2]) };

    auto p0{ ci.vt0(ij[1]) },
        t00{ cj.vt0(cj.indices[ij[3] * 3]) },
        t10{ cj.vt0(cj.indices[ij[3] * 3 + 1]) },
        t20{ cj.vt0(cj.indices[ij[3] * 3 + 2]) };

    Vector<scalar, 4> w_p{ 1.0, pr[0], pr[1], pr[2] };
    Vector<scalar, 4> w_t0{ 1.0, t0r[0], t0r[1], t0r[2] };
    Vector<scalar, 4> w_t1{ 1.0, t1r[0], t1r[1], t1r[2] };
    Vector<scalar, 4> w_t2{ 1.0, t2r[0], t2r[1], t2r[2] };

    vector<vec3> surface_x{ p, f.t0, f.t1, f.t2 }, surface_xhat{ p0, t00, t10, t20 }, surface_X{ pr, t0r, t1r, t2r };

    vector<pair<Vector4i, Vector<scalar, 4>>> dpdx{ { cid_p, w_p }, { cid_t0, w_t0 }, { cid_t1, w_t1 }, { cid_t2, w_t2 } };
    vec12 ga, gb;
    ga.setZero(12);
    gb.setZero(12);
    mat12 ha, hb, hab;
    ha.setZero(12, 12);
    hb.setZero(12, 12);
    hab.setZero(12, 12);
    Vector<scalar, -1> gaf, gbf;
    gaf.setZero(12);
    gbf.setZero(12);
    Matrix<scalar, -1, -1> haf, hbf, habf;
    haf.setZero(12, 12);
    hbf.setZero(12, 12);
    habf.setZero(12, 12);
    Vector<scalar, -1> gac, gbc;
    gac.setZero(12);
    gbc.setZero(12);
    Matrix<scalar, -1, -1> hac, hbc, habc;
    hac.setZero(12, 12);
    hbc.setZero(12, 12);
    habc.setZero(12, 12);
    AIPC::IpcFrictionConstraintOp3D* friction_constraint;
    AIPC::IpcConstraintOp3D* constraint;
    if (pt_type == ipc::PointTriangleDistanceType::P_T0) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPPConstraint(0, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }

    else if (pt_type == ipc::PointTriangleDistanceType::P_T1) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPPConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }
    else if (pt_type == ipc::PointTriangleDistanceType::P_T2) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPPConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }
    else if (pt_type == ipc::PointTriangleDistanceType::P_E0) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPEConstraint(0, 0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }
    else if (pt_type == ipc::PointTriangleDistanceType::P_E1) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPEConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }
    else if (pt_type == ipc::PointTriangleDistanceType::P_E2) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 3, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPEConstraint(0, 0, 3, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }
    else {
        friction_constraint = new AIPC::IpcPTFConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        constraint = new AIPC::IpcPTConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
    }


    constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gac, gbc);
    constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hac, hbc, habc);

    friction_constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gaf, gbf);
    friction_constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, haf, hbf, habf);

    
    ga = gaf + gac;
    gb = gbf + gbc;
    ha = haf + hac;
    hb = hbf + hbc;
    hab = habf + habc;

    ga /= barrier::d_hat;
    gb /= barrier::d_hat;
    ha /= barrier::d_hat;
    hb /= barrier::d_hat;
    hab /= barrier::d_hat;

    return {ga, gb, ha, hb, hab};
}

void compare_lan(const vec12 &ga, const vec12& gb, const mat12 &ha, const mat12 &hb, const mat12 &hab, const vec12 & gradp, const vec12 &gradt, const mat12 &hess_p, const mat12 &hess_t, const mat12 &off_diag) {
    bool b0 = ::fd::compare_gradient(ga, gradp);
    bool b1 = ::fd::compare_gradient(gb, gradt);

    bool b2 = fd::compare_hessian(ha, hess_p);
    bool b3 = fd::compare_hessian(hb, hess_t);
    bool b4 = fd::compare_hessian(hab, off_diag);

    if (!b0) {
        spdlog::error("gradient p error");
    }
    if (!b1) {
        spdlog::error("gradient t error");
    }
    if (!b2) {
        spdlog::error("hessian p error");
    }
    if (!b3) {
        spdlog::error("hessian t error");
    }
    if (!b4) {
        spdlog::error("hessian off_diag error");
    }
}

void compare_lan_ee(vec12 &ga, vec12 &gb, mat12 &ha, mat12 &hb, mat12 &hab, vec12 &grad_0, vec12 &grad_1, mat12 &hess_0, mat12 &hess_1, mat12 &off_diag, ipc::EdgeEdgeDistanceType ee_type, scalar mollifier) {
    bool b0 = ::fd::compare_gradient(ga, grad_0);
    bool b1 = ::fd::compare_gradient(gb, grad_1);

    bool b2 = fd::compare_hessian(ha, hess_0);
    bool b3 = fd::compare_hessian(hb, hess_1);
    bool b4 = fd::compare_hessian(hab, off_diag);
    const auto to_int = [](const ipc::EdgeEdgeDistanceType& type) -> int{
        if (type == ipc::EdgeEdgeDistanceType::EA0_EB) {
            return 0;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA1_EB) {
            return 1;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA_EB0) {
            return 2;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA_EB1) {
            return 3;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA0_EB0) {
            return 4;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA0_EB1) {
            return 5;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA1_EB0) {
            return 6;
        }
        else if (type == ipc::EdgeEdgeDistanceType::EA1_EB1) {
            return 7;
        }
        else {
            return 8;
        }
    };
#ifndef LANS_DIRECT
    // b4 = b4 || mollifier != 1.0;
    // b3 = b3 || mollifier != 1.0;
    // b2 = b2 || mollifier != 1.0;
    // b1 = b1 || mollifier != 1.0;
    // b0 = b0 || mollifier != 1.0;
    

    if (!b0) {
        spdlog::error("ee gradient p error, {}, molli = {}", to_int(ee_type), mollifier);
    }
    if (!b1) {
        spdlog::error("ee gradient t error, {}, molli = {}", to_int(ee_type), mollifier);
    }
    if (!b2) {
        spdlog::error("ee hessian p error, {}, molli = {}", to_int(ee_type), mollifier);
    }
    if (!b3) {
        spdlog::error("ee hessian t error, {}, molli = {}", to_int(ee_type), mollifier);
    }
    if (!b4) {
        globals.params_int["ee error " + to_string(to_int(ee_type))] ++;
        spdlog::error("ee hessian off_diag error, {}, molli = {}", to_int(ee_type), mollifier);
    }
    
#endif

}
tuple<vec12, vec12, mat12, mat12, mat12> ee_ipc_friction_constraint(AffineBody &ci, AffineBody &cj, ipc::EdgeEdgeDistanceType ee_type, scalar eps_x, scalar mollifier) {
    
    Vector4i cid_ei0{ 0, 0, 0, 0 };
    Vector4i cid_ei1{ 0, 0, 0, 0 };
    Vector4i cid_ej0{ 4, 4, 4, 4 };
    Vector4i cid_ej1{ 4, 4, 4, 4 };

    auto ei0r{ ci.vertices(ci.edges[ij[1] * 2]) },
        ei1r{ ci.vertices(ci.edges[ij[1] * 2 + 1]) },
        ej0r{ cj.vertices(cj.edges[ij[3] * 2]) },
        ej1r{ cj.vertices(cj.edges[ij[3] * 2 + 1]) };

    auto ei00{ ci.vt0(ci.edges[ij[1] * 2]) },
        ei10{ ci.vt0(ci.edges[ij[1] * 2 + 1]) },
        ej00{ cj.vt0(cj.edges[ij[3] * 2]) },
        ej10{ cj.vt0(cj.edges[ij[3] * 2 + 1]) };

    Vector<scalar, 4> w_ei0{ 1.0, ei0r[0], ei0r[1], ei0r[2] };
    Vector<scalar, 4> w_ei1{ 1.0, ei1r[0], ei1r[1], ei1r[2] };
    Vector<scalar, 4> w_ej0{ 1.0, ej0r[0], ej0r[1], ej0r[2] };
    Vector<scalar, 4> w_ej1{ 1.0, ej1r[0], ej1r[1], ej1r[2] };

    vector<vec3> surface_x{ ee[0], ee[1], ee[2], ee[3] }, surface_xhat{ ei00, ei10, ej00, ej10 }, surface_X{ ei0r, ei1r, ej0r, ej1r };
    vector<vec3> sx = surface_x, sX = surface_X;
    vector<pair<Vector4i, Vector<scalar, 4>>> dpdx{ { cid_ei0, w_ei0 }, { cid_ei1, w_ei1 }, { cid_ej0, w_ej0 }, { cid_ej1, w_ej1 } }, px = dpdx;
    vec12 ga, gb;
    ga.setZero(12);
    gb.setZero(12);
    mat12 ha, hb, hab;
    ha.setZero(12, 12);
    hb.setZero(12, 12);
    hab.setZero(12, 12);
    Vector<scalar, -1> gaf, gbf;
    gaf.setZero(12);
    gbf.setZero(12);
    Matrix<scalar, -1, -1> haf, hbf, habf;
    haf.setZero(12, 12);
    hbf.setZero(12, 12);
    habf.setZero(12, 12);
    Vector<scalar, -1> gac, gbc;
    gac.setZero(12);
    gbc.setZero(12);
    Matrix<scalar, -1, -1> hac, hbc, habc;
    hac.setZero(12, 12);
    hbc.setZero(12, 12);
    habc.setZero(12, 12);
    AIPC::IpcFrictionConstraintOp3D* friction_constraint;
    AIPC::IpcConstraintOp3D* constraint;
    if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB0) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPPConstraint(0, 0, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
        constraint = new AIPC::IpcPPMConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB1) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPPConstraint(0, 0, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

        constraint = new AIPC::IpcPPMConstraint(0, 0, 1, 3, 2, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
        // perm = { 0, 1, 3, 2 };
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB0) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPPConstraint(0, 1, 2, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

        constraint = new AIPC::IpcPPMConstraint(0, 1, 0, 2, 3, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
        // perm = { 1, 0, 2, 3 };
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB1) {
        friction_constraint = new AIPC::IpcPPFConstraint(0, 1, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPPConstraint(0, 1, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

        constraint = new AIPC::IpcPPMConstraint(0, 1, 0, 3, 2, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
        // perm = { 1, 0, 3, 2 };
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB0) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 2, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPEConstraint(0, 2, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);

        constraint = new AIPC::IpcPEMConstraint(0, 2, 3, 0, 1, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
        // perm = { 2, 3, 0, 1 };
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB1) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 3, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPEConstraint(0, 3, 0, 1, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
        
        constraint = new AIPC::IpcPEMConstraint(0, 3, 2, 0, 1, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
        // perm = { 3, 2, 0, 1 };
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA0_EB) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPEConstraint(0, 0, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
        constraint = new AIPC::IpcPEMConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
    }
    else if (ee_type == ipc::EdgeEdgeDistanceType::EA1_EB) {
        friction_constraint = new AIPC::IpcPEFConstraint(0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcPEConstraint(0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
        
        constraint = new AIPC::IpcPEMConstraint(0, 1, 0, 2, 3, px, sx, sX, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);
        // perm = { 1, 0, 2, 3 };
    }
    else {
        friction_constraint = new AIPC::IpcEEFConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.mu, globals.dt, globals.evh, 1.0, 1.0);
        // constraint = new AIPC::IpcEEConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0);
        constraint = new AIPC::IpcEEMConstraint(0, 0, 1, 2, 3, dpdx, surface_x, surface_X, barrier::d_hat, globals.kappa, globals.dt, 1.0, mollifier, eps_x);

    }
    // if (false) {
    if (ee_type == ipc::EdgeEdgeDistanceType::EA_EB0 || ee_type == ipc::EdgeEdgeDistanceType::EA_EB1) {
        Matrix<scalar, -1, -1> _habf, _habc;
        _habf.setZero(12, 12);
        _habc.setZero(12, 12);
        friction_constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gbf, gaf);
        friction_constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hbf, haf, _habf);
        constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gbc, gac);
        constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hbc, hac, _habc);
        habf = _habf; // .transpose();
        habc = _habc; //.transpose();
    }
    else {
        friction_constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gaf, gbf);
        friction_constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, haf, hbf, habf);
        constraint->gradient({}, surface_x, surface_X, surface_xhat, {}, gac, gbc);
        constraint->hessian({}, surface_x, surface_X, surface_xhat, {}, hac, hbc, habc);
    }

    ga = gaf + gac;
    gb = gbf + gbc;
    ha = haf + hac;
    hb = hbf + hbc;
    hab = habf + habc;

    ga /= barrier::d_hat;
    gb /= barrier::d_hat;
    ha /= barrier::d_hat;
    hb /= barrier::d_hat;
    hab /= barrier::d_hat;
    return {ga, gb, ha, hb, hab};
}
#endif