#include "scalar_types.h"
#include "hl.h"
#include <iostream>
#include <tuple>
#include <vector>
using namespace std;
mat9 Hl(const vec3& e0o, const vec3& e1o, const vec3& e2o, scalar l) {
    mat3 z33 = mat3::Zero();
    mat3 i33 = mat3::Identity();

    mat3 l02 = -l * (e2o * e0o.transpose()) / (e2o.squaredNorm() * e0o.squaredNorm());
    mat3 l12 = -l * (e2o * e1o.transpose()) / (e2o.squaredNorm() * e1o.squaredNorm());

    vec3 e0o_unit = e0o.normalized();
    vec3 e1o_unit = e1o.normalized();
    mat3 block = i33 - (e0o_unit * e0o_unit.transpose()) - (e1o_unit * e1o_unit.transpose());
    
    mat3 l00 = -l * block / e0o.squaredNorm();
    mat3 l11 = -l * block / e1o.squaredNorm();

    mat9 ret; 
    ret.setZero();
    ret.block<3, 3>(0, 0) = l00;
    ret.block<3, 3>(3, 3) = l11;
    ret.block<3, 3>(0, 6) = l02;
    ret.block<3, 3>(3, 6) = l12;

    ret.block<3, 3>(6, 3) = l12.transpose();
    ret.block<3, 3>(6, 0) = l02.transpose();
    
    return ret;
}

vec9 gl(scalar l, const vec3& e2o) {
    vec3 z3 = vec3::Zero();
    vec9 ret; 
    ret << z3, z3, (e2o * l) / e2o.squaredNorm();
    return ret;
}

mat12 hessian_pt(const vec3& p, const vec3& t0, const vec3& t1, const vec3& t2, scalar l) {
    vec3 e0p, e1p, e2p;
    auto C = C_vf(p, t0, t1, t2);
    e0p = C.col(0);
    e1p = C.col(1);
    e2p = C.col(2);


    q4 dcdx_s = dcvfdx_s(p, t0, t1, t2);
    

    auto del_term = dcdx_delta_vf(p, t0, t1, t2);
    mat9x12 simple_term;

    for (int i = 0; i < 3; i ++) for (int j = 0; j < 4; j ++) {
        simple_term.block<3, 3>(3 * i, 3 * j) = dcdx_s(i, j) * mat3::Identity();
        //simple_term.block<3, 3>(3 * i, 3 * j) = mat3::Zero();
    }
    auto H = Hl(e0p, e1p, e2p, l);
    auto g = gl(l, e2p);

    H *= 2 * l;
    H += g * g.transpose() * 2;
    
    return simple_term.transpose() * H * simple_term -del_term.transpose() * H * del_term;
}


std::vector<std::tuple<vec9, scalar>> eigHl (const vec3& e0p, const vec3& e1p, const vec3& e2p, scalar l) {
    scalar e0pn = e0p.squaredNorm();
    scalar e1pn = e1p.squaredNorm();
    scalar e2pn = e2p.squaredNorm();

    scalar f12 = sqrt(1.0 + 4.0 * e1pn / e2pn);
    scalar f02 = sqrt(1.0 + 4.0 * e0pn / e2pn);

    scalar lam0 = -l / (2.0 * e1pn) * (1.0 + f12);
    scalar lam1 = -l / (2.0 * e1pn) * (1.0 - f12);
    scalar lam2 = -l / (2.0 * e0pn) * (1.0 + f02);
    scalar lam3 = -l / (2.0 * e0pn) * (1.0 - f02);

    scalar lam4 = 2.0; 

    vec3 z31 = vec3::Zero();
    scalar omega0, omega1, omega2, omega3;

    omega0 = lam0 / (lam0 - l / e2pn);
    omega1 = lam1 / (lam1 - l / e2pn);
    omega2 = lam2 / (lam2 - l / e2pn);
    omega3 = lam3 / (lam3 - l / e2pn);
    
    vec9 q0, q1, q2, q3, q4;
    q0 << z31, e2p, omega0 * e1p;
    q1 << z31, e2p, omega1 * e1p;
    q2 << e2p, z31, omega2 * e0p;
    q3 << e2p, z31, omega3 * e0p;
    q4 = gl(l, e2p);


    lam0 *= 2 * l;
    lam1 *= 2 * l;
    lam2 *= 2 * l;
    lam3 *= 2 * l;

    return {
        {q0, lam0},
        {q1, lam1},
        {q2, lam2},
        {q3, lam3},
        {q4, lam4}
    };
}

mat9 mat9_from_eigs(const std::vector<std::tuple<vec9, scalar>>& eigs, int keep) {
    mat9 ret = mat9::Zero();
    for (auto& [q, lam] : eigs) {
        if (keep > 0 && lam < 0) continue;
        if (keep < 0 && lam > 0) continue;
        scalar qTq = q.squaredNorm();
        ret += lam / qTq * q * q.transpose();
    }
    return ret;
}

mat12 hessian_pt_eig(const vec3& p, const vec3& t0, const vec3& t1, const vec3& t2, scalar l, int keep) {
    // keep: 0: all, 1: positive, -1: negative
    vec3 e0p, e1p, e2p;
    auto C = C_vf(p, t0, t1, t2);
    e0p = C.col(0);
    e1p = C.col(1);
    e2p = C.col(2);


    q4 dcdx_s = dcvfdx_s(p, t0, t1, t2);
    

    auto del_term = dcdx_delta_vf(p, t0, t1, t2);
    mat9x12 simple_term;

    for (int i = 0; i < 3; i ++) for (int j = 0; j < 4; j ++) {
        simple_term.block<3, 3>(3 * i, 3 * j) = dcdx_s(i, j) * mat3::Identity();
        //simple_term.block<3, 3>(3 * i, 3 * j) = mat3::Zero();
    }
    // auto H = Hl(e0p, e1p, e2p, l);
    
    auto eigs = eigHl(e0p, e1p, e2p, l);
    auto g = gl(l, e2p);
    
    auto H = mat9_from_eigs(eigs, keep);


    // H *= 2 * l;
    // H += g * g.transpose() * 2;
    if (keep == 0) {

        return simple_term.transpose() * H * simple_term -del_term.transpose() * H * del_term;
    }    
    else {
        mat12 term1 = simple_term.transpose() * H * simple_term;
        auto H_neg = mat9_from_eigs(eigs, -keep);
        return term1 - del_term.transpose() * H_neg * del_term; 
    }
}