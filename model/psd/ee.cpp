#include "scalar_types.h"
#include "hl.h"

mat3 C_ee(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;
    vec3 e2 = x2 - x0;

    scalar alpha = e0.dot(e1) / e0.dot(e0);
    vec3 e1perp = e1 - alpha * e0;
    
    mat22 A;
    A << e0.dot(e0), e0.dot(e1),
         e0.dot(e1), e1.dot(e1);
    vec2 b(e0.dot(e2), e1.dot(e2));

    vec2 beta_gamma = A.inverse() * b;
    vec3 e2perp = e2 - beta_gamma[0] * e0 - beta_gamma[1] * e1;
    mat3 ret;
    ret << e0, e1perp, e2perp;
    return ret;
}

vec12 dadx(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;

    scalar al = e1.dot(e0) / e0.dot(e0);
    scalar e0n2 = e0.dot(e0);

    vec3 dadx0 = 2.0 * al * e0 - e1;
    vec3 dadx1 = -dadx0;
    vec3 dadx2 = -e0;
    vec3 dadx3 = -dadx2;

    vec12 ret;
    ret << dadx0 / e0n2, dadx1 / e0n2, dadx2 / e0n2, dadx3 / e0n2;
    return ret;
}

vec12 dbdx(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;
    vec3 e2 = x2 - x0;

    auto bg = beta_gamma_ee(x0, x1, x2, x3);
    scalar bet = bg[0];

    scalar e1n2 = e1.dot(e1);
    scalar e0n2 = e0.dot(e0);
    scalar e0d1 = e0.dot(e1);
    scalar e0d2 = e0.dot(e2);
    scalar e1d2 = e1.dot(e2);
    scalar e0x12 = e0.cross(e1).squaredNorm();

    vec3 dbdx0 = scalar(2.0) * bet * (e1n2 * e0 - e0d1 * e1) + e1 * (e0d1 + e1d2) - e1n2 * (e2 + e0);
    vec3 dbdx1 = scalar(2.0) * bet * (e0d1 * e1 - e1n2 * e0) - e1 * e1d2 + e1n2 * e2;
    vec3 dbdx2 = scalar(2.0) * bet * (e0n2 * e1 - e0d1 * e0) + e0 * (e1n2 + e1d2) - scalar(2.0) * e0d2 * e1 - e0d1 * (e1 - e2);
    vec3 dbdx3 = scalar(2.0) * bet * (e0d1 * e0 - e0n2 * e1) - e0 * e1d2 + scalar(2.0) * e1 * e0d2 - e0d1 * e2;
    vec12 ret; 
    ret << dbdx0 / e0x12, dbdx1 / e0x12, dbdx2 / e0x12, dbdx3 / e0x12;
    return ret; 
}

vec12 dgdx(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;
    vec3 e2 = x2 - x0;

    scalar gam = e0.dot(e1) / (e0.dot(e0) * e1.dot(e1) - e0.dot(e1) * e0.dot(e1));

    scalar e1n2 = e1.dot(e1);
    scalar e0n2 = e0.dot(e0);
    scalar e0d1 = e0.dot(e1);
    scalar e0d2 = e0.dot(e2);
    scalar e1d2 = e1.dot(e2);
    scalar e0x12 = e0.cross(e1).squaredNorm();

    vec3 dgdv0 = scalar(2.0) * gam * (e1n2 * e0 - e0d1 * e1) - scalar(2.0) * e1d2 * e0 - e0n2 * e1 + e0d2 * e1 + e0d1 * (e2 + e0);
    vec3 dgdv1 = scalar(2.0) * gam * (e0d1 * e1 - e1n2 * e0) + scalar(2.0) * e1d2 * e0 - e0d2 * e1 - e0d1 * e2;
    vec3 dgdv2 = scalar(2.0) * gam * (e0n2 * e1 - e0d1 * e0) - e0d1 * e0 + e0d2 * e0 + e0n2 * (e1 - e2);
    vec3 dgdv3 = scalar(2.0) * gam * (e0d1 * e0 - e0n2 * e1) - e0d2 * e0 + e0n2 * e2;

    vec12 ret; 
    ret << dgdv0 / e0x12, dgdv1 / e0x12, dgdv2 / e0x12, dgdv3 / e0x12;
    return ret;
}

mat9x12 dcdx_delta_ee(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;

    mat3 z3 = mat3::Zero();
    vec3 z3v = vec3::Zero();
    
    auto da = dadx(x0, x1, x2, x3);
    auto db = dbdx(x0, x1, x2, x3);
    auto dg = dgdx(x0, x1, x2, x3);
    // vec3 dadx0 = da.col(0), dadx1 = da.col(1), dadx2.col(2), dadx3 = da.col(3);
    // vec3 dbdx0 = db.col(0), dbdx1 = db.col(1), dbdx2.col(2), dbdx3 = db.col(3);
    // vec3 dgdx0 = dg.col(0), dgdx1 = dg.col(1), dgdx2.col(2), dgdx3 = dg.col(3);
    
    mat9x12 ret;
    ret << z3, z3, z3, z3,
        -e0 * da.transpose(),
        -e0 * db.transpose() -e1 * dg.transpose();

    // ret[idx(0, 0)] = z3;
    // ret[idx(0, 1)] = z3;
    // ret[idx(0, 2)] = z3;
    // ret[idx(0, 3)] = z3;

    // ret[idx(1, 0)] = -e0 * dadx0.transpose();
    // ret[idx(1, 1)] = -e0 * dadx1.transpose();
    // ret[idx(1, 2)] = -e0 * dadx2.transpose();
    // ret[idx(1, 3)] = -e0 * dadx3.transpose();

    // ret[idx(2, 0)] = -e0 * dbdx0.transpose() - e1 * dgdx0.transpose();
    // ret[idx(2, 1)] = -e0 * dbdx1.transpose() - e1 * dgdx1.transpose();
    // ret[idx(2, 2)] = -e0 * dbdx2.transpose() - e1 * dgdx2.transpose();
    // ret[idx(2, 3)] = -e0 * dbdx3.transpose() - e1 * dgdx3.transpose();

    return ret;
}

vec2 beta_gamma_ee(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;
    vec3 e2 = x2 - x0;

    mat22 A;
    A << e0.dot(e0), e0.dot(e1),
         e0.dot(e1), e1.dot(e1);
    vec2 b(e0.dot(e2), e1.dot(e2));

    vec2 beta_gamma = A.inverse() * b;
    return beta_gamma;
}

q4 dceedx_s(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x0;
    vec3 e1 = x3 - x2;
    vec3 e2 = x2 - x0;

    scalar al = e0.dot(e1) / e0.dot(e0);
    auto bet_gam = beta_gamma_ee(x0, x1, x2, x3);
    scalar bet = bet_gam(0), gam = bet_gam(1);
    q4 ret;
    ret << -1.0,  1.0,  0.0,  0.0,
              al,  -al,  -1.0,  1.0,
              bet - 1.0, -bet,  gam + 1.0, -gam;

    return ret;
}
