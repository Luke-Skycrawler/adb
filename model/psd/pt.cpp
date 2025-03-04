#include "scalar_types.h"
#include "hl.h"
using namespace std; 
using namespace Eigen;


vec2 beta_gamma_pt(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;
    vec3 e2 = x0 - x2;

    scalar alpha = e0.dot(e1) / e0.squaredNorm();

    scalar e0Te0 = e0.squaredNorm();
    scalar e0Te1 = e0.dot(e1);
    scalar e1Te1 = e1.squaredNorm();

    mat22 A;
    A << e0Te0, e0Te1, 
         e0Te1, e1Te1;

    vec2 b(e0.dot(e2), e1.dot(e2));

    vec2 beta_gamma = A.inverse() * b;
    return beta_gamma;
}

mat3 C_vf(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;
    vec3 e2 = x0 - x2;

    scalar alpha = e0.dot(e1) / e0.squaredNorm();

    scalar e0Te0 = e0.squaredNorm();
    scalar e0Te1 = e0.dot(e1);
    scalar e1Te1 = e1.squaredNorm();

    mat22 A;
    A << e0Te0, e0Te1, 
         e0Te1, e1Te1;

    vec2 b(e0.dot(e2), e1.dot(e2));

    vec2 beta_gamma = A.inverse() * b;

    vec3 e2perp = e2 - beta_gamma(0) * e0 - beta_gamma(1) * e1;

    mat3 ret;
    ret << e0, e1 - alpha * e0, e2perp;
    return ret;
}

q4 dcvfdx_s(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;
    vec3 e2 = x0 - x2;

    scalar alpha = e0.dot(e1) / e0.squaredNorm();

    scalar e0Te0 = e0.squaredNorm();
    scalar e0Te1 = e0.dot(e1);
    scalar e1Te1 = e1.squaredNorm();

    mat22 A;
    A << e0Te0, e0Te1, 
         e0Te1, e1Te1;

    vec2 b(e0.dot(e2), e1.dot(e2));

    vec2 beta_gamma = A.inverse() * b;
    scalar beta = beta_gamma(0);
    scalar gamma = beta_gamma(1);

    q4 M;
    M << 0.0, 1.0, -1.0, 0.0,
        0.0, -alpha, alpha - 1.0, 1.0,
        1.0, -beta, beta + gamma - 1.0, -gamma;

    return M;
}

vec12 dalpha_dx(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;

    vec3 e0h = e0.normalized();
    scalar term = scalar(1.0) / e0.squaredNorm();
    vec12 ret; 
    ret <<
        vec3::Zero(),
        term * (e1 - scalar(2.0) * e0h * e0h.dot(e1)),
        term * (scalar(2.0) * e0h * e0h.dot(e1) - e1 - e0),
        term * e0
    ;
    return ret;
}

vec12 dbeta_dx(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;
    vec3 e2 = x0 - x2;

    scalar e0Te0 = e0.dot(e0);
    scalar e0Te1 = e0.dot(e1);
    scalar e1Te1 = e1.dot(e1);
    mat22 A;
    A << e0Te0, e0Te1, e0Te1, e1Te1;
    vec2 b(e0.dot(e2), e1.dot(e2));

    vec2 beta_gamma = A.inverse() * b;
    scalar bet = beta_gamma[0];

    scalar e1n2 = e1.squaredNorm();
    scalar e0n2 = e0.squaredNorm();
    scalar e0d1 = e0.dot(e1);
    scalar e0d2 = e0.dot(e2);
    scalar e1d2 = e1.dot(e2);
    scalar e0x12 = e0.cross(e1).squaredNorm();

    vec3 dbdv0 = (e1n2 * e0 - e0d1 * e1);
    vec3 dbdv1 = (scalar(2.0) * bet * (e0d1 * e1 - e1n2 * e0) - e1d2 * e1 + e1n2 * e2);
    vec3 dbdv2 = (scalar(2.0) * bet * (e1n2 * e0 + e0n2 * e1 - e0d1 * (e0 + e1)) + e1d2 * (e0 + e1) - scalar(2.0) * e0d2 * e1 - e1n2 * (e0 + e2) + e0d1 * (e1 + e2));
    vec3 dbdv3 = scalar(2.0) * bet * (e0d1 * e0 - e0n2 * e1) - e1d2 * e0 + scalar(2.0) * e0d2 * e1 - e0d1 * e2;

    vec12 ret; 
    ret << dbdv0 / e0x12, dbdv1 / e0x12, dbdv2 / e0x12, dbdv3 / e0x12;
    return ret; 
}

vec12 dgamma_dx(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;
    vec3 e2 = x0 - x2;

    scalar e0Te0 = e0.dot(e0);
    scalar e0Te1 = e0.dot(e1);
    scalar e1Te1 = e1.dot(e1);
    Eigen::Matrix2d A;
    A << e0Te0, e0Te1, e0Te1, e1Te1;
    Eigen::Vector2d b(e0.dot(e2), e1.dot(e2));

    Eigen::Vector2d beta_gamma = A.inverse() * b;
    scalar gam = beta_gamma[1];

    scalar e1n2 = e1.squaredNorm();
    scalar e0n2 = e0.squaredNorm();
    scalar e0d1 = e0.dot(e1);
    scalar e0d2 = e0.dot(e2);
    scalar e1d2 = e1.dot(e2);
    scalar e0x12 = e0.cross(e1).squaredNorm();

    vec3 dgdv0 = -e0d1 * e0 + e0n2 * e1;
    vec3 dgdv1 = scalar(2.0) * gam * (e0d1 * e1 - e1n2 * e0) + scalar(2.0) * e1d2 * e0 - e0d2 * e1 - e0d1 * e2;
    vec3 dgdv2 = scalar(2.0) * gam * (e1n2 * e0 - e0d1 * (e0 + e1) + e0n2 * e1) - scalar(2.0) * e1d2 * e0 + e0d2 * (e0 + e1) + e0d1 * (e0 + e2) - e0n2 * (e1 + e2);
    vec3 dgdv3 = scalar(2.0) * gam * (e0d1 * e0 - e0n2 * e1) - e0d2 * e0 + e0n2 * e2;

    vec12 ret;
    ret << dgdv0 / e0x12, dgdv1 / e0x12, dgdv2 / e0x12, dgdv3 / e0x12;
    return ret;
}

mat9x12 dcdx_delta_vf(const vec3& x0, const vec3& x1, const vec3& x2, const vec3& x3) {
    vec3 e0 = x1 - x2;
    vec3 e1 = x3 - x2;
    
    auto da = dalpha_dx(x0, x1, x2, x3);
    auto db = dbeta_dx(x0, x1, x2, x3);
    auto dg = dgamma_dx(x0, x1, x2, x3);

    // vec3 dadx1 = da.col(0), dadx2 = da.col(1), dadx3 = da.col(2);
    // vec3 dbdx0 = db.col(0), dbdx1 = db.col(1), dbdx2 = db.col(2), dbdx3 = db.col(3);
    // vec3 dgdx0 = dg.col(0), dgdx1 = dg.col(1), dgdx2 = dg.col(2), dgdx3 = dg.col(3);

    
    mat3 z3 = mat3::Zero();

    mat9x12 ret;
    ret << z3, z3, z3, z3,
        -e0 * da.transpose(),
        -e0 * db.transpose() - e1 * dg.transpose();
    // ret[idx(0, 0)] = z3;
    // ret[idx(0, 1)] = z3;
    // ret[idx(0, 2)] = z3;
    // ret[idx(0, 3)] = z3;

    // ret[idx(1, 0)] = z3;
    // ret[idx(1, 1)] = -e0 * dadx1.transpose();
    // ret[idx(1, 2)] = -e0 * dadx2.transpose();
    // ret[idx(1, 3)] = -e0 * dadx3.transpose();

    // ret[idx(2, 0)] = -e0 * dbdx0.transpose() - e1 * dgdx0.transpose();
    // ret[idx(2, 1)] = -e0 * dbdx1.transpose() - e1 * dgdx1.transpose();
    // ret[idx(2, 2)] = -e0 * dbdx2.transpose() - e1 * dgdx2.transpose();
    // ret[idx(2, 3)] = -e0 * dbdx3.transpose() - e1 * dgdx3.transpose();
    return ret;
}
