#include "hl.h"
#include <tuple>
using namespace std;
using namespace Eigen;
scalar signed_distance(const vec3& e0, const vec3& e1, const vec3& e2) {
    e2.dot(Eigen::cross(e0, e1)) / (Eigen::cross(e0, e1).norm());
}

class PSDTest {

    std::tuple<mat12, vec12> test_vf(const vector<vec3> &x, ) {
        vector<mat3> dcdx_delta(12);
        vec3 e0p, e1p, e2p;
        auto C = C_vf(x[0], x[1], x[2], x[3]);
        e0p = C.col(0);
        e1p = C.col(1);
        e2p = C.col(2);
    
    
        q4 dcdx_s = dcvfdx_s(x[0], x[1], x[2], x[3]);
        l = signed_distance(e0p, e1p, e2p);
        dcdx_delta_vf(x[0], x[1], x[2], x[3], dcdx_delta.data());
        auto H = Hl(e0p, e1p, e2p, l);
        auto g = gl(l, e2p);
    
        H *= 2 * l;
        H += g * g.transpose() * 2;
        
    }
    std::tuple<mat12, vec12> test_ee(const vector<vec3> &x, ) {
        vector<mat3> dcdx_delta(12);
        vec3 e0p, e1p, e2p;
        auto C = C_ee(x[0], x[1], x[2], x[3]);
        e0p = C.col(0);
        e1p = C.col(1);
        e2p = C.col(2);
    
    
        q4 dcdx_s = dceedx_s(x[0], x[1], x[2], x[3]);
        l = signed_distance(e0p, e1p, e2p);
        dcdx_delta_ee(x[0], x[1], x[2], x[3], dcdx_delta.data());
        auto H = Hl(e0p, e1p, e2p, l);
        auto g = gl(l, e2p);
    
        H *= 2 * l;
        H += g * g.transpose() * 2;

        mat9x12 simple_term, del_term;
        for (int i = 0; i < 3; i ++) for (int j = 0; j < 4; j ++) {
            simple_term.block<3, 3>(3 * i, 3 * j) = dcdx_s[3 * i + j] * mat3::Identity();
        }

        
        mat12 A1 = del_term.transpose() * H * del_term;
        mat12 A0 = simple_term.transpose() * H * simple_term;

        mat12 hess = A0 - A1; 
        vec12 grad = simple_term.transpose() * g * 2 * l;

        return {hess, grad};

    }

};
    
