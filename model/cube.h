#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <array>
using namespace Eigen;
using vec3 = Vector3d;
using mat3 = Matrix3d;
using q4 = std::array<vec3, 4>;
struct Cube {
    mat3 A;
    vec3 p, dimensions, f, tau;
    double mass, scale, Ic, toi, alpha;
    static int indices[36], edges[24];
    static const int n_vertices = 8, n_faces = 12, n_edges = 12;
    Vector<double, 12> barrier_gradient, dq, grad;
    Matrix<double, 12, 12> hess;
    q4 q, q0, dqdt;
    void prepare_q_array();
    static const vec3* vertices()
    {
        static const vec3 v[] = {
            vec3(-0.5f, -0.5f, -0.5f),
            vec3(-0.5f, -0.5f, 0.5f),
            vec3(-0.5f, 0.5f, -0.5f),
            vec3(-0.5f, 0.5f, 0.5f),
            vec3(0.5f, -0.5f, -0.5f),
            vec3(0.5f, -0.5f, 0.5f),
            vec3(0.5f, 0.5f, -0.5f),
            vec3(0.5f, 0.5f, 0.5f)
        };
        return v;
    }
    static const int* faces()
    {
        static const int fs[] = {
            0, 1, 3, 2,
            4, 5, 1, 0,
            4, 0, 2, 6,
            2, 3, 7, 6,
            1, 5, 7, 3,
            5, 4, 6, 7
        };
        return fs;
    }
    // FIXME: probably switch to a static function
    Cube(double scale = 1.0f)
        : mass(1000.0f), scale(scale), p(0.0f, 0.0f, 0.0f)
    {
        A.setIdentity(3, 3);
        q = {
            p, 
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        };
        q0 = q;
        for (int i= 0; i < 4; i++)
            dqdt[i].setZero(3);
        
        dimensions.setOnes(3, 1);
        dimensions *= scale;
        Ic = mass * scale * scale / 12;
    }

    inline vec3 vt1(int i) const {
        mat3 a;
        vec3 b = q[0];
        a << q[1] , q[2], q[3];
        return a * vertices() [i] + b;
    }
    
    inline vec3 vt2(int i) const {
        mat3 a;
        vec3 b = q[0];
        b += dq.segment<3>(0);
        a << q[1] , q[2], q[3];
        for (int i = 1; i < 4; i ++){
            a.col(i - 1) += dq.segment<3>(i * 3);
        }
        return a * vertices() [i] + b;
    }

    inline vec3 vt0(int i) const {
        mat3 a;
        vec3 b = q0[0];
        a << q0[1] , q0[2], q0[3];
        return a * vertices() [i] + b;
    }
    
    double vg_collision_time();
    VectorXd q_tile(double dt, const vec3 &f) const;
    static void gen_indices()
    {
        for (int i = 0; i < 6; i++) {
            Cube::indices[i * 6 + 0] = faces()[i * 4 + 0];
            Cube::indices[i * 6 + 1] = faces()[i * 4 + 1];
            Cube::indices[i * 6 + 2] = faces()[i * 4 + 2];
            Cube::indices[i * 6 + 3] = faces()[i * 4 + 2];
            Cube::indices[i * 6 + 4] = faces()[i * 4 + 3];
            Cube::indices[i * 6 + 5] = faces()[i * 4 + 0];
        }
        for (int i = 0; i < 3; i++) {
            int di = 1 << (2 - i);
            for (int j = 0; j < 4; j++) {
                int I = i * 4 + j;
                int e0 = faces()[I];
                Cube::edges[I * 2] = e0;
                Cube::edges[I * 2 + 1] = e0 + di;
            }
        }
    }
};