#pragma once
#include <Eigen/Eigen>
#include <vector>
using namespace Eigen;
using vec3 = Vector3d;
using mat3 = Matrix3d;

struct Cube {
    mat3 A, q_dot, q_next, dq;
    vec3 p, p_next, p_dot, dimensions, f, tau, dp;
    double mass, scale;
    static int indices[36], edges[24];
    static const int n_vertices = 8, n_faces = 12, n_edges = 12;
    Vector<double, 12> barrier_gradient;
    Matrix<double, 12, 12> hess;
    vec3 q[4], q0[4];
    void cat_all(){
        q[0] = p_next;
        for (  int i = 0; i < 3; i++) {
            q[i] = q_next.col(i);
        }
    }
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
        : mass(1000.0f), scale(scale), p(0.0f, 0.0f, 0.0f), p_dot(0.0f, 0.0f, 0.0f)
    {
        A.setIdentity(3, 3);
        dimensions.setOnes(3, 1);
        dimensions *= scale;
    }

    inline vec3 vi(int i, bool increment = false) const
    {
        const auto& q(increment ? (q_next - dq) : q_next);
        const auto& p(increment ? (p_next - dp) : p_next);
        return q * vertices()[i] + p;
    }

    double vf_collision_detect(const vec3& dp, const mat3& dq);
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