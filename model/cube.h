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
    static int indices[36];
    static const int n_vertices = 8, n_faces = 12;
    static const vec3* vertices() {
        static const vec3 v[] ={
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
    static const int *faces(){
        static const int fs[] = {
            0,1,3,2,
            4,5,1,0,
            2,3,7,6,
            4,0,2,6,
            1,5,7,3,
            5,4,6,7
        };
        return fs;
    }
    // FIXME: probably switch to a static function
    Cube(double scale = 1.0f): mass(1.0f), scale(scale), p(0.0f, 0.0f, 0.0f), p_dot(0.0f, 0.0f, 0.0f) {
        A.setIdentity(3, 3);
        dimensions.setOnes(3, 1);
        dimensions *= scale;
    }
    
    double vf_collision_detect(const vec3 &dp, const mat3 &dq);
    static void gen_indices() {
        for(int i = 0; i < 6; i++) {
            Cube::indices[i * 6 + 0] = faces()[i * 4 + 0];
            Cube::indices[i * 6 + 1] = faces()[i * 4 + 1];
            Cube::indices[i * 6 + 2] = faces()[i * 4 + 2];
            Cube::indices[i * 6 + 3] = faces()[i * 4 + 2];
            Cube::indices[i * 6 + 4] = faces()[i * 4 + 3];
            Cube::indices[i * 6 + 5] = faces()[i * 4 + 0];
        }
    }
};