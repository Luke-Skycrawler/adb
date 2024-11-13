#pragma once
#include <vector>
// #include <array>

#ifndef TESTING 
#define EIGEN_USE_MKL_ALL
#endif

#include "../view/shader.h"
#include "scalar_types.h"

struct AffineBody {
    mat3 A;
    vec3 p;
    scalar mass, Ic;
    std::vector<int> indices, edges;
    virtual const vec3 vertices(int i) const = 0;
    virtual void draw(Shader& shader) const = 0;
    std::vector<vec3> v_transformed;
    int n_edges, n_vertices, n_faces;
    vec12 dq, grad;
    mat12 hess;
    q4 q, q0, dqdt;
    vec12 q_tile(scalar dt, const vec3 &f) const;
    inline vec3 vt0(int i) const {
        mat3 a;
        vec3 b = q0[0];
        a << q0[1] , q0[2], q0[3];
        return a * vertices(i) + b;
    }

    inline vec3 vt1(int i) const {
        mat3 a;
        vec3 b = q[0];
        a << q[1] , q[2], q[3];
        return a * vertices(i) + b;
    }
    
    inline vec3 vt2(int i) const {
        mat3 a;
        vec3 b = q[0];
        b += dq.segment<3>(0);
        a << q[1] , q[2], q[3];
        for (int i = 1; i < 4; i ++){
            a.col(i - 1) += dq.segment<3>(i * 3);
        }
        return a * vertices(i) + b;
    }

    inline void project_vt1()
    {
        mat3 a;
        vec3 b = q[0];
        a << q[1], q[2], q[3];
        for (int i = 0; i < n_vertices; i++) {
            v_transformed[i] = a * vertices(i) + b;
        }
    }
    inline void project_vt2()
    {
        mat3 a;
        vec3 b = q[0];
        b += dq.segment<3>(0);
        a << q[1], q[2], q[3];
        for (int i = 1; i < 4; i++) {
            a.col(i - 1) += dq.segment<3>(i * 3);
        }
        for (int i = 0; i < n_vertices; i++) {
            v_transformed[i] = a * vertices(i) + b;
        }
    }
    inline void project_vt0()
    {
        mat3 a;
        vec3 b = q0[0];
        a << q0[1], q0[2], q0[3];
        for (int i = 0; i < n_vertices; i++) {
            v_transformed[i] = a * vertices(i) + b;
        }
    }

    Face face(int triangle_id, bool use_line_search_increment = false, bool batch = false) const;
    Edge edge(int eid, bool use_line_search_increment = false, bool batch = false) const;
    AffineBody(int n_vertices, int n_faces, int n_edges, std::vector<int> indices = {}, std::vector<int> edges = {})
        : mass(1000.0), Ic(1000.0), p(0.0f, 0.0f, 0.0f), indices(indices), edges(edges), n_vertices(n_vertices), n_edges(n_edges), n_faces(n_faces)
    {
        v_transformed.resize(n_vertices);
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
    }
};

