#pragma once
#include "affine_body.h"
using namespace Eigen;
struct Cube: AffineBody {
    double scale, toi;
    static const int n_vertices = 8, n_faces = 12, n_edges = 12;
    Vector<double, 12> barrier_gradient;
    inline const vec3 vertices(int i) const {return _vertices()[i];}
    static const vec3* _vertices()
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
        : scale(scale), AffineBody(8, 12, 12, _indices, _edges){
        Ic = mass * scale * scale / 12;
    }

    static unsigned * _indices, *_edges;
    static void gen_indices()
    {
        _indices = new unsigned[n_faces * 3];
        _edges = new unsigned[n_edges * 2];
        for (int i = 0; i < 6; i++) {
            const int * f = faces();
            _indices[i * 6 + 0] = f[i * 4 + 0];
            _indices[i * 6 + 1] = f[i * 4 + 1];
            _indices[i * 6 + 2] = f[i * 4 + 2];
            _indices[i * 6 + 3] = f[i * 4 + 2];
            _indices[i * 6 + 4] = f[i * 4 + 3];
            _indices[i * 6 + 5] = f[i * 4 + 0];
        }
        for (int i = 0; i < 3; i++) {
            int di = 1 << (2 - i);
            for (int j = 0; j < 4; j++) {
                int I = i * 4 + j;
                int e0 = faces()[I];
                _edges[I * 2] = e0;
                _edges[I * 2 + 1] = e0 + di;
            }
        }
    }
    void draw(Shader &shader) const;
};