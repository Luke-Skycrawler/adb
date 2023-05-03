#pragma once
#include "affine_body.h"
#include "cuda_globals.cuh"
#include <vector>
using lu = std::array<vec3, 2>;

inline luf to_luf(const lu& a)
{
    return {
        make_float3(a[0][0], a[0][1], a[0][2]),
        make_float3(a[1][0], a[1][1], a[1][2])
    };
}

inline vec3f to_vec3f(const vec3& a)
{
    return make_float3(a[0], a[1], a[2]);
}
inline Facef to_facef(const Face& f)
{
    return {
        to_vec3f(f.t0),
        to_vec3f(f.t1),
        to_vec3f(f.t2)
    };
}

inline cudaAffineBody to_cabd(AffineBody &a){
   cudaAffineBody b;
   for (int i = 0; i < 4; i ++){
       b.q[i] = to_vec3f(a.q[i]);
       b.q0[i] = to_vec3f(a.q0[i]);
       b.dqdt[i] = to_vec3f(a.dqdt[i]);
       b.q_update[i] = to_vec3f(a.q[i] + a.dq.segment<3>(i * 3));
   }
   b.mass = a.mass;
   b.Ic = a.Ic;
   b.n_vertices = a.n_vertices;
   b.n_faces = a.n_faces;
   b.n_edges = a.n_edges;

   return b;
}

template <typename T>
std::vector<T> from_thrust(thrust::device_vector<T> &a)
{
    thrust::host_vector<T> b = a;
    std::vector<T> ret;
    ret.resize(b.size());
    for (int i = 0; i < b.size(); i++) {
        ret[i] = b[i];
    }
    return ret;
}
template <typename T>
std::vector<T> from_thrust(thrust::host_vector<T>& b)
{
    std::vector<T> ret;
    ret.resize(b.size());
    for (int i = 0; i < b.size(); i++) {
        ret[i] = b[i];
    }
    return ret;
}


// template <typename T>
// thrust::device_vector<T> to_thrust(std::vector<T> &b);

