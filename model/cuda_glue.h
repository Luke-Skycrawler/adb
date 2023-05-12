#pragma once
// works in conjunction with eigen and cuda environment
#include "affine_body.h"
#include "cuda_globals.cuh"
#include <vector>
#include <memory>
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



#ifndef TESTING
// function declarations


void initialize_primitives(
    const std::vector<std::unique_ptr<AffineBody>>& cubes
);
void initialize_aabbs(
    const std::vector<lu>& aabbs
);

// template <typename T>
// thrust::device_vector<T> to_thrust(std::vector<T> &b);
void project_glue(int vtn);
void cuda_ipc_glue();
void hess_cuda(int n_cubes, float dt, float* grads, float* hess);
void cuda_solve(Eigen::VectorXd& dq, Eigen::SparseMatrix<double>& sparse_hess, Eigen::VectorXd& r);

void glue_vf_col_set(
    std::vector<int>& vilist, std::vector<int>& fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J,
    std::vector<std::array<vec3, 4>>& pts,
    std::vector<std::array<int, 4>>& idx,
    int tid = 0);

void get_submat_glue(
    int ii, int jj,
    float* submat12x12);
#endif