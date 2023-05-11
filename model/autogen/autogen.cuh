#pragma once
#include "../cuda_header.cuh"

// Symbolically generated derivatives;
namespace autogen {
    __host__ __device__ void point_plane_distance_gradient(
        float v01,
        float v02,
        float v03,
        float v11,
        float v12,
        float v13,
        float v21,
        float v22,
        float v23,
        float v31,
        float v32,
        float v33,
        float g[12]);

    inline __host__ __device__ void point_plane_distance_gradient(
        vec3f p,
        vec3f t0,
        vec3f t1,
        vec3f t2,
        float g[12])
    {
        point_plane_distance_gradient(p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t2.x, t2.y, t2.z, g);
    }

    __host__ __device__ void point_plane_distance_hessian(
        float v01,
        float v02,
        float v03,
        float v11,
        float v12,
        float v13,
        float v21,
        float v22,
        float v23,
        float v31,
        float v32,
        float v33,
        float H[144]);
    
    inline __host__ __device__ void point_plane_distance_hessian(
        vec3f p,
        vec3f t0,
        vec3f t1,
        vec3f t2,
        float H[144])
    {
        point_plane_distance_hessian(p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t2.x, t2.y, t2.z, H);
    }
    
    __host__ __device__ void point_line_distance_gradient_3D(
        float v01,
        float v02,
        float v03,
        float v11,
        float v12,
        float v13,
        float v21,
        float v22,
        float v23,
        float g[9]);
    
    inline __host__ __device__ void point_line_distance_gradient_3D(
        vec3f p,
        vec3f t0,
        vec3f t1,
        float g[9])
    {
        point_line_distance_gradient_3D(p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, g);
    }
    
    __host__ __device__ void point_line_distance_hessian_3D(
        float v01,
        float v02,
        float v03,
        float v11,
        float v12,
        float v13,
        float v21,
        float v22,
        float v23,
        float H[81]);
    
    inline __host__ __device__ void point_line_distance_hessian_3D(
        vec3f p,
        vec3f t0,
        vec3f t1,
        float H[81])
    {
        point_line_distance_hessian_3D(p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, H);
    }
    __host__ __device__ void line_line_distance_gradient(
        float v01,
        float v02,
        float v03,
        float v11,
        float v12,
        float v13,
        float v21,
        float v22,
        float v23,
        float v31,
        float v32,
        float v33,
        float g[12]);

    inline __host__ __device__ void line_line_distance_gradient(
        vec3f p0,
        vec3f p1,
        vec3f q0,
        vec3f q1,
        float g[12])
    {
        line_line_distance_gradient(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, q0.x, q0.y, q0.z, q1.x, q1.y, q1.z, g);
    }
    __host__ __device__ void line_line_distance_hessian(
        float v01,
        float v02,
        float v03,
        float v11,
        float v12,
        float v13,
        float v21,
        float v22,
        float v23,
        float v31,
        float v32,
        float v33,
        float H[144]);

    inline __host__ __device__ void line_line_distance_hessian(
        vec3f p0,
        vec3f p1,
        vec3f q0,
        vec3f q1,
        float H[144])
    {
        line_line_distance_hessian(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, q0.x, q0.y, q0.z, q1.x, q1.y, q1.z, H);
    }

} // namespace autogen
