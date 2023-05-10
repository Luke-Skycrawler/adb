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

} // namespace autogen
