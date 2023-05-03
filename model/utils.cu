#include "cuda_globals.cuh"
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

using namespace ipc;

__device__ __host__ float vf_distance(vec3f _v, Facef f, PointTriangleDistanceType& pt_type);
__device__ luf intersection(const luf &a, const luf &b);
__device__ luf affine(luf aabb, cudaAffineBody &c, int vtn);
__device__ luf compute_aabb(const Facef& f, float d_hat_sqrt);
__device__ luf compute_aabb(const Edgef& e, float d_hat_sqrt);
