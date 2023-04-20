#pragma once
#include <cuda/std/array>
#include <thrust/device_vector.h>


using i2 = cuda::std::array<int, 2>;
using i4 = cuda::std::array<int, 4>;
struct CollisionSets {
    thrust::device_vector<i2> pt_set[7];
    thrust::device_vector<i2> pt_set_body_index[7];
    /* corresponds to enum class PointTriangleDistanceType {
        P_T0, 
        P_T1, 
        P_T2, 
        P_E0, 
        P_E1, 
        P_E2, 
        P_T
    };*/
    thrust::device_vector<i2> ee_set[9];
    thrust::device_vector<i2> ee_set_body_index[9];
    /* corresponds to enum class EdgeEdgeDistanceType {
        EA0_EB0, 
        EA0_EB1, 
        EA1_EB0, 
        EA1_EB1, 
        EA_EB0,
        EA_EB1,
        EA0_EB,
        EA1_EB,
        EA_EB
    }; */
    
};

struct CsrSparseMatrix
{
	int rows = 0;
	int cols = 0;
	int nnz = 0;
	thrust::device_vector<int> outer_start;
	thrust::device_vector<int> inner;
	thrust::device_vector<float> values;
};

struct CudaGlobals {
    CollisionSets collision_sets;
    CsrSparseMatrix hess;
    thrust::device_vector<float> b;
    thrust::device_vector<i2> lut;
};    

extern CudaGlobals cuda_globals;
