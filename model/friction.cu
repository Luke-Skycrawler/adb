#include "cuda_globals.cuh"

// return basis are all 3x2 
__host__ __device__ void point_point_tangent_basis(
    vec3f p0, vec3f p1, 
    float3& b0, float3& b1
) {
    auto p01 = p1 - p0;
    auto ex = make_flaot3(1.0f, 0.0f, 0.0f);
    auto ey = make_flaot3(0.0f, 1.0f, 0.0f);
    auto cx = cross(p01, ex);
    auto cy = cross(p01, ey);
    auto nx = dot(cx, cx);
    auto ny = dot(cy, cy);
    if (nx > ny) {
        b0 = normalize(cx);
        b1 = normalize(cross(p01, cx));
    } else {
        b0 = normalize(cy);
        b1 = normalize(cross(p01, cy));
    }
}
__host__ __device__ void point_edge_tangent_basis(
    vec3f p, vec3f e0, vec3f e1, 
    float3 &b0, float3 &b1
) {
    auto e = e1 - e0;
    b0 = normalize(e);
    b1 = normalize(cross(e, p - e0));
}
__host__ __device__ void edge_edge_tangent_basis(
    vec3f ei0, vec3f ei1, vec3f ej0, vec3f ej1,
    float3 &b0, float3 &b1
)   {
    auto ei = ei1 - ei0;
    auto ej = ej1 - ej0;
    b0 = normalize(ei);
    auto normal = cross(ei , ej);
    b1 = normalize(cross(normal, ei));
}
__host__ __device__ void point_triangle_tangent_basis(
    vec3f p, vec3f t0, vec3f t1, vec3f t2,
    float3 &b0, float3 &b1
)  {
    auto e0 = t1 - t0;
    auto normal = cross(e0, t2 - t0);
    b0 = normalize(e0);
    b1 = normalize(cross(normal, e0));
}

__host__ __device__ float point_edge_closest_point(
    vec3f p, vec3f e0, vec3f e1
) {
    
}

__host__ __device__ edge_edge_closest_point(
    vec3f ei0, vec3f ei1, vec3f ej0, vec3f ej1, float &a, float &b
) {

}
__host__ __device__ point_triangle_closest_point(
    vec3f p, vec3f t0, vec3f t1, vec3f t2, float &a, float &b
) {

}

__forceinline__ __host__ __device__ void to_float9(
    vec3f b0, vec3f b1, float lams[3],
    float *ret
) {
    // pack to save bandwidth
    ret[0] = b0.x;
    ret[1] = b0.y;
    ret[2] = b0.z;
    ret[3] = b1.x;
    ret[4] = b1.y;
    ret[5] = b1.z;
    ret[6] = lams[0];
    ret[7] = lams[1];
    ret[8] = lams[2];
}

__forceinline__ __host__ __device__ void Tk_from_float9(
    float *Tk_float9, float3 *ret[8]
){
    
    // Tk^T = 
    // [[ -b0, l0 b0, l1 b0, l2 b0 ]
    // [  -b1, l0 b1, l1 b1, l2 b1 ]]
    auto b0 = make_float3(Tk_float9[0], Tk_float9[1], Tk_float9[2]);
    auto b1 = make_float3(Tk_float9[3], Tk_float9[4], Tk_float9[5]);
    auto lams = Tk_float9 + 6;
    ret[0] = -b0; ret[2] = lams[0] * b0; ret[4] = lams[1] * b0; ret[6] = lams[2] * b0;
    ret[1] = -b1; ret[3] = lams[0] * b1; ret[5] = lams[1] * b1; ret[7] = lams[2] * b1;
}

__host__ __device__ pt_ustack(cudaAffineBody  &ci, cudaAffineBody &cj, int vi, int fj, float3 u[4]) {
    // NOTE: should call project vt2 then project vt0 before calling this function
    Facef tt2{cj.triangle_updated(fj)};
    Facef tt0{cj.triangle(fj)};
    vec3f pt2{ci.updated(vi)};
    vec3f pt0{ci.projected(pt2)};

    u[0] = pt2 - pt0;
    u[1] = tt2.t0 - tt0.t0;
    u[2] = tt2.t1 - tt0.t1;
    u[3] = tt2.t2 - tt0.t2;
} 

__host__ __device__ ee_ustack(cudaAffineBody &ci, cudaAffineBody &cj, int ei, int ej, flaot3 u[4]) {
    Edgef ei0{ci.edge(ei)}, ei2{ci.edge_updated(ei)};
    Edgef ej0{cj.edge(ej)}, ej2{cj.edge_updated(ej)};
    u[0] = ei2.e0 - ei0.e0;
    u[1] = ei2.e1 - ei0.e1;
    u[2] = ej2.e0 - ej0.e0;
    u[3] = ej2.e1 - ej0.e1;
}


__host__ __device__ void pt_uktk(
    cudaAffineBody &ci, cudaAffineBody &cj,
    i2 p,
    int pt_type,
    float *ret_Tk_float9,   // float9 format
    float &ux, float &uy
    // float dist , float dt
) {
    float3 u[4];
    pt_ustack(ci, cj, p[0], p[1], u );

    vec3f v {ci.updated[p[0]]};
    Facef f {cj.triangle_updated(p[1])};
    float lams[3] {0.0f};
    vec3f b0, b1;
    float a,b;

    switch (pt_type) {
        case 0: 
            // P_T0
            lams[0] = 1.0f;
            point_point_tangent_basis(v, f.t0, b0, b1);
            break;
        case 1: 
            // P_T1
            lams[1] = 1.0f;
            point_point_tangent_basis(v, f.t1, b0, b1);
            break;
        case 2:
        // P_T2
            lams[2] = 1.0f;
            point_point_tangent_basis(v, f.t2, b0, b1);
            break;
        case 3:
            // P_E0
            a = point_edge_closest_point(v, f.t0, f.t1);
            lams[0] = 1.0f - a;
            lams[1] = a;
            point_edge_tangent_basis(v, f.t0, f.t1, b0, b1);
            break;
        case 4:
            // P_E1
            a = point_edge_closest_point(v, f.t1, f.t2);
            lams[1] = 1.0f - a;
            lams[2] = a;
            point_edge_tangent_basis(v, f.t1, f.t2, b0, b1);
            break;
        case 5: 
            // P_E2
            a = point_edge_closest_point(v, f.t2, f.t0);
            lams[2] = 1.0f - a;
            lams[0] = a;
            point_edge_tangent_basis(v, f.t2, f.t0, b0, b1);
            break;
        case 6:
        // P_T
            point_triangle_closest_point(v, f.t0, f.t1, f.t2, a, b);
            lams[0] = 1.0f - a - b;
            lams[1] = a;
            lams[2] = b;
            point_triangle_tangent_basis(v, f.t0, f.t1, f.t2, b0, b1);
    }
    // Tk^T = 
    // [[ -b0, l0 b0, l1 b0, l2 b0 ]
    // [  -b1, l0 b1, l1 b1, l2 b1 ]]
    to_float9(b0, b1, lams, ret_Tk_float9);
    // uk = Tk^T * (u0, u1, u2, u3)^T
    ux = -dot(b0, u[0])  + lams[0] * dot(b0, u[1]) + lams[1] * dot(b0, u[2]) + lams[2] * dot(b0, u[3]);
    uy = -dot(b1, u[0])  + lams[0] * dot(b1, u[1]) + lams[1] * dot(b1, u[2]) + lams[2] * dot(b1, u[3]);
}