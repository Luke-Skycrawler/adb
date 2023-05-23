#include "cuda_header.cuh"
static __constant__ float mu = 0.2f, evh = 1e-3f * 1e-2f;

// return basis are all 3x2 
__host__ __device__ void point_point_tangent_basis(
    vec3f p0, vec3f p1, 
    float3& b0, float3& b1
) {
    auto p01 = p1 - p0;
    auto ex = make_float3(1.0f, 0.0f, 0.0f);
    auto ey = make_float3(0.0f, 1.0f, 0.0f);
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
    auto e = e1 - e0;
    return dot(p - e0, e) / dot(e, e);
}

__host__ __device__ void edge_edge_closest_point(
    vec3f ei0, vec3f ei1, vec3f ej0, vec3f ej1, float &a, float &b
) {
    auto eij = ei0 - ej0;
    auto ei = ei1 - ei0;
    auto ej = ej1 - ej0;

    auto a00= dot(ei, ei);
    auto a01= -dot(ei, ej);
    auto a11 = dot(ej, ej);

    auto b0 = -dot(ei, eij);
    auto b1 = dot(ej, eij);
    float J = a00 * a11 - a01 * a01;
    a = (b0 * a11 - b1 * a01) / J;
    b = (a00 * b1 - b0 * a01) / J;
}
__host__ __device__ void point_triangle_closest_point(
    vec3f p, vec3f t0, vec3f t1, vec3f t2, float &a, float &b
) {
    auto normal = cross(t1 - t0, t2 - t0);
    auto at1 = CUDA_ABS(dot(normal, cross(t2 - t0, p - t0)));
    auto at2 = CUDA_ABS(dot(normal, cross(t0 - t1, p - t0)));
    auto at = dot(normal, normal);
    a = at1 / at;
    b = at2 / at;
}

__forceinline__ __host__ __device__ void to_float8(
    vec3f b0, vec3f b1, float lams[2],
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
    // ret[8] = lams[2];
}

__forceinline__ __host__ __device__ void Tk_pt_from_float8(
    float *Tk_float8, float3 ret[8]
){
    
    // Tk^T = 
    // [[ -b0, l0 b0, l1 b0, l2 b0 ]
    // [  -b1, l0 b1, l1 b1, l2 b1 ]]
    auto b0 = make_float3(Tk_float8[0], Tk_float8[1], Tk_float8[2]);
    auto b1 = make_float3(Tk_float8[3], Tk_float8[4], Tk_float8[5]);
    auto lams = Tk_float8 + 6;
    float lams2 = 1.0f - lams[0] - lams[1];
    ret[0] = b0 * -1.0f; ret[2] = b0 * lams[0]; ret[4] = b0 * lams[1]; ret[6] = b0 * lams2;
    ret[1] = b1 * -1.0f; ret[3] = b1 * lams[0]; ret[5] = b1 * lams[1]; ret[7] = b1 * lams2;
}

__forceinline__ __host__ __device__ void Tk_ee_from_float8(
    float *Tk_float8, float3 ret[8]
){
    // Tk^T = 
    // [[ -(1 - l0)b0, -l0 b0, (1 - l1) b0, l1 b0 ]
    // [  -(1 - l0)b1, -l0 b1, (1 - l1) b1, l1 b1 ]]
    auto b0 = make_float3(Tk_float8[0], Tk_float8[1], Tk_float8[2]);
    auto b1 = make_float3(Tk_float8[3], Tk_float8[4], Tk_float8[5]);
    auto lams = Tk_float8 + 6;
    ret[0] = b0 * -(1.0f - lams[0]); ret[2] = b0 * - lams[0]; ret[4] = b0 * (1.0f - lams[1]); ret[6] = b0 * lams[1];
    ret[1] = b1 * -(1.0f - lams[0]); ret[3] = b1 * - lams[0]; ret[5] = b1 * (1.0f - lams[1]); ret[7] = b1 * lams[1];
}

__host__ __device__ void pt_ustack(cudaAffineBody  &ci, cudaAffineBody &cj, int vi, int fj, float3 u[4]) {
    // NOTE: should call project vt2 then project vt0 before calling this function
    Facef tt2{cj.triangle_updated(fj)};
    Facef tt0{cj.triangle(fj)};
    vec3f pt2{ci.updated[vi]};
    vec3f pt0{ci.projected[vi]};

    u[0] = pt2 - pt0;
    u[1] = tt2.t0 - tt0.t0;
    u[2] = tt2.t1 - tt0.t1;
    u[3] = tt2.t2 - tt0.t2;
} 

__host__ __device__ void ee_ustack(cudaAffineBody &ci, cudaAffineBody &cj, int ei, int ej, float3 u[4]) {
    Edgef ei0{ci.edge(ei)}, ei2{ci.edge_updated(ei)};
    Edgef ej0{cj.edge(ej)}, ej2{cj.edge_updated(ej)};
    u[0] = ei2.e0 - ei0.e0;
    u[1] = ei2.e1 - ei0.e1;
    u[2] = ej2.e0 - ej0.e0;
    u[3] = ej2.e1 - ej0.e1;
}


__host__ __device__ void pt_uktk(
    #ifdef TESTING
    vec3f pts[4], float& dist_ret,
    #else
    cudaAffineBody &ci, cudaAffineBody &cj,
    i2 p,
    #endif
    int pt_type,
    float *ret_Tk_float8,   // float8 format
    float &ux, float &uy
    
    // float dist , float dt
) {
    #ifndef TESTING
    float3 u[4];
    pt_ustack(ci, cj, p[0], p[1], u );

    vec3f v {ci.updated[p[0]]};
    Facef f {cj.triangle_updated(p[1])};
    #else 
    vec3f v {pts[0]};
    Facef f {pts[1], pts[2], pts[3]};
    #endif
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
    #ifdef TESTING
    auto tp = f.t0 * lams[0] + f.t1 * lams[1] + f.t2 * lams[2];
    auto closest = dot(v - tp, v - tp);
    dist_ret = closest;
    #else 
    // Tk^T = 
    // [[ -b0, l0 b0, l1 b0, l2 b0 ]
    // [  -b1, l0 b1, l1 b1, l2 b1 ]]
    to_float8(b0, b1, lams, ret_Tk_float8);
    // uk = Tk^T * (u0, u1, u2, u3)^T
    ux = -dot(b0, u[0])  + lams[0] * dot(b0, u[1]) + lams[1] * dot(b0, u[2]) + lams[2] * dot(b0, u[3]);
    uy = -dot(b1, u[0])  + lams[0] * dot(b1, u[1]) + lams[1] * dot(b1, u[2]) + lams[2] * dot(b1, u[3]);
    #endif
}

__host__ __device__ void ee_uktk(
    #ifndef TESTING
    cudaAffineBody &ci, cudaAffineBody &cj,
    i2 p,
    #else 
    vec3f ees[4], float &dist_ret,
    #endif
    int ee_type,
    float *ret_Tk_float8,   // float8 format
    float &ux, float &uy
    // float dist , float dt
) {

    #ifndef TESTING
    float3 u[4];
    ee_ustack(ci, cj, p[0], p[1], u );

    Edgef ei{ci.edge_updated(p[0])}, ej {cj.edge_updated(p[1])};
    #else 
    Edgef ei{ees[0], ees[1]}, ej{ees[2], ees[3]};
    #endif
    float lams[2] {0.0f};
    vec3f b0, b1;
    float a,b;

    switch (ee_type) {
        case 0: 
        // EA0_EB0
        point_point_tangent_basis(ei.e0, ej.e0, b0, b1);
        break;
        case 1:
        // EA0_EB1
        point_point_tangent_basis(ei.e0, ej.e1, b0, b1);
        lams[1] = 1.0f;
        break;
        case 2:
        // EA1_EB0
        point_point_tangent_basis(ei.e1, ej.e0, b0, b1);
        lams[0] = 1.0f;
        break;
        case 3:
        // EA1_EB1
        point_point_tangent_basis(ei.e1, ej.e1, b0, b1);
        lams[0] = 1.0f;
        lams[1] = 1.0f;
        break;
        case 4:
        // EA_EB0
        point_edge_tangent_basis(ej.e0, ei.e0, ei.e1, b0, b1);
        a = point_edge_closest_point(ej.e0, ei.e0, ei.e1);
        lams[0] = a;
        break;
        case 5:
        // EA_EB1
        point_edge_tangent_basis(ej.e1, ei.e0, ei.e1, b0, b1);
        a = point_edge_closest_point(ej.e1, ei.e0, ei.e1);
        lams[0] = a;
        lams[1] = 1.0f;
        break;
        case 6:
        // EA0_EB
        point_edge_tangent_basis(ei.e0, ej.e0, ej.e1, b0, b1);
        a = point_edge_closest_point(ei.e0, ej.e0, ej.e1);
        lams[1] = a;
        break;
        case 7:
        // EA1_EB
        point_edge_tangent_basis(ei.e1, ej.e0, ej.e1, b0, b1);
        a = point_edge_closest_point(ei.e1, ej.e0, ej.e1);
        lams[0] = 1.0f;
        lams[1] = a;
        break;
        case 8:
        // EA_EB
        edge_edge_tangent_basis(ei.e0, ei.e1, ej.e0, ej.e1, b0, b1);
        edge_edge_closest_point(ei.e0, ei.e1, ej.e0, ej.e1, a, b);
        lams[0] = a;
        lams[1] = b;
    }
    // lams[0] = CUDA_MAX(CUDA_MIN(lams[0], 1.0f), 0.0f);
    // lams[1] = CUDA_MAX(CUDA_MIN(lams[1], 1.0f), 0.0f);

    #ifdef TESTING
    auto pei = ei.e0 * (1.0f - lams[0]) + ei.e1 * lams[0];
    auto pej = ej.e0 * (1.0f - lams[1]) + ej.e1 * lams[1];
    auto closest = dot(pei - pej, pei - pej);
    dist_ret = closest;
    #else 
    // Tk^T = 
    // [[ -(1 - l0)b0, -l0 b0, (1 - l1) b0, l1 b0 ]
    // [  -(1 - l0)b1, -l0 b1, (1 - l1) b1, l1 b1 ]]
    to_float8(b0, b1, lams, ret_Tk_float8);

    // uk = Tk^T * (u0, u1, u2, u3)^T
    ux = -(1.0f - lams[0])  * dot(b0, u[0])  - lams[0] * dot(b0, u[1]) + (1.0f - lams[1]) * dot(b0, u[2]) + lams[1] * dot(b0, u[3]);
    uy = -(1.0f - lams[0]) * dot(b1, u[0])  - lams[0] * dot(b1, u[1]) + (1.0f - lams[1]) * dot(b1, u[2]) + lams[1] * dot(b1, u[3]);
    #endif
}

__host__ __device__ float f1_over_x(float x, float epsv_times_h) {
    if (CUDA_ABS(x) >= epsv_times_h) {
        return 1 / x;
    }
    return (-x / epsv_times_h + 2) / epsv_times_h;    
} 

__forceinline__ __host__ __device__ float2 Tki(
    float3 Tk[8],
    int I
)  {
    int i = I  / 3, j = i % 3;
    auto t0 {u[i *2 ]}, t1 {u[i * 2 + 1]};
    switch (j) {
        case 0:
        return make_float2(t0.x, t1.x);
        case 1:
        return make_float2(t0.y, t1.y);
        case 2:
        return make_float2(t0.z, t1.z);
    }
}
__forceinline__ __host__ __device__ dot(float2 x, float2 y) {
    return x.x * y.x + x.y * y.y;
}
__host__ __device__ void friciton (
    float2 u,
    float lam,
    float Tk[8],
    int pt_1_ee_0,
    float *g, float *H,
) {
    float uk = CUDA_SQRT(u.x * u.x + u.y * u.y);
    auto f1 = f1_over_x(uk, evh);
    
    float3 Tk[8];
    if (pt_1_ee_0) {
        Tk_pt_from_float8(Tk);
    }
    else {
        Tk_ee_from_float8(Tk);
    }
    auto scale = lam * mu * f1;
    for (int i = 0; i < 12; i++) {
        // g += F_k = mu * contact_lambda * f1 * Tk * uk;
        auto ti = Tki(Tk, i);
        g[i] += scale * dot(ti, u);
    }

    if (uk >= evh) {
        
        float2 ut = make_float2(-u.y, u.x);
        auto scale_by_uk_square = scale / (uk * uk);
        // Dk_hessian = scale / (uk * uk) * Tk * ut * ut^T * Tk^T
        // = scale / (uk * uk) * f * f^T, where f_12x1 = Tk * ut
        for (int j = 0; j < 12; j ++) {
            auto tj = Tki(Tk, j);
            auto fj = dot(tj, ut);
            
            for (int i = 0 ; i < 12; i++) {
                int I = i + j * 12;
                auto ti = Tki(Tk, i);
                auto fi = dot(ti, ut);
                H[I] += scale_by_uk_square * fi * fj;
            }
        }
    }
    else if (uk == 0) {
        // Dk_hessian = scale * Tk * Tk^T
        for (int j = 0; j < 12; j ++) {
            auto tj = Tki(Tk, j);
            for (int i = 0; i < 12; i ++){
                int I = i + j * 12;
                auto ti = Tki(Tk, i);
                H[I] += scale * dot(ti, tj);
            }
        }
    } else {
        float f2_term = -1.0f / (evh * evh);

        // Dk_hessian = Tk * M2x2 * Tk^T

        float M2x2 {
            f1 + u.x * u.x * f2_term / uk, u.x * u.y * f2_term / uk,
            u.x * u.y * f2_term / uk, f1 + u.y * u.y * f2_term / uk
        };
        // don't apply psd projection since ac - b^2 > 0 forall eph < 0.5
        for (int j = 0; j <12; j ++) {
            auto tj = Tki(Tk, j);
            for (int i = 0; i < 12; i ++) {
                auto ti = Tki(Tk, i);
                int I = i + j * 12;
                H[I] += M2x2[0] * ti.x * tj.x + M2x2[2] * ti.y * tj.x  + M2x2[1] * ti.x * tj.y + M2x2[3] * ti.y * tj.y;
            }
        }
    }
}