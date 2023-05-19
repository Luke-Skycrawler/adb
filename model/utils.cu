#include "cuda_globals.cuh"
#include "autogen/autogen.cuh"

static __constant__ float kappa = 1e9f, dt_static = 1e-2f;


__device__ luf intersection(const luf& a, const luf& b)
{
    vec3f l, u;
    l = make_float3(
        CUDA_MAX(a.l.x, b.l.x),
        CUDA_MAX(a.l.y, b.l.y),
        CUDA_MAX(a.l.z, b.l.z));
    u = make_float3(
        CUDA_MIN(a.u.x, b.u.x),
        CUDA_MIN(a.u.y, b.u.y),
        CUDA_MIN(a.u.z, b.u.z));
    return { l, u };
}


__host__ __device__ luf merge(luf a, luf b) {
    vec3f l = make_float3(
        CUDA_MIN(a.l.x, b.l.x),
        CUDA_MIN(a.l.y, b.l.y),
        CUDA_MIN(a.l.z, b.l.z)
    );
    vec3f u = make_float3(
        CUDA_MAX(a.u.x, b.u.x),
        CUDA_MAX(a.u.y, b.u.y),
        CUDA_MAX(a.u.z, b.u.z)
    );
    return luf{ l, u };
}


__device__ luf affine(luf aabb, cudaAffineBody& c, int vtn)
{
    vec3f cull[8];
    vec3f l, u;
    auto q{ vtn == 2 ? c.q_update : vtn == 0? c.q0: c.q };
    // vtn == 2, q update 
    // vtn == 3, merge q update and q
    // vtn == 1, q
    // vtn == 0, q0

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                auto I = (i << 2) | (j << 1) | k;
                cull[I] = make_float3(
                    i ? aabb.u.x : aabb.l.x,
                    j ? aabb.u.y : aabb.l.y,
                    k ? aabb.u.z : aabb.l.z);
                cull[I] = matmul(q, cull[I]);
            }
    for (int i = 0; i < 8; i++) {
        if (i == 0) {
            l = u = cull[i];
        }
        else {
            l.x = CUDA_MIN(l.x, cull[i].x);
            l.y = CUDA_MIN(l.y, cull[i].y);
            l.z = CUDA_MIN(l.z, cull[i].z);
            u.x = CUDA_MAX(u.x, cull[i].x);
            u.y = CUDA_MAX(u.y, cull[i].y);
            u.z = CUDA_MAX(u.z, cull[i].z);
        }
    }
    if (vtn == 3) {
        auto updated =  affine(aabb, c, 2);
        l.x = CUDA_MIN(l.x, updated.l.x);
        l.y = CUDA_MIN(l.y, updated.l.y);
        l.z = CUDA_MIN(l.z, updated.l.z);
        u.x = CUDA_MAX(u.x, updated.u.x);
        u.y = CUDA_MAX(u.y, updated.u.y);
        u.z = CUDA_MAX(u.z, updated.u.z);
        return {l, u};
    }
    return { l, u };
}

__device__ __host__ float vf_distance(vec3f _v, Facef f, int& _pt_type)
{
    auto n = unit_normal(f);
    auto d = dot(n, _v - f.t0);
    auto a1 = area_x2(f.t1, f.t0, f.t2);
    auto v = _v - n * d;
    d = d * d;
    // float a2 = ((f[0] - v).cross(f[1] - v).norm() + (f[1] - v).cross(f[2] - v).norm() + (f[2] - v).cross(f[0] - v).norm());
    // auto a2 = area_x2(f[0], f[1], v) + area_x2(f[1], f[2], v) + area_x2(f[2], f[0], v);
    auto _a1 = dot(cross(f.t0 - v, f.t1 - v), n);
    auto _a2 = dot(cross(f.t1 - v, f.t2 - v), n);
    auto _a3 = dot(cross(f.t2 - v, f.t0 - v), n);
    bool inside = _a1 * _a2 > 0.0f && _a2 * _a3 > 0.0f;
    int pt_type;
    // if (a2 > a1 + 1e-8) {
    if (!inside) {
        // projection outside of triangle

        auto d_ab = h(f.t0, f.t1, v);
        auto d_bc = h(f.t1, f.t2, v);
        auto d_ac = h(f.t0, f.t2, v);

        auto d_a = ab(v, f.t0);
        auto d_b = ab(v, f.t1);
        auto d_c = ab(v, f.t2);

        auto dab = is_obtuse_triangle(f.t0, f.t1, v) ? CUDA_MIN(d_a, d_b) : d_ab;
        auto dbc = is_obtuse_triangle(f.t2, f.t1, v) ? CUDA_MIN(d_c, d_b) : d_bc;
        auto dac = is_obtuse_triangle(f.t0, f.t2, v) ? CUDA_MIN(d_a, d_c) : d_ac;

        auto d_projected = CUDA_MIN3(dab, dbc, dac);
        d += d_projected * d_projected;

        if (d_projected == d_ab)
            // pt_type = ipc::PointTriangleDistanceType::P_E0;
            pt_type = 3;
        else if (d_projected == d_bc)
            // pt_type = ipc::PointTriangleDistanceType::P_E1;
            pt_type = 4;
        else if (d_projected == d_ac)
            // pt_type = ipc::PointTriangleDistanceType::P_E2;
            pt_type = 5;
        else if (d_projected == d_a)
            // pt_type = ipc::PointTriangleDistanceType::P_T0;
            pt_type = 0;
        else if (d_projected == d_b)
            // pt_type = ipc::PointTriangleDistanceType::P_T1;
            pt_type = 1;
        else
            // pt_type = ipc::PointTriangleDistanceType::P_T2;
            pt_type = 2;
    }
    else
        // pt_type = ipc::PointTriangleDistanceType::P_T;
        pt_type = 6;
    // _pt_type = static_cast<cuda::std::underlying_type_t<ipc::PointTriangleDistanceType>>(pt_type);
    _pt_type = pt_type;
    return d;
}

__device__ __host__ luf compute_aabb(const Edgef& e, float d_hat_sqrt)
{
    vec3f l, u;
    l = make_float3(
        CUDA_MIN(e.e0.x, e.e1.x),
        CUDA_MIN(e.e0.y, e.e1.y),
        CUDA_MIN(e.e0.z, e.e1.z));
    u = make_float3(
        CUDA_MAX(e.e0.x, e.e1.x),
        CUDA_MAX(e.e0.y, e.e1.y),
        CUDA_MAX(e.e0.z, e.e1.z));
    return { l - d_hat_sqrt , u + d_hat_sqrt};
}

__device__ __host__ luf compute_aabb(const Facef& f, float d_hat_sqrt)
{
    vec3f l, u;
    l = make_float3 (
        CUDA_MIN3(f.t0.x, f.t1.x, f.t2.x),
        CUDA_MIN3(f.t0.y, f.t1.y, f.t2.y),
        CUDA_MIN3(f.t0.z, f.t1.z, f.t2.z));
    u = make_float3 (
        CUDA_MAX3(f.t0.x, f.t1.x, f.t2.x),
        CUDA_MAX3(f.t0.y, f.t1.y, f.t2.y),
        CUDA_MAX3(f.t0.z, f.t1.z, f.t2.z));
    return { l - d_hat_sqrt, u + d_hat_sqrt };
}


__host__ __device__ void cudaAffineBody::q_minus_qtiled(float3 dq[4])
{
    float h = dt_static;
    float h2 = h * h;
    for (int i = 0; i < 4; i++) {
        dq[i] = q_update[i] - (q0[i] + dqdt[i] * h);
        if (i == 0) dq[0] = dq[0] - make_float3(0.0f, -9.8f, 0.0f) * h2;
    }
}

__host__ __device__ float orthogonal_energy(float3 q[4], float dt) {
    float E = 0;
    for (int i = 1; i < 4; i ++)
        for (int j = 1; j < 4; j ++) {
            float t = dot(q[i], q[j]) - kronecker(i, j);
            E += t * t;
        }
    return E * kappa * dt * dt;
}

__host__ __device__ void orthogonal_grad(float3 q[4], float dt, float ret[12])
{
    auto h2 = dt * dt;
    ret[0] = ret[1] = ret[2] = 0.0f;
    for (int i = 1; i < 4; i++) {
        float3 g = make_float3(0.0f, 0.0f, 0.0f);
        for (int j = 1; j < 4; j++) {
            g = g + q[j] * (dot(q[i], q[j]) - kronecker(i, j));
        }

        g = g * (4 * kappa * h2);
        ret[i * 3 + 0] = g.x;
        ret[i * 3 + 1] = g.y;
        ret[i * 3 + 2] = g.z;
    }
}

__host__ __device__ void orthogonal_hess(float3 q[4], float dt, float ret[144])
{
    auto h2 = dt * dt;
    for (int i = 0; i < 144; i ++ ) ret[i]  = 0.0f;
    for (int i = 1; i < 4; i++)
        for (int j = 1; j < 4; j++) {
            float h[9]{ 0.0f };
            if (i == j) {

                for (int k = 1; k < 4; k++) {
                    float w = k == i ? 2.0f : 1.0f;
                    float _q[3]{ q[k].x, q[k].y, q[k].z };
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++) {
                            auto qii = _q[ii], qjj = _q[jj];
                            h[ii + jj * 3] += w * qii * qjj;
                        }
                }
                for (int ii = 1; ii < 4; ii++) {
                    h[ii * 4] += dot(q[i], q[i]) - 1.0f;
                }
            }
            else {
                float d = dot(q[i], q[j]);
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++) {
                        auto qjii = ii == 0 ? q[j].x : (ii == 1 ? q[j].y : q[j].z);
                        auto qijj = jj == 0 ? q[i].x : (jj == 1 ? q[i].y : q[i].z);
                        h[ii + jj * 3] = qjii * qijj + kronecker(ii, jj) * d;
                    }
            }
            auto scale = 4 * kappa * h2;
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++) {
                    ret[(i * 3 + ii) + (j * 3 + jj) * 12] = scale * h[ii + jj * 3];
                }
        }
}

__host__ __device__ float inertia(cudaAffineBody &c, float dt) {
    if (c.mass < 0) return 0.0f;
    float ret;

    float3 dq[4];
    c.q_minus_qtiled(dq);
    for (int i= 0; i < 4; i ++) {
        float w = 0.5 * (i == 0 ? c.mass : c.Ic);
        ret += w * dot(dq[i], dq[i]);
    }
    return ret + orthogonal_energy( c.q_update, dt);
}
__host__ __device__ void inertia_grad(cudaAffineBody& c, float dt, float ret[12])
{
    float3 g[4];
    c.q_minus_qtiled(g);
    for (int i = 0; i < 4; i++) {
        float w = i == 0 ? c.mass : c.Ic;
        ret[i * 3 + 0] += w * g[i].x;
        ret[i * 3 + 1] += w * g[i].y;
        ret[i * 3 + 2] += w * g[i].z;
    }
}

__host__ __device__ void inertia_hess(cudaAffineBody& c, float ret[144])
{
    for (int i = 0; i < 3; i++) {
        ret[i * 13] += c.mass;
    }
    for (int i = 3; i < 12; i++) {
        ret[i * 13] += c.Ic;
    }
}


__forceinline__ __host__ __device__ int rc12(int i, int j) {
    return j * 12+ i;
}
__forceinline__ __host__ __device__ int rc9(int i, int j) {
    return j * 9 + i;
}
__forceinline__ __host__ __device__ int rc6(int i, int j) {
    return j * 6 + i;
}

__host__ __device__ void put6(float *src6x6, i2 ij_src, float* dst12x12, i2 ij_dst) {
    int i = ij_src[0] * 3, j = ij_src[1] * 3;
    int ii = ij_dst[0] * 3, jj = ij_dst[1] * 3;
    for (int c = 0; c <3; c++) for (int r = 0; r < 3; r++) {
        dst12x12[rc12(ii + r, jj + c)] = src6x6[rc6(i + r, j + c)];
    }
}

__host__ __device__ void put9(float *src9x9, i2 ij_src, float* dst12x12, i2 ij_dst) {
    int i = ij_src[0] * 3, j = ij_src[1] * 3;
    int ii = ij_dst[0] * 3, jj = ij_dst[1] * 3;
    for (int c = 0; c <3; c++) for (int r = 0; r < 3; r++) {
        dst12x12[rc12(ii + r, jj + c)] = src9x9[rc9(i + r, j + c)];
    }
}

namespace dev {

__host__ __device__ float barrier_derivative_d(float x)
{
    if (x >= dev::d_hat)
        return 0.0f;
    return -(x - dev::d_hat) * dev::kappa * (2 * CUDA_LOG(x / dev::d_hat) + (x - dev::d_hat) / x) / (dev::d_hat * dev::d_hat);
}
__host__ __device__ float barrier_second_derivative(float d)
{
    if (d >= dev::d_hat)
        return 0.0f;
    return -dev::kappa * (2 * CUDA_LOG(d / dev::d_hat) + (d - dev::d_hat) / d + (d - dev::d_hat) * (2 / d + dev::d_hat / d / d)) / (dev::d_hat * dev::d_hat);
}

__host__ __device__ float barrier_function(float d)
{
    if (d >= dev::d_hat) return 0.0;
    return dev::kappa * -(d - dev::d_hat) * (d - dev::d_hat) * CUDA_LOG(d / dev::d_hat) / (dev::d_hat * dev::d_hat);
}

__host__ __device__ float point_triangle_distance(vec3f p, vec3f t0, vec3f t1, vec3f t2, int type) {

    // crude
    return vf_distance(p, {t0, t1, t2}, type);

    auto normal = cross(t1 - t0, t2 - t0);
    auto pt = p - t0;
    float a = dot(normal, pt);
    return a * a / dot(normal, normal);
}
__host__ __device__ void point_triangle_distance_gradient(vec3f p, vec3f t0, vec3f t1, vec3f t2, float* pt_grad, int type, float * local_grad)
{
    for (int i = 0; i < 12; i++) pt_grad[i] = 0.0f;
    switch (type) {
    case 0:
        // P_T0
        point_point_distance_gradient(p, t0, local_grad);
        for (int i = 0; i < 6; i++) pt_grad[i] = local_grad[i];
        break;
    case 1:
        point_point_distance_gradient(p, t1, local_grad);
        for (int i = 0; i < 3; i++) {
            pt_grad[i] = local_grad[i];
            pt_grad[i + 6] = local_grad[i + 3];
        }
        break;
    case 2:
        point_point_distance_gradient(p, t2, local_grad);
        for (int i = 0; i < 3; i++) {
            pt_grad[i] = local_grad[i];
            pt_grad[i + 9] = local_grad[i + 3];
        }
        break;
    case 3:
        // P_E0
        autogen::point_line_distance_gradient_3D(p, t0, t1, local_grad);
        for (int i = 0; i < 9; i++) {
            pt_grad[i] = local_grad[i];
        }
        break;
    case 4:
        // P_E1
        autogen::point_line_distance_gradient_3D(p, t1, t2, local_grad);
        for (int i = 0; i < 3; i++) {
            pt_grad[i] = local_grad[i];
            pt_grad[i + 6] = local_grad[i + 3];
            pt_grad[i + 9] = local_grad[i + 6];
        }
        break;
    case 5:
        // P_E2
        autogen::point_line_distance_gradient_3D(p, t2, t0, local_grad);
        for (int i = 0; i < 3; i++) {
            pt_grad[i] = local_grad[i];
            pt_grad[i + 3] = local_grad[i + 3];
            pt_grad[i + 9] = local_grad[i + 6];
        }
        break;
    case 6:
        // P_T
        autogen::point_plane_distance_gradient(
            p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t2.x,
            t2.y, t2.z, pt_grad);
        break;
    case 7:
        // AUTO
        //type = point_triangle_distance_type(p, t0, t1, t2);
        //point_triangle_distance_gradient(p, t0, t1, t2, pt_grad, type);
        //break;
        printf("error: auto not supported");
    }
}

__host__ __device__ void point_triangle_distance_hessian(vec3f p, vec3f t0, vec3f t1, vec3f t2, float* pt_hess, int type, float *local_hess)
{
    for (int i = 0; i < 144; i++) pt_hess[i] = 0.0f;

    switch ( type)
    {
    case 0:
        // P_T0
        point_point_distance_hessian(p, t0, local_hess);

        put6(local_hess, {0,0}, pt_hess, {0,0});
        put6(local_hess, {0,1}, pt_hess, {0,1});
        put6(local_hess, {1,0}, pt_hess, {1,0});
        put6(local_hess, {1,1}, pt_hess, {1,1});

        break;
    case 1:
        // P_T1
        point_point_distance_hessian(p, t1, local_hess);

        put6(local_hess, {0,0}, pt_hess, {0,0});
        put6(local_hess, {0,1}, pt_hess, {0,2});
        put6(local_hess, {1,0}, pt_hess, {2,0});
        put6(local_hess, {1,1}, pt_hess, {2,2});
        break;
    case 2:
        // P_T2
        point_point_distance_hessian(p, t2, local_hess);
        put6(local_hess, {0,0}, pt_hess, {0,0});
        put6(local_hess, {0,1}, pt_hess, {0,3});
        put6(local_hess, {1,0}, pt_hess, {3,0});
        put6(local_hess, {1,1}, pt_hess, {3,3});
        break;  
    case 3: 
        // P_E0
        autogen::point_line_distance_hessian_3D(p, t0, t1, local_hess);
        put9(local_hess, {0,0}, pt_hess, {0,0});
        put9(local_hess, {0,1}, pt_hess, {0,1});
        put9(local_hess, {0,2}, pt_hess, {0,2});
        put9(local_hess, {1,0}, pt_hess, {1,0});
        put9(local_hess, {1,1}, pt_hess, {1,1});
        put9(local_hess, {1,2}, pt_hess, {1,2});
        put9(local_hess, {2,0}, pt_hess, {2,0});
        put9(local_hess, {2,1}, pt_hess, {2,1});
        put9(local_hess, {2,2}, pt_hess, {2,2});
        break;  
    case 4: 
        // P_E1
        autogen::point_line_distance_hessian_3D(p, t1, t2, local_hess);
        put9(local_hess, {0,0}, pt_hess, {0,0});
        put9(local_hess, {0,1}, pt_hess, {0,2});
        put9(local_hess, {0,2}, pt_hess, {0,3});
        put9(local_hess, {1,0}, pt_hess, {2,0});
        put9(local_hess, {1,1}, pt_hess, {2,2});
        put9(local_hess, {1,2}, pt_hess, {2,3});
        put9(local_hess, {2,0}, pt_hess, {3,0});
        put9(local_hess, {2,1}, pt_hess, {3,2});
        put9(local_hess, {2,2}, pt_hess, {3,3});
        break;
    case 5: 
        // P_E2
        autogen::point_line_distance_hessian_3D(p, t2, t0, local_hess);
        put9(local_hess, {0,0}, pt_hess, {0,0});
        put9(local_hess, {0,1}, pt_hess, {0,3});
        put9(local_hess, {0,2}, pt_hess, {0,1});
        put9(local_hess, {1,0}, pt_hess, {3,0});
        put9(local_hess, {1,1}, pt_hess, {3,3});
        put9(local_hess, {1,2}, pt_hess, {3,1});
        put9(local_hess, {2,0}, pt_hess, {1,0});
        put9(local_hess, {2,1}, pt_hess, {1,3});
        put9(local_hess, {2,2}, pt_hess, {1,1});
        break;
    case 6: 
        // P_T
        autogen::point_plane_distance_hessian(
            p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t2.x,
            t2.y, t2.z, pt_hess);
        break;
    case 7: 
        printf("error: not supposed to support auto");
        //// AUTO
        //type = point_triangle_distance_type(p, t0, t1, t2);
        //point_triangle_distance_hessian(p, t0, t1, t2, pt_hess, type);
        //break;
    }
}

__host__ __device__ int point_triangle_distance_type(vec3f p, vec3f t0, vec3f t1, vec3f t2)
{
    int type = 0;
    vf_distance(p, { t0, t1, t2 }, type);
    return type;
    // crude
}

__host__ __device__ void point_point_distance_gradient(vec3f p, vec3f q, float* pp_grad)
{
    // no autogen this time
    auto t = (p - q) * 2.0f;
    pp_grad[0] = t.x;
    pp_grad[1] = t.y;
    pp_grad[2] = t.z;
    pp_grad[3] = -t.x;
    pp_grad[4] = -t.y;
    pp_grad[5] = -t.z;
}
__host__ __device__ void point_point_distance_hessian(vec3f p, vec3f q, float* pp_hess)
{
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            pp_hess[i + j * 6] = i == j ? i < 3 ? 2.0f : -2.0f : 0.0f;
            // column major
        }
}

__host__ __device__ int edge_edge_distance_type(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1)
{

    
    // EA0_EB0, ///< The edges are closest at vertex 0 of edge A and 0 of edge B.
    // EA0_EB1, ///< The edges are closest at vertex 0 of edge A and 1 of edge B.
    // EA1_EB0, ///< The edges are closest at vertex 1 of edge A and 0 of edge B.
    // EA1_EB1, ///< The edges are closest at vertex 1 of edge A and 1 of edge B.
    // /// The edges are closest at the interior of edge A and vertex 0 of edge B.
    // EA_EB0,
    // /// The edges are closest at the interior of edge A and vertex 1 of edge B.
    // EA_EB1,
    // /// The edges are closest at vertex 0 of edge A and the interior of edge B.
    // EA0_EB,
    // /// The edges are closest at vertex 1 of edge A and the interior of edge B.
    // EA1_EB,
    // EA_EB, ///< The edges are closest at an interior point of edge A and B.
    // AUTO   ///< Automatically determine the closest pair.

    auto u = ea1 - ea0;
    auto v = eb1 - eb0;
    auto w = ea0 - eb0;
    auto a = dot(u, u);
    auto b = dot(u, v);
    auto c = dot(v, v);
    auto d = dot(u, w);
    auto e = dot(v, w);
    auto D = a * c - b * b;
    auto tD = D;

    int default_case = 8;

    float sN = (b * e - c * d), tN;
    if (sN <= 0.0f) {
        tN = e;
        tD = c;
        default_case = 6;
    } else if (sN >= D) {
        tN = e + b;
        tD = c;
        default_case = 7;
    } else {
        tN = a * e - b * d;
        auto tmp = cross(u, v);
        if (tN > 0.0 && tN < tD && dot(tmp, tmp) < 1e-20f * a * c) {
            // avoid nearly parallel edge-edge
            if (sN  < D / 2) {
                tN = e;
                tD = c;
                default_case = 6;
            } else {
                tN = e + b;
                tD = c;
                default_case = 7;
            }
        }
    }

    if (tN<= 0.0f) {
        if (-d <= 0.0f) {
            return 0;
        }
        else if (-d >= a) {
            return 2;
        }
        else {
            return 4;
        }
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0f) {
            return 1;
        } else if ((-d + b) >= a) {
            return 3;
        } else {
            return 5;
        }
    }
    return default_case;
}

__host__ __device__ float edge_edge_distance(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, int type) {
    float t;
    switch (type) {
    case 0: // EA0_EB0
        return dot(ea0 - eb0, ea0 - eb0);
        case 1: // EA0_EB1
        return dot(ea0 - eb1, ea0 - eb1);
        case 2: // EA1_EB0
        return dot(ea1 - eb0, ea1 - eb0);
        case 3: // EA1_EB1
        return dot(ea1 - eb1, ea1 - eb1);
        case 4: // EA_EB0
        t = h(ea0, ea1, eb0);
        return t * t;
        case 5: // EA_EB1
        t = h(ea0, ea1, eb1);
        return t * t;
        case 6: // EA0_EB
        t=  h(eb0, eb1, ea0);
        return t * t;

        case 7: // EA1_EB
        t =  h(eb0, eb1, ea1);
        return t * t;

        case 8: // EA_EB
        auto normal = cross(ea1 - ea0, eb1 - eb0);
        t = dot(normal, ea0 - eb0);
        return t * t / dot(normal, normal);
        case 9: // AUTO
        printf("ee distance error: auto not supported");
    }
}

__host__ __device__ void edge_edge_distance_gradient(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float* ee_grad, int type, float *local_grad) {
    for(int i = 0; i < 12; i++) {
        local_grad[i] = 0.0f;
    }
    switch (type) {
        case 0: // EA0_EB0
        point_point_distance_gradient(ea0, eb0, local_grad);
        for (int i = 0; i <3;i ++) {
            ee_grad[i] = local_grad[i];
            ee_grad[i + 6] = local_grad[i + 3];
        }
        break;
        case 1: // EA0_EB1
        point_point_distance_gradient(ea0, eb1, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i] = local_grad[i];
            ee_grad[i + 9] = local_grad[i + 3];
        }
        break;
        case 2: // EA1_EB0
        point_point_distance_gradient(ea1, eb0, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i + 3] = local_grad[i];
            ee_grad[i + 6] = local_grad[i + 3];
        }
        break;
        case 3: // EA1_EB1
        point_point_distance_gradient(ea1, eb1, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i + 3] = local_grad[i];
            ee_grad[i + 9] = local_grad[i + 3];
        }
        break;
        case 4: // EA_EB0
        autogen::point_line_distance_gradient_3D(eb0, ea0, ea1, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i + 6] = local_grad[i];
            ee_grad[i] = local_grad[i + 3];
            ee_grad[i + 3] = local_grad[i  + 6];
        }
        break;
        case 5: // EA_EB1
        autogen::point_line_distance_gradient_3D(eb1, ea0, ea1, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i + 9] = local_grad[i];
            ee_grad[i] = local_grad[i + 3];
            ee_grad[i + 3] = local_grad[i  + 6];
        }
        break;
        case 6: // EA0_EB
        autogen::point_line_distance_gradient_3D(ea0, eb0, eb1, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i] = local_grad[i];
            ee_grad[i + 6] = local_grad[i + 3];
            ee_grad[i + 9] = local_grad[i  + 6];
        }
        break;
        case 7: // EA1_EB
        autogen::point_line_distance_gradient_3D(ea1, eb0, eb1, local_grad);
        for (int i = 0; i < 3; i ++) {
            ee_grad[i + 3] = local_grad[i];
            ee_grad[i + 6] = local_grad[i + 3];
            ee_grad[i + 9] = local_grad[i  + 6];
        }
        break;
        case 8: // EA_EB
        autogen::line_line_distance_gradient( ea0, ea1, eb0, eb1, ee_grad);
        break;
        case 9: // AUTO
        printf("ee distance error: auto not supported");

    }
}
__host__ __device__ void edge_edge_distance_hessian(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float* ee_hess, int type, float *local_hess) {
    for(int i = 0; i < 144; i++) {
        local_hess[i] = 0.0f;
    }
    switch (type) {
    case 0: // EA0_EB0
        point_point_distance_hessian(ea0, eb0, local_hess);
        put6(local_hess, { 0, 0 }, ee_hess, { 0, 0 });
        put6(local_hess, { 0, 1 }, ee_hess, { 0, 2 });
        put6(local_hess, { 1, 0 }, ee_hess, { 2, 0 });
        put6(local_hess, { 1, 1 }, ee_hess, { 2, 2 });
        break;
    case 1: // EA0_EB1
        point_point_distance_hessian(ea0, eb1, local_hess);
        put6(local_hess, { 0, 0 }, ee_hess, { 0, 0 });
        put6(local_hess, { 0, 1 }, ee_hess, { 0, 3 });
        put6(local_hess, { 1, 0 }, ee_hess, { 3, 0 });
        put6(local_hess, { 1, 1 }, ee_hess, { 3, 3 });

        break;
    case 2: // EA1_EB0
        point_point_distance_hessian(ea1, eb0, local_hess);
        put6(local_hess, { 0, 0 }, ee_hess, { 1, 1 });
        put6(local_hess, { 0, 1 }, ee_hess, { 1, 2 });
        put6(local_hess, { 1, 0 }, ee_hess, { 2, 1 });
        put6(local_hess, { 1, 1 }, ee_hess, { 2, 2 });
        break;
    case 3: // EA1_EB1
        point_point_distance_hessian(ea1, eb1, local_hess);
        put6(local_hess, { 0, 0 }, ee_hess, { 1, 1 });
        put6(local_hess, { 0, 1 }, ee_hess, { 1, 3 });
        put6(local_hess, { 1, 0 }, ee_hess, { 3, 1 });
        put6(local_hess, { 1, 1 }, ee_hess, { 3, 3 });
        break;
    case 4: // EA_EB0
        autogen::point_line_distance_hessian_3D(eb0, ea0, ea1, local_hess);
        put9(local_hess, { 0, 0 }, ee_hess, { 2, 2 });
        put9(local_hess, { 0, 1 }, ee_hess, { 2, 0 });
        put9(local_hess, { 0, 2 }, ee_hess, { 2, 1 });
        put9(local_hess, { 1, 0 }, ee_hess, { 0, 2 });
        put9(local_hess, { 1, 1 }, ee_hess, { 0, 0 });
        put9(local_hess, { 1, 2 }, ee_hess, { 0, 1 });
        put9(local_hess, { 2, 0 }, ee_hess, { 1, 2 });
        put9(local_hess, { 2, 1 }, ee_hess, { 1, 0 });
        put9(local_hess, { 2, 2 }, ee_hess, { 1, 1 });
        break;
    case 5: // EA_EB1
        autogen::point_line_distance_hessian_3D(eb1, ea0, ea1, local_hess);
        put9(local_hess, { 0, 0 }, ee_hess, { 3, 3 });
        put9(local_hess, { 0, 1 }, ee_hess, { 3, 0 });
        put9(local_hess, { 0, 2 }, ee_hess, { 3, 1 });
        put9(local_hess, { 1, 0 }, ee_hess, { 0, 3 });
        put9(local_hess, { 1, 1 }, ee_hess, { 0, 0 });
        put9(local_hess, { 1, 2 }, ee_hess, { 0, 1 });
        put9(local_hess, { 2, 0 }, ee_hess, { 1, 3 });
        put9(local_hess, { 2, 1 }, ee_hess, { 1, 0 });
        put9(local_hess, { 2, 2 }, ee_hess, { 1, 1 });
        break;
    case 6: // EA0_EB
        autogen::point_line_distance_hessian_3D(ea0, eb0, eb1, local_hess);
        put9(local_hess, { 0, 0 }, ee_hess, { 0, 0 });
        put9(local_hess, { 0, 1 }, ee_hess, { 0, 2 });
        put9(local_hess, { 0, 2 }, ee_hess, { 0, 3 });
        put9(local_hess, { 1, 0 }, ee_hess, { 2, 0 });
        put9(local_hess, { 1, 1 }, ee_hess, { 2, 2 });
        put9(local_hess, { 1, 2 }, ee_hess, { 2, 3 });
        put9(local_hess, { 2, 0 }, ee_hess, { 3, 0 });
        put9(local_hess, { 2, 1 }, ee_hess, { 3, 2 });
        put9(local_hess, { 2, 2 }, ee_hess, { 3, 3 });

        break;
    case 7: // EA1_EB
        autogen::point_line_distance_hessian_3D(ea1, eb0, eb1, local_hess);
        put9(local_hess, { 0, 0 }, ee_hess, { 1, 1 });
        put9(local_hess, { 0, 1 }, ee_hess, { 1, 2 });
        put9(local_hess, { 0, 2 }, ee_hess, { 1, 3 });
        put9(local_hess, { 1, 0 }, ee_hess, { 2, 1 });
        put9(local_hess, { 1, 1 }, ee_hess, { 2, 2 });
        put9(local_hess, { 1, 2 }, ee_hess, { 2, 3 });
        put9(local_hess, { 2, 0 }, ee_hess, { 3, 1 });
        put9(local_hess, { 2, 1 }, ee_hess, { 3, 2 });
        put9(local_hess, { 2, 2 }, ee_hess, { 3, 3 });
        break;
    case 8: // EA_EB
        autogen::line_line_distance_hessian(ea0, ea1, eb0, eb1, ee_hess);
        break;
    case 9: // AUTO
        printf("ee hess error: AUTO not implemented\n");
    }
}


__host__ __device__ float edge_edge_cross_squarednorm(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1) {
    auto ea = ea1 - ea0;
    auto eb = eb1 - eb0;
    auto ea_cross_eb = cross(ea, eb);
    return dot(ea_cross_eb, ea_cross_eb);
}

__host__ __device__ float edge_edge_mollifier(float x, float eps_x)    {
    float x_div_eps_x = x / eps_x;
    return (-x_div_eps_x + 2.0) * x_div_eps_x;
}

__host__ __device__ float edge_edge_mollifier_gradient(float x, float eps_x)
{
    float one_div_eps_x = 1.0 / eps_x;
    return 2.0 * one_div_eps_x * (-one_div_eps_x * x + 1.0);
}

__host__ __device__ float edge_edge_mollifier_hessian(float x, float eps_x)
{
    return -2.0 / (eps_x * eps_x);
}

__host__ __device__ float edge_edge_mollifier(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float eps_x) {
    auto ee_cross_norm_sqr = edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
    if (ee_cross_norm_sqr < eps_x) {
        return edge_edge_mollifier(ee_cross_norm_sqr, eps_x);
    } else {
        return 1.0f;
    }
}

__host__ __device__ void edge_edge_mollifier_gradient(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float eps_x, float *grad) {
    auto ee_cross_norm_sqr = edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
    if (ee_cross_norm_sqr < eps_x) {
        autogen::edge_edge_cross_squarednorm_gradient(ea0, ea1, eb0, eb1, grad);
        auto scale = edge_edge_mollifier_gradient(ee_cross_norm_sqr, eps_x);
        for (int i = 0; i < 12; i ++) grad[i] *= scale;
    } else {
        for (int i = 0; i < 12; i ++) grad[i] = 0.0f;
    }
}

__host__ __device__ void edge_edge_mollifier_hessian(vec3f ea0, vec3f ea1, vec3f eb0, vec3f eb1, float eps_x, float *grad_input, float *hess) {
    auto ee_cross_norm_sqr = edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
    if (ee_cross_norm_sqr < eps_x) {

        autogen::edge_edge_cross_squarednorm_hessian(ea0, ea1, eb0, eb1, hess);
        auto scale = edge_edge_mollifier_gradient(ee_cross_norm_sqr, eps_x);
        for (int i = 0; i < 144; i ++) hess[i] *= scale;

        scale = edge_edge_mollifier_hessian(ee_cross_norm_sqr, eps_x);

        for (int I = 0; I < 144; I++) {
            int i = i % 12, j = I / 12;
            hess[i] += scale * grad_input[i] * grad_input[j];
        }
    } else {
        for (int i = 0; i < 144; i ++) hess[i] = 0.0f;
    }
}
} // namespace dev