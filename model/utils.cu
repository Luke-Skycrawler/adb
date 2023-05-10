#include "cuda_globals.cuh"
#include "autogen/autogen.cuh"

static __constant__ float kappa = 1e9f, dt = 1e-2f;


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
__device__ luf affine(luf aabb, cudaAffineBody& c, int vtn)
{
    vec3f cull[8];
    vec3f l, u;
    auto q{ vtn == 2 ? c.q_update : c.q };
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
    return { l, u };
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
    return { l, u };
}


__host__ __device__ float dev::barrier_function(float d)
{
    if (d >= dev::d_hat) return 0.0;
    return dev::kappa * -(d - dev::d_hat) * (d - dev::d_hat) * log(d / dev::d_hat) / (dev::d_hat * dev::d_hat);
}

//__device__ void cudaAffineBody::q_minus_qtiled(float3 dq[4])
//{
//    __constant__ float  h =  cuda_globals->dt;
//    __constant__ float h2 = h * h;
//    for (int i = 0; i < 4; i++) {
//        dq[i] = q_update[i] - (q[i] + dqdt[i] * h + cuda_globals->gravity * h2);
//    }
//}
__host__ __device__ void cudaAffineBody::q_minus_qtiled(float3 dq[4])
{
    float h = dt;
    float h2 = h * h;
    for (int i = 0; i < 4; i++) {
        dq[i] = q_update[i] - (q0[i] + dqdt[i] * h);
        if (i == 0) dq[0] = dq[0] - make_float3(0.0f, -9.8f, 0.0f) * h2;
    }
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
namespace dev {
//__device__ __constant__ float kappa = 1e-1f, d_hat = 1e-4f, d_hat_sqr = 1e-2f;

__host__ __device__ float barrier_derivative_d(float x)
{
    if (x >= d_hat)
        return 0.0f;
    return -(x - d_hat) * kappa * (2 * log(x / d_hat) + (x - d_hat) / x) / (d_hat * d_hat);
}
__host__ __device__ float barrier_second_derivative(float d)
{
    if (d >= d_hat)
        return 0.0f;
    return -kappa * (2 * log(d / d_hat) + (d - d_hat) / d + (d - d_hat) * (2 / d + d_hat / d / d)) / (d_hat * d_hat);
}

__device__ float point_triangle_distance(vec3f p, vec3f t0, vec3f t1, vec3f t2) {return 0.0f;}
__device__ void point_triangle_distance_gradient(vec3f p, vec3f t0, vec3f t1, vec3f t2, float *pt_grad) {
    autogen::point_plane_distance_gradient(
        p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t2.x,
        t2.y, t2.z, pt_grad);
}

__device__ void point_triangle_distance_hessian(vec3f p, vec3f t0, vec3f t1, vec3f t2, float *pt_hess){
    autogen::point_plane_distance_hessian(
        p.x, p.y, p.z, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t2.x,
        t2.y, t2.z, pt_hess);
}
}

