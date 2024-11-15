#define CUDA_SOURCE
#include "bounds3.h"

using namespace Eigen;
namespace cuda {
using namespace cuda;
func int cubic_roots(scalar roots[3], scalar coef[4], scalar x0, scalar x1);
    
inline func scalar area(const vec3& a, const vec3& b, const vec3& c)
{
    return (c - a).cross(b - a).norm();
}

void func cubic_binomial(const scalar a[3], const scalar b[3], scalar polynomial[4])
{
    // for (int i = 0; i < 2; i++)
    //     for (int j = 0; j < 2; j++)
    //         for (int k = 0; k < 2; k++) {
    //             scalar c11 = i ? a[0] : b[0];
    //             scalar c22 = j ? a[1] : b[1];
    //             scalar c33 = k ? a[2] : b[2];
    //             // int I = (i<< 2) + (j << 1) + k;
    //             scalar t = c11 * c22 * c33;
    //             int J = i + k + j;
    //             polynomial[J] += t;
    //         }
    polynomial[0] += b[0] * b[1] * b[2];
    polynomial[1] += a[0] * b[1] * b[2] + b[0] * b[1] * a[2] + b[0] * a[1] * b[2];
    polynomial[2] += a[0] * a[1] * b[2] + b[0] * a[1] * a[2] + a[0] * b[1] * a[2];
    polynomial[3] += a[0] * a[1] * a[2];
}

func Vector<scalar, 4> det_polynomial(const mat3& a, const mat3& b)
{
    scalar pos_polynomial[4]{ 0.0 }, neg_polynomial[4]{ 0.0 };
    scalar c11c22c33[2][3]{
        { a(0, 0), a(1, 1), a(2, 2) },
        { b(0, 0), b(1, 1), b(2, 2) }
    },
        c12c23c31[2][3]{
            { a(0, 1), a(1, 2), a(2, 0) },
            { b(0, 1), b(1, 2), b(2, 0) }
        },
        c13c21c32[2][3]{
            { a(0, 2), a(1, 0), a(2, 1) },
            { b(0, 2), b(1, 0), b(2, 1) }
        };
    scalar c11c23c32[2][3]{
        { a(0, 0), a(1, 2), a(2, 1) }, { b(0, 0), b(1, 2), b(2, 1) }
    },
        c12c21c33[2][3]{
            { a(0, 1), a(1, 0), a(2, 2) }, { b(0, 1), b(1, 0), b(2, 2) }
        },
        c13c22c31[2][3]{
            { a(0, 2), a(1, 1), a(2, 0) }, { b(0, 2), b(1, 1), b(2, 0) }
        };
    cubic_binomial(
        c11c22c33[0],
        c11c22c33[1],
        pos_polynomial);
    cubic_binomial(
        c12c23c31[0],
        c12c23c31[1],
        pos_polynomial);
    cubic_binomial(
        c13c21c32[0],
        c13c21c32[1],
        pos_polynomial);
    cubic_binomial(
        c11c23c32[0],
        c11c23c32[1],
        neg_polynomial);
    cubic_binomial(
        c12c21c33[0],
        c12c21c33[1],
        neg_polynomial);
    cubic_binomial(
        c13c22c31[0],
        c13c22c31[1],
        neg_polynomial);
    Vector<scalar, 4> ret;
    for(int i = 0; i < 4; i++) ret(i) = pos_polynomial[i] - neg_polynomial[i];
    return ret;
}

func bool inside(const Face& f, const vec3& v)
{
    scalar a1 = area(f.t1, f.t0, f.t2);
    scalar a2 = area(f.t0, f.t1, v) + area(f.t1, f.t2, v) + area(f.t2, f.t0, v);
    return a2 <= a1 + 1e-8;
}

func bool verify_root_pt(const vec3& _v, const Face& f)
{
    auto n = f.unit_normal();
    scalar d = n.dot(_v - f.t0);

    vec3 v = _v - d * n;
    d = d * d;
    return inside(f, v);
}

func bool verify_root_ee(
    const Edge& ei,
    const Edge& ej)
{
    const auto cross = [](const Edge& ei, const Edge& ej) {
        vec3 vei{ ei.e1 - ei.e0 };
        vec3 vej0{ ej.e0 - ei.e0 };
        vec3 vej1{ ej.e1 - ei.e0 };
        return vei.cross(vej0).dot(vei.cross(vej1)) < 0.0;
    };
    return cross(ei, ej) && cross(ej, ei);
}

inline func vec3 linerp(const vec3& p_t1, const vec3& p_t0, scalar t)
{
    return t * p_t1 + (1 - t) * p_t0;
}
Face func linerp(const Face& t_t1, const Face& t_t0, scalar t)
{
    return Face{
        linerp(t_t1.t0, t_t0.t0, t),
        linerp(t_t1.t1, t_t0.t1, t),
        linerp(t_t1.t2, t_t0.t2, t),
    };
}
Edge func linerp(const Edge& e_t1, const Edge& e_t0, scalar t)
{
    return Edge{
        linerp(e_t1.e0, e_t0.e0, t),
        linerp(e_t1.e1, e_t0.e1, t),
    };
}

func int build_and_solve_4_points_coplanar(
    const vec3& p0_t0,
    const vec3& p1_t0,
    const vec3& p2_t0,
    const vec3& p3_t0,

    const vec3& p0_t1,
    const vec3& p1_t1,
    const vec3& p2_t1,
    const vec3& p3_t1,

    scalar roots[3])
{
    mat3 a1, a2, a3, a4;
    mat3 b1, b2, b3, b4;

    b1 << p1_t0, p2_t0, p3_t0;
    b2 << p0_t0, p2_t0, p3_t0;
    b3 << p0_t0, p1_t0, p3_t0;
    b4 << p0_t0, p1_t0, p2_t0;

    a1 << p1_t1, p2_t1, p3_t1;
    a2 << p0_t1, p2_t1, p3_t1;
    a3 << p0_t1, p1_t1, p3_t1;
    a4 << p0_t1, p1_t1, p2_t1;

    a1 -= b1;
    a2 -= b2;
    a3 -= b3;
    a4 -= b4;

    Vector<scalar, 4> t = det_polynomial(a1, b1) - det_polynomial(a2, b2) + det_polynomial(a3, b3) - det_polynomial(a4, b4);
    scalar root = 1.0;
    int found = cuda::cubic_roots(roots, t.data(), static_cast<scalar>(0.0), static_cast<scalar>(1.0));
    return found;
}

func scalar pt_collision_time(
    const vec3& p0,
    const Face& t0,

    const vec3& p1,
    const Face& t1)
{
    scalar roots[3];
    int found = build_and_solve_4_points_coplanar(
        p0, t0.t0, t0.t1, t0.t2,
        p1, t1.t0, t1.t1, t1.t2,
        roots);
    bool true_root = false;
    scalar root = 1.0;
    for(int i = 0; i < found && !true_root; i++) {
        root = roots[i];
        true_root = verify_root_pt(linerp(p1, p0, root), linerp(t1, t0, root));
    }
    return found && true_root ? root : 1.0;
}

func scalar ee_collision_time(
    const Edge& ei0,
    const Edge& ej0,
    const Edge& ei1,
    const Edge& ej1)
{
    scalar roots[3];
    int found = build_and_solve_4_points_coplanar(
        ei0.e0, ei0.e1, ej0.e0, ej0.e1,
        ei1.e0, ei1.e1, ej1.e0, ej1.e1,
        roots);
    scalar root = 1.0;
    bool true_root = false;
    for(int i = 0; i < found && !true_root; i++) {
        root = roots[i];
        true_root = verify_root_ee(linerp(ei1, ei0, root), linerp(ej1, ej0, root));
    }
    return found && true_root ? root : 1.0;
}

func scalar pt_collision_time(
    vec3 p0_t0,
    vec3 t0_t0,
    vec3 t1_t0,
    vec3 t2_t0,
    vec3 p0_t1,
    vec3 t0_t1,
    vec3 t1_t1,
    vec3 t2_t1)

{
    scalar roots[3];
    int found = build_and_solve_4_points_coplanar(
        p0_t0, t0_t0, t1_t0, t2_t0,
        p0_t1, t0_t1, t1_t1, t2_t1,
        roots);
    bool true_root = false;
    scalar root = 1.0;
    for(int i = 0; i < found && !true_root; i++) {
        root = roots[i];
        Face t0{ t0_t0, t1_t0, t2_t0 }, t1{ t0_t1, t1_t1, t2_t1 };
        true_root = verify_root_pt(linerp(p0_t1, p0_t0, root), linerp(t1, t0, root));
    }
    return found && true_root ? root : 1.0;
}

func scalar ee_collision_time(
    vec3 ei0_t0,
    vec3 ei1_t0,
    vec3 ej0_t0,
    vec3 ej1_t0,
    vec3 ei0_t1,
    vec3 ei1_t1,
    vec3 ej0_t1,
    vec3 ej1_t1)
{
    scalar roots[3];
    int found = build_and_solve_4_points_coplanar(
        ei0_t0, ei1_t0, ej0_t0, ej1_t0,
        ei0_t1, ei1_t1, ej0_t1, ej1_t1,
        roots);
    scalar root = 1.0;
    bool true_root = false;
    for(int i = 0; i < found && !true_root; i++) {
        root = roots[i];
        Edge ei0{ ei0_t0, ei1_t0 }, ei1{ ei0_t1, ei1_t1 }, ej0{ ej0_t0, ej1_t0 }, ej1{ ej0_t1, ej1_t1 };
        true_root = verify_root_ee(linerp(ei1, ei0, root), linerp(ej1, ej0, root));
    }
    return found && true_root ? root : 1.0;
}

__global__ void ccd_toi(int nvi, int nfj, int* vilist, int* fjlist, lu* viaabbs, lu* fjaabbs, vec3* v0s, vec3* v1s, Face* f0s, Face* f1s, scalar* toi)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x, int j = threadIdx.y + blockDim.y * blockIdx.y;
    int vi = vilist[i], fj = fjlist[j];
    if(i < nvi && j < nfj) {

        if(intersects(viaabbs[i], fjaabbs[j])) {
            auto &v0{ v0s[i] }, &v{ v1s[i] };
            auto &f0{ f0s[j] }, &f{ f1s[j] };
            scalar t = pt_collision_time(v0, f0, v, f);
            atomic_min<scalar>(toi, t);
        }
    }
}
scalar cuda_pt_list_toi(int nvi, int nfj, int* vilist, int* fjlist, lu* viaabbs, lu* fjaabbs, vec3* v0s, vec3* v1s, Face* f0s, Face* f1s)
{
    scalar* toi;

    int *dev_vilist, *dev_fjlist;
    lu *dev_viaabbs, *dev_fjaabbs;
    vec3 *dev_v0s, *dev_v1s;
    Face *dev_f0s, *dev_f1s;
    cudaMalloc(&dev_vilist, sizeof(int) * nvi);
    cudaMalloc(&dev_fjlist, sizeof(int) * nfj);
    cudaMalloc(&dev_viaabbs, sizeof(lu) * nvi);
    cudaMalloc(&dev_fjaabbs, sizeof(lu) * nfj);
    cudaMalloc(&dev_v0s, sizeof(vec3) * nvi);
    cudaMalloc(&dev_v1s, sizeof(vec3) * nvi);
    cudaMalloc(&dev_f0s, sizeof(Face) * nfj);
    cudaMalloc(&dev_f1s, sizeof(Face) * nfj);
    cudaMalloc(&toi, sizeof(scalar));

    cudaMemcpy(dev_vilist, vilist, sizeof(int) * nvi, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fjlist, fjlist, sizeof(int) * nfj, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_viaabbs, viaabbs, sizeof(lu) * nvi, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fjaabbs, fjaabbs, sizeof(lu) * nfj, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v0s, v0s, sizeof(vec3) * nvi, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v1s, v1s, sizeof(vec3) * nvi, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f0s, f0s, sizeof(Face) * nfj, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_f1s, f1s, sizeof(Face) * nfj, cudaMemcpyHostToDevice);
    scalar* ret = new scalar;
    *ret = 1.0;
    cudaMemcpy(toi, ret, sizeof(scalar), cudaMemcpyHostToDevice);
    int gx, gy, bx, by;
    if(nvi > 32) {
        gx = (nvi + 31) / 32;
        bx = 32;
    }
    else {
        gx = 1;
        bx = nvi;
    }
    if(nfj > 32) {
        gy = (nfj + 31) / 32;
        by = 32;
    }
    else {
        gy = 1;
        by = nfj;
    }
    dim3 grid_dim(gx, gy), block_dim(bx, by);
    ccd_toi<<<grid_dim, block_dim>>>(nvi, nfj, dev_vilist, dev_fjlist, dev_viaabbs, dev_fjaabbs, dev_v0s, dev_v1s, dev_f0s, dev_f1s, toi);
    cudaMemcpy(ret, toi, sizeof(scalar), cudaMemcpyDeviceToHost);

    cudaFree(dev_vilist);
    cudaFree(dev_fjlist);
    cudaFree(dev_viaabbs);
    cudaFree(dev_fjaabbs);
    cudaFree(dev_v0s);
    cudaFree(dev_v1s);
    cudaFree(dev_f0s);
    cudaFree(dev_f1s);
    cudaFree(toi);
    return *ret;
}
};
