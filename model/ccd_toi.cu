#include "cuda_header.cuh"



inline __host__ __device__ vec3f linerp(vec3f pt1, vec3f pt0, float t){
    return pt1 * t + pt0 * (1.0f - t);
}
inline __host__ __device__ Facef linerp(Facef tt1, Facef tt0, float t) {
    return Facef {
        linerp(tt1.t0, tt0.t0, t),
        linerp(tt1.t1, tt0.t1, t),
        linerp(tt1.t2, tt0.t2, t),
    };
}
inline __host__ __device__ Edgef linerp(Edgef et1, Edgef et0, float t) {
    return Edgef {
        linerp(et1.e0, et0.e0, t),
        linerp(et1.e1, et0.e1, t),
    };
}

__host__ __device__ void cubic_binomial (float a[3], float b[3], float ret_polynomial[4]){
    ret_polynomial[0] += b[0] * b[1] * b[2];
    ret_polynomial[1] += a[0] * b[1] * b[2] + b[0] * b[1] * a[2] + b[0] * a[1] * b[2];
    ret_polynomial[2] += a[0] * a[1] * b[2] + b[0] * a[1] * a[2] + a[0] * b[1] * a[2];
    ret_polynomial[3] += a[0] * a[1] * a[2];
}

__forceinline__ __host__ __device__ int rc_3(int i, int j) {
    return 3 * j + i;
}
__host__ __device__ void det_polynomial(
    // const mat3& a, const mat3& b
    float *a, float *b, float *ret_polynomial
)
{
    float pos_polynomial[4]{ 0.0 }, neg_polynomial[4]{ 0.0 };
    float c11c22c33[2][3]{
        { a[rc_3(0, 0)], a[rc_3(1, 1)], a[rc_3(2, 2)] },
        { b[rc_3(0, 0)], b[rc_3(1, 1)], b[rc_3(2, 2)] }
    },
        c12c23c31[2][3]{
            { a[rc_3(0, 1)], a[rc_3(1, 2)], a[rc_3(2, 0)] },
            { b[rc_3(0, 1)], b[rc_3(1, 2)], b[rc_3(2, 0)] }
        },
        c13c21c32[2][3]{
            { a[rc_3(0, 2)], a[rc_3(1, 0)], a[rc_3(2, 1)] },
            { b[rc_3(0, 2)], b[rc_3(1, 0)], b[rc_3(2, 1)] }
        };
    float c11c23c32[2][3]{
        { a[rc_3(0, 0)], a[rc_3(1, 2)], a[rc_3(2, 1)] }, { b[rc_3(0, 0)], b[rc_3(1, 2)], b[rc_3(2, 1)] }
    },
        c12c21c33[2][3]{
            { a[rc_3(0, 1)], a[rc_3(1, 0)], a[rc_3(2, 2)] }, { b[rc_3(0, 1)], b[rc_3(1, 0)], b[rc_3(2, 2)] }
        },
        c13c22c31[2][3]{
            { a[rc_3(0, 2)], a[rc_3(1, 1)], a[rc_3(2, 0)] }, { b[rc_3(0, 2)], b[rc_3(1, 1)], b[rc_3(2, 0)] }
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
    for (int i = 0; i < 4; i++) ret_polynomial[i] = pos_polynomial[i] - neg_polynomial[i];
}

__host__ __device__ int cubic_roots(float roots[3], float coef[4], float x0, float x1);


__host__ __device__ int build_and_solve_4_points_coplanar(
    const vec3f& p0_t0,
    const vec3f& p1_t0,
    const vec3f& p2_t0,
    const vec3f& p3_t0,

    const vec3f& p0_t1,
    const vec3f& p1_t1,
    const vec3f& p2_t1,
    const vec3f& p3_t1,

    float roots[3])
{

    float a1[9] {
        p1_t1.x - p1_t0.x, p1_t1.y - p1_t0.y, p1_t1.z - p1_t0.z,
        p2_t1.x - p2_t0.x, p2_t1.y - p2_t0.y, p2_t1.z - p2_t0.z,
        p3_t1.x - p3_t0.x, p3_t1.y - p3_t0.y, p3_t1.z - p3_t0.z
    },
    a2[9] {
        p0_t1.x - p0_t0.x, p0_t1.y - p0_t0.y, p0_t1.z - p0_t0.z,
        p2_t1.x - p2_t0.x, p2_t1.y - p2_t0.y, p2_t1.z - p2_t0.z,
        p3_t1.x - p3_t0.x, p3_t1.y - p3_t0.y, p3_t1.z - p3_t0.z
    }, 
    a3[9] {
        p0_t1.x - p0_t0.x, p0_t1.y - p0_t0.y, p0_t1.z - p0_t0.z,
        p1_t1.x - p1_t0.x, p1_t1.y - p1_t0.y, p1_t1.z - p1_t0.z,
        p3_t1.x - p3_t0.x, p3_t1.y - p3_t0.y, p3_t1.z - p3_t0.z
    },
    a4[9] {
        p0_t1.x - p0_t0.x, p0_t1.y - p0_t0.y, p0_t1.z - p0_t0.z,
        p1_t1.x - p1_t0.x, p1_t1.y - p1_t0.y, p1_t1.z - p1_t0.z,
        p2_t1.x - p2_t0.x, p2_t1.y - p2_t0.y, p2_t1.z - p2_t0.z
    };
    float b1[9] {
        p1_t0.x, p1_t0.y, p1_t0.z,
        p2_t0.x, p2_t0.y, p2_t0.z,
        p3_t0.x, p3_t0.y, p3_t0.z
    }, 
    b2[9] {
        p0_t0.x, p0_t0.y, p0_t0.z,
        p2_t0.x, p2_t0.y, p2_t0.z,
        p3_t0.x, p3_t0.y, p3_t0.z
    },
    b3[9] {
        p0_t0.x, p0_t0.y, p0_t0.z,
        p1_t0.x, p1_t0.y, p1_t0.z,
        p3_t0.x, p3_t0.y, p3_t0.z
    },
    b4[9] {
        p0_t0.x, p0_t0.y, p0_t0.z,
        p1_t0.x, p1_t0.y, p1_t0.z,
        p2_t0.x, p2_t0.y, p2_t0.z
    }; 
    
    float ret_polynomial[4] {0.0f};
    float tmp_polynomial[4] {0.0f};

    det_polynomial(a1, b1, ret_polynomial);
    det_polynomial(a2, b2, tmp_polynomial);
    for (int i = 0 ; i < 4; i ++) ret_polynomial[i] -= tmp_polynomial[i];
    det_polynomial(a3, b3, tmp_polynomial);
    for (int i = 0 ; i < 4; i ++) ret_polynomial[i] += tmp_polynomial[i];
    det_polynomial(a4, b4, tmp_polynomial);
    for (int i = 0 ; i < 4; i ++) ret_polynomial[i] -= tmp_polynomial[i];

    double root = 1.0;
    int found = cubic_roots(roots, ret_polynomial, 0.0, 1.0);
    static const auto eval = [](float x, float c[4]) {
        return c[0] + x * (c[1] + x * (c[2] + x * c[3]));
    };
    if (found < 0 || found > 3) found = 3;
    #ifdef TESTING
    float y[3];
    for (int i = 0; i < found; i ++ ){
        y[i] = eval(roots[i], ret_polynomial);
    }
    #endif
    return found;
}


__device__ __host__ bool _cross(const Edgef &ei, const Edgef &ej){
    auto vei = ei.e1 - ei.e0;
    auto vej0 = ej.e0 - ei.e0;
    auto vej1 = ej.e1 - ei.e0;
    return dot(cross(vei, vej0), cross(vei, vej1)) < 0.0f;
}

__device__ __host__ bool verify_root_ee(
    const Edgef &ei, 
    const Edgef &ej
) {
    return _cross(ei, ej) && _cross(ej, ei);
}

__forceinline__ __device__ __host__ bool inside(const Facef &f, const vec3f &p) {
    auto f01 = cross(f.t0 - p, f.t1- p);
    auto f12 = cross(f.t1 - p, f.t2- p);
    auto f20 = cross(f.t2 - p, f.t0- p);
    return dot(f01, f12) >= 0.0f && dot(f12, f20) >= 0.0f;
}

__device__ __host__ bool verify_root_pt(
    const vec3f &p, const Facef &f
 ) {
    auto n = unit_normal(f);
    double d = dot(n, p - f.t0);
    auto v = p - n * d;
    return inside(f, v);
 }
__device__ __host__ float pt_collision_time(
    const vec3f &p0,
    const Facef &t0, 
    const vec3f &p1,
    const Facef &t1
){
    float roots[3] {1,1,1};
    int found = build_and_solve_4_points_coplanar(p0, t0.t0, t0.t1, t0.t2, p1, t1.t0, t1.t1, t1.t2, roots);
    bool true_root = false;
    float root;
    for (int i = 0; (i < found) && !true_root; i++) {
        root = roots[i];
        true_root = verify_root_pt(linerp(p1, p0, root), linerp(t1, t0, root));
    }
    return found && true_root? root: 1.0f;
}

__device__ __host__ float ee_collision_time(
    const Edgef &ei0, 
    const Edgef &ej0,
    const Edgef &ei1,
    const Edgef &ej1
){
    float roots[3]{1,1,1};
    int found = build_and_solve_4_points_coplanar(
        ei0.e0, ei0.e1, ej0.e0, ej0.e1,
        ei1.e0, ei1.e1, ej1.e0, ej1.e1,
        roots
    );
    float root = 1.0f;
    float real_roots[3] {1,1,1};
    int cnt = 0;
    for (int i = 0; i < found; i++) {
        root = roots[i];
        bool true_root = verify_root_ee(linerp(ei1, ei0, root), linerp(ej1, ej0, root));
        if (true_root)
            real_roots[cnt ++] = root;
    }
    return CUDA_MIN3(real_roots[0], real_roots[1], real_roots[2]);
}