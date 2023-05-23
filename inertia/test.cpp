#include "pch.h"
#include <random>
#include "../model/othogonal_energy.h"
#include "../model/cube.h"
#include <vector>
#include "../model/cuda_header.cuh"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;
using namespace Eigen;
inline void Cube::draw(Shader& shader) const {}
vector<unsigned> Cube::_edges = {}, Cube::_indices = {};

inline vec3f to_vec3f(const vec3& a)
{
    return make_float3(a[0], a[1], a[2]);
}
inline Facef to_facef(const Face& f)
{
    return {
        to_vec3f(f.t0),
        to_vec3f(f.t1),
        to_vec3f(f.t2)
    };
}
__host__ __device__ void cudaAffineBody::q_minus_qtiled(float3 dq[4])
{
    float h = 1e-2;
    float h2 = h * h;
    for (int i = 0; i < 4; i++) {
        dq[i] = q_update[i] - (q[i] + dqdt[i] * h + make_float3(0.0f, -9.8f, 0.0f) * h2);
    }
}

mat3 rotation(double a, double b, double c)
{
    auto s1 = sin(a);
    auto s2 = sin(b);
    auto s3 = sin(c);

    auto c1 = cos(a);
    auto c2 = cos(b);
    auto c3 = cos(c);

    mat3 R;
    R << c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2,
        c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3,
        -s2, c2 * s3, c2 * c3;
    // R = R.transpose();
    return R;
}


static const double kappa = 1e9;
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
struct double3 {
    double x, y, z;
};
double3 operator+(double3 a, double3 b) {
    return double3 {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}
double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
double3 operator*(double3 a, double b) {
    return double3 {
        a.x * b,
        a.y * b,
        a.z * b
    };
}
__host__ __device__ void orthogonal_grad(double3 q[4], double dt, double ret[12])
{
    auto h2 = dt * dt;
    ret[0] = ret[1] = ret[2] = 0.0;
    for (int i = 1; i < 4; i++) {
        double3 g {0.0, 0.0, 0.0};
        for (int j = 1; j < 4; j++) {
            g = g + q[j] * (dot(q[i], q[j]) - (i == j ? 1.0 : 0.0));
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
inline cudaAffineBody to_cabd(AffineBody &a){
   cudaAffineBody b;
   for (int i = 0; i < 4; i ++){
       b.q[i] = to_vec3f(a.q[i]);
       b.q0[i] = to_vec3f(a.q0[i]);
       b.dqdt[i] = to_vec3f(a.dqdt[i]);
       b.q_update[i] = to_vec3f(a.q[i] + a.dq.segment<3>(i * 3));
   }
   b.mass = a.mass;
   b.Ic = a.Ic;
   b.n_vertices = a.n_vertices;
   b.n_faces = a.n_faces;
   b.n_edges = a.n_edges;

   return b;
}

class inertiaTest : public ::testing::Test {
public:
    std::vector<std::unique_ptr<AffineBody>> cubes;
    std::vector<cudaAffineBody> cabds;
    
    int n_cubes;
    default_random_engine gen;
    static uniform_real_distribution<double> dist;
    static const double space_range[2];

protected:
    void SetUp() override
    {
        n_cubes = 50;
        Cube::gen_indices();
        for (int i = 0; i < n_cubes; i++) {
            unique_ptr<AffineBody> a;
            a = make_unique<Cube>();
            double aa, bb, cc;
            double p0, p1, p2;

            double aa_t2, bb_t2, cc_t2;
            double p0_t2, p1_t2, p2_t2;

            aa = dist(gen) * M_PI * 2, bb = dist(gen) * M_PI * 2, cc = dist(gen) * M_PI * 2;
            p0 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            p1 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            p2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];

            aa_t2 = dist(gen) * M_PI * 2, bb_t2 = dist(gen) * M_PI * 2, cc_t2 = dist(gen) * M_PI * 2;
            p0_t2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            p1_t2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];
            p2_t2 = dist(gen) * (space_range[1] - space_range[0]) + space_range[0];

            mat3 r = rotation(aa, bb, cc);
            mat3 r_t2 = rotation(aa_t2, bb_t2, cc_t2);

            for (int i = 0; i < 3; i++) {
                a->q[i + 1] = r.col(i);
                a->dq.segment<3>(3 * (i + 1)) = r.col(i);
            }
            a->dq.segment<3>(0) = vec3(p0_t2, p1_t2, p2_t2);
            a->q[0] = vec3(p0, p1, p2);
            a->q0 = a->q;

            cabds.push_back(to_cabd(*a));
            cubes.push_back(move(a));

        }
        
    }
};
TEST_F(inertiaTest, test_against_ref) {
    float *ret = new float [12];
    double ret_double[12];
    for (int i = 0; i < n_cubes; i ++) {
        auto &c {*cubes[i]};
        const float dt = 1e-2;
        auto g_ref = othogonal_energy::grad (c.q) * dt *dt;
        for (int j = 0; j < 12; j++) {
            ret[j] = 0;
            ret_double[j] = 0;
        }
        double3 q[4];
        orthogonal_grad(cabds[i].q, dt, ret);
        // for (int i = 0; i < 4; i++) q[i] = { c.q[i][0], c.q[i][1], c.q[i][2] };
        auto &qf = cabds[i].q;
        for (int i = 0; i < 4; i++) q[i] = { qf[i].x, qf[i].y, qf[i].z };
        orthogonal_grad(q, 1e-2, ret_double);
        vec12 g_act = Map<Vector<float, 12>>(ret).cast<double>();
        vec12 g_double = Map<vec12>(ret_double);
        // EXPECT_TRUE(g_ref.isApprox(g_act, 1e-3)) << "\nref norm = " << g_ref.norm() << " \n diff norm = " << (g_act - g_ref).norm();
        EXPECT_TRUE(g_ref.isApprox(g_double, 1e-3)) << "\nref norm = " << g_ref.norm() << " \n diff norm = " << (g_double - g_ref).norm();
        EXPECT_TRUE(g_act.isApprox(g_double, 1e-3)) << "\nref norm = " << g_double.norm() << " \n diff norm = " << (g_double - g_act).norm();
        //EXPECT_TRUE((g_ref - g_act).norm() < 1e-1) << "\nref " << g_ref.transpose() << " \n act " << g_act.transpose();
    }
    delete[] ret;
}
uniform_real_distribution<double> inertiaTest ::dist(0.0, 1.0);
const double inertiaTest::space_range[2]{ -3.0, 3.0 };

TEST_F(inertiaTest, test_hess) {
    float *ret = new float[144];
    for (int i = 0; i < n_cubes; i ++) {
        auto &c {*cubes[i]};
        const float dt = 1e-2;
        fill (ret, ret + 144, 0.0f);
        auto g_ref = othogonal_energy::hessian (c.q) * dt *dt;
        orthogonal_hess(cabds[i].q, dt, ret);
        mat12 g_act = Map<Matrix<float, 12, 12>>(ret).cast<double>();
        EXPECT_TRUE(g_ref.isApprox(g_act, 1e-3)); //  << "ref " << g_ref.transpose() << " \n act " << g_act.transpose() ;
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}