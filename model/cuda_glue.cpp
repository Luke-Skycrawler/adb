#include "cuda_glue.h"
#include <vector>
#include <spdlog/spdlog.h>
using namespace std;
using namespace Eigen;
void initialize_aabbs(const vector<lu> &aabbs) {
    cudaMallocManaged(&host_cuda_globals.aabbs, aabbs.size() * sizeof(luf));
    vector<luf> host_aabbs;
    host_aabbs.resize(aabbs.size());
    for (int i = 0; i < aabbs.size(); i++) {
        host_aabbs[i] = to_luf(aabbs[i]);
    }
    cudaMemcpy(host_cuda_globals.aabbs, host_aabbs.data(), host_aabbs.size() * sizeof(luf), cudaMemcpyHostToDevice);
}
void initialize_primitives(
    const vector<unique_ptr<AffineBody>>& cubes)
{
    int ptot = 0, etot = 0, ftot = 0;
    vector<cudaAffineBody> &host_cubes { host_cuda_globals.host_cubes};
    host_cubes.resize(cubes.size());
    for (int i = 0; i < cubes.size(); i++) {
        auto& c{ *cubes[i] };
        ptot += c.n_vertices;
        etot += c.n_edges;
        ftot += c.n_faces;
        host_cubes[i] = to_cabd(c);
    }

    cudaMallocManaged(&host_cuda_globals.vertices_at_rest, ptot * sizeof(vec3f));
    cudaMallocManaged(&host_cuda_globals.projected_vertices, ptot * sizeof(vec3f));
    cudaMallocManaged(&host_cuda_globals.updated_vertices, ptot * sizeof(vec3f));
    
    cudaMallocManaged(&host_cuda_globals.edges, etot * 2 * sizeof(int));
    cudaMallocManaged(&host_cuda_globals.faces, ftot * 3 * sizeof(int));
    host_cuda_globals.n_vertices = ptot;
    host_cuda_globals.n_edges = etot;
    host_cuda_globals.n_faces = ftot;
    
    int start = 0, estart = 0, fstart = 0;
    vector<vec3f> vertices;
    vector<int> edges, faces;
    vertices.resize(ptot);
    edges.resize(etot * 2);
    faces.resize(ftot * 3);

    for (int i = 0; i < cubes.size(); i++) {
        auto& c{ *cubes[i] };
        auto& cabd{ host_cubes[i] };
        cabd.projected = host_cuda_globals.projected_vertices + start;
        cabd.vertices = host_cuda_globals.vertices_at_rest + start;
        cabd.updated = host_cuda_globals.updated_vertices + start;
        
        cabd.edges = host_cuda_globals.edges + estart;
        cabd.faces = host_cuda_globals.faces + fstart;
         
        for (int j = 0; j < c.n_vertices; j++) {
            vertices[j + start] = to_vec3f(c.vertices(j));
        }
        start += c.n_vertices;
        for (int j = 0; j < c.n_edges * 2; j++) {
            edges[j + estart] = c.edges[j];
        }
        estart += c.n_edges * 2;

        for (int j = 0; j < c.n_faces * 3; j++) {
            faces[j + fstart] = c.indices[j];
        }
        fstart += c.n_faces * 3;
    }

    cudaMemcpy(host_cuda_globals.vertices_at_rest, vertices.data(), vertices.size() * sizeof(vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(host_cuda_globals.edges, edges.data(), edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(host_cuda_globals.faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(host_cuda_globals.cubes, host_cubes.data(), host_cubes.size() * sizeof(cudaAffineBody), cudaMemcpyHostToDevice);
    
}

void init_dev_cubes(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes)
{
    for (int i = 0; i < n_cubes; i++) {
        auto& b{ host_cuda_globals.host_cubes[i] };
        auto& a{ *cubes[i] };
        for (int i = 0; i < 4; i++) {
            b.q[i] = to_vec3f(a.q[i]);
            b.q0[i] = to_vec3f(a.q0[i]);
            b.dqdt[i] = to_vec3f(a.dqdt[i]);
            b.q_update[i] = to_vec3f(a.q[i] + a.dq.segment<3>(i * 3));
        }
    }
    cudaMemcpy(host_cuda_globals.cubes, host_cuda_globals.host_cubes.data(), sizeof(cudaAffineBody) * n_cubes, cudaMemcpyHostToDevice);
}

static bool strict = false;
void cuda_hess_glue(
    int n_cubes,
    const vector<unique_ptr<AffineBody>>& cubes,
    float dt)
{
    vector<cudaAffineBody> cabs(n_cubes);
    float* grads = new float[n_cubes * 12];
    float* hess = new float[n_cubes * 144];
    // for (int i = 0; i < n_cubes; i++) cabs[i] = to_cabd(*cubes[i]);
    // cudaMemcpy(host_cuda_globals.cubes, cabs.data(), sizeof(cudaAffineBody) * cabs.size(), cudaMemcpyHostToDevice);
    init_dev_cubes(n_cubes, cubes);
    cuda_inert_hess_glue(n_cubes, dt, grads, hess);
    for (int k = 0; k < n_cubes; k++) {
        auto& c{ *cubes[k] };
        vec12 ref = c.grad, act = Map<Vector<float, 12>>(grads + k * 12).cast<double>();
        mat12 h_ref = c.hess, h_act = Map<Matrix<float, 12, 12>>(hess + k * 144).cast<double>();
        auto norm = (ref - act).norm();
        auto h_norm = (h_ref - h_act).norm();
        if (!strict) {
            if (!act.isApprox(ref, 1e-3) && ref.norm() > 1e-3)
                spdlog::warn("grad norm too large: diff = {}, ref_norm = {}", norm, ref.norm());
            if (!h_act.isApprox(h_ref, 1e-3) && h_ref.norm() > 1e-3)
                spdlog::warn("hess norm too large: {}", h_norm);
        }
        else {
            assert(norm < 1e-2);
            assert(h_norm < 1e-2);
        }
        if (host_cuda_globals.params["use_cuda_inertia"]){
            c.grad = act;
            c.hess = h_act;
        }
    }
    if (!host_cuda_globals.params["use_cuda_inertia"]) {
        for (int i = 0; i < n_cubes; i++) {
            // copy c.hess and c.grad to grads and hess
            for (int j = 0; j < 12; j++) grads[i * 12 + j] = cubes[i]->grad(j);
            for (int j = 0; j < 12; j++)
                for (int k = 0; k < 12; k++) hess[i * 144 + j + k * 12] = cubes[i]->hess(j, k);
        }
        cudaMemcpy(host_cuda_globals.b, grads, sizeof(float) * 12 * n_cubes, cudaMemcpyHostToDevice);
        cudaMemcpy(host_cuda_globals.hess_diag, hess, sizeof(float) * 144 * n_cubes, cudaMemcpyHostToDevice);
    } 
    delete[] grads;
    delete[] hess;
}
