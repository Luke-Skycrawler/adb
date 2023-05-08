#include "cuda_glue.h"
#include <vector>
using namespace std;

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
    cudaMallocManaged(&host_cuda_globals.edges, etot * 2 * sizeof(int));
    cudaMallocManaged(&host_cuda_globals.faces, ftot * 3 * sizeof(int));

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