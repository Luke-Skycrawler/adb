#include "affine_body.h"
#include "../view/mesh.h"

struct AffineObject: AffineBody {
    Mesh &mesh;
    AffineObject(Mesh &mesh):
    mesh(mesh), 
    AffineBody(mesh.vertices.size(), mesh.indices.size() / 3, 0, mesh.indices, {}) {
        extract_edges();
        Ic = mass / 12;
        vert_rest.resize(mesh.vertices.size()); 
        for (int i = 0; i < vert_rest.size(); i++) {
            auto &p = mesh.vertices[i].Position;
            vert_rest[i] = vec3(p[0], p[1], p[2]);
        }
    }
    std::vector<int> _edges;
    void extract_edges();
    std::vector<vec3> vert_rest;
    const vec3 vertices(int i) const;
    void set_vertices() const;
    // mesh.vertices <- v_transformed 
    void draw(Shader &shader) const;
};