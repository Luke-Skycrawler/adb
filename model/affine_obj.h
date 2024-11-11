#include "affine_body.h"
#include "../view/mesh.h"

struct AffineObject: AffineBody {
    Mesh &mesh;
    AffineObject(Mesh &mesh):
    mesh(mesh), 
    AffineBody(mesh.vertices.size(), mesh.indices.size() / 3, 0, mesh.indices, {}) {
        extract_edges();
        Ic = mass / 12;
    }
    std::vector<int> _edges;
    void extract_edges();
    const vec3 vertices(int i) const;
    void draw(Shader &shader) const;
};