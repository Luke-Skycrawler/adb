#include "affine_body.h"
#include "../view/model.h"
#include "../view/mesh.h"

struct AffineObject: AffineBody {
    Mesh &mesh;
    AffineObject(Mesh &mesh):
    mesh(mesh), 
    AffineBody(mesh.vertices.size(), mesh.indices.size() / 3, 0, mesh.indices.data(), nullptr) {
        extract_edges();
    }
    std::vector<unsigned> _edges;
    void extract_edges();
};