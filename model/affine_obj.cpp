#include "affine_obj.h"
#include <set>
#include <array>
using namespace std;
void AffineObject::extract_edges() {
    static set<array<int,2>> e;
    e.clear();
    static const auto insert = [&](int a, int b){
        e.insert({min(a, b), max(a, b)});
    };
    for (int i = 0; i < n_faces; i ++){
        auto t0 = indices[3 * i], t1 = indices[3 * i + 1], t2 = indices[3 * i + 2]; 
        insert(t0, t1);
        insert(t1, t2);
        insert(t0, t2);
    }
    edges.reserve(e.size() * 2);
    for (auto &ei: e) {
        edges.push_back(ei[0]);
        edges.push_back(ei[1]);
    }
    n_edges = edges.size() / 2; 
}

const vec3 AffineObject::vertices(int i) const{
    // glm::vec3 p = mesh.vertices[i].Position;
    //  return vec3(p[0], p[1], p[2]);
    return vert_rest[i];
}

void AffineObject::set_vertices() const {
    for (int i = 0; i < n_vertices; i++) {
        auto p = v_transformed[i];
        mesh.vertices[i].Position = glm::vec3(p[0], p[1], p[2]);
        
    }
}
void AffineObject::draw(Shader& shader) const
{
    set_vertices();
    mesh.setupMesh();
    mesh.Draw(shader);
}