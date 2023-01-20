#include "affine_obj.h"
#include <set>
#include <array>
using namespace std;
void AffineObject::extract_edges() {
    set<array<unsigned,2>> e;
    static const auto insert = [&](unsigned a, unsigned b){
        e.insert({min(a, b), max(a, b)});
    };
    for (int i = 0; i < n_faces; i ++){
        auto t0 = indices[3 * i], t1 = indices[3 * i + 1], t2 = indices[3 * i + 2]; 
        insert(t0, t1);
        insert(t1, t2);
        insert(t0, t2);
    }
    _edges.reserve(e.size() * 2);
    for (auto &ei: e) {
        _edges.push_back(ei[0]);
        _edges.push_back(ei[1]);
    }
    n_edges = _edges.size() / 2; 
    edges = _edges.data();
}

const vec3 AffineObject::vertices(int i) const{
    auto &p = mesh.vertices[i].Position;
    return vec3(p[0], p[1], p[2]);
}

void AffineObject::draw(Shader& shader) const
{
    mesh.Draw(shader);
}