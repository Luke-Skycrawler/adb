#include "affine_body.h"
using lu = std::array<vec3, 2>;


struct Intersection {
    int i, j;
    lu cull;
    inline bool operator==(Intersection &b) const {
        return i == b.i && j == b.j;
    }
    inline bool operator<(Intersection& b) const {
        return i < b.i || (i == b.i && j < b.j);
    }
    inline Intersection &operator=(const Intersection& b) {
        i = b.i; 
        j = b.j;
        cull = b.cull;
        return *this;
    }
};

struct BoundingBox {
    int body;
    double p;
    bool true_for_l_false_for_u;
};

lu compute_aabb(const AffineBody& c);
lu affine(const lu& aabb, q4& q);
lu affine(lu aabb, AffineBody& c, int vtn);


void intersect_brute_force(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu> aabbs,
    std::vector<Intersection> ret,
    int vtn);
void intersect_sort(
    int n_cubes,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    const std::vector<lu> aabbs,
    std::vector<Intersection> ret,
    int vtn);
