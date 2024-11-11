#include "affine_body.h"
#include <array>
#include <vector>
#include "iaabb.h"
void pt_col_set_task(
    int vi, int fj, int I, int J,
    // const AffineBody& ci, const AffineBody& cj,
    const vec3& v, const Face& f,
    const lu& aabb_i, const lu& aabb_j,
    std::vector<q4>& pts,
    std::vector<i4>& idx);
inline void ee_col_set_task(
    int ei, int ej, int I, int J,
    // const AffineBody& ci, const AffineBody& cj,
    const Edge& eii, const Edge& ejj,
    const lu& aabb_i, const lu& aabb_j, std::vector<q4>& ees,
    std::vector<i4>& eidx);

    
scalar ee_col_time(
    std::vector<int>& eilist, std::vector<int>& ejlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J, 
    std::vector<int>& vertex_starting_index, std::vector<vec3>& vt1_buffer);
scalar vf_col_time(
    std::vector<int>& vilist, std::vector<int>& fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J, 
    std::vector<int> &vertex_starting_index, std::vector<vec3> &vt1_buffer);

void vf_col_set(
    std::vector<int>& vilist, std::vector<int>& fjlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J,
    std::vector<q4>& pts,
    std::vector<i4>& idx);

void ee_col_set(
    std::vector<int>& eilist, std::vector<int>& ejlist,
    const std::vector<std::unique_ptr<AffineBody>>& cubes,
    int I, int J,
    std::vector<q4>& ees,
    std::vector<i4>& eidx);
