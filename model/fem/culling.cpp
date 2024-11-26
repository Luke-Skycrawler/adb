#include "../scalar_types.h"
#include "fem.h"
using namespace std;
using namespace Eigen;

scalar time_culling_ee(const std::vector<i2>& eibi, const std::vector<i2>& ejbj, const std::vector<int>& vertex_starting_index, const std::vector<vec3>& vt1_buffer, const std::vector<vec3>& vt2_buffer,
    const std::vector<FEMObject>& objs)
{
    return 1.0;
}

scalar time_culling_pt(const std::vector<i2>& pibi, const std::vector<i2>& tjbj, const std::vector<int>& vertex_starting_index, const std::vector<vec3>& vt1_buffer, const std::vector<vec3>& vt2_buffer,
    const std::vector<FEMObject>& objs)
{
    return 1.0;
}
