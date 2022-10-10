#include <Eigen>

using namespace Eigen;
using vec3 = Vector3f;
using mat3 = Matrix3f;

struct Cube{
    vec3 a_col[3];
    vec3 translation;
    vec3 q_dot[3];
    vec3 vc;
    vec3 dimensions;
    Cube(float scale = 1.0f, vec3 &vc = vec3(0.0f)): vc(vc){}
}