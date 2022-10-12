#include <glm/glm.hpp>
#include <Eigen/Eigen>
mat4 from_eigen(Matrix3f &eig_matrix) {
    glm::mat3 a = glm::make_mat3(eig_matrix.data());
    glm::mat4 ret(a);
    return ret;
}

