#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Eigen>
glm::mat4 from_eigen(Eigen::Matrix3f &eig_matrix){
    glm::mat3 a = glm::make_mat3(eig_matrix.data());
    glm::mat4 ret(a);
    return ret;
}

