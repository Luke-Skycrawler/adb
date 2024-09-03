#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Eigen>
#include "affine_body.h"
glm::mat4 from_eigen(mat3 &eig_matrix){
    glm::mat3 a = glm::make_mat3(eig_matrix.data());
    glm::mat4 ret(a);
    return ret;
}

glm::vec3 from_eigen(vec3 &eig_vec) {
    glm::vec3 v = glm::make_vec3(eig_vec.data());
    return v;
}
