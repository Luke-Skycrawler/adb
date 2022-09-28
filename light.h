#ifndef LIGHT_H
#define LIGHT_H
#ifdef LIGHT_CPP
#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"

using namespace std;

const unsigned int SCR_WIDTH=800;
const unsigned int SCR_HEIGHT=600;
void renderCube(int light);
#endif
class Light
{
public:
    Shader lightingShader;
    Shader lightCubeShader;

    Light(glm::vec3 LightPositions[], int num);
    void SetShaderValue(Camera &camera);    
    glm::vec3* pointLightPositions;
    void Draw(Camera &camera);

private:
    int LightNum;
};

#endif