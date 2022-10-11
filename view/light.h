#ifndef LIGHT_H
#define LIGHT_H
#include "shader.h"
#include "camera.h"
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