#pragma once
#include "camera.h"
// consts
static const int SHADOW_WIDTH = 800, SHADOW_HEIGHT = 600;
static const char *varyings[] = {
    "selected_alias"
    // "alias"
    // "TexCoords","selected_alias","a","b","c"
};
static const unsigned int SCR_WIDTH = 800;
static const unsigned int SCR_HEIGHT = 600;

#ifdef _MAIN_CPP
#define VARIABLE_LOCATOR static
#else 
#define VARIABLE_LOCATOR extern
#endif
VARIABLE_LOCATOR unsigned int depthMapFBO, depthMap;
VARIABLE_LOCATOR int objectType = 0;
VARIABLE_LOCATOR bool postrender = false, edge = false, skybox = false, model_draw = false,
     display_corner = true, Motion = false, feedback = false, cursor_hidden = true;
// settings
// camera
VARIABLE_LOCATOR Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
VARIABLE_LOCATOR float lastX = SCR_WIDTH / 2.0f;
VARIABLE_LOCATOR float lastY = SCR_HEIGHT / 2.0f;
VARIABLE_LOCATOR bool firstMouse = true;

// timing
VARIABLE_LOCATOR float deltaTime = 0.0f, lastFrame = 0.0f;

// lighting
VARIABLE_LOCATOR glm::vec3 LightPositions[] = {
    glm::vec3(1.2f, 1.0f, 2.0f),
    glm::vec3(1.2f, 2.0f, 0.0f),
    glm::vec3(-1.2f, 2.0f, 2.0f),
    glm::vec3(-1.2f, 2.0f, 0.0f)};

VARIABLE_LOCATOR glm::vec3 &lightPos(LightPositions[0]);



