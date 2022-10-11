#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include <string>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void gen_preview_framebuffer();
void renderPlane();
void renderCube(int light = 0);
void click_callback(GLFWwindow *window, int button, int action, int mods);
unsigned int Feedback_Initialize(unsigned int *_vbo = NULL, unsigned int *_xfb = NULL);
unsigned int loadCubemap(std::vector<std::string> faces);
unsigned int loadTexture(char const *path);
