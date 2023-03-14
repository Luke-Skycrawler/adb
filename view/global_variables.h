#pragma once
#ifndef TESTING
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
#include "../model/cube.h"
#include "../model/affine_obj.h"
#include <vector>
#include <memory>
#include <map>
#include "model.h"
#include <string>
#include <omp.h>
#include <string>
#include "../model/spatial_hashing.h"
#define _INCLUDE_IAABB_H_
#ifdef _INCLUDE_IAABB_H_
#include "../model/iaabb.h"
#endif

struct HessBlock {
    int i, j;
    Matrix<double, 12, 1> block;
    HessBlock(int _i, int _j, const Matrix<double, 12, 1> hess): i(_i), j(_j) {
        block = hess;
    }
};

struct GlobalVariableMainCPP{
    #ifndef GOOGLE_TEST
    // control
    bool postrender, edge, skybox, model_draw, display_corner, motion, feedback, cursor_hidden;
    float deltaTime, lastFrame;
    int objectType;

    // timing
    unsigned int depthMapFBO, depthMap;

    // camera
    Camera camera;
    float lastX, lastY;
    bool firstMouse;
    GlobalVariableMainCPP(): camera(glm::vec3(0.0f, 6.0f, 10.0f)){
        postrender = false, edge = false, skybox = false, model_draw = false, display_corner = true, motion = false, feedback = false, cursor_hidden = true;        
        lastX = SCR_WIDTH / 2.0f;
        lastY = SCR_HEIGHT / 2.0f;
        firstMouse = true;
    }
    #endif
    double dt;
    int max_iter, tot_iter, ts, set_size;
    std::vector<HessBlock> hess_triplets;
    std::vector<unique_ptr<AffineBody>> cubes;
        std::map<std::string, unique_ptr<Model>> loaded_models;

    vec3 gravity;
    double alpha, beta;
    double kappa, d_hat, safe_factor, mu, eps_x, backoff, evh;
    bool vg_fric, pt_fric, ee_fric;
    bool col_set, upper_bound, line_search, sparse, dense, ee, pt, ground, psd, damp, full_ccd, align_com, player;
    int log, iaabb;
    vector<omp_lock_t> writelock_cols;
    unique_ptr<spatial_hashing> sh;
    std::string scene, trace_folder;
    std::vector<std::array<unsigned, 2>> edges, points, triangles;
#ifdef _INCLUDE_IAABB_H_
    std::vector<lu> aabbs;
#endif

};
#ifndef GOOGLE_TEST
#ifdef _MAIN_CPP
#define VARIABLE_LOCATOR
VARIABLE_LOCATOR glm::vec3 LightPositions[]
= {
   glm::vec3(5.0f, 5.0f, 6.0f),
   glm::vec3(1.2f, 2.0f, 0.0f),
   glm::vec3(-1.2f, 2.0f, 2.0f),
   glm::vec3(-1.2f, 2.0f, 0.0f)};
#else 
#define VARIABLE_LOCATOR extern
VARIABLE_LOCATOR glm::vec3 LightPositions[];
#endif

VARIABLE_LOCATOR GlobalVariableMainCPP globals;

// lighting
static glm::vec3 &lightPos(LightPositions[0]);

#endif
#endif