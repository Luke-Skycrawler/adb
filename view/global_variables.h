#pragma once
#ifndef TESTING
#ifndef ABDTK
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
#include "../model/affine_body.h"
#include <vector>
#include <memory>
#include <map>
#include "model.h"
#include <string>
#include <string>
#include "../model/spatial_hashing.h"
#define _INCLUDE_IAABB_H_
#ifdef _INCLUDE_IAABB_H_
#include "../model/iaabb.h"
#endif

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

    // control
    float lastX, lastY;
    bool firstMouse;

    // light
    glm::vec3 light_positions[4];
    GlobalVariableMainCPP(): camera(glm::vec3(0.0f, 6.0f, 10.0f)), 
        light_positions{
            glm::vec3(5.0f, 5.0f, 6.0f),
            glm::vec3(1.2f, 2.0f, 0.0f),
            glm::vec3(-1.2f, 2.0f, 2.0f),
            glm::vec3(-1.2f, 2.0f, 0.0f)}
    {
        postrender = false, edge = false, skybox = false, model_draw = false, display_corner = true, motion = false, feedback = false, cursor_hidden = true;        
        lastX = SCR_WIDTH / 2.0f;
        lastY = SCR_HEIGHT / 2.0f;
        firstMouse = true;
    }
    #endif
    scalar dt;
    int max_iter, tot_iter, ts, set_size, starting_ts, ending_ts;
    std::vector<unique_ptr<AffineBody>> cubes;
        std::map<std::string, unique_ptr<Model>> loaded_models;

    vec3 gravity;
    scalar alpha, beta;
    scalar kappa, d_hat, safe_factor, mu, eps_x, backoff, evh;
    Eigen::Vector<scalar, 4> aggregate_time;
    bool vg_fric, pt_fric, ee_fric;
    bool col_set, upper_bound, line_search, sparse, dense, ee, pt, ground, psd, damp, full_ccd, align_com, player;

    // bundle the rest parameters in here
    // no intellecense but speed up compiling
    std::map<std::string, scalar> params_double;
    std::map<std::string, int> params_int;
    
    int log, iaabb;
    unique_ptr<spatial_hashing> sh;
    std::string scene, trace_folder;
    std::vector<std::array<unsigned, 2>> edges, points, triangles;
#ifdef _INCLUDE_IAABB_H_
    std::vector<lu> aabbs;
#endif

};




// alocate globals
#ifndef GOOGLE_TEST
#ifdef _MAIN_CPP
#define VARIABLE_LOCATOR
#else 
#define VARIABLE_LOCATOR extern
#endif



#ifndef _GLOBAL_VARIABLE_DEFINE_ONLY_
VARIABLE_LOCATOR GlobalVariableMainCPP globals;
#endif

#endif
#endif
#endif