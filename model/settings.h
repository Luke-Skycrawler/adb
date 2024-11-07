#pragma once
#include "affine_body.h"
#include <string>
#include <map>

struct ABDTKLocalConfig{
    bool g = true;
    vec3 gravity = vec3(0.0, g ? -9.8 : 0.0, 0.0);
    bool col_set = true;
    bool line_search = true;
    bool upper_bound = true;
    bool sparse = true;
    bool dense = false;
    bool ee = true;
    bool pt = true;
    bool psd = true; 
    bool ground = true;
    bool damp = false;
    scalar backoff = 0.8;
    bool full_ccd = true;
    bool align_com = false;
    int log = 1;
    bool player = false;
    std::string trace_folder = "trace";
    int iaabb = 3;
    scalar evh = 1e-3;
    scalar kappa = 1e-4;
    bool vg_fric = true;
    bool pt_fric = true;
    bool ee_fric = true;

    scalar dt = 1e-2;
    int max_iter = 1000;
    scalar alpha = 5e-2;
    scalar beta = 2.5e-2;
    scalar safe_factor = 1.0;
    scalar mu = 0.0;
    scalar eps_x = 1e-3;
    
    std::map<std::string, scalar> params_double = {
        {"tol", 1e-2},
        {"dhat_sqr", 1e-2},
        {"kappa", 1e-1},
        {"c1", 1e-4},
        {"max_uk", 0.0},
        {"dq_tol", 0.0}
    };
    std::map<std::string, int> params_int = {
        {"clip", 0},
        {"gfk", 1},
        {"hdk", 1}
    };    

};
#ifndef ABDTK
#include "../view/global_variables.h"
#else 
static const ABDTKLocalConfig globals;
#endif
