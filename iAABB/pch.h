//
// pch.h
//

#pragma once
#define TESTING
#include "gtest/gtest.h"
#include <vector>
#include <array>
struct Globals {
    std::vector<std::array<int, 2>> points, edges, triangles;
    double mu, evh, dt;
    bool ground;
};

static struct CoreIPCLocalConfig{
    bool psd;
    scalar eps_x;
    CoreIPCLocalConfig(): psd(true), eps_x(1e-3){}
} globals;

//#define _BODY_WISE_


