//
// pch.h
//

#pragma once
#define TESTING
#include "gtest/gtest.h"
#include <vector>
#include <array>
struct Globals {
    std::vector<std::array<unsigned, 2>> points, edges, triangles;
    double mu, evh, dt;
    bool ground;
};
//#define _BODY_WISE_


