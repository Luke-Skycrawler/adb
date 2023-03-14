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
};
//#define _BODY_WISE_