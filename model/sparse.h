#pragma once
#include <map>
#include <array>

inline int starting_offset(int i, int j, const std::map<std::array<int, 2>, int>& lut, int* outers)
{
    auto it = lut.find({ i, j });
    int k = it->second;
    return k * 12 + outers[j * 12];
}

inline int stride(int j, int* outers)
{
    return outers[j * 12 + 1] - outers[j * 12];
    // full 12x12 matrix, no overflow issue
}
