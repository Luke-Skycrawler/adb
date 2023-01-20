#pragma once
#include <string>
#include "../model/cube.h"
#include "../view/global_variables.h"
unique_ptr<Cube> spinning_cube();
void cube_blocks(int n);
void customize(std::string file);
