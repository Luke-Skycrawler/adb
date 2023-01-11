#include "cube.h"
#include "othogonal_energy.h"
#include <vector>
#include <memory>
void implicit_euler(std::vector<std::unique_ptr<AffineBody>>& cubes, double dt);
