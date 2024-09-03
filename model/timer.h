#include <chrono>
#include <string>

using namespace std::chrono;
#define DURATION_TO_DOUBLE(X) (duration_cast<duration<scalar>>(high_resolution_clock::now() - (X)).count() * 1000)
