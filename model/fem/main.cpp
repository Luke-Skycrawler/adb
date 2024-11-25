#include "fem.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
// #include "polyscope/point_cloud.h"
using namespace Eigen;
using namespace std;

namespace ps = polyscope;

static const scalar dt = 1e-2;
int main() {
    // int n_proc = omp_get_num_procs();
    // omp_set_num_threads(n_proc);
    // setNbThreads(n_proc);
    // initParallel();

    TetFEM fem("assets/bar2.tobj");
    ps::options::groundPlaneHeightMode = ps::GroundPlaneHeightMode::Manual;
    ps::options::groundPlaneHeight = -0.5;
    ps::init();
    ps::view::lookAt(glm::vec3(-1.0, 1.0, 0.0), glm::vec3(10.0, -0.5, 0.0));

    auto *mesh = ps::registerSurfaceMesh("bar", fem.V, fem.F);
    while (! ps::windowRequestsClose()) {
        fem.step(dt);
        mesh -> updateVertexPositions(fem.v_deformed);
        ps::frameTick();
    }
}