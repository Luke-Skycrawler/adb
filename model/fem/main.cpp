#include "fem.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
// #include "polyscope/point_cloud.h"
using namespace Eigen;
using namespace std;

namespace ps = polyscope;

static const scalar dt = 1e-2;

int test_FEMObject()
{
    // int n_proc = omp_get_num_procs();
    // omp_set_num_threads(n_proc);
    // setNbThreads(n_proc);
    // initParallel();

    FEMObject fem("assets/bar2.tobj");
    ps::options::groundPlaneHeightMode = ps::GroundPlaneHeightMode::Manual;
    ps::options::groundPlaneHeight = -0.5;
    ps::init();
    // ps::view::lookAt(glm::vec3(-1.0, 1.0, 0.0), glm::vec3(10.0, -0.5, 0.0));

    fem.A.setZero();
    fem.triplets.resize(0);
    fem.stiffness_kernel();
    fem.A.setFromTriplets(fem.triplets.begin(), fem.triplets.end());

    auto *mesh = ps::registerSurfaceMesh("bar", fem.V, fem.F);
    int frame = 0;

    while(!ps::windowRequestsClose()) {
        fem.step(dt);
        cout << "frame: " << frame++ << endl;
        mesh->updateVertexPositions(fem.v_deformed);
        ps::frameTick();
    }
    return 0;
}

int test_FEMSimulator()
{
    FEMSimulator sim("../test_cases/fem.json");
    ps::options::groundPlaneHeightMode = ps::GroundPlaneHeightMode::Manual;
    ps::options::groundPlaneHeight = -0.5;
    ps::init();

    vector<ps::SurfaceMesh*> ps_meshes;

    for(int i = 0; i < sim.objects.size(); i++) {
        auto& obj = sim.objects[i];
        auto* mesh = ps::registerSurfaceMesh("bar" + to_string(i), obj->V, obj->F);

        ps_meshes.push_back(mesh);
    }

    int frame = 0;
    while(!ps::windowRequestsClose()) {
        sim.step(dt);
        for(int i = 0; i < sim.objects.size(); i++) {
            ps_meshes[i]->updateVertexPositions(sim.objects[i]->v_deformed);
        }
        ps::frameTick();
    }
    return 0;
}
int main()
{
    test_FEMSimulator();
    return 0;
}