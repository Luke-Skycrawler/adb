#include "time_integrator.h"
#include "barrier.h"
#include "spdlog/spdlog.h"
#include "collision.h"
#include "../view/global_variables.h"
#include "marcros_settings.h"


using namespace std;
using namespace barrier;
using namespace Eigen;

VectorXd& cat_all(vector<Cube> & cubes) {
    VectorXd *ret = new VectorXd;
    for (auto &c : cubes) {
        
    }
    return *ret;
}
void implicit_euler(vector<Cube> & cubes, double dt) {
    bool term_cond;
    int iter = 0;
    double sup_dq;
    do {
        for i 
        term_cond = sup_dq > 1e-6 && iter < globals.max_iter;
    }
    while (! term_cond);
}