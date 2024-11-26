#include "fem.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using namespace std;
using namespace Eigen;
using json = nlohmann::json;

static const scalar tol = 1e-6;
FEMSimulator ::FEMSimulator(const std::string& config_file)
{
    std::ifstream f(config_file);

    json data = json::parse(f);

    for(auto& obj : data) {
        string tobj = obj["tobj"];

        auto fem_obj = make_unique<FEMObject>(tobj);
        vec3 dp{ 0.0, 0.0, 0.0 };
        if(obj.find("p") != obj.end()) {
            dp = vec3(obj["p"][0], obj["p"][1], obj["p"][2]);
        }
        for(int i = 0; i < fem_obj->n_nodes; i++) {

            fem_obj->vtxs[i].x += dp;
        }
        objects.push_back(move(fem_obj));
    }
    n_dofs = 0;
    for(auto& obj : objects) {
        obj->sys_offset = n_dofs;
        n_dofs += obj->n_nodes * 3;
    }
    A = SparseMatrix<scalar>(n_dofs, n_dofs);
    b = Vector<scalar, -1>(n_dofs);
    dx = Vector<scalar, -1>(n_dofs);
}
void FEMSimulator::combine_triplets()
{
    triplets.resize(0);
    int n_triplets = 0;
    for(auto& obj : objects) {
        n_triplets += obj->triplets.size();
    }

    triplets.reserve(n_triplets);
    for(auto& obj : objects) {
        for(auto& t : obj->triplets) {
            triplets.push_back({ t.row() + obj->sys_offset, t.col() + obj->sys_offset, t.value() });
        }
    }
}
void FEMSimulator::step(scalar dt)
{
    bool term_cond = false;
    int iter = 0;
    do {
        cout << "newton iter" << iter++ << endl;
        A.setZero();
        b.setZero();
        triplets.clear();

        for(auto& obj : objects) {
            obj->compute_f();
            obj->triplets.clear();
            obj->stiffness_kernel();
        }

        combine_triplets();
        for(int i = 0; i < n_dofs; i++) {
            triplets.push_back({ i, i, 1.0f / (dt * dt) });
        }
        A.setFromTriplets(triplets.begin(), triplets.end());

        for(auto& obj : objects) {
            for(int i = obj->sys_offset; i < obj->sys_offset + obj->n_nodes * 3; i++) {
                int ii = (i - obj->sys_offset) / 3;
                int d = (i - obj->sys_offset) % 3;
                auto& vtx = obj->vtxs[ii];

                vec3 dv = vtx.v_n - vtx.v;

                if(obj->vtxs[ii].M_inv == 0.0f) {
                    b(i) = 0.0;
                }
                else {
                    b(i) = (1.0 / dt) * dv[d] + vtx.f[d];
                }
            }
        }
        solve();
        add_dx_dv(dt);
        term_cond = dx.norm() < tol * objects.size();
    } while(!term_cond);

    for(auto& obj : objects) {
        for(int i = 0; i < obj->n_nodes; i++) {
            obj->vtxs[i].v_n = obj->vtxs[i].v;
            obj->v_deformed[i] = obj->vtxs[i].x;
        }
    }
}

void FEMSimulator::add_dx_dv(scalar dt)
{
    for(auto& obj : objects) {
        obj->dx = dx.segment(obj->sys_offset, obj->n_nodes * 3);
        obj->add_dx_dv(dt);
    }
}