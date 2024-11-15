#define _USE_MATH_DEFINES
#include <cmath>
#include "../model/affine_obj.h"
#include "../model/cube.h"
#include "../model/geometry.h"
#include "global_variables.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include "cnpy.h"
using namespace std;
using json = nlohmann::json;

Eigen::Matrix<scalar, -1, -1> load(const string& npy){
    auto phi = cnpy::npy_load(npy);
    int nc = phi.shape[1], nr = phi.shape[0];

    if (phi.word_size == sizeof(float)) {
        Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> Phi{phi.data<float>(), nr, nc};
        return Phi.cast<scalar>();
    }
    else {
        Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> Phi{phi.data<double>(), nr, nc};
        return Phi.cast<scalar>();
    }
    
}

static const vec3 omega(0.0, 0.0, 20.0);
unique_ptr<Cube> spinning_cube()
{
    auto _cube = make_unique<Cube>();
    auto& cube = *_cube;
    auto a = skew(omega);
    for (int i = 0; i < 3; i++) cube.dqdt[i + 1] = a[i];
    cube.q[0] = vec3(0.0, 1.0, 0.0);
    cube.q0 = cube.q;
    return _cube;
}

void cube_blocks(int n)
{
    globals.cubes.clear();
    for (int i = 0; i < n; i++) {
        auto _cube = make_unique<Cube>();
        auto& cube = *_cube;
        auto R = skew(omega);
        for (int j = 0; j < 3; j++) cube.dqdt[j + 1] = R[j];
        cube.dqdt[0] = vec3(0.0, i * -5.0, 0.0);

        cube.q[0] = vec3(i * 0.5, 1.0 * (i * 1.5 + 0.1), i * 0.5);
        cube.q0 = cube.q;
        globals.cubes.push_back(move(_cube));
    }
}
void customize(string file)
{
    vector<Cube> cubes;
    std::fstream f(file);
    json data = json::parse(f);
    globals.cubes.clear();
    for (auto &it: data) {
        bool isobj = it.find("obj") != it.end();
        string objfile = isobj ? it["obj"] : "";
        unique_ptr<AffineBody> _a;
        if (isobj) {

            if (globals.loaded_models.find(objfile) == end(globals.loaded_models)) {
                auto model = make_unique<Model>(objfile);
                globals.loaded_models[objfile] = move(model);
            }
            auto& mesh{ globals.loaded_models[objfile]->meshes[0] };
            _a = make_unique<AffineObject>(mesh);

            if(it.find("npy") != it.end()) {
                assert(it.find("lambda") != it.end());
                auto npy = it["npy"];
                auto lam_npy = it["lambda"];
                _a -> Phi = load(npy);
                _a -> lam = load(lam_npy);
                assert(_a -> lam.rows() == _a -> Phi.cols());
                // filter out interior vertices (the npy Phi correspond to a mesh, with surface vertices are at the front)
                _a -> Phi.conservativeResize(_a -> n_vertices, NoChange);
            }
            
        }
        else
            _a = make_unique<Cube>();

        auto& a = *_a;
        array<scalar, 3> omega = {0.0, 0.0, 20.0};
        omega = it["omega"];
        auto p = it["p"];
        auto p_dot = it["p_dot"];
        bool use_euler = it.find("euler") != it.end();

        bool mass_ = it.find("mass") != end(it);
        if (mass_) {
            scalar m = it["mass"];
            a.mass = m;
            a.Ic = m / 12;
        }
        
        a.dqdt[0] = vec3(p_dot[0], p_dot[1], p_dot[2]);
        a.q[0] = vec3(p[0], p[1], p[2]);
        auto R = skew(vec3(omega[0], omega[1], omega[2]));
        for (int i = 0; i < 3; i++) a.dqdt[i + 1] = R[i];
        if (use_euler) {
            auto ds = it["euler"];
            scalar  aa = ds[0] / 180.0 * M_PI, 
                    bb = ds[1] / 180.0 * M_PI, 
                    cc = ds[2] / 180.0 * M_PI;
            mat3 rotate = rotation(cc, bb, aa);
            for (int i = 0; i < 3; i++) a.q[i + 1] = rotate.col(i);
        }
        else {
            scalar theta = use_euler ? 0.0 : it["theta"] / 180.0 * M_PI;
            scalar s = sin(theta), c = cos(theta);
            a.q[1](0) = c;
            a.q[2](1) = c;
            a.q[2](0) = -s;
            a.q[1](1) = s;
        }
        a.q0 = a.q;
        globals.cubes.push_back(move(_a));
    }
}