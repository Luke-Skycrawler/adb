#include "fem.h"
#include <fstream>
#include <igl/boundary_facets.h>
#include <vector>
#include <array>
using namespace std;
using namespace Eigen;

void TOBJLoader::import_tobj(const string& filename)
{
    ifstream file(filename);
    string line;
    vector<vec3> vertices;
    vector<array<int, 4>> tets;
    while(getline(file, line)) {
        istringstream iss(line);
        string type;
        iss >> type;
        if(type == "v") {
            scalar v0, v1, v2;
            iss >> v0 >> v1 >> v2;
            vertices.push_back(vec3(v0, v1, v2));
        }
        else if(type == "t") {
            int t0, t1, t2, t3;
            iss >> t0 >> t1 >> t2 >> t3;
            tets.push_back({ t0, t1, t2, t3 });
        }
    }
    V.resize(vertices.size(), 3);
    T.resize(tets.size(), 4);
    for(int i = 0; i < vertices.size(); i++) {
        V.row(i) = vertices[i];
    }
    for(int i = 0; i < tets.size(); i++) {
        T.row(i) = Eigen::Vector4i{ tets[i][0], tets[i][1], tets[i][2], tets[i][3] };
    }
}

TOBJLoader::TOBJLoader(const string& filename)
{
    import_tobj(filename);
    n_nodes = V.rows();
    n_tets = T.rows();

    v_deformed.resize(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        v_deformed[i] = V.row(i);
    }
    igl::boundary_facets(T, F);
    cout << filename << "loaded, " << n_nodes << " nodes, " << n_tets << " tets\n";
}
