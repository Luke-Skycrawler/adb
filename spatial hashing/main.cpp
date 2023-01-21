#include <spdlog/spdlog.h>
#include "../model/spatial_hashing.h"
using namespace std;

int main(void) {
    vector<vec3> ps;
    vector<vec3> ts;
    static const int n_p = 20, n_t = 20;
    ps.resize(n_p);
    int i = 0;
    for (auto &p : ps) {
        p = { i * 0.5, i * 0.5, i * 0.5 };
        i++;
        spatial_hashing::register_vertex(p, 0, i);
    }
    ts.push_back({ 0.0, 0.0, 0.0 });
    ts.push_back({ 0.0, 1.0, 0.0 });
    ts.push_back({ 10.0, 10.0, 10.0 });
    auto t = spatial_hashing::query_triangle(ts[0], ts[1], ts[2], 1, 0.0);
    for (auto& u : t) {
        spdlog::info("{}, {}", u.body, u.pid);
    }
	return 0;
}