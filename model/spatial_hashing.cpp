#include "spatial_hashing.h"
#include "barrier.h"
#include <iostream>
#include <set>
#include <unordered_map>

using namespace std;
using namespace spatial_hashing;



namespace spatial_hashing {
static unordered_map<hi, unique_ptr<BodyGroup>> table;
static const double MIN_XYZ = -10.0f, MAX_XYZ = 10.0f, dx = 0.02;

hi hash(const vec3i& grid_index)
{
    unsigned mask = 0x1fffff, ret = 0;
    for (int i = 0; i < 3; i++) {
        ret = ret << 21;
        ret |= (grid_index(i) & mask);
    }
    return ret;
}

vec3i tovec3i(const vec3& f)
{
    vec3i u = ((f.array() - MIN_XYZ).max(0) / dx).cast<int>();
    return u;
}


std::vector<unsigned>& register_group(unsigned body, hi h)
{
    auto it = table.find(h);
    if (it == end(table)) {
        auto container_grid = make_unique<BodyGroup>(body, nullptr);
        table[h] = move(container_grid);
        it = table.find(h);
    }
    auto &container_grid = *(it -> second);
    auto jt = container_grid.find(body);
    if (jt != end(container_grid)) {
        return *(jt -> second);
    }
    else {
        container_grid[body] = make_unique<std::vector<unsigned>>();
        jt = container_grid.find(body);
        return *(jt -> second);
    }
}

void register_interval(const vec3i& l, const vec3i& u, const Primitive& t)
{
    int body = t.body;
    
    for (int i = l(0); i <= u(0); i++) for (int j = l(1); j <= u(1); j++) for (int k = l(2); k <= u(2); k++) {
        auto idx = hash(vec3i(i, j, k));
        auto &g{register_group(body, idx)};
        g.push_back(t.body);
    }
}

vector<Primitive> query_interval(const vec3i& l, const vec3i& u, int body_exl)
{
    vector<Primitive> ret;
    for (int i = l(0); i <= u(0); i++) for (int j = l(1); j <= u(1); j++) for (int k = l(2); k <= u(2); k++) {
        unsigned h = hash(vec3i(i, j, k));
        auto it = table.find(h);
        if (it == table.end()) return ret;
        auto &container_grid = *(it->second);
        for (auto &jt: container_grid) {
            auto body = jt.first;
            if (body_exl != body) for (auto kt: *(jt.second))
                ret.push_back({kt, body});
        }
    }
    return ret;
}

void remove_all_entries()
{
    table.clear();
}
}; // namespace spatial_hashing