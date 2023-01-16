#include "spatial_hashing.h"
#include "barrier.h"
#include <iostream>
#include <set>
#include <unordered_map>

using namespace std;
using namespace spatial_hashing;



namespace spatial_hashing {
static unordered_map<hi, unique_ptr<BodyGroup>> pt_table;
static unordered_map<hi, unique_ptr<BodyGroup>> ee_table;

static const double MIN_XYZ = -10.0f, MAX_XYZ = 10.0f, dx = 0.02;

hi hash(const vec3i& grid_index)
{
    hi mask = 0x1fffff, ret = 0;
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

std::vector<unsigned>& register_group(unsigned body, hi h, unordered_map<hi, unique_ptr<BodyGroup>>& table)
{
    auto it = table.find(h);
    if (it == end(table)) {
        auto container_grid = make_unique<BodyGroup>();
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

void register_interval(const vec3i& l, const vec3i& u, const Primitive& t, unordered_map<hi, unique_ptr<BodyGroup>>& table)
{
    auto body = t.body;
    
    for (int i = l(0); i <= u(0); i++) for (int j = l(1); j <= u(1); j++) for (int k = l(2); k <= u(2); k++) {
        auto idx = hash(vec3i(i, j, k));
        auto& g{ register_group(body, idx, table) };
        g.push_back(t.body);
    }
}

vector<Primitive> query_interval(const vec3i& l, const vec3i& u, int body_exl, unordered_map<hi, unique_ptr<BodyGroup>>& table)
{
    vector<Primitive> ret;
    for (int i = l(0); i <= u(0); i++) for (int j = l(1); j <= u(1); j++) for (int k = l(2); k <= u(2); k++) {
        auto h = hash(vec3i(i, j, k));
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
    pt_table.clear();
    ee_table.clear();
}

void register_edge(const vec3& a, const vec3& b, unsigned body, unsigned pid)
{
    vec3i ia(tovec3i(a)), ib(tovec3i(b));
    vec3i u = ia.cwiseMax(ib);
    vec3i l = ia.cwiseMin(ib);
    register_interval(l, u, { pid, body }, ee_table);
}

vector<Primitive> query_edge(const vec3& a, const vec3& b, int group_exl, double dhat)
{
    auto _u = a.cwiseMax(b).array() + dhat;
    auto _l = a.cwiseMin(b).array() - dhat;
    auto u = tovec3i(_u);
    auto l = tovec3i(_l);
    return query_interval(l, u, group_exl, ee_table);
}

void register_vertex(const vec3& a, unsigned body, unsigned pid)
{
    vec3i ia(tovec3i(a));
    register_interval(ia, ia, { pid, body }, pt_table);
}

vector<Primitive> query_triangle(const vec3& a, const vec3& b, const vec3& c, int group_exl, double dhat)
{
    auto _u = a.cwiseMax(b).cwiseMax(c).array() + dhat;
    auto _l = a.cwiseMin(b).cwiseMin(c).array() - dhat;
    auto u = tovec3i(_u);
    auto l = tovec3i(_l);
    return query_interval(l, u, group_exl, pt_table);
}
};
