#include "spatial_hashing.h"
#include "barrier.h"
#include <iostream>
#include <set>
#include <unordered_map>

using namespace std;
using namespace spatial_hashing;

static unordered_map<unsigned, vector<Group>*> table;

namespace spatial_hashing {
static const double MIN_XYZ = -10.0f, MAX_XYZ = 10.0f;

unsigned hash(const vec3i& grid_index)
{
    unsigned mask = 0x3ff, ret = 0;
    for (int i = 0; i < 3; i++) {
        ret = ret << 10;
        ret |= (grid_index(i) & mask);
    }
    return ret;
}

vec3i tovec3i(const vec3& f)
{
    vec3i u;
    for (int i = 0; i < 3; i++) {
        // u(i) = int(max(MIN_XYZ * 512, min(MAX_XYZ,f(i) * 512 ) * 512));
        u(i) = int(f(i) * 512);
    }
    return u;
}

void register_interval(const vec3i& l, const vec3i& u, const Primitive& t,
    int ts)
{
    int group = t.group;
    for (int i = l(0); i <= u(0); i++) {
        for (int j = l(1); j <= u(1); j++) {
            for (int k = l(2); k <= u(2); k++) {
                auto& g(register_group(group, hash(vec3i(i, j, k)), ts));
                g.primitives.push_back(t);
                int __ = 0;
            }
        }
    }
}
Group& register_group(int group, unsigned h, int ts)
{
    auto i = table.find(h);
    if (i == table.end()) {
        vector<Group>* v = table[h] = new vector<Group>;
        v->push_back(Group(group, ts));
        return v->back();
    }
    for (auto& g : *table[h]) {
        if (g.body == group) {
            g.timestamp = ts;
            return g;
        }
    }

    table[h]->push_back(Group(group, ts));
    return table[h]->back();
}

vector<Primitive> query_interval(const vec3i& l, const vec3i& u, int group_exl,
    int ts)
{
    vector<Primitive> ret;
    for (int i = l(0); i <= u(0); i++) {
        for (int j = l(1); j <= u(1); j++) {
            for (int k = l(2); k <= u(2); k++) {
                unsigned h = hash(vec3i(i, j, k));
                if (table.find(h) != table.end()) {
                    for (auto& g : *table[h]) {
                        if (g.body != group_exl) {
                            for (auto& p : g.primitives) {
                                if (p.timestamp == ts) {
                                    ret.push_back(p);
                                    // FIXME: possible redundent elements, use set to eliminate
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return ret;
}

void remove_all_entries()
{
    // cout << "printing entries of hash table" << endl;
    for (auto& t : table) {
        // cout << (t.first) << endl;
        delete t.second;
    }
    table.clear();
}
}; // namespace spatial_hashing