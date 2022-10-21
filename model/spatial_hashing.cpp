#include <unordered_map>
#include <set>
#include "spatial_hashing.h"
#include "barrier.h"
using namespace std;
using namespace spatial_hashing;

static const unordered_map<unsigned, vector<Group>> table;

namespace spatial_hashing{
    static const float MIN_XYZ = -10.0f, MAX_XYZ = 10.0f;

    unsigned hash(const Vector3i &grid_index){
        unsigned mask = 0x3ff, ret = 0;
        ret = ret << 10;
        ret |= (grid_index(i) & mask);
        return ret;
    }

    Vector3i ftoi(const vec3 &f){
        Eigen::Vector3i u;
        for (int i =0; i < 3; i ++){
            u(i) = max(MIN_XYZ, min(f(i), MAX_XYZ));
        } 
        return u;
    }

    void register_interval(const Vector3i &l, const Vector3i &u, const Primitive &t) {
        int group = t.group;
        for(int i = l(0); i < u(0); i ++){
            for(int j = l(1); j < u(1); j++){
                for (int  k = l(2); k < l(2); k++){
                    auto &g = register_group(group, hash(i, j, k));
                    g.insert(t);
                }
            }
        }
    } 
    void register_group(int group, unsigned h){
        auto i = table.find(h);
        if (i == table.end()) {
            table.at(h) = new vector<Group>;
            h.push_basck(group);
        }
        else {
            table[h].push_back(group);
        }
    }

    set<Primitive> query_interval(const Vector3i &l, const Vector3i &u, int group_exl, int ts) {
        set<Primitive> ret;
        for (int i = l(0); i < u(0); i ++){
            for (int j = l(1); j < u(1); j ++){
                for (int k = l(2); k < u(2); k ++) {
                    unsigned h = hash(i,j,k);
                    if (table.find(h) != table.end()) {
                        for (auto g: table[h]){
                            if (g.body != group_exl) {
                                for (auto &p: g.primitives){
                                    if (p.timestep == ts) {
                                        ret.insert(p);
                                        // FIXME: possible redundent elements, use set to eliminate
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }    
    }
};