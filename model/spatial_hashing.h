#include <vector>
#include <set>
#include "cube.h"
struct Primitive{
    int timestamp, type, id, group;
};
struct Group{
    int body, timestamp;
    std::vector<Primitive> primitives;
    void insert(const Primitive &t) {
        primitives.push_back(t);
    }
    Group(int group, int ts): body(group), timestamp(ts) {}
};


namespace spatial_hashing{
    using vec3i = Vector<int, 3>;
    // int regi(const vec3 &p);
    // int regi_tri(const mat3 &t);  
    // int regi_edge(const vec3 &v1, const vec3 &v2);
    // // unsigned int tovec3i(const vec3 &f);
    // std::vector<Primitive> query(const Eigen::vec3i &max_xyz, const Eigen::vec3i &min_xyz);
    

    vec3i tovec3i(const vec3 &f);
    unsigned int hash(const vec3i &grid_index);
    Group& register_group(int group, unsigned h, int ts);
    void register_interval(const vec3i &l, const vec3i &u, const Primitive &t, int ts);
    std::vector<Primitive> query_interval(const vec3i &l, const vec3i &u, int group_exl, int timestep);
    
    void reigster_triangle(const vec3 &a, const vec3 &b, const vec3 &c, const Primitive &t, int ts){
        vec3i ia(tovec3i(a)), ib(tovec3i(b)), ic(tovec3i(c));
        vec3i u = ia.cwiseMax(ib).cwiseMax(ic);
        vec3i l = ia.cwiseMin(ib).cwiseMin(ic);
        register_interval(l, u, t, ts);
    }
    
    void register_edge(const vec3 &a, const vec3 &b, const Primitive &t, int ts){
        vec3i ia(tovec3i(a)), ib(tovec3i(b));
        vec3i u = ia.cwiseMax(ib);
        vec3i l = ia.cwiseMin(ib);
        register_interval(l, u, t, ts);
    }
};
