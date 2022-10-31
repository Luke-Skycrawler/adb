#pragma once
#include <vector>
#include <set>
#include "cube.h"
struct Primitive{
    int timestamp, type, id, group;
    Primitive(int timestamp, int id, int group, int type = 0): timestamp(timestamp), id(id), group(group), type(type) {}
    
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
    
    inline void reigster_triangle(const vec3 &a, const vec3 &b, const vec3 &c, const Primitive &t, int ts){
        vec3i ia(tovec3i(a)), ib(tovec3i(b)), ic(tovec3i(c));
        vec3i u = ia.cwiseMax(ib).cwiseMax(ic);
        vec3i l = ia.cwiseMin(ib).cwiseMin(ic);
        register_interval(l, u, t, ts);
    }
    
    inline void register_edge(const vec3 &a, const vec3 &b, const Primitive &t, int ts){
        vec3i ia(tovec3i(a)), ib(tovec3i(b));
        vec3i u = ia.cwiseMax(ib);
        vec3i l = ia.cwiseMin(ib);
        register_interval(l, u, t, ts);
    }

    inline std::vector<Primitive> query_edge(const vec3 &a, const vec3 &b, int group_exl, int timestep){
        vec3i ia(tovec3i(a)), ib(tovec3i(b));
        vec3i u = ia.cwiseMax(ib);
        vec3i l = ia.cwiseMin(ib);
        return query_interval(l, u, group_exl, timestep);
    }

    inline void register_triangle_orbit(const vec3 &a0, const vec3 &b0, const vec3 &c0, const vec3 &a1, const vec3 &b1, const vec3 &c1, const Primitive &t, int ts){
        vec3i ia0(tovec3i(a0)), 
            ib0(tovec3i(b0)), 
            ic0(tovec3i(c0)), 
            ia1(tovec3i(a1)), 
            ib1(tovec3i(b1)), 
            ic1(tovec3i(c1));

        vec3i u = ia0.cwiseMax(ib0).cwiseMax(ic0).cwiseMax(ia1).cwiseMax(ib1).cwiseMax(ic1);
        vec3i l = ia0.cwiseMin(ib0).cwiseMin(ic0).cwiseMin(ia1).cwiseMin(ib1).cwiseMin(ic1);
        register_interval(l, u, t, ts);
    }

    inline void register_vertex(const vec3 &a, const Primitive &t, int ts) {
        vec3i ia(tovec3i(a));
        register_interval(ia, ia, t, ts);
    }
};
