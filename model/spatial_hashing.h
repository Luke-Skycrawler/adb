#include <vector>
#include "cube.h"
struct Primitive{
    int timestamp, type, id, group;
};
struct Group{
    int body, timestamp;
    std::vector<Premitive> primitives;
    void insert(const Premitive &t) {
        primitives.push_back(t);
    }
};


namespace spatial_hashing{
    // int regi(const vec3 &p);
    // int regi_tri(const mat3 &t);  
    // int regi_edge(const vec3 &v1, const vec3 &v2);
    // // unsigned int ftoi(const vec3 &f);
    // std::vector<Primitive> query(const Eigen::Vector3i &max_xyz, const Eigen::Vector3i &min_xyz);
    

    Vector3i ftoi(const vec3 &f);
    unsigned int hash(const Vector3i &grid_index);
    void register_group(int group, unsigned h);
    void register_interval(const Vector3i &l, const Vector3i &u, const Primitive &t);
    std::vector<Primitive> query_interval(const Vector3i &l, cosnt Vector3i &u, int group_exl, int timestep);
    

};
