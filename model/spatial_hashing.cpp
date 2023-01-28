#include "spatial_hashing.h"
#include "barrier.h"
#include <iostream>
#include <set>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <atomic>
using namespace std;
hi spatial_hashing::hash(const vec3i& grid_index)
{
    static const hi mask = (1 << vec3_compressed_bits) - 1;
    hi ret = 0;
    for (int i = 0; i < 3; i++) {
        ret = ret << vec3_compressed_bits;
        ret |= (grid_index(i) & mask);
    }
    return ret;
}

vec3i spatial_hashing::tovec3i(const vec3& f)
{
    vec3i u = ((f.array() - MIN_XYZ).max(0.0) / dx).cast<int>();
    return u;
}

spatial_hashing::spatial_hashing(
    int vec3_compressed_bits, int n_buffer,
    double MIN_XYZ, double MAX_XYZ, double dx)
    : vec3_compressed_bits(vec3_compressed_bits), n_entries(1 << (vec3_compressed_bits * 3)), n_overflow_buffer(8), n_l1_bitmap(n_entries >> 10), n_l2_bitmap(n_entries >> 5), n_buffer(n_buffer), MIN_XYZ(MIN_XYZ), MAX_XYZ(MAX_XYZ), dx(dx), count(new atomic<element_type>[n_entries]), overflow(new Primitive[n_overflow_buffer]), hashtable(new Primitive[n_entries * n_buffer]), bitmap_l1(new bool[n_l1_bitmap]), bitmap_l2(new bool[n_l2_bitmap])
{
}

void spatial_hashing::register_interval(const vec3i& l, const vec3i& u, const Primitive& t)
{
    for (int i = l(0); i <= u(0); i++)
        for (int j = l(1); j <= u(1); j++)
            for (int k = l(2); k <= u(2); k++) {
                auto idx = hash(vec3i(i, j, k));          
                element_type c = count[idx] ++;
                if (c < n_buffer - 1) {
                    if (c == 0) {
                        // new entry on the bitmaps
                        auto l1_pos = idx >> 10, l2_pos = idx >> 5;
                        bitmap_l1[l1_pos] = true;
                        bitmap_l2[l2_pos] = true;
                    }
                    hashtable[idx * n_buffer + c] = t;
                }
                else {
                    spdlog::error("overflow at {},{},{}", i, j, k);
                    int c = count_overflow++;
                    if (c < n_overflow_buffer)
                        overflow[c] = t;
                }
            }
}

vector<Primitive> spatial_hashing::query_interval(const vec3i& l, const vec3i& u, element_type body_exl)
{
    const auto cmp = [](const Primitive& a, const Primitive& b) {
        return a.body < b.body || (a.body == b.body && a.pid < b.pid);
    };
    set<Primitive, decltype(cmp)> ret(cmp);
    vector<Primitive> val;
    for (int i = l(0); i <= u(0); i++)
        for (int j = l(1); j <= u(1); j++)
            for (int k = l(2); k <= u(2); k++) {
                auto h = hash(vec3i(i, j, k));
                for (int _i = 0; _i < count[h]; _i++) {
                    auto offset = h * n_buffer + _i;
                    if (hashtable[offset].body != body_exl)
                        ret.insert(hashtable[offset]);
                }
                if (count[h] == n_buffer - 1) {
                    spdlog::error("hash table overflow occured at {}, {}, {}", i, j, k);
                    // hack, dump the whole overflow buffer
                    for (int i = 0; i < count_overflow; i++)
                        if (overflow[i].body != body_exl)
                            ret.insert(overflow[i]);
                }
            }
    for (auto& a : ret) val.push_back(a);
    return val;
}

void spatial_hashing::remove_all_entries()
{
    count_overflow = 0;
    for (int i = 0; i < n_l1_bitmap; i++)
        if (bitmap_l1[i]) {
            bitmap_l1[i] = false;
            for (int j = 0; j < 1 << 5; j++)
                if (bitmap_l2[(i << 5) + j]) {
                    bitmap_l2[(i << 5) + j] = false;
                    for (int k = 0; k < 1 << 5; k++) count[(i << 10) + (j << 5) + k] = 0;
                }
        }
}

void spatial_hashing::register_edge(const vec3& a, const vec3& b, element_type body, element_type pid)
{
    vec3i ia(tovec3i(a)), ib(tovec3i(b));
    vec3i u = ia.cwiseMax(ib);
    vec3i l = ia.cwiseMin(ib);
    register_interval(l, u, { pid, body });
}

void spatial_hashing::register_edge_trajectory(
    const vec3& a0, const vec3& b0,
    const vec3& a1, const vec3& b1,
    element_type body, element_type pid)
{
    vec3i ia0(tovec3i(a0)), ib0(tovec3i(b0));
    vec3i ia1(tovec3i(a0)), ib1(tovec3i(b0));
    vec3i u = ia0.cwiseMax(ib0).cwiseMax(ia1).cwiseMax(ib1);
    vec3i l = ia0.cwiseMin(ib0).cwiseMin(ia1).cwiseMin(ib1);
    register_interval(l, u, { pid, body });
}

vector<Primitive> spatial_hashing::query_edge(const vec3& a, const vec3& b, element_type group_exl, double dhat)
{
    vec3 _u = a.cwiseMax(b).array() + dhat;
    vec3 _l = a.cwiseMin(b).array() - dhat;
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    return query_interval(l, u, group_exl);
}

void spatial_hashing::register_vertex(const vec3& a, element_type body, element_type pid)
{
    vec3i ia(tovec3i(a));
    register_interval(ia, ia, { pid, body });
}

vector<Primitive> spatial_hashing::query_triangle(const vec3& a, const vec3& b, const vec3& c, element_type group_exl, double dhat)
{
    vec3 _u = a.cwiseMax(b).cwiseMax(c).array() + dhat;
    vec3 _l = a.cwiseMin(b).cwiseMin(c).array() - dhat;
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    return query_interval(l, u, group_exl);
}

vector<Primitive> spatial_hashing::query_triangle_trajectory(
    const vec3& a0, const vec3& b0, const vec3& c0,
    const vec3& a1, const vec3& b1, const vec3& c1,
    element_type group_exl)
{
    vec3 _u0 = a0.cwiseMax(b0).cwiseMax(c0);
    vec3 _l0 = a0.cwiseMin(b0).cwiseMin(c0);

    vec3 _u1 = a1.cwiseMax(b1).cwiseMax(c1);
    vec3 _l1 = a1.cwiseMin(b1).cwiseMin(c1);

    vec3i u = tovec3i(_u0.cwiseMax(_u1));
    vec3i l = tovec3i(_l0.cwiseMin(_l1));
    return query_interval(l, u, group_exl);
}

vector<Primitive> spatial_hashing::query_edge_trajectory(
    const vec3& a0, const vec3& b0,
    const vec3& a1, const vec3& b1,
    element_type group_exl)
{
    vec3 _u = a0.cwiseMax(b0).cwiseMax(a1).cwiseMax(b1);
    vec3 _l = a0.cwiseMin(b0).cwiseMin(a1).cwiseMin(b1);
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    return query_interval(l, u, group_exl);
}
