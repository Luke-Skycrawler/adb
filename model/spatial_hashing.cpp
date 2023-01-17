#include "spatial_hashing.h"
#include "barrier.h"
#include <iostream>
#include <set>
#include <unordered_map>
#include <spdlog/spdlog.h>

using namespace std;
using namespace spatial_hashing;

namespace spatial_hashing {

// static unordered_map<hi, unique_ptr<BodyGroup>> pt_table;
// static unordered_map<hi, unique_ptr<BodyGroup>> ee_table;

static const int vec3_compressed_bits = 8, n_entries = 1 << (vec3_compressed_bits * 3), n_overflow_buffer = 8,
                 n_l1_bitmap = n_entries >> 10, n_l2_bitmap = n_entries >> 5, n_buffer = 16;
static element_type count[n_entries], count_overflow;
static Primitive overflow[n_overflow_buffer], hashtable[n_entries * n_buffer];

static bool bitmap_l1[n_l1_bitmap], bitmap_l2[n_l2_bitmap];

static const double MIN_XYZ = -10.0f, MAX_XYZ = 10.0f, dx = 0.2;

hi hash(const vec3i& grid_index)
{
    static const hi mask = (1 << vec3_compressed_bits) - 1;
    hi ret = 0;
    for (int i = 0; i < 3; i++) {
        ret = ret << vec3_compressed_bits;
        ret |= (grid_index(i) & mask);
    }
    return ret;
}

vec3i tovec3i(const vec3& f)
{
    vec3i u = ((f.array() - MIN_XYZ).max(0.0) / dx).cast<int>();
    return u;
}

void register_interval(const vec3i& l, const vec3i& u, const Primitive& t)
{
    for (int i = l(0); i <= u(0); i++)
        for (int j = l(1); j <= u(1); j++)
            for (int k = l(2); k <= u(2); k++) {
                auto idx = hash(vec3i(i, j, k));
                auto& c = count[idx];
                if (c < n_buffer - 1) {
                    if (c == 0) {
                        // new entry on the bitmaps
                        auto l1_pos = idx >> 10, l2_pos = idx >> 5;
                        bitmap_l1[l1_pos] = true;
                        bitmap_l2[l2_pos] = true;
                    }
                    hashtable[idx * n_buffer + c++] = t;
                }
                else {
                    spdlog::error("overflow at {},{},{}", i, j, k);
                    overflow[count_overflow++] = t;
                }
            }
}

vector<Primitive> query_interval(const vec3i& l, const vec3i& u, element_type body_exl)
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

void remove_all_entries()
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

void register_edge(const vec3& a, const vec3& b, element_type body, element_type pid)
{
    vec3i ia(tovec3i(a)), ib(tovec3i(b));
    vec3i u = ia.cwiseMax(ib);
    vec3i l = ia.cwiseMin(ib);
    register_interval(l, u, { pid, body });
}

vector<Primitive> query_edge(const vec3& a, const vec3& b, element_type group_exl, double dhat)
{
    vec3 _u = a.cwiseMax(b).array() + dhat;
    vec3 _l = a.cwiseMin(b).array() - dhat;
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    return query_interval(l, u, group_exl);
}

void register_vertex(const vec3& a, element_type body, element_type pid)
{
    vec3i ia(tovec3i(a));
    register_interval(ia, ia, { pid, body });
}

vector<Primitive> query_triangle(const vec3& a, const vec3& b, const vec3& c, element_type group_exl, double dhat)
{
    vec3 _u = a.cwiseMax(b).cwiseMax(c).array() + dhat;
    vec3 _l = a.cwiseMin(b).cwiseMin(c).array() - dhat;
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    return query_interval(l, u, group_exl);
}
};
