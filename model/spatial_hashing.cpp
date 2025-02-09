#include "spatial_hashing.h"
#include "barrier.h"
#include <iostream>
#include <set>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <atomic>
#include <unordered_set>
#include <omp.h>
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
    scalar MIN_XYZ, scalar MAX_XYZ, scalar dx, int set_size)
    : vec3_compressed_bits(vec3_compressed_bits), n_entries(1 << (vec3_compressed_bits * 3)), n_overflow_buffer(8), n_l1_bitmap(n_entries >> 8), n_l2_bitmap(n_entries >> 3), n_buffer(n_buffer), MIN_XYZ(MIN_XYZ), MAX_XYZ(MAX_XYZ), dx(dx), count(new atomic<element_type>[n_entries]), overflow(new Primitive[n_overflow_buffer]), hashtable(new Primitive[n_entries * n_buffer]), bitmap_l1(new bool[n_l1_bitmap]), bitmap_l2(new bool[n_l2_bitmap]), set_size(set_size), count_non_atomic(new element_type[n_entries])
{
    auto n_proc = omp_get_num_procs();
    sets = new unordered_set<int>[n_proc];
    collisions = new vector<Primitive>[n_proc];
    for (int i = 0; i < n_proc; i++) {
        sets[i].max_load_factor(0.5);
        sets[i].reserve(n_buffer * set_size);
        collisions[i].reserve(n_buffer * set_size);
    }
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
                        auto l1_pos = idx >> 8, l2_pos = idx >> 3;
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

void spatial_hashing::query_interval(const vec3i& l, const vec3i& u, element_type body_exl, std::vector<Primitive>& val)
{
    // const auto cmp = [](const Primitive& a, const Primitive& b) {
    //     return a.body < b.body || (a.body == b.body && a.pid < b.pid);
    // };
    // unordered_set<Primitive, decltype(cmp)> ret(cmp);
    auto tid = omp_get_thread_num();
    unordered_set<int>& ret = sets[tid];
    val.resize(0);
    ret.clear();
    // ret.reserve(n_buffer * set_size);
    // ret.max_load_factor(0.5);

    for (int i = l(0); i <= u(0); i++)
        for (int j = l(1); j <= u(1); j++)
            for (int k = l(2); k <= u(2); k++) {
                auto h = hash(vec3i(i, j, k));
                int cnt = min(int(count_non_atomic[h]), n_buffer);
                for (int _i = 0; _i < cnt; _i++) {
                    auto offset = h * n_buffer + _i;
                    auto p = hashtable[offset].pbody;
                    // int ele = (static_cast<int>(p.body) << 16) | static_cast<int>(p.pid);
                    auto ele = hashtable[offset].as_uint;
                    if (p.body != body_exl  && ret.find(ele)== ret.end()) {
                        // ret.insert(ele);
                        val.push_back({ele});
                    }
                }
                if (cnt >= n_buffer) {
                    spdlog::error("hash table overflow occured at {}, {}, {}", i, j, k);
                    // hack, dump the whole overflow buffer
                    cnt = count_overflow;
                    for (int i = 0; i < min(cnt, n_overflow_buffer); i++)
                    {
                        auto o = overflow[i].pbody;
                        auto ele = overflow[i].as_uint;
                        if (o.body != body_exl)
                            // ret.insert(ele);
                            val.push_back({ele});
                    }
                }
            }

    std::sort(val.begin(), val.end());
    val.erase(std::unique(val.begin(), val.end()), val.end());
    // val.reserve(ret.size());
    // for (auto& a : ret) val.push_back({ static_cast<element_type>(a & 0xffff), static_cast<element_type>(a >> 16) });
    // for (auto& a : ret) val.push_back({ a });
}

void spatial_hashing::remove_all_entries()
{
    count_overflow = 0;
    fill(count_non_atomic, count_non_atomic + n_entries, 0);
    for (int i = 0; i < n_l1_bitmap; i++)
        if (bitmap_l1[i]) {
            bitmap_l1[i] = false;
            for (int j = 0; j < 1 << 5; j++)
                if (bitmap_l2[(i << 5) + j]) {
                    bitmap_l2[(i << 5) + j] = false;
                    for (int k = 0; k < 1 << 3; k++) {
                        count_non_atomic[(i << 8) + (j << 3) + k] = count[(i << 8) + (j << 3) + k];
                        count[(i << 8) + (j << 3) + k] = 0;
                    }
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

void spatial_hashing::query_edge(const vec3& a, const vec3& b, element_type group_exl, scalar dhat,
    std::vector<Primitive>& ret)
{
    vec3 _u = a.cwiseMax(b).array() + dhat;
    vec3 _l = a.cwiseMin(b).array() - dhat;
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    query_interval(l, u, group_exl, ret);
}

void spatial_hashing::register_vertex(const vec3& a, element_type body, element_type pid)
{
    vec3i ia(tovec3i(a));
    register_interval(ia, ia, { pid, body });
}

void spatial_hashing::query_triangle(const vec3& a, const vec3& b, const vec3& c, element_type group_exl, scalar dhat,
    std::vector<Primitive>& ret)
{
    vec3 _u = a.cwiseMax(b).cwiseMax(c).array() + dhat;
    vec3 _l = a.cwiseMin(b).cwiseMin(c).array() - dhat;
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    return query_interval(l, u, group_exl, ret);
}

void spatial_hashing::query_triangle_trajectory(
    const vec3& a0, const vec3& b0, const vec3& c0,
    const vec3& a1, const vec3& b1, const vec3& c1,
    element_type group_exl,
    std::vector<Primitive>& ret)
{
    vec3 _u0 = a0.cwiseMax(b0).cwiseMax(c0);
    vec3 _l0 = a0.cwiseMin(b0).cwiseMin(c0);

    vec3 _u1 = a1.cwiseMax(b1).cwiseMax(c1);
    vec3 _l1 = a1.cwiseMin(b1).cwiseMin(c1);

    vec3i u = tovec3i(_u0.cwiseMax(_u1));
    vec3i l = tovec3i(_l0.cwiseMin(_l1));
    query_interval(l, u, group_exl, ret);
}

void spatial_hashing::query_edge_trajectory(
    const vec3& a0, const vec3& b0,
    const vec3& a1, const vec3& b1,
    element_type group_exl,
    std::vector<Primitive>& ret)
{
    vec3 _u = a0.cwiseMax(b0).cwiseMax(a1).cwiseMax(b1);
    vec3 _l = a0.cwiseMin(b0).cwiseMin(a1).cwiseMin(b1);
    vec3i u = tovec3i(_u);
    vec3i l = tovec3i(_l);
    query_interval(l, u, group_exl, ret);
}



func void SpatialHash::register_prim(const lu& bb, const Entry &body_prim) {
    const i3 l = bb.lower.array().floor().cast<int>(); 
    const i3 u = bb.upper.array().floor().cast<int>();
    for (int ii = l[0]; ii <= u[0]; ii ++){
        for (int jj = l[1]; jj <= u[1]; jj ++){
            for (int kk = l[2]; kk <= u[2]; kk ++){
                int morton_code = morton({ii, jj, kk});
                push_back(morton_code, body_prim);
            }
        }
    }
}

func void SpatialHash::clear() {
    for (int i = 0; i < hash_table.size(); i++) {
        hash_table[i].next = -1;
        hash_table[i].cnt = 0;
    }
}

func void SpatialHash::query(const lu & bb, const Entry &body_prim) {
    const i3 l = bb.lower.array().floor().cast<int>(); 
    const i3 u = bb.upper.array().floor().cast<int>();
    for (int ii = l[0]; ii <= u[0]; ii ++){
        for (int jj = l[1]; jj <= u[1]; jj ++){
            for (int kk = l[2]; kk <= u[2]; kk ++){
                int morton_code = morton({ii, jj, kk});
                
                int cnt = hash_table[morton_code].cnt;
                for (int i = 0; i < cnt; i++) {
                    Entry entry = hash_table[morton_code].entries[i];
                    append_collision_ee(body_prim, entry);
                }
                // ignore overflow for now
            }
        }
    }
}

func void SpatialHash::query(const vec3 & p, const Entry &body_prim) {
    const i3 pi = p.array().floor().cast<int>(); 
    int morton_code = morton(pi);
    
    int cnt = hash_table[morton_code].cnt;
    for (int i = 0; i < cnt; i++) {
        Entry entry = hash_table[morton_code].entries[i];
        append_collision_pt(body_prim, entry);
    }
    // ignore overflow for now
}

func void SpatialHash::append_collision_pt(const Entry &bpi, const Entry &bpj) {
    i4 ij; 
    ij << bpi.body_prim, bpj.body_prim;
    if (bpi.bb.overlaps(bpj.bb) && ij[0] != ij[2]) {
        idx.push_back(ij);
    }
}

func void SpatialHash::append_collision_ee(const Entry &bpi, const Entry &bpj) {
    i4 ij; 
    ij << bpi.body_prim, bpj.body_prim;
    if (bpi.bb.overlaps(bpj.bb) && ij[0] != ij[2]) {
        eidx.push_back(ij);
    }
}

func void SpatialHash::push_back(int idx, const Entry &entry) {
    TableData &td = hash_table[idx];
    if (td.cnt < cap) {
        // FIXME: concurrency
        td.entries[td.cnt] = entry;
        td.cnt ++;
    } else {
        int nxt = td.next;
        td.next = overflow_cnt ++;
        OverflowData &od = overflow_table[td.next];
        od.data = entry;
        od.next = nxt;         
    }
}

inline func uint64_t split_by_3(unsigned int a){
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}
    
inline func uint64_t morton_encode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    uint64_t answer = 0;
    answer |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2;
    return answer;
}


func int SpatialHash::morton(const i3 &xyz) {
    int x = xyz[0] + bound, y = xyz[1] + bound, z = xyz[2] + bound;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (z < 0) z = 0;
    if (x >= bound * 2) x = bound * 2 - 1; 
    if (y >= bound * 2) y = bound * 2 - 1;
    if (z >= bound * 2) z = bound * 2 - 1;
    
    unsigned ux = static_cast<unsigned>(x);
    unsigned uy = static_cast<unsigned>(y);
    unsigned uz = static_cast<unsigned>(z);

    auto ui = morton_encode_magicbits(ux, uy, uz);
    return static_cast<int>(ui);
}

void SpatialHash::collision_detect() {
    pts.resize(0);
    idx.resize(0);
    ees.resize(0);
    eidx.resize(0);
    vidx.resize(0);
    {
        // point-triangle
        clear();
        for (int b = 0; b < n_cubes; b ++) {
            for (int t = 0; t < cubes[b]->n_faces; t ++) {
                // Face f = cubes[b].face(dof, t);
                Face f = cubes[b]->face(t, false, false);
                i2 body_prim;
                body_prim << b, t;
                lu bb{ compute_aabb(f, d_hat_sqrt) };
                register_prim(bb, { body_prim, bb });
            }
        }
        for (int b = 0; b < n_cubes; b ++) {
            for (int p = 0; p < cubes[b]->n_vertices; p ++) {
                // vec3 x = cubes[b].vert(dof, p);
                vec3 x = cubes[b]-> vt1(p);
                i2 body_prim;
                body_prim << b, p;
                query(x, {body_prim, {x, x}});
            }
        }
    }

    {
        // edge-edge
        clear(); 

        for (int b = 0; b < n_cubes; b ++) {
            for (int e = 0; e < cubes[b]->n_edges; e ++) {
                Edge edge = cubes[b] -> edge(e, false, false);
                i2 body_prim;
                body_prim << b, e;
                lu bb = compute_aabb(edge, d_hat_sqrt / 2.0);
                register_prim(bb, {body_prim, bb});
            }
        }
        for (int b = 0; b < n_cubes; b ++) {
            for (int e = 0; e < cubes[b]->n_edges; e ++) {
                // Edge edge = cubes[b].edge(dof, e);
                Edge edge = cubes[b]->edge(e, false, false);
                i2 body_prim;
                body_prim << b, e;
                lu bb = compute_aabb(edge, d_hat_sqrt / 2.0);
                query(bb, {body_prim, bb});
            }
        }
    }
    
    {
        // point-ground
        for (int b = 0; b < n_cubes; b++) {
            for (int p = 0; p < cubes[b]->n_vertices; p++) {
                // vec3 x = cubes[b].vert(dof, p);
                vec3 x = cubes[b]->vt1(p);
                i2 body_prim;
                body_prim << b, p;
                if (x[1] < d_hat_sqrt) {
                    vidx.push_back(body_prim);
                }                
            }
        }
    }
}