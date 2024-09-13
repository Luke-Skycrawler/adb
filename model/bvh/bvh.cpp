#include <vector>
#include <algorithm>
#include <map>


#include "bvh.h"

inline int longest_axis(const vec3& v)
{
    auto lmax = abs(v[0]);
    int ret(0);
    for( unsigned i=1; i < 3; ++i )
    {
        auto l = abs(v[i]);
        if( l > lmax )
        {
            ret = i;
            lmax = l;
        }
    }
    return ret;
}

inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound[0];
    n.y = bound[1];
    n.z = bound[2];
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

class MedianBVHBuilder
{	
public:

    void build(BVH& bvh, const bounds3* items, int n);

private:

    bounds3 calc_bounds(const bounds3* bounds, const int* indices, int start, int end);

    int partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    int partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    int partition_sah(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);

    int build_recursive(BVH& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent);
};

//////////////////////////////////////////////////////////////////////

void MedianBVHBuilder::build(BVH& bvh, const bounds3* items, int n)
{
    bvh.max_depth = 0;
    bvh.max_nodes = 2*n-1;
    bvh.num_nodes = 0;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_parents = new int[bvh.max_nodes];
    bvh.node_counts = NULL;

    // root is always in first slot for top down builders
    bvh.root = 0;

    if (n == 0)
        return;
    
    std::vector<int> indices(n);
    for (int i=0; i < n; ++i)
        indices[i] = i;

    build_recursive(bvh, items, &indices[0], 0, n, 0, -1);
}


bounds3 MedianBVHBuilder::calc_bounds(const bounds3* bounds, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i=start; i < end; ++i)
        u = bounds_union(u, bounds[indices[i]]);

    return u;
}

struct PartitionPredicateMedian
{
    PartitionPredicateMedian(const bounds3* bounds, int a) : bounds(bounds), axis(a) {}

    bool operator()(int a, int b) const
    {
        return bounds[a].center()[axis] < bounds[b].center()[axis];
    }

    const bounds3* bounds;
    int axis;
};


int MedianBVHBuilder::partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start+end)/2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(&bounds[0], axis));

    return k;
}	
    
struct PartitionPredictateMidPoint
{
    PartitionPredictateMidPoint(const bounds3* bounds, int a, float m) : bounds(bounds), axis(a), mid(m) {}

    bool operator()(int index) const 
    {
        return bounds[index].center()[axis] <= mid;
    }

    const bounds3* bounds;
    int axis;
    float mid;
};


int MedianBVHBuilder::partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices+start, indices+end, PartitionPredictateMidPoint(&bounds[0], axis, mid));

    int k = upper-indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end)
        k = (start+end)/2;


    return k;
}

int MedianBVHBuilder::build_recursive(BVH& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent)
{
    assert(start < end);

    const int n = end-start;
    const int node_index = bvh.num_nodes++;

    assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(bounds, indices, start, end);
    
    const int kMaxItemsPerLeaf = 1;

    if (n <= kMaxItemsPerLeaf)
    {
        bvh.node_lowers[node_index] = make_node(b.lower, indices[start], true);
        bvh.node_uppers[node_index] = make_node(b.upper, indices[start], false);
        bvh.node_parents[node_index] = parent;
    }
    else    
    {
        //int split = partition_midpoint(bounds, indices, start, end, b);
        int split = partition_median(bounds, indices, start, end, b);
        //int split = partition_sah(bounds, indices, start, end, b);

        if (split == start || split == end)
        {
            // partitioning failed, split down the middle
            split = (start+end)/2;
        }
    
        int left_child = build_recursive(bvh, bounds, indices, start, split, depth+1, node_index);
        int right_child = build_recursive(bvh, bounds, indices, split, end, depth+1, node_index);
        
        bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);
        bvh.node_parents[node_index] = parent;
    }

    return node_index;
}

// create only happens on host currently, use bvh_clone() to transfer BVH To device
BVH bvh_create(const bounds3* bounds, int num_bounds)
{
    BVH bvh;
    memset(&bvh, 0, sizeof(bvh));

    MedianBVHBuilder builder;
    //LinearBVHBuilderCPU builder;
    builder.build(bvh, bounds, num_bounds);

    return bvh;
}

void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;
	delete[] bvh.bounds;

    bvh.node_lowers = NULL;
    bvh.node_uppers = NULL;
    bvh.max_nodes = 0;
    bvh.num_nodes = 0;
    bvh.num_bounds = 0;
}

void bvh_refit_recursive(BVH& bvh, int index, const bounds3* bounds)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b)
    {
        const int leaf_index = lower.i;

        (vec3&)lower = bounds[leaf_index].lower;
        (vec3&)upper = bounds[leaf_index].upper;
    }
    else
    {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_recursive(bvh, left_index, bounds);
        bvh_refit_recursive(bvh, right_index, bounds);

        // compute union of children
        const vec3& left_lower = (vec3&)bvh.node_lowers[left_index];
        const vec3& left_upper = (vec3&)bvh.node_uppers[left_index];

        const vec3& right_lower = (vec3&)bvh.node_lowers[right_index];
        const vec3& right_upper = (vec3&)bvh.node_uppers[right_index];

        // union of child bounds
        vec3 new_lower = left_lower.cwiseMin(right_lower);
        vec3 new_upper = left_upper.cwiseMax(right_upper);
        
        // write new BVH nodes
        (vec3&)lower = new_lower;
        (vec3&)upper = new_upper;
    }
}

void bvh_refit_host(BVH& bvh, const bounds3* b)
{
    bvh_refit_recursive(bvh, 0, b);
}

uint64_t bvh_create_host(vec3* lowers, vec3* uppers, int num_bounds)
{
    BVH* bvh = new BVH();
    memset(bvh, 0, sizeof(BVH));

    bvh->context = NULL;

    bvh->lowers = lowers;
    bvh->uppers = uppers;
    bvh->num_bounds = num_bounds;

    bvh->bounds = new bounds3[num_bounds];  

    for (int i=0; i < num_bounds; ++i)
    {
        bvh->bounds[i].lower = lowers[i];
        bvh->bounds[i].upper = uppers[i];
    }

    MedianBVHBuilder builder;
    builder.build(*bvh, bvh->bounds, num_bounds);

    return (uint64_t)bvh;
}

void bvh_refit_host(uint64_t id)
{
    BVH* bvh = (BVH*)(id);

    for (int i=0; i < bvh->num_bounds; ++i)
    {
        bvh->bounds[i] = bounds3();
        bvh->bounds[i].lower = bvh->lowers[i];
        bvh->bounds[i].upper = bvh->uppers[i];
    }

    bvh_refit_host(*bvh, bvh->bounds);
}
