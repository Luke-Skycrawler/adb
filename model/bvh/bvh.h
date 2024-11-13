#pragma once
#include "float.h"
#ifndef CUDA_SOURCE
#include "../affine_body.h"
#else
#include "../cuda/common.cuh"
#endif
struct BVHPackedNodeHalf
{
	float x;
	float y;
	float z;
	int i : 31;
	int b : 1;
};

struct bounds3
{
	inline func bounds3() : lower( FLT_MAX, FLT_MAX, FLT_MAX)
						           , upper(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}

	inline func bounds3(const vec3& lower, const vec3& upper) : lower(lower), upper(upper) {}

	inline func vec3 center() const { return 0.5*(lower+upper); }
	inline func vec3 edges() const { return upper-lower; }


	inline func bool empty() const { return lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2]; }

	inline func bool overlaps(const vec3& p) const
	{
		if (p[0] < lower[0] ||
			p[1] < lower[1] ||
			p[2] < lower[2] ||
			p[0] > upper[0] ||
			p[1] > upper[1] ||
			p[2] > upper[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	inline func bool overlaps(const bounds3& b) const
	{
		if (lower[0] > b.upper[0] ||
			lower[1] > b.upper[1] ||
			lower[2] > b.upper[2] ||
			upper[0] < b.lower[0] ||
			upper[1] < b.lower[1] ||
			upper[2] < b.lower[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}


	vec3 lower;
	vec3 upper;

};


struct BVH
{
    BVHPackedNodeHalf* node_lowers;
    BVHPackedNodeHalf* node_uppers;

	// used for fast refits
	int* node_parents;
	int* node_counts;
	
	int max_depth;
	int max_nodes;
    int num_nodes;

	int root;

    vec3* lowers;
	vec3* uppers;
	bounds3* bounds;
	int num_bounds;

	void* context;
};

inline BVH bvh_get(uint64_t id)
{
    return *(BVH*)(id);
}

BVH bvh_create(const bounds3* bounds, int num_bounds);

void bvh_destroy_host(BVH& bvh);

void bvh_refit_host(BVH& bvh, const bounds3* bounds);


// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct bvh_query_t
{
    bvh_query_t()
    {
    }
    bvh_query_t(int)
    {
    } // for backward pass

    BVH bvh;

	// BVH traversal stack:
	int stack[32];
	int count;

    // inputs
	bool is_ray;
    vec3 input_lower;	// start for ray
    vec3 input_upper;	// dir for ray

	int bounds_nr;
};

inline bvh_query_t bvh_query(
    uint64_t id, bool is_ray, const vec3& lower, const vec3& upper)
{
    // This routine traverses the BVH tree until it finds
	// the first overlapping bound. 

    // initialize empty
	bvh_query_t query;

	query.bounds_nr = -1;

	BVH bvh = bvh_get(id);

	query.bvh = bvh;
	query.is_ray = is_ray;


    // if no bvh nodes, return empty query.
    if (bvh.num_nodes == 0)
    {
		query.count = 0;
		return query;
	}

    // optimization: make the latest
	
	query.stack[0] = bvh.root;
	query.count = 1;
    query.input_lower = lower;
    query.input_upper = upper;

    bounds3 input_bounds(query.input_lower, query.input_upper);

    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
		const int node_index = query.stack[--query.count];

		BVHPackedNodeHalf node_lower = bvh.node_lowers[node_index];
		BVHPackedNodeHalf node_upper = bvh.node_uppers[node_index];

		vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
		vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        bounds3 current_bounds(lower_pos, upper_pos);

		{
	        if (!input_bounds.overlaps(current_bounds))
				// Skip this box, it doesn't overlap with our target box.
				continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
			// found very first leaf index.
			// Back up one level and return 
			query.stack[query.count++] = node_index;
			return query;
        }
        else
        {	
		  query.stack[query.count++] = left_index;
		  query.stack[query.count++] = right_index;
		}
	}	

	return query;
}

inline bvh_query_t bvh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper)
{
	return bvh_query(id, false, lower, upper);
}

inline bounds3 bounds_union(const bounds3& a, const bounds3& b) 
{
	return bounds3(a.lower.cwiseMin(b.lower), a.upper.cwiseMax(b.upper));
}

inline bool bvh_query_next(bvh_query_t& query, int& index)
{
    BVH bvh = query.bvh;
	
	bounds3 input_bounds(query.input_lower, query.input_upper);

    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
        const int node_index = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = bvh.node_lowers[node_index];
        BVHPackedNodeHalf node_upper = bvh.node_uppers[node_index];

        vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
        vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        bounds3 current_bounds(lower_pos, upper_pos);

		{
	        if (!input_bounds.overlaps(current_bounds))
				// Skip this box, it doesn't overlap with our target box.
				continue;
		}

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        if (node_lower.b)
        {
            // found leaf
            query.bounds_nr = left_index;
			index = left_index;
            return true;
        }
        else
        {

            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
    return false;
}

