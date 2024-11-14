#include "scalar_types.h"

namespace cuda{

    struct ThreadLocalToI {
        scalar* toi;
    
        int *dev_vilist, *dev_fjlist;
        lu *dev_viaabbs, *dev_fjaabbs;
        vec3 *dev_v0s, *dev_v1s;
        Face *dev_f0s, *dev_f1s;
    
        ThreadLocalToI(int nx = 2048, int ny = 2048);
        // reserve memory for (nx, ny) primitive list
        ~ThreadLocalToI();

        scalar pt_list_toi(int nvi, int nfj, int* vilist, int* fjlist, lu* viaabbs, lu* fjaabbs, vec3* v0s, vec3* v1s, Face* f0s, Face* f1s);
        cudaStream_t stream;
    };
};
