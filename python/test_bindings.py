import abdtk
import numpy as np
import ipctk
if __name__ == "__main__":
    arr = np.array([
        [0.4360, 0.0259, 0.5497],
        [0.4353, 0.4204, 0.3303],
        [0.2046, 0.6193, 0.2997],
        [0.2668, 0.6211, 0.5291]
    ])
    ij = np.arange(4)
    pt_type = abdtk.PointTriangleDistanceType.P_T
    # pt_type = pt_type.P_T
    pt = [arr[i] for i in range(4)]
    ij = [i for i in range(4)]
    dist = ipctk.point_plane_distance(arr[0], arr[1], arr[2], arr[3])
    H, g = abdtk.ipc_hess_pt_12x12(pt, ij, pt_type, dist)
    np.printoptions(precision=4, suppress=True)
    print(H)
    print(g)
    
