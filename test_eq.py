import numpy as np 

x = np.array(
    [
        [0.2, 0.0],
        [0.8, 1.0],
        [0.0, 0.8],
        [1.0, 0.2],
        [0.5, 0.5] 
    ],
    dtype = float
)

fibras = np.array(
    [ 
        [0, 5, 1],
        [2, 5, 3],
    ],
    dtype = int
)

subfibras = np.array(
    [
        [0,5],
        [5,1],
        [2,5],
        [5,3]
    ],
    dtype = int
)

nin_sf = np.array(
    [0,1,2,3],
    dtype = int
)

nin_sf_dir = np.array(
    [-1, +1, -1, +1],
    dtype = int
)

mask_x_fron = np.array(
    [True, True, True, True, False],
    dtype=bool
)

mask_x_in = np.logical_not(mask_x_fron) # solo para este caso?

# deformacion macro 
F = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype = float
)


# hago la afin
Fx = np.matmul( x, np.transpose(F) ) # Fx_ij = x_ik F_kj  -> dado un nodo i: Fx_j = x_k F_kj

# ahora calculo las fuerzas...