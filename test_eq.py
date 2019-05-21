import numpy as np 

x0 = np.array(
    [
        [0.0, 0.0],
        [0.5, 1.0],
        [0.0, 1.0],
        [1.0, 0.5],
        [0.5, 0.5] 
    ],
    dtype = float
)

fibras = np.array(
    [ 
        [0, 4, 1],
        [2, 4, 3],
    ],
    dtype = int
)

sf_ns = np.array(
    [
        [0,4],
        [4,1],
        [2,4],
        [4,3]
    ],
    dtype = int
)

num_sf = 4

# direcciones y longitudes de cada segmento
sf_dr0 = np.zeros( (num_sf,2), dtype=float )
sf_dl0 = np.zeros( num_sf, dtype=float )
for i, sf_ns_i in enumerate(sf_ns):
    x0_ini = x0[ sf_ns_i[0] ] 
    x0_fin = x0[ sf_ns_i[1] ]
    dr0 = x0_fin - x0_ini
    sf_dr0[i] = dr0
    sf_dl0[i] = np.sqrt(np.dot(dr0,dr0))



nin_sf = np.array(
    [0,1,2,3],
    dtype = int
)

nin_sf_dir = np.array(
    [-1, +1, -1, +1],
    dtype = int
)

mask_fron = np.array(
    [True, True, True, True, False],
    dtype=bool
)

mask_inter = np.logical_not(mask_fron) # solo para este caso?

# deformacion macro 
F = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype = float
)


# hago la afin
x = np.matmul( x0, np.transpose(F) ) # xij = x0_ik F_kj  -> dado un nodo i: x_j = F_kj x0_j = x0_k F_kj 

# direcciones y longitudes de cada segmento
sf_dr = np.zeros( (num_sf,2), dtype=float )
sf_dl = np.zeros( num_sf, dtype=float )
sf_lam = np.zeros( num_sf, dtype=float )
sf_dir = np.zeros( (num_sf,2), dtype=float )
for i, sf_ns_i in enumerate(sf_ns):
    x_ini = x[ sf_ns_i[0] ] 
    x_fin = x[ sf_ns_i[1] ]
    dr = x_fin - x_ini
    sf_dr[i] = dr
    sf_dl[i] = np.sqrt(np.dot(dr,dr))
    sf_lam[i] = sf_dl[i] / sf_dl0[i]
    sf_dir[i] = sf_dr[i] / sf_dl[i]


# ahora calculo las fuerzas sobre el nodo interseccion 
def fuerza(lam, k=1.0):
    return k*(lam-1.0)

k_fib = 0.1
m_nod = 1.0 
c_nod = 2.0 * np.sqrt(k_fib * m_nod)

fza = np.zeros(2, dtype=float)
for sf in nin_sf:
    lam_sf = sf_lam[sf]
    dir_sf = sf_dir[sf] * nin_sf_dir[sf]
    fza = fza + fuerza(lam_sf, k_fib)*dir_sf

# a preparar pasos de pseudo-tiempo
x_n2 = x 
x_n1 = x 
x_n0 = x 

