import numpy as np 
from matplotlib import pyplot as plt

L_rve = 1.0

num_nodos = 5

x0 = L_rve * np.array(
    [
        [0.0, 0.0],
        [0.5, 1.0],
        [0.0, 1.0],
        [1.0, 0.5],
        [0.5, 0.5] 
    ],
    dtype = float
)

num_fibras = 2

fibras = np.array(
    [ 
        [0, 4, 1],
        [2, 4, 3],
    ],
    dtype = int
)

num_sfs = 4

sfs = np.array(
    [
        [0,4],
        [4,1],
        [2,4],
        [4,3]
    ],
    dtype = int
)


# direcciones y longitudes de cada segmento
sf_dr0 = np.zeros( (num_sfs,2), dtype=float )
sf_dl0 = np.zeros( num_sfs, dtype=float )
for i_sf in range(num_sfs):
    sf_i = sfs[i_sf]
    x0_ini = x0[ sf_i[0] ] 
    x0_fin = x0[ sf_i[1] ]
    dr0 = x0_fin - x0_ini
    sf_dr0[i_sf] = dr0
    sf_dl0[i_sf] = np.sqrt(np.dot(dr0,dr0))



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
x_a = np.matmul( x0, np.transpose(F) ) # xij = x0_ik F_kj  -> dado un nodo i: x_j = F_kj x0_j = x0_k F_kj 

# direcciones y longitudes de cada segmento

# ahora calculo las fuerzas sobre el nodo interseccion 
def fuerza(lam, k=1.0):
    return k*(lam-1.0)

# propiedad de las fuerzas de las fibras
k_fib = 0.1

# a preparar iteraciones
# pseudoviscosidad
psv = 2*np.sqrt(k_fib)
tol = L_rve*0.01
ref_small = tol*5 # referencia para desplazamiento pequenos
ref_big = L_rve*0.1 # referencia para desplazamientos grandes
ref_div = 0.9 # referencia para divergencia de desplazamientos
maxiter = 100
# adjudico memoria de algunos arrays
FzaRes = np.zeros(2, dtype=float) # fuerza resultante sobre el nodo
sf_dr = np.zeros( (num_sfs,2), dtype=float ) # vectores de las subfibras
sf_dl = np.zeros( num_sfs, dtype=float ) # longitudes de las subfibras
sf_lam = np.zeros( num_sfs, dtype=float ) # elongaciones de las subfibras
sf_dir = np.zeros( (num_sfs,2), dtype=float ) # direcciones de las subfibras
x_it = np.zeros( (num_nodos, 2), dtype=float )
rec_x = np.zeros( (maxiter, num_nodos, 2), dtype=float )
# iteraciones
x_it1 = x_a # iteracion previa "it-1" (valores conocidos)
rec_x[0] = x_it1
# voy a resolver cada nodo por separado segun sus fuerzas en la iteracion "it-1"
for n in range(num_nodos):
    # nodos de dirichlet van con la deformacion afin
    if mask_fron[n]:
        x_it[n] = x_a[n] 
    # el nodo interseccion se mueve segun las fuerzas
    it = 0
    switch_primera_iteracion = False
    while True:
        it += 1
        if it>= maxiter:
            print "maxiter alcanzado" 
            break
        # calcular los segmentos (direcciones, longitudes, elongaciones)
        # en la iteracion anterior "it-1"
        for j_sf in range(num_sfs):
            sf_j = sfs[j_sf]
            x_ini = x_it1[ sf_j[0] ] 
            x_fin = x_it1[ sf_j[1] ]
            dr = x_fin - x_ini
            sf_dr[j_sf] = dr
            sf_dl[j_sf] = np.sqrt(np.dot(dr,dr))
            sf_lam[j_sf] = sf_dl[j_sf] / sf_dl0[j_sf]
            sf_dir[j_sf] = sf_dr[j_sf] / sf_dl[j_sf]
        # calcular las fuerzas 
        FzaRes[:] = 0.0
        for j_sf in nin_sf:
            lam_sf = sf_lam[j_sf]
            versor = sf_dir[j_sf] * nin_sf_dir[j_sf]
            FzaRes = FzaRes + fuerza(lam_sf, k_fib)*versor
        # muevo el nodo en base a la fuerza resultante sobre el
        dx_it = FzaRes/psv
        dl_it = np.sqrt(np.dot(dx_it, dx_it))
        if not switch_primera_iteracion:
            switch_primera_iteracion = True 
            dx_it1 = dx_it/ref_div 
            dl_it1 = dl_it/ref_div
            it -= 1 
            continue 
        rel_dl = dl_it/dl_it1 
        if dl_it<ref_small:
            print "desplazamiento pequeno"
            pass 
        elif dl_it>ref_big:
            print "desplazamiento grande" 
            psv = psv*2.0 
            it -= 1 
            continue 
        elif rel_dl<ref_div:
            print "desplazamiento no convergente"
            psv = psv*2.0 
            it -= 1 
            continue 
        x_it[4] = x_it1[4] + dx_it 
        rec_x[it] = x_it
        x_it1 = x_it
        dx_it1 = dx_it 
        dl_it1 = dl_it 


fig = plt.figure()
ax = fig.add_subplot(111)

xx = np.zeros(maxiter) 
yy = np.zeros(maxiter)
for it in range(maxiter):
    xx[it] = rec_x[it,4,0]
    yy[it] = rec_x[it,4,1]

ax.plot(xx,yy, label="x", marker="")
ax.legend()
ax.set_xlim(left=-0.1, right=1.1)
ax.set_ylim(bottom=-0.1, top=1.1)
plt.show()
        
print FzaRes
print x_it


