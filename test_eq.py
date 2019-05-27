import numpy as np 
from matplotlib import pyplot as plt

# longitud del rve
L_rve = 1.0

# numero de nodos
num_nodos = 8

# numero de nodos frontera
num_fron = 6

# numero de nodos interseccion (o internos)
num_nin = 2

# array mask de nodos frontera
mask_fron = np.array(
    [True, True, True, True, True, True, False, False],
    dtype=bool
)

# array mask de nodos interseccion
mask_inter = np.logical_not(mask_fron) # solo para este caso?

# coordenadas de los nodos
x0 = L_rve * np.array(
    [
        [0.0, 0.1],
        [1.0, 0.9],
        [0.0, 0.7],
        [0.3, 0.0],
        [0.7, 1.0],
        [1.0, 0.3],
        [0.1915, 0.2532],
        [0.8085, 0.7468]
    ],
    dtype = float
)

# conectividad de nodos: subfibras que tocan a cada nodo
nodos_sf = np.array(
    [
        [0],
        [2],
        [3],
        [4],
        [5],
        [6],
        [0,1,3,4],
        [1,2,5,6]
    ],
    dtype = int
)

# sentido de las subfibras que tocan a cada nodo
# +1: subfibra saliente del nodo (segun su vector dada su conectividad)
# -1: subfibra entrante
nodos_sf_dir = np.array(
    [
        [+1],
        [-1],
        [+1],
        [-1],
        [+1],
        [-1],
        [-1, +1, -1, +1],
        [-1, +1, -1, +1]
    ],
    dtype = float
)

# numero de fibras
num_fibras = 3

# conectividad de fibras (nodos en orden)
fibras = np.array(
    [ 
        [0, 6, 7, 1],
        [2, 6, 3],
        [4, 7, 5]
    ],
    dtype = int
)

# numero de subfibras
num_sfs = 7

# conectividad de subfibras (nodos de a pares)
sfs = np.array(
    [
        [0,6],
        [6,7],
        [7,1],
        [2,6],
        [6,3],
        [4,7],
        [7,5]
    ],
    dtype = int
)

# rigidez de las subfibras
sfs_k = np.ones(num_sfs, dtype=float) 

# direcciones y longitudes iniciales de cada segmento
sfs_dr0 = np.zeros( (num_sfs,2), dtype=float )
sfs_dl0 = np.zeros( num_sfs, dtype=float )
for i_sf in range(num_sfs):
    sf_i = sfs[i_sf]
    x0_ini = x0[ sf_i[0] ] 
    x0_fin = x0[ sf_i[1] ]
    dr0 = x0_fin - x0_ini
    sfs_dr0[i_sf] = dr0
    sfs_dl0[i_sf] = np.sqrt(np.dot(dr0,dr0))



# deformacion macro 
F = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype = float
)


# coordenadas bajo deformacion afin
x_a = np.matmul( x0, np.transpose(F) ) # xij = x0_ik F_kj  -> dado un nodo i: x_j = F_kj x0_j = x0_k F_kj 


# funcion para calcular la fuerza que ejerce una subfibra al estirarse
def tension_subfibra(lam, k=1.0):
    if lam>=1.0:
        kef = k
    else: 
        kef = 0.1*k 
    return kef*(lam-1.0)



# a preparar iteraciones
# pseudoviscosidad
psv = 2.0*np.sqrt(np.max(sfs_k)) * np.ones(num_nodos, dtype=float)
tol = L_rve*1.0e-3
ref_small = tol*10.0 # referencia para desplazamiento pequenos
ref_big = L_rve*1.0e-1 # referencia para desplazamientos grandes
ref_div = 0.9 # referencia para divergencia de desplazamientos
maxiter = 100

# adjudico memoria de algunos arrays
Tresul = np.zeros( 2, dtype=float) # fuerza resultante sobre el nodo
sf_dr = np.zeros( (num_sfs,2), dtype=float ) # vectores de las subfibras
sf_dl = np.zeros( num_sfs, dtype=float ) # longitudes de las subfibras
sf_lam = np.zeros( num_sfs, dtype=float ) # elongaciones de las subfibras
sf_t = np.zeros( num_sfs, dtype=float ) # tensiones de las subfibras
sf_a = np.zeros( (num_sfs,2), dtype=float ) # direcciones de las subfibras
x_it1 = np.zeros( (num_nodos, 2), dtype=float ) # posiciones en iteracion anterior "it-1"
x_it = np.zeros( (num_nodos, 2), dtype=float ) # posiciones a calcular en iteracion "it"
dx_it = np.zeros( (num_nodos,2), dtype=float) # cambio de posicion de "it-1" a "it"
dx_it1 = np.zeros( (num_nodos,2), dtype=float) # cambio de posicion de "it-2" a "it-1"
dl_it1 = np.zeros( num_nodos, dtype=float ) # magnitud de desplazamiento dx1
dl_it = np.zeros( num_nodos, dtype=float ) # magnitud de desplazamiento dx
rel_dl = np.zeros( num_nodos, dtype=float ) # relacion de magnitudes de desplazamiento "dl_it/dl_it1"
rec_x = np.zeros( (maxiter+1, num_nodos, 2), dtype=float ) # array para el "movimiento" en iteraciones
flag_small_dx = np.zeros( num_nodos, dtype=bool) # flag para indicar que hubo nodos con desplazamientos pequenisimos
flag_big_dx = np.zeros( num_nodos, dtype=bool) # flag para indicar que hubo nodos con desplazamientos demasiado grandes
flag_divergence = np.zeros( num_nodos, dtype=bool) # flag para indicar que hubo nodos con desplazamientos divergentes

# preparo valor de solucion en iteracion previa (asigno coordenada inicial excepto nodos de dirichlet)
# es necesario para calcular el primer dx_it
x_it1[:] = x0 # iteracion previa "it-1" (valores conocidos)
x_it1[mask_fron] = x_a[mask_fron]
rec_x[0] = x_it1
# voy a resolver cada nodo por separado segun sus fuerzas en la iteracion "it-1"

# muevo los nodos de dirichlet con la deformacion afin 
x_it[mask_fron] = x_a[mask_fron]

# el resto de los nodos se mueve de a poco en iteraciones
it = 0 
flag_primera_iteracion = True 
flag_big_dx[:] = False 
flag_small_dx[:] = False 
flag_divergence[:] = False
while True:
    # calcular tensiones y direcciones de los segmentos en "it-1"
    for j_sf in range(num_sfs):
        sf_j = sfs[j_sf]
        x_ini = x_it1[ sf_j[0] ] 
        x_fin = x_it1[ sf_j[1] ]
        dr = x_fin - x_ini
        sf_dr[j_sf] = dr
        sf_dl[j_sf] = np.sqrt(np.dot(dr,dr))
        sf_lam[j_sf] = sf_dl[j_sf] / sfs_dl0[j_sf]
        sf_t[j_sf] = tension_subfibra(sf_lam[j_sf], sfs_k[j_sf])
        sf_a[j_sf] = sf_dr[j_sf] / sf_dl[j_sf]
    # para cada nodo
    # calcular las tensiones resultantes sobre los nodos en "it-1"
    # y moverlo segun su propia pseudoviscosidad
    for n in nodos_sf:
        # no me interesa saber la tension sobre nodos de frontera
        if mask_fron[n]:
            dx_it[n] = 0.0
        else:
            # para los demas hago la sumatoria
            Tresul[:] = 0.0
            for j_sf in nodos_sf:
                t_sf = sf_t[j_sf]
                versor = sf_a[j_sf] * nodos_sf_dir[j_sf]
                Tresul = Tresul + t_sf*versor
            # mover el nodo
            dx_it[n] = Tresul/psv[n]
            dl_it[n] = np.sqrt( np.dot(dx_it[n],dx_it[n]))
    # ahora que tengo los desplazamientos hago chequeos 
    # primero si es la primera iteracion
    if flag_primera_iteracion:
        flag_primera_iteracion = False 
        for n in range(num_nodos):
            dx_it1[n] = dx_it[n]/ref_div 
            dl_it1[n] = np.sqrt( np.dot(dx_it1[n],dx_it1[n]))
    # si no es la primera iteracion la unica diferencia es que ya voy a tener dx_it1 y dl_it1 
    # ahora calculo la relacion 
    rel_dl = dl_it/dl_it1 # element-wise
    


for n in range(num_nodos):
    if mask_fron[n]: # nodos de dirichlet van con la deformacion afin
        x_it[n] = x_a[n]
    else: # el resto de los nodos buscan el equilibrio de fuerzas
        it = 0
        switch_primera_iteracion = False
        while True:
            it += 1
            # calcular los segmentos (direcciones, longitudes, elongaciones)
            # en la iteracion anterior "it-1"
            for j_sf in range(num_sfs):
                sf_j = sfs[j_sf]
                x_ini = x_it1[ sf_j[0] ] 
                x_fin = x_it1[ sf_j[1] ]
                dr = x_fin - x_ini
                sf_dr[j_sf] = dr
                sf_dl[j_sf] = np.sqrt(np.dot(dr,dr))
                sf_lam[j_sf] = sf_dl[j_sf] / sfs_dl0[j_sf]
                sf_t[j_sf] = tension_subfibra(sf_lam[j_sf], sfs_k[j_sf])
                sf_a[j_sf] = sf_dr[j_sf] / sf_dl[j_sf]
            # calcular las fuerzas 
            Tresul[:] = 0.0
            for j_sf in nodos_sf:
                t_sf = sf_t[j_sf]
                versor = sf_a[j_sf] * nodos_sf_dir[j_sf]
                Tresul = Tresul + t_sf*versor
            # muevo el nodo en base a la fuerza resultante sobre el
            dx_it = Tresul/psv
            dl_it = np.sqrt(np.dot(dx_it, dx_it))
            if not switch_primera_iteracion:
                switch_primera_iteracion = True 
                dx_it1 = dx_it/ref_div 
                dl_it1 = dl_it/ref_div
            rel_dl = dl_it/dl_it1 
            if dl_it<ref_small:
                print "desplazamiento pequeno"
                pass 
            elif dl_it>ref_big:
                print "desplazamiento grande" 
                psv = psv*2.0 
                it -= 1 
                continue 
            elif rel_dl>ref_div:
                print "desplazamiento no convergente"
                psv = psv*2.0 
                it -= 1 
                continue
            # incremento x
            x_it[n] = x_it1[n] + dx_it 
            # imprimo
            print it, Tresul, x_it1[n], x_it[n]
            rec_x[it] = x_it
            err = dl_it 
            if err<tol:
                break
            if it>= maxiter:
                print "maxiter alcanzado" 
                break
            x_it1[:] = x_it
            dx_it1 = dx_it 
            dl_it1 = dl_it 

performed_iters = it

fig = plt.figure()
ax = fig.add_subplot(111)

xx = np.zeros(performed_iters+1) 
yy = np.zeros(performed_iters+1)
for it in range(performed_iters+1):
    xx[it] = rec_x[it,2,0]
    yy[it] = rec_x[it,2,1]

ax.plot(xx, label="x", marker=".", linewidth=1)
ax.legend()
# ax.set_xlim(left=-0.1, right=1.1)
# ax.set_ylim(bottom=-0.1, top=1.1)
plt.show()


