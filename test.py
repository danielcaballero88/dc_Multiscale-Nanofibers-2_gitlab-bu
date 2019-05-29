from Malla import Nodos, Subfibras, Malla, Iterador
import numpy as np

nod_coors = [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2],
    [1, 1]
]

nod_tipos = [1,1,1,1,2]

n = Nodos(nod_coors,nod_tipos) 

print "num nodos: ", n.num
print "tipo nodos: ", n.tipos 
print "coordenadas: ", n.x
print "mask fronteras: ", n.mask_fr 
print "mask intersecciones: ", n.mask_in 
print "num fron y num in: ", n.num_fr, n.num_in
print "---"

conec = [
    [0,4],
    [4,2],
    [3,4],
    [4,1]
]

s = Subfibras(conec, n.x0, [0.1])

print "ne: ", s.ne 
print "ie. ", s.ie 
print "je: ", s.je 
print "---"


m = Malla(n, s, 0.1)

F = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype=float
)

m.mover_nodos_frontera(F) 

i = Iterador(len(nod_coors), m.nodos.x, m, 0.0001, 0.5, 0.9, 100, 1.0e-6)

i.iterar() 

ts = m.calcular_tracciones_de_subfibras()
print m.calcular_tracciones_sobre_nodos(ts)
print m.calcular_incremento()

print "---"
m.set_x(i.x)

ts = m.calcular_tracciones_de_subfibras()
print m.calcular_tracciones_sobre_nodos(ts)
print m.calcular_incremento()