from Malla import Nodos, Subfibras, Malla, Iterador
import numpy as np

nod_coors = [
    [0, 0],
    [5, 0],
    [12, 0],
]

nod_tipos = [1,2,1]

n = Nodos(nod_coors,nod_tipos) 

print "num nodos: ", n.num
print "tipo nodos: ", n.tipos 
print "coordenadas: ", n.x
print "mask fronteras: ", n.mask_fr 
print "mask intersecciones: ", n.mask_in 
print "num fron y num in: ", n.num_fr, n.num_in
print "---"

conec = [
    [0,1],
    [1,2],
]

parcon = [
    [1.0], # subfibra 1 
    [2.0] # subfibra 2
]

s = Subfibras(conec, n.x0, parcon)

print "ne: ", s.ne 
print "ie. ", s.ie 
print "je: ", s.je 
print "---"

pseudovis = [2.0, 2.0, 2.0]

m = Malla(n, s, pseudovis)

F = np.array(
    [
        [20.0/12.0, 0.0],
        [0.0, 1.0]
    ],
    dtype=float
)

m.mover_nodos_frontera(F) 

ref_pequeno = 0.1
ref_grande = 5.0
ref_diverge = 0.9
max_iter = 100
tolerancia = 0.00001
i = Iterador(len(nod_coors), m.nodos.x, m, ref_pequeno, ref_grande, ref_diverge, max_iter, tolerancia)

i.iterar() 

ts = m.calcular_tracciones_de_subfibras()
print m.calcular_tracciones_sobre_nodos(ts)
print m.calcular_incremento()

print "---"
m.set_x(i.x)

ts = m.calcular_tracciones_de_subfibras()
print m.calcular_tracciones_sobre_nodos(ts)
print m.calcular_incremento()