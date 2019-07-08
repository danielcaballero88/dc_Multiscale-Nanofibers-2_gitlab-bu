import numpy as np
from Malla_simplificada import Malla as Ms, Iterador
from Malla_completa import Malla as Mc
from matplotlib import pyplot as plt

parcon = [0.1e9, 0.1e3] # Et, Eb
eccon = 0 # lineal con reclutamiento
pseudovisc = 0.2e7

mc = Mc.leer_de_archivo("Malla.txt")

ms = Ms()
ms.simplificar_malla_completa(mc, parcon, eccon, pseudovisc)

# ms.guardar_en_archivo("Malla_simplificada.txt")

# ms = Ms.leer_de_archivo("Malla_simplificada.txt")

# # iterador
# n_sis = ms.nodos.num # numero de variables a resolver (en este caso son arrays de len 2: x e y)
# semilla = ms.nodos.x0
# sistema = ms
# ref_grande = ms.L * 1.0e-1
# ref_divergente = 0.95
# max_iters = 1000
# tolerancia = ms.L*1.0e-16
# ref_pequeno = tolerancia
# ite = Iterador(n_sis, sistema, ref_pequeno, ref_grande, ref_divergente, max_iters, tolerancia)
# # ---


# deformacion
F = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.5]
    ],
    dtype=float
)


# ms.deformar_afin_frontera(F)

# x = ite.iterar()
# ms.set_x(x)

print "---"

ms.deformar_afin(F)

# ms.psv = ms.psv / 100.
# x = ite.iterar()
# ms.set_x(x)

trac_sfs, dirs_sfs = ms.calcular_tracciones_de_subfibras()
trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs, dirs_sfs)
print np.max(trac_nods)

# ms.iterar(numiters=100, delta=0.001)

# trac_sfs = ms.calcular_tracciones_de_subfibras()
# trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs)
# print np.max(trac_nods)


n=44
print n, ms.nodos.x[n], trac_nods[n]

# for i in range(5):
#     ms.iterar(numiters=1, delta=0.1)
#     trac_sfs, dirs_sfs = ms.calcular_tracciones_de_subfibras()
#     trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs, dirs_sfs)
#     print i, n, ms.nodos.x[n], trac_nods[n]

# for i in range(20):
#     ms.iterar(numiters=1, delta=0.01)
#     trac_sfs, dirs_sfs = ms.calcular_tracciones_de_subfibras()
#     trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs, dirs_sfs)
#     print i, n, ms.nodos.x[n], trac_nods[n]

for i in range(100):
    ms.iterar(numiters=1, delta=0.005)
    trac_sfs, dirs_sfs = ms.calcular_tracciones_de_subfibras()
    trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs, dirs_sfs)
    print i, n, np.max(trac_nods)

for i in range(100):
    ms.iterar(numiters=1, delta=0.0001)
    trac_sfs, dirs_sfs = ms.calcular_tracciones_de_subfibras()
    trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs, dirs_sfs)
    print i, n, np.max(trac_nods)

# ms.iterar(numiters=1, delta=0.001)

# for n in range(ms.nodos.num):
#     print n, ":", ms.nodos.x[n], trac_nods[n]

ms.graficar(F)
