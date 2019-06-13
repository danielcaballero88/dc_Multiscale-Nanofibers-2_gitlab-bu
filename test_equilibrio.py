import numpy as np
from Malla_equilibrio_2 import Malla
from matplotlib import pyplot as plt

parcon = [0.1e9, 0.1e6, 0.0] # Et, Eb y lamr (que va a ser calculado fibra a fibra)
eccon = 0 # lineal con reclutamiento
pseudovisc = 0.1e11
m = Malla.leer_de_archivo_malla_completa("Malla.txt", parcon, eccon, pseudovisc)

m.guardar_en_archivo("Malla_simplificada.txt")

# m.graficar0()

F = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype=float
)

m.mover_nodos_frontera(F)
dx = m.calcular_incremento()
m.set_x(m.nodos.x+dx)

print dx
print m.nodos.x0
print m.nodos.x

m.graficar(F)

# coors = m.nodos.x
# subfibs = m.sfs
# # grafico las subfibras
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(subfibs.num):
#     xx = list() # valores x
#     yy = list() # valores y
#     # son dos nodos por subfibra
#     # n0 = subfibs[i][0]
#     # n1 = subfibs[i][1]
#     n0, n1 = subfibs.get_con_sf(i)
#     r0 = coors[n0]
#     r1 = coors[n1]
#     xx = [r0[0], r1[0]]
#     yy = [r0[1], r1[1]]
#     p = ax.plot(xx, yy, label=str(i))
# ax.legend(loc="upper left", numpoints=1, prop={"size":6})
# plt.show()