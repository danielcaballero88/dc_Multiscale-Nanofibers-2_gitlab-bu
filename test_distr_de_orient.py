from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np

L = 100.0
Dm = 1.0


mc = Mc(L, Dm)

devangle = 0. * np.pi / 180.
dl = 0.05 * L

for i in range(10):
    mc.make_capa2(dl, Dm, devangle, 0.3)

print len(mc.fibs.con)

# mc = Mc.leer_de_archivo("Malla.txt")

# mc.cambiar_capas(10)

# start = time.clock()
# mc.intersectar_fibras()
# print time.clock() - start

# mc.guardar_en_archivo("Malla.txt")

phis = mc.calcular_orientaciones()
print phis

phis, binwidth, frecs = mc.calcular_distribucion_de_orientaciones(bindata=18)
print phis
print binwidth
print frecs
print np.sum(frecs)


max_phi = np.max(phis)
index = np.where( phis == max_phi)
print index

fig1, ax1 = plt.subplots()
mc.pre_graficar_bordes(fig1, ax1)
mc.pre_graficar_interfibras(fig1, ax1, lamr_min = 1.0, lamr_max = 1.3)
mc.pre_graficar_nodos_interseccion(fig1, ax1)

fig2, ax2 = plt.subplots()
mc.pre_graficar_bordes(fig2, ax2)
mc.pre_graficar_fibras(fig2, ax2, lamr_min=1.0, lamr_max=1.3)
mc.pre_graficar_nodos_interseccion(fig2, ax2)

fig3, ax3 = plt.subplots()
bars1 = ax3.bar(phis, frecs, binwidth*0.5)

plt.show()