from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np

L = 100.0
Dm = 1.0


ncaps = 10
L = 50.
devang_deg = 10.
dl_rel = 1.
nm = 1
nombrearchivo = "mallas/" + \
                "L_" + "{:08.1f}".format(L) + \
                "_dlrel_" + "{:05.2f}".format(dl_rel) + \
                "_devang_" + "{:05.2f}".format(devang_deg) + \
                "_ncaps_" + "{:07d}".format(ncaps) + \
                "_nm_" + "{:07d}".format(nm) + \
                ".txt"

mc = Mc.leer_de_archivo(archivo=nombrearchivo)

phis = mc.calcular_orientaciones()
mean = np.mean(phis)
stdev = np.std(phis)
print phis
print mean
print stdev


phis, binwidth, frecs = mc.calcular_distribucion_de_orientaciones(bindata=9)
print phis
print binwidth
print frecs
print np.sum(frecs)


max_phi = np.max(phis)
index = np.where( phis == max_phi)
print index

fig1, ax1 = plt.subplots()
mc.pre_graficar_bordes(fig1, ax1)
mc.pre_graficar_fibras(fig1, ax1, lamr_min = None, lamr_max = None, byn=False, color_por="capa")
mc.pre_graficar_nodos_interseccion(fig1, ax1)


# frecs = np.array(frecs, dtype=float) / float(np.sum(frecs))
frecs = [frec/float(np.sum(frecs)) for frec in frecs]
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, polar=True)
phis_polar = phis*2
phis_polar[9:] = [phi_polar + np.pi for phi_polar in phis]
frecs_polar = frecs*2
phis_polar += phis_polar[:1]
frecs_polar += frecs_polar[:1]
ax3.plot(phis_polar, frecs_polar, linewidth=2, linestyle="solid", color="k", marker="o")
ax3.fill(phis_polar, frecs_polar, color="gray", alpha=0.2)

plt.show()