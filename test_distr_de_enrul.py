from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np

def get_histograma_lamr(mc, lamr_min=None, lamr_max=None, nbins=5, opcion="fibras"):
    if opcion=="fibras":
        lamsr = mc.calcular_enrulamientos()
        lrs, dlr, conteo = mc.calcular_distribucion_de_enrulamiento(lamr_min=lamr_min, lamr_max=lamr_max, n=nbins)
    elif opcion=="interfibras":
        lamsr = mc.calcular_enrulamientos_de_interfibras()
        lrs, dlr, conteo = mc.calcular_distribucion_de_enrulamiento_de_interfibras(lamr_min=lamr_min, lamr_max=lamr_max, n=nbins)
    else:
        raise ValueError
    # print np.max
    frecs = np.array(conteo, dtype=float) / float(np.sum(conteo))
    # print lrs
    # print dlr
    # print conteo
    # print np.array(conteo, dtype=float)/float(np.sum(conteo))
    # print np.sum(conteo)
    max_lamr = np.max(lamsr)
    index = np.where( lamsr == max_lamr)
    # print index, max_lamr
    return lrs, dlr, conteo, frecs

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
                "_i.txt"

mc = Mc.leer_de_archivo(archivo=nombrearchivo)

lrs, dlr, conteo, frecs = get_histograma_lamr(mc, lamr_min=1.0, lamr_max=2.0, nbins=10, opcion="interfibras")

# fig1, ax1 = plt.subplots()
# mc.pre_graficar_bordes(fig1, ax1)
# mc.pre_graficar_interfibras(fig1, ax1, lamr_min = lamr_min, lamr_max = lamr_max)
# mc.pre_graficar_nodos_interseccion(fig1, ax1)

fig2, ax2 = plt.subplots()
mc.pre_graficar_bordes(fig2, ax2)
mc.pre_graficar_interfibras(fig2, ax2, lamr_min=None, lamr_max=None, byn=False, color_por="lamr")
mc.pre_graficar_nodos_interseccion(fig2, ax2, markersize=2)

fig3, ax3 = plt.subplots()
bars1 = ax3.bar(lrs, frecs, dlr*0.8)
# ax3.set_ylim(bottom=0.0, top=1.0)
# ax3.set_xlim(left=1.0, right=2.0)


for lr, frec in zip(lrs,frecs):
    print "{:10.4f}{:10.4f}".format(lr, frec)


plt.show()