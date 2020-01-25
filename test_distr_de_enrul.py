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

def graficar_histograma(x, y, figax=None, x_offset=0., dx_mult=1., edgecolor="k", color="k", label=None):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #
    if figax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, polar=False)
    else:
        fig, ax = figax
    dx = x[1]-x[0]
    ax.bar(x + x_offset, y, alpha=0.5, width=dx*dx_mult, edgecolor=edgecolor, color=color, label=label)
    ax.tick_params(axis='both', which='major', pad=15)
    return fig, ax

nombresarchivos = ["mallas/uniforme.txt", "mallas/stdev_0.1.txt", "mallas/stdev_0.2.txt"]

for nombrearchivo in nombresarchivos:
    print "leyendo malla"
    mc = Mc.leer_de_archivo(archivo=nombrearchivo)
    print "pregraficando"
    lrs, dlr, conteo, frecs = mc.get_histograma_lamr(lamr_min=None, lamr_max=None, nbins=10, opcion="fibras")
    fig, ax = graficar_histograma(lrs, frecs)

plt.show()

# fig1, ax1 = plt.subplots()
# mc.pre_graficar_bordes(fig1, ax1)
# mc.pre_graficar_interfibras(fig1, ax1, lamr_min = lamr_min, lamr_max = lamr_max)
# mc.pre_graficar_nodos_interseccion(fig1, ax1)

# fig2, ax2 = plt.subplots()
# mc.pre_graficar_bordes(fig2, ax2)
# mc.pre_graficar_interfibras(fig2, ax2, lamr_min=None, lamr_max=None, byn=False, color_por="lamr")
# mc.pre_graficar_nodos_interseccion(fig2, ax2, markersize=2)

# fig3, ax3 = plt.subplots()
# bars1 = ax3.bar(lrs, frecs, dlr*0.8)
# # ax3.set_ylim(bottom=0.0, top=1.0)
# # ax3.set_xlim(left=1.0, right=2.0)


# for lr, frec in zip(lrs,frecs):
#     print "{:10.4f}{:10.4f}".format(lr, frec)


# plt.show()