from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import truncnorm

SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def graficar_histograma(x, y, figax=None, x_offset=0., dx_mult=1., edgecolor="k", color="k", label=None):
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

cwd = "/home/dancab/Documents/academia/doctorado/articulos/multiscale_nanofibers_randomRVE_2/Analisis_geometria/07_alineacion_mallas_y_pdfs/"
nombresarchivos = ["uniforme.txt", "stdev_0.2.txt", "stdev_0.1.txt"]
nombresarchivos = [cwd + nombrearchivo for nombrearchivo in nombresarchivos]
scales = [10.*np.pi, 0.2*np.pi, 0.1*np.pi]
nombresfigs = [cwd + "analisis_alineacion_" + item + "_distribucion.pdf" for item in ("uniforme", "moderada", "alta")]

npuntos_pdf = 1001
thetas_pdf = np.linspace(0., np.pi, npuntos_pdf)


for i, nombrearchivo in enumerate(nombresarchivos):
    print "leyendo malla"
    mc = Mc.leer_de_archivo(archivo=nombrearchivo)
    print "pregraficando"
    ths, dth, conteo, frecs, pdf = mc.get_histograma_orientaciones(nbins=10, opcion="fibras")
    print np.sum(conteo)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    graficar_histograma(ths, pdf, figax=(fig,ax))
    scale = scales[i]
    curva_pdf = truncnorm.pdf(thetas_pdf, -0.5*np.pi/scale, 0.5*np.pi/scale, loc=0.5*np.pi, scale=scale)
    ax.plot(thetas_pdf, curva_pdf, linewidth=2, color="r")
    # ax.set_xlabel(r"Orientacion")
    ax.set_xticks([0., 0.5*np.pi, np.pi])
    ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1]+.2, .2))
    # ax.set_yticks(ax.get_ylim())
    ax.set_xlim(left=0., right=np.pi)
    nombrefigura = nombresfigs[i]
    # plt.show() # para debug
    fig.savefig(nombrefigura, bbox="tight")

# plt.show()


# L = 100.0
# Dm = 1.0


# ncaps = 10
# L = 50.
# devang_deg = 10.
# dl_rel = 1.
# nm = 1
# nombrearchivo = "mallas/" + \
#                 "L_" + "{:08.1f}".format(L) + \
#                 "_dlrel_" + "{:05.2f}".format(dl_rel) + \
#                 "_devang_" + "{:05.2f}".format(devang_deg) + \
#                 "_ncaps_" + "{:07d}".format(ncaps) + \
#                 "_nm_" + "{:07d}".format(nm) + \
#                 ".txt"

# mc = Mc.leer_de_archivo(archivo=nombrearchivo)

# phis = mc.calcular_orientaciones()
# mean = np.mean(phis)
# stdev = np.std(phis)
# print phis
# print mean
# print stdev


# phis, binwidth, frecs = mc.calcular_distribucion_de_orientaciones(bindata=9)
# print phis
# print binwidth
# print frecs
# print np.sum(frecs)


# max_phi = np.max(phis)
# index = np.where( phis == max_phi)
# print index

# fig1, ax1 = plt.subplots()
# mc.pre_graficar_bordes(fig1, ax1)
# mc.pre_graficar_fibras(fig1, ax1, lamr_min = None, lamr_max = None, byn=False, color_por="capa")
# mc.pre_graficar_nodos_interseccion(fig1, ax1)


# # frecs = np.array(frecs, dtype=float) / float(np.sum(frecs))
# frecs = [frec/float(np.sum(frecs)) for frec in frecs]
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, polar=True)
# phis_polar = phis*2
# phis_polar[9:] = [phi_polar + np.pi for phi_polar in phis]
# frecs_polar = frecs*2
# phis_polar += phis_polar[:1]
# frecs_polar += frecs_polar[:1]
# ax3.plot(phis_polar, frecs_polar, linewidth=2, linestyle="solid", color="k", marker="o")
# ax3.fill(phis_polar, frecs_polar, color="gray", alpha=0.2)

# plt.show()