from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import truncnorm

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
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


if False: # opcion para graficar las distribuciones de tres mallas en un directorio particular
    cwd = "/home/dancab/Documents/academia/doctorado/articulos/multiscale_nanofibers_randomRVE_2/Analisis_geometria/08_alineacion_mallas_y_pdfs/"
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

if True: # opcion para leer una malla (o varias) segun parametros y graficar histogramas
    Dm = 0.1
    nfibs = 0.3

    ncapss = [2]
    Ls = [250.]
    devangs_deg = [20.]
    dls_rel = [5.]

    nmallas = 1

    cwd = "/home/dancab/Documents/academia/doctorado/articulos/multiscale_nanofibers_randomRVE_2/Analisis_geometria/13_comparacion_geometria_sim_exp/"

    for ncaps in ncapss:
        for L in Ls:
            for dl_rel in dls_rel:
                dl = dl_rel * Dm
                for i, devang_deg in enumerate(devangs_deg):
                    devang = devang_deg*np.pi/180.
                    for nm in range(1,nmallas+1):
                        print "ncaps={:05d}  L = {:08.2f}  devang = {:05.2f}  dl_rel = {:05.2f}  nm = {:07d}".format(ncaps, L, devang_deg, dl_rel, nm)
                        mc = Mc(L, Dm, nfibs, dl, devang)
                        nombrearchivo = cwd + \
                                        "L_" + "{:08.1f}".format(L) + \
                                        "_dlrel_" + "{:05.2f}".format(dl_rel) + \
                                        "_devang_" + "{:05.2f}".format(devang_deg) + \
                                        "_ncaps_" + "{:07d}".format(ncaps) + \
                                        "_nm_" + "{:07d}".format(nm) + \
                                        ".txt"
                        print "leyendo malla"
                        mc = Mc.leer_de_archivo(archivo=nombrearchivo)
                        print "pregraficando"
                        ths, dth, conteo, frecs, pdf = mc.get_histograma_orientaciones(nbins=18, opcion="fibras", csv_file=nombrearchivo[:-4]+"_histograma_virtual_orientaciones.csv")
                        fig = plt.figure(figsize=(8,8))
                        ax = fig.add_subplot(111)
                        graficar_histograma(ths*180/np.pi, frecs*100, figax=(fig,ax))
                        ax.set_xlabel(r"Orientation angle (${}^\circ$)")
                        ax.set_xlim(left=0., right=np.pi)
                        ax.set_xticks(range(0,181,30))
                        ax.set_ylabel("Fiber Percentage")
                        ax.set_ylim([0,50])
                        ax.set_yticks(range(0,51,10))
                        fig.tight_layout()
                        fig.savefig(nombrearchivo[:-4]+"_Histograma_virtual_orientaciones.pdf")

    plt.show()