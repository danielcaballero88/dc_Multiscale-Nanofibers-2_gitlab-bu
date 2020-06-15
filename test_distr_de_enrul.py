from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
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

if False: # esto es para graficar alguna malla o mallas de archivos

    cwd = "mallas/"
    nombrearchivo = "000_malla_prueba_i.txt"
    nombresarchivos = [cwd+nombrearchivo]


    for i, nombrearchivo in enumerate(nombresarchivos):
        print "leyendo malla"
        mc = Mc.leer_de_archivo(archivo=nombrearchivo)
        print "pregraficando"
        lrs, dlr, conteo, frecs, pdf = mc.get_histograma_lamr(lamr_min=1., lamr_max=1.05, nbins=10, binwidth=None, opcion="interfibras")
        for x,y in zip(lrs,frecs):
            print ("{:20.8f}"*2).format(x,y)
        print np.sum(conteo), len(mc.fibs.con)
        # frecs[-1] = 0. # para que quede mas lindo el grafico
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
        fig, ax1 = graficar_histograma(lrs, frecs, figax=(fig,ax1))
        ax1.set_xlabel(r"Reclutamiento ($\lambda_r$)")
        ax1.set_ylabel("Fraccion de fibras")
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # fig.tight_layout()
        # nombrefigura = nombresfigs[i]
        # plt.show()
        # fig.savefig(nombrefigura, bbox="tight")

    plt.show()


if True: # esto es para leer de archivos pero para muchas mallas con parametros diferentes
    Dm = 0.1
    nfibs = 0.3

    ncapss = [2]
    Ls = [250.]
    devangs_deg = [20.]
    dls_rel = [5.]

    nmallas = 1

    cwd = "/home/dancab/Documents/academia/doctorado/articulos/multiscale_nanofibers_randomRVE_2/Analisis_geometria/13_comparacion_geometria_sim_exp/"
    cwd = "mallas/"

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
                        lrs, dlr, conteo, frecs, pdf = mc.get_histograma_lamr(lamr_min=1., lamr_max=1.8, nbins=10, binwidth=None, opcion="fibras", csv_file=nombrearchivo[:-4]+"_histograma_virtual_reclutamiento.csv")
                        # frecs[-1] = 0. # para que quede mas lindo el grafico
                        fig = plt.figure(figsize=(8,6))
                        ax = fig.add_subplot(111)
                        fig, ax = graficar_histograma(lrs, frecs, figax=(fig,ax))
                        ax.set_xlabel(r"Initial tortuosity")
                        ax.set_ylabel("Fiber fraction")
                        # ax.set_xlim(left=1., right=list_maxlamsr[i])
                        # ax.set_xticks(list_xticks[i])
                        fig.tight_layout()
                        # fig.savefig(nombresfigs[i])
                        # plt.show()


    plt.show()




if False: # Este es para crear mallas y graficar sus distr de reclut
    Dm = 1.0
    nfibs = 0.1

    ncapss = [1]
    Ls = [200.]
    devangs_deg = [17.]
    dls_rel = [5.]

    fundisor = None

    nmallas = 3

    cwd = "mallas/"

    start = time.time()
    for ncaps in ncapss:
        for L in Ls:
            for dl_rel in dls_rel:
                dl = dl_rel * Dm
                for devang_deg in devangs_deg:
                    devang = devang_deg*np.pi/180.
                    for nm in range(1,nmallas+1):
                        print "ncaps={:05d}  L = {:08.2f}  devang = {:05.2f}  dl_rel = {:05.2f}  nm = {:07d}".format(ncaps, L, devang_deg, dl_rel, nm)
                        mc = Mc(L, Dm, nfibs, dl, devang, fundisor=None)
                        for i in range(1,ncaps+1):
                            mc.make_capa2(dl, Dm, devang, nfibs, orient_distr=fundisor)
                        # mc.intersectar_fibras()
                        nombrearchivo = cwd + \
                                        "L_" + "{:08.1f}".format(L) + \
                                        "_dlrel_" + "{:05.2f}".format(dl_rel) + \
                                        "_devang_" + "{:05.2f}".format(devang_deg) + \
                                        "_ncaps_" + "{:07d}".format(ncaps) + \
                                        "_nm_" + "{:07d}".format(nm) + \
                                        ".txt"
                        # mc.guardar_en_archivo(nombrearchivo)
                        lrs, dlr, conteo, frecs, pdf = mc.get_histograma_lamr(lamr_min=1., lamr_max=1.8, nbins=10, binwidth=None, opcion="interfibras")
                        fig = plt.figure(figsize=(8,6))
                        ax = fig.add_subplot(111)
                        fig, ax = graficar_histograma(lrs, frecs, figax=(fig,ax))
                        ax.set_xlabel(r"Reclutamiento ($\lambda_r$)")
                        ax.set_ylabel("Fraccion de fibras")
                        fig.tight_layout()


    plt.show()
