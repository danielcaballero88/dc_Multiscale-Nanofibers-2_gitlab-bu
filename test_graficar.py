from Malla_completa import Malla as Mc
from Mallita import Mallita as Ms
import time
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




if False:
    cwd = "mallas/"
    # nombresarchivos = ["malla_prueba.txt", "malla_prueba_i.txt"]
    nombresarchivos = ["malla_prueba.txt"]
    nombresarchivos = [cwd + nombrearchivo for nombrearchivo in nombresarchivos]
    # # nombresfigs = [cwd + "analisis_alineacion_" + item + "_malla.pdf" for item in ("uniforme", "moderada", "alta")]
    # # nombresfigs = ["volfrac_0.1_i.pdf", "volfrac_0.3_i.pdf"]

    for i, nombrearchivo in enumerate(nombresarchivos):
        print "leyendo malla"
        mc = Mc.leer_de_archivo(archivo=nombrearchivo)
        print "pregraficando"
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        mc.pre_graficar_bordes(fig, ax)
        # mc.pre_graficar_fibras(fig, ax, ncapas=None, byn=True, color_por="capa", colores_cm=[(1,0,0), (0,0,1), (1,0,0)], barracolor=False)
        colores_cm = [(1,0,0), (0,1,0), (0,0,1), (0,0,0), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.4), (0.4, 0.4, 0.4)]*10
        mc.pre_graficar_interfibras(fig, ax, byn=False, color_por="nada", colormap="Dark2", barracolor=False, colores_cm=colores_cm , ncolores_cm=len(colores_cm))
        mc.pre_graficar_nodos_interseccion(fig, ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(nombrearchivo[:-4]+".pdf")
        # intersectada
        print "leyendo malla intersectada"
        nombrearchivo = nombrearchivo[:-4] + "_i.txt"
        mc = Mc.leer_de_archivo(archivo=nombrearchivo)
        print "pregraficando"
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        mc.pre_graficar_bordes(fig, ax)
        colores_cm = [(0,0,1), (0,0.5,0.5), (0,1,0), (0.5,0.5,0), (1,0,0)]
        mc.pre_graficar_fibras(fig, ax, byn=False, color_por="capa", barracolor=True, colores_cm=colores_cm , ncolores_cm=len(colores_cm))
        mc.pre_graficar_nodos_interseccion(fig, ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(nombrearchivo[:-4]+".pdf")

plt.show()

# nombrearchivo = cwd + "000_malla_prueba_i.txt"
# print "leyendo malla"
# mc = Mc.leer_de_archivo(nombrearchivo)
# print "pregraficando"
# fig = plt.figure()
# # ax = fig.add_subplot(121)
# # mc.pre_graficar_bordes(fig, ax)
# # mc.pre_graficar_fibras(fig, ax, byn=True, color_por="nada", barracolor=False)
# # mc.pre_graficar_nodos_interseccion(fig, ax)
# ax = fig.add_subplot(111)
# mc.pre_graficar_bordes(fig, ax)
# mc.pre_graficar_interfibras(fig, ax, byn=False, color_por="lamr", barracolor=True)
# mc.pre_graficar_nodos_interseccion(fig, ax)



# nombrearchivo = cwd + "000_malla_prueba_i_s.txt"
# print "leyendo mallita"
# ms = Ms.leer_desde_archivo(nombrearchivo)
# print "pregraficando"
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ms.pre_graficar_bordes(fig, ax)
# colores_cm = ["blue", "red"]
# ms.pre_graficar(fig, ax, lam_min=1., lam_max=None, maxnfibs=2000, color_por="lamr", barracolor=True, colores_cm=colores_cm)

if True: # para graficar mallas deformadas por el fortran
    cwd = "/home/dancab/Documents/academia/doctorado/articulos/multiscale_nanofibers_randomRVE_2/Calibracion_para_paper/mallas/"
    # nombrearchivo = "L_000100.0_dlrel_01.00_devang_10.00_ncaps_0000005_nm_0000001_i_s_save_{:04d}.txt"
    nombrearchivo = "L_000150.0_dlrel_05.00_devang_17.00_ncaps_0000005_nm_0000001_i_s_save_{:04d}.txt"
    numeros = [11]
    nombresarchivos = [cwd + nombrearchivo.format(numero) for numero in numeros]

    # nombresarchivos = nombresarchivos + [cwd + "malla_test.txt"]

    for nombrearchivo in nombresarchivos:
        print "leyendo mallita: ", nombrearchivo
        ms = Ms.leer_desde_archivo(nombrearchivo)
        print "pregraficando"
        Fmacro = ms.Fmacro
        h = 10.*Fmacro[0,0]
        v = 6.*Fmacro[1,1]
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ms.pre_graficar_bordes(fig, ax)
        colores_cm = ["blue", "red"]
        # colores_cm = ["blue", "green", "yellow", "orange", "red"]
        ms.pre_graficar(fig, ax, color_por="lam_ef", linewidth=1.5,
                        # lam_min=0.0, lam_max=100.,
                        barracolor=True, colormap="rainbow", colores_cm=colores_cm, maxnfibs=3000,
                        afin=True, colorafin="k", linewidthafin=1.5)
        # ax.set_xticks([-.5*ms.L*Fmacro[0,0], 0.5*ms.L*Fmacro[0,0]])
        # ax.set_yticks([-.5*ms.L*Fmacro[1,1], 0.5*ms.L*Fmacro[1,1]])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlim([-75., 75.])
        # ax.set_ylim([-75., 75.])
        ax.set_title(r"$\varepsilon = {:5.3f}$".format(Fmacro[0,0]-1.))
        # modificar colorbar
        cb = fig.axes[1]
        cb.set_ylabel("Tension [MPa]")
        tickvalues = np.arange(1,1.041,0.01)
        tickvalues = cb.get_yticks()
        ticklabels = tickvalues * 75.
        ticklabels[-1] = 75.
        ticklabels = ["{:4.1f}".format(item) for item in ticklabels]
        cb.set_yticklabels(ticklabels)
        fig.tight_layout()
        # plt.show()
        fig.savefig(nombrearchivo[:-4]+".png")
        fig.savefig(nombrearchivo[:-4]+".pdf")

    # plt.show()

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ms.pre_graficar_bordes(fig, ax)
    colores_cm = ["blue", "red"]
    ms.pre_graficar(fig, ax, color_por="reclutamiento", barracolor=False, colormap="rainbow", colores_cm=colores_cm, maxnfibs=4000, colorafin="gray", linewidthafin=1)

    # for i in [6]:
    #     nombrearchivo = cwd + "000_malla_prueba_i_s_e_" + "{:07d}".format(i) + ".txt"
    #     print "leyendo mallita"
    #     ms = Ms.leer_desde_archivo(nombrearchivo)
    #     print "pregraficando"
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     Fmacro = np.array([ [1. + .02*float(i-1), 0.0], [0.0, 1.0 - .02*float(i-1)] ])
    #     ms.pre_graficar_bordes(fig, ax, Fmacro=Fmacro)
    #     # Fmacro = None
    #     colores_cm = ["blue", "red"]
    #     ms.pre_graficar(fig, ax, Fmacro=Fmacro, lam_min=0.9, lam_max=None, maxnfibs=2000, color_por="lam_ef", barracolor=True, colormap="rainbow", colores_cm=colores_cm)

    plt.show()



if False: # leer y graficar mallas segun los parametros (nombres estandar)

    Dm = 0.1
    nfibs = 0.1

    ncapss = [20]
    Ls = [50.]
    devangs_deg = [20.]
    dls_rel = [5.]

    nmallas = 1

    cwd = "mallas/"

    # nombresfigs = [cwd + "analisis_reclutamiento_devangmax_" + item + "_malla.pdf" for item in ("5", "10", "20")]

    start = time.time()
    for ncaps in ncapss:
        for Ladim in Ls:
            L = Ladim*Dm
            for dl_rel in dls_rel:
                dl = dl_rel * Dm
                for devang_deg in devangs_deg:
                    devang = devang_deg*np.pi/180.
                    for nm in range(1,nmallas+1):
                        print "ncaps={:05d}  L = {:08.2f}  devang = {:05.2f}  dl_rel = {:05.2f}  nm = {:07d}".format(ncaps, L, devang_deg, dl_rel, nm)
                        nombrearchivo = cwd + \
                                        "L_" + "{:08.1f}".format(Ladim) + \
                                        "_dlrel_" + "{:05.2f}".format(dl_rel) + \
                                        "_devang_" + "{:05.2f}".format(devang_deg) + \
                                        "_ncaps_" + "{:07d}".format(ncaps) + \
                                        "_nm_" + "{:07d}".format(nm) + \
                                        ".txt"
                        print nombrearchivo
                        print "leyendo malla"
                        mc = Mc.leer_de_archivo(archivo=nombrearchivo)
                        print "pregraficando"
                        fig1 = plt.figure(figsize=(6,6))
                        ax1 = fig1.add_subplot(111)
                        mc.pre_graficar_bordes(fig1, ax1)
                        # colores_cm = [(1,0,0), (0,0,1), (1,0,0)]
                        mc.pre_graficar_fibras(fig1, ax1, ncapas=None, byn=True, color_por="capa", colores_cm=None, barracolor=False)
                        # mc.pre_graficar_nodos_frontera(fig1, ax1, markersize=6)
                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        fig1.tight_layout()
                        fig1.savefig(nombrearchivo[:-4]+"_capas.pdf")
                        #
                        print "armando segundo plot"
                        fig2 = plt.figure(figsize=(6,6))
                        ax2 = fig2.add_subplot(111)
                        mc.pre_graficar_bordes(fig2, ax2)
                        # colores_cm = [(1,0,0), (0,1,0), (0,0,1), (0,0,0), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.4), (0.4, 0.4, 0.4)]*10
                        # colores_cm = [(0,0,0), (0.6, 0.2, 0.2), (0.2, 0.2, 0.6), (0.4, 0.4, 0.4)]
                        colores_cm = ["blue", "red"]
                        ncolores_cm = 100
                        mc.pre_graficar_fibras(fig2, ax2, byn=False, ncapas=2, color_por="nada", lamr_min=1., lamr_max=1.8, barracolor=True, colores_cm=colores_cm , ncolores_cm=ncolores_cm)
                        # mc.pre_graficar_nodos_interseccion(fig2, ax2, markersize=6)
                        # mc.pre_graficar_nodos_frontera(fig2, ax2, markersize=6)
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        fig2.tight_layout()
                        fig2.savefig(nombrearchivo[:-4]+"_fibras_2_capas.pdf")
                        #
                        # fig1.savefig(nombrearchivo[:-6] + ".pdf")
                        # fig2.savefig(nombrearchivo[:-4] + ".pdf")
    print "tiempo pregraf: ", time.time() - start

    # plt.show()


    # # ploteo la malla original
    # print "leyendo malla"
    # mc = Mc.leer_de_archivo("Malla.txt")
    # fig1,ax1 = plt.subplots()
    # mc.pre_graficar_bordes(fig1, ax1)
    # mc.pre_graficar_fibras(fig1, ax1, byn=True, color_por="capa")
    # # mc.pre_graficar_nodos_interseccion(fig1,ax1)

    # # ploteo la malla original
    # print "leyendo malla"
    # mc = Mc.leer_de_archivo("Malla.txt")
    # fig12,ax12 = plt.subplots()
    # mc.pre_graficar_bordes(fig12, ax12)
    # mc.pre_graficar_fibras(fig12, ax12, byn=False, color_por="lamr")
    # # mc.pre_graficar_nodos_interseccion(fig12,ax12)

    # # ploteo la malla intersectada por fortran
    # print "leyendo malla intersectada"
    # mc = Mc.leer_de_archivo("Malla_i.txt")
    # fig2,ax2 = plt.subplots()
    # mc.pre_graficar_bordes(fig2, ax2)
    # mc.pre_graficar_interfibras(fig2, ax2, byn=False, color_por="lamr", barracolor=True)
    # mc.pre_graficar_nodos_interseccion(fig2,ax2)

    # # ploteo la malla simplificada por fortran
    # print "leyendo mallita"
    # ms = Ms.leer_desde_archivo("Malla_i_s.txt")
    # fig3,ax3 = plt.subplots()
    # ms.pre_graficar_bordes(fig3, ax3)
    # ms.pre_graficar_0(fig3,ax3, plotnodos=True, maxnfibs=20000)
    # # ms.pre_graficar(fig3,ax3, lam_min=1.0, lam_max=1.2, maxnfibs=20000)

    # # ploteo la malla simplificada por fortran
    # print "leyendo mallita"
    # ms = Ms.leer_desde_archivo("Malla_i_s_e.txt")
    # fig4,ax4 = plt.subplots()
    # ms.pre_graficar_bordes(fig4, ax4)
    # ms.pre_graficar(fig4,ax4, maxnfibs=20000)

    # plt.show()

    # # ploteo afin
    # Fmacro = np.array(
    #     [
    #         [1.2, 0.0],
    #         [0.0, 1.0]
    #     ]
    # )
    # rafin = np.matmul(ms.nodos.r0, np.transpose(Fmacro))
    # ms.nodos.r = rafin
    # fig4, ax4 = plt.subplots()
    # ms.pre_graficar(fig4,ax4,lam_min=1.0, lam_max=1.2, maxnfibs=20000)



    # rafin = np.matmul(ms.nodos.r0, np.transpose(Fmacro))
    # for i in range(ms.nodos.n):
    #     dr = ms.nodos.r[i] - rafin[i]
    #     print dr


    # fig1.savefig("Malla.pdf", bbox_inches='tight')
    # fig12.savefig("Malla2.pdf", bbox_inches='tight')
    # fig2.savefig("Malla_i.pdf", bbox_inches='tight')
    # fig3.savefig("Malla_s.pdf", bbox_inches='tight')
    # fig4.savefig("Malla_sd.pdf", bbox_inches='tight')
    # plt.show()

    # muevo el nodo 1 y calculo A y b
    # r1 = ms.nodos.r.copy()
    # r1[0] = [50., -10.]
    # print r1
    # Ag, bg = ms.calcular_A_b(r1)

    # # fid = open("Ab.txt", "w")
    # n = ms.nodos.n
    # # m = 2*n
    # # for i in range(m):
    # #     linea = "".join( "{:20.8e}".format(val) for val in Ag[i,:] )
    # #     linea += "{:20.8e}".format(bg[i,0])
    # #     fid.write(linea+"\n")
    # # fid.close()

    # dr = np.linalg.solve(Ag,bg)
    # dr = dr.reshape(-1,2)

    # for i in range(n):
    #     print i, dr[i,:]



