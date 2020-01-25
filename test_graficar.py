from Malla_completa import Malla as Mc
from Mallita import Mallita as Ms
import time
import numpy as np
import matplotlib.pyplot as plt

nombresarchivos = ["mallas/uniforme.txt", "mallas/stdev_0.1.txt", "mallas/stdev_0.2.txt"]

for nombrearchivo in nombresarchivos:
    print "leyendo malla"
    mc = Mc.leer_de_archivo(archivo=nombrearchivo)
    print "pregraficando"
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    mc.pre_graficar_bordes(fig, ax1)
    mc.pre_graficar_fibras(fig, ax1, byn=True, color_por="capa", colores_cm=[(1,0,0), (0,0,1), (1,0,0)], barracolor=False)
    nombrefigura = nombrearchivo[:-4] + ".pdf"
    fig.savefig(nombrefigura, bbox="tight")

# plt.show()

# Dm = 1.0
# nfibs = 0.3

# ncapss = [10]
# Ls = [50.]
# devangs_deg = [5.]
# dls_rel = [1.]
# nmallas = 1

# start = time.time()
# for ncaps in ncapss:
#     for L in Ls:
#         for dl_rel in dls_rel:
#             dl = dl_rel * Dm
#             for devang_deg in devangs_deg:
#                 devang = devang_deg*np.pi/180.
#                 for nm in range(1,nmallas+1):
#                     print "ncaps={:05d}  L = {:08.2f}  devang = {:05.2f}  dl_rel = {:05.2f}  nm = {:07d}".format(ncaps, L, devang_deg, dl_rel, nm)
#                     nombrearchivo = "mallas/" + \
#                                     "L_" + "{:08.1f}".format(L) + \
#                                     "_dlrel_" + "{:05.2f}".format(dl_rel) + \
#                                     "_devang_" + "{:05.2f}".format(devang_deg) + \
#                                     "_ncaps_" + "{:07d}".format(ncaps) + \
#                                     "_nm_" + "{:07d}".format(nm) + \
#                                     ".txt"
#                     nombrearchivo = "mallas/stdev_0.5.txt"
#                     print nombrearchivo
#                     print "leyendo malla"
#                     mc = Mc.leer_de_archivo(archivo=nombrearchivo)
#                     print "pregraficando"
#                     fig = plt.figure(figsize=(10,4))
#                     ax1 = fig.add_subplot(121)
#                     mc.pre_graficar_bordes(fig, ax1)
#                     mc.pre_graficar_fibras(fig, ax1, byn=True, color_por="capa", colores_cm=[(1,0,0), (0,0,1), (1,0,0)])
#                     # ax2 = fig.add_subplot(122)
#                     # nombrearchivo = "mallas/" + \
#                     #                 "L_" + "{:08.1f}".format(L) + \
#                     #                 "_dlrel_" + "{:05.2f}".format(dl_rel) + \
#                     #                 "_devang_" + "{:05.2f}".format(devang_deg) + \
#                     #                 "_ncaps_" + "{:07d}".format(ncaps) + \
#                     #                 "_nm_" + "{:07d}".format(nm) + \
#                     #                 "_i.txt"
#                     # # nombrearchivo = "Malla.txt"
#                     # mc = Mc.leer_de_archivo(archivo=nombrearchivo)
#                     # mc.pre_graficar_bordes(fig, ax2)
#                     # mc.pre_graficar_interfibras(fig, ax2, byn=False, color_por="interfibra")
#                     # mc.pre_graficar_nodos_interseccion(fig, ax2, markersize=4)
# print "tiempo pregraf: ", time.time() - start

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



