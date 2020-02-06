from Malla_completa import Malla as Mc
import time
import numpy as np
import matplotlib.pyplot as plt
from OrientationDistributionFunctions import NormalTruncada

# fid1 = open("mallas/datos.txt", "w")

# IMPORTANTE: la funcion distribucion de orientacion fundisor es periodica en pi: theta = theta + pi
# porque cada fibra se genera con un angulo respecto de la horizontal que no tiene un sentido tipo vector,
# sino que es el angulo de una recta
# fundisor = NormalTruncada(loc=0.5*np.pi, scale=0.1*np.pi, lower=0., upper=np.pi)
fundisor = None
# =====
# Calcular mallas y escribirlas
Dm = 1.0
nfibs = 0.1

ncapss = [2]
Ls = [200.]
devangs_deg = [20.]
dls_rel = [1.]

nmallas = 1

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
                    mc = Mc(L, Dm)
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
                    nombrearchivo = cwd + "000_malla_prueba.txt"
                    mc.guardar_en_archivo(nombrearchivo)
print "tiempo generacion: ", time.time() - start

# mc = Mc.leer_de_archivo("Malla.txt")
# start = time.time()
# mc.intersectar_fibras()
# print "tiempo interseccion: ", time.time() - start
# mc.guardar_en_archivo("Malla_intersectada.txt")
# # =====

# # mc = Mc.leer_de_archivo("Malla_intersectada.txt")
# start = time.time()
# fig, ax = plt.subplots()
# mc.pre_graficar_fibras(fig, ax, byn=True, color_por="capa")
# mc.pre_graficar_nodos_interseccion(fig,ax)
# print "tiempo pregraficar: ", time.time() - start
# # mc2 = Mc.leer_de_archivo("Malla_inter2.txt")
# # fig,ax = plt.subplots()
# # mc2.pre_graficar_fibras(fig,ax,byn=False, color_por="fibra")
# # mc2.pre_graficar_nodos_interseccion(fig,ax)
# # ax.set_facecolor('black')
# plt.show()


# fid1.close()

# start = time.clock()
# m.intersectar_fibras()
# print time.clock() - start

# m.guardar_en_archivo("Malla.txt")
# =====



# infbs_con = m.calcular_conectividad_de_interfibras()
# for i, infb_con in enumerate(infbs_con):
#     print i, ":", infb_con





# mc.graficar(lamr_min=None, lamr_max=None, byn=True)