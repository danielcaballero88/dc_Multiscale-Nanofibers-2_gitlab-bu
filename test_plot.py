from Malla_completa import Malla as Mc
import time
import numpy as np
import matplotlib.pyplot as plt

# # =====
# # Leer mallas y plotearlas
# Ls = np.arange(10, 10+1, 10, dtype=float)
# for L in Ls:
#     for nummalla in range(1,11):
#         fig, ax = plt.subplots()
#         nombrearchivo = "mallas/L_" + "{:08.4f}".format(L) + "_nm_" + "{:02d}".format(nummalla) + ".txt"
#         m = Mc.leer_de_archivo(nombrearchivo)
#         m.pre_graficar_bordes(fig, ax)
#         m.pre_graficar_fibras(fig, ax)

# nombrearchivo = "mallas_to_plot/01_L_" + "{:08.4f}".format(100) + "_nm_" + "{:02d}".format(10) + ".txt"
# fig, ax = plt.subplots()
# m = Mc.leer_de_archivo(nombrearchivo)
# m.pre_graficar_bordes(fig, ax)
# m.pre_graficar_fibras(fig, ax)

# nombrearchivo = "mallas_to_plot/02_L_" + "{:08.4f}".format(100) + "_nm_" + "{:02d}".format(10) + ".txt"
# fig, ax = plt.subplots()
# m = Mc.leer_de_archivo(nombrearchivo)
# m.pre_graficar_bordes(fig, ax)
# m.pre_graficar_fibras(fig, ax)

# nombrearchivo = "mallas_to_plot/03_L_" + "{:08.4f}".format(100) + "_nm_" + "{:02d}".format(10) + ".txt"
# fig, ax = plt.subplots()
# m = Mc.leer_de_archivo(nombrearchivo)
# m.pre_graficar_bordes(fig, ax)
# m.pre_graficar_fibras(fig, ax)


# =====
# Leer mallas y plotearlas

fig, ax = plt.subplots(3,3)

fig2, ax2 = plt.subplots(3,3)

L = 100.0
Dm = 1.0
nfibs = 50

dls_rel = (0.04, 0.03, 0.02)
devangs_deg = (10., 15., 20.)

for i,dl_rel in enumerate(dls_rel):
    dl = dl_rel * L
    for j,devang_deg in enumerate(devangs_deg):
        devang = devang_deg*np.pi/180.
        for nm in range(1,2):
            print "{:05.2f}  {:05.2f}  {:07d}".format(dl_rel,devang_deg,nm)
            nombrearchivo = "mallas_to_plot/dl_" + "{:05.2f}".format(dl_rel) + \
                            "_devang_" + "{:05.2f}".format(devang_deg) + \
                            "_nm_" + "{:07d}".format(nm) + \
                            ".txt"
            mc = Mc.leer_de_archivo(nombrearchivo)
            mc.pre_graficar_bordes(fig, ax[i,j])
            mc.pre_graficar_fibras(fig, ax[i,j], lamr_min=1.0, lamr_max=2.0, byn=True, barracolor=False)
            # tambien voy a calcular el histograma a cada mall
            lrs, dlr, frecs = mc.calcular_distribucion_de_enrulamiento(lamr_min=1.0, lamr_max=2.0, n=10)
            frecs = np.array(frecs, dtype=float) / float(np.sum(frecs))
            ax2[i,j].bar(lrs, frecs, dlr*0.8)



plt.show()

# start = time.clock()
# m.intersectar_fibras()
# print time.clock() - start

# m.guardar_en_archivo("Malla.txt")
# =====

# # =====
# # Leer malla de archivo
# m = Mc.leer_de_archivo("Malla_con_problemas.txt")
# # =====

# infbs_con = m.calcular_conectividad_de_interfibras()
# for i, infb_con in enumerate(infbs_con):
#     print i, ":", infb_con

# m.graficar(lamr_min=1.0, lamr_max=1.2)