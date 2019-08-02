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

nombrearchivo = "mallas_to_plot/01_L_" + "{:08.4f}".format(100) + "_nm_" + "{:02d}".format(10) + ".txt"
fig, ax = plt.subplots()
m = Mc.leer_de_archivo(nombrearchivo)
m.pre_graficar_bordes(fig, ax)
m.pre_graficar_fibras(fig, ax)

nombrearchivo = "mallas_to_plot/02_L_" + "{:08.4f}".format(100) + "_nm_" + "{:02d}".format(10) + ".txt"
fig, ax = plt.subplots()
m = Mc.leer_de_archivo(nombrearchivo)
m.pre_graficar_bordes(fig, ax)
m.pre_graficar_fibras(fig, ax)

nombrearchivo = "mallas_to_plot/03_L_" + "{:08.4f}".format(100) + "_nm_" + "{:02d}".format(10) + ".txt"
fig, ax = plt.subplots()
m = Mc.leer_de_archivo(nombrearchivo)
m.pre_graficar_bordes(fig, ax)
m.pre_graficar_fibras(fig, ax)


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