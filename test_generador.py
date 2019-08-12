from Malla_completa import Malla as Mc
import time
import numpy as np
import matplotlib.pyplot as plt

# fid1 = open("mallas/datos.txt", "w")

# =====
# Calcular mallas y escribirlas
L = 100.0
Dm = 1.0
devangle = 10. * np.pi / 180.
dl = 0.02 * L
nfibs = 20


dls_rel = (0.02, 0.03, 0.04)
devangs_deg = (10., 15., 20.)

dls_rel = [0.04]
devangs_deg = [20.]

for dl_rel in dls_rel:
    dl = dl_rel * L
    for devang_deg in devangs_deg:
        devang = devang_deg*np.pi/180.
        for nm in range(1,2):
            print "{:05.2f}  {:05.2f}  {:07d}".format(dl_rel,devang_deg,nm)
            mc = Mc(L, Dm)
            for i in range(20):
                mc.make_capa2(dl, Dm, devang, nfibs)
            nombrearchivo = "mallas/dl_" + "{:05.2f}".format(dl_rel) + \
                            "_devang_" + "{:05.2f}".format(devang_deg) + \
                            "_nm_" + "{:07d}".format(nm) + \
                            ".txt"
            # mc.guardar_en_archivo(nombrearchivo)

# fid1.close()

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

fig, ax = plt.subplots()
mc.pre_graficar_capas(fig, ax, byn=True)
plt.show()
# mc.graficar(lamr_min=None, lamr_max=None, byn=True)