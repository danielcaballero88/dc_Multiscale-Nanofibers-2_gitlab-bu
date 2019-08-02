from Malla_completa import Malla as Mc
import time
import numpy as np

L = 190. # micrones
Dm = 1.0

diam = Dm
seglen = 0.01*L
devang = 5. * np.pi / 180. # 5 grados

# fid1 = open("mallas/datos.txt", "w")

# =====
# Calcular mallas y escribirlas
Ls = np.arange(100, 100+1, 10, dtype=float)
for L in Ls:
    for nummalla in range(1,11):
        m = Mc(L, Dm)
        for i in range(1,2): # capas
            m.make_capa2(seglen, Dm, devang, 0.3)
            volfra = m.calcular_fraccion_de_volumen_de_una_capa(m.caps.con[-1])
            numfibs = len(m.caps.con[0])
            print "{:08.4f} {:6d} {:10.5f} {:6d}".format(L, nummalla, volfra, numfibs)
        # guardo en archivo
        # nombrearchivo = "mallas/L_" + "{:08.4f}".format(L) + "_nm_" + "{:02d}".format(nummalla) + ".txt"
        nombrearchivo = "mallas/devang_" + "{:06.4f}".format(devang*180./np.pi) + "_nm_" + "{:02d}".format(nummalla) + ".txt"
        # fid1.write( "{:08.4f}{:6d}{:10.5f}{:6d}".format(L,nummalla,volfra,numfibs) + "\n")
        m.guardar_en_archivo(nombrearchivo)

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

m.graficar(lamr_min=None, lamr_max=None)