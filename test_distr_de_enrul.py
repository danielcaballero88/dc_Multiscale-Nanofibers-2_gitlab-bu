from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np

L = 100.0
Dm = 1.0
mc = Mc(L, Dm)
devangle = 19. * np.pi / 180.
dl = 0.04 * L
nfibs = 100
for i in range(1):
    mc.make_capa2(dl, Dm, devangle, nfibs)

# mc = Mc.leer_de_archivo("Malla.txt")

# mc.cambiar_capas(10)

# start = time.clock()
# mc.intersectar_fibras()
# print time.clock() - start

nombrearchivo = "mallas/dl_" + "{:05.2f}".format(dl) + \
                "_devang_" + "{:05.2f}".format(devangle*180./np.pi) + \
                "_nf_" + "{:07d}".format(nfibs) + \
                ".txt"
# mc.guardar_en_archivo(nombrearchivo)

lamr_min = 1.0
lamr_max = 1.5
n = 5

lamsr = mc.calcular_enrulamientos()
print np.max

lrs, dlr, frecs = mc.calcular_distribucion_de_enrulamiento(lamr_min=lamr_min, lamr_max=lamr_max, n=n)
print lrs
print dlr
print frecs
print np.array(frecs, dtype=float)/float(np.sum(frecs))
print np.sum(frecs)



max_lamr = np.max(lamsr)
index = np.where( lamsr == max_lamr)
print index, max_lamr

# fig1, ax1 = plt.subplots()
# mc.pre_graficar_bordes(fig1, ax1)
# mc.pre_graficar_interfibras(fig1, ax1, lamr_min = lamr_min, lamr_max = lamr_max)
# mc.pre_graficar_nodos_interseccion(fig1, ax1)

fig2, ax2 = plt.subplots()
mc.pre_graficar_bordes(fig2, ax2)
mc.pre_graficar_fibras(fig2, ax2, lamr_min=lamr_min, lamr_max=lamr_max, byn=False)
mc.pre_graficar_nodos_interseccion(fig2, ax2)

frecs = np.array(frecs, dtype=float) / float(np.sum(frecs))
fig3, ax3 = plt.subplots()
bars1 = ax3.bar(lrs, frecs, dlr*0.8)
ax3.set_ylim(bottom=0.0, top=1.0)
ax3.set_xlim(left=1.0, right=2.0)


for lr, frec in zip(lrs,frecs):
    print "{:10.4f}{:10.4f}".format(lr, frec)


plt.show()