from Malla_completa import Malla as Mc
from Mallita import Mallita as Ms
import time
import numpy as np
import matplotlib.pyplot as plt


# L = 100.
# Dm = 1.
# nfibs = 10
# dl = 0.1 * L
# devang = 10. * np.pi/180.
# ncaps = 1
# mc = Mc(L,Dm)
# for nc in range(1,1+ncaps):
#     mc.make_capa2(dl, Dm, devang, nfibs)
# mc.guardar_en_archivo("Malla.txt")




# ploteo la malla original
print "leyendo malla"
mc = Mc.leer_de_archivo("Malla.txt")
fig1,ax1 = plt.subplots()
mc.pre_graficar_bordes(fig1, ax1)
mc.pre_graficar_fibras(fig1, ax1, byn=True, color_por="capa")
mc.pre_graficar_nodos_interseccion(fig1,ax1)

# ploteo la malla intersectada por fortran
print "leyendo malla intersectada"
mc = Mc.leer_de_archivo("Malla_i.txt")
fig2,ax2 = plt.subplots()
mc.pre_graficar_bordes(fig2, ax2)
mc.pre_graficar_interfibras(fig2, ax2, byn=False, color_por="lamr", barracolor=True)
mc.pre_graficar_nodos_interseccion(fig2,ax2)

# ploteo la malla simplificada por fortran
print "leyendo mallita"
ms = Ms.leer_desde_archivo("Malla_s.txt")
fig3,ax3 = plt.subplots()
ms.pre_graficar_bordes(fig3, ax3)
ms.pre_graficar_0(fig3,ax3, plotnodos=True, maxnfibs=20000)
# ms.pre_graficar(fig3,ax3, lam_min=1.0, lam_max=1.2, maxnfibs=20000)

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


fig1.savefig("Malla.pdf", bbox_inches='tight')
fig2.savefig("Malla_i.pdf", bbox_inches='tight')
fig3.savefig("Malla_s.pdf", bbox_inches='tight')
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



