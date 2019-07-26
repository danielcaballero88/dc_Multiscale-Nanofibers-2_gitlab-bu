from Malla_completa import Malla as Mc
import time

pi = 3.1416

L = 1.0 # pongo valor 1 aca y las demas medidas las hago relativas a esta
Dm = 0.01


m = Mc(L, Dm)

# =====
# Calcular malla
start = time.clock()
for i in range(1): # capas
    m.make_capa(0.05*L, Dm, pi*0.1, 85)
print time.clock() - start

fravol = m.calcular_fraccion_de_volumen_de_una_capa(m.caps.con[0])
print fravol


# start = time.clock()
# m.intersectar_fibras()
# print time.clock() - start

m.guardar_en_archivo("Malla.txt")
# =====

# # =====
# # Leer malla de archivo
# m = Mc.leer_de_archivo("Malla_con_problemas.txt")
# # =====

# infbs_con = m.calcular_conectividad_de_interfibras()
# for i, infb_con in enumerate(infbs_con):
#     print i, ":", infb_con

m.graficar(lamr_min=1.0, lamr_max=1.2)