from Malla_completa import Malla as Mc
import time

pi = 3.1416

L = 1.0


m = Mc(L)

start = time.clock()
for i in range(1):
    m.make_capa(0.05*L, pi*0.1, 50)
print time.clock() - start

# for i in range(10): # capas
#     for j in range(10): # fibras
#         m.make_fibra(0.05*L, pi*0.1)
#         m.trim_fibra_at_frontera(m.fibs.con[-1])

# m.guardar_en_archivo()

# m = Rve_generador2.Malla.leer_de_archivo()

start = time.clock()
m.intersectar_fibras()
print time.clock() - start

m.guardar_en_archivo("Malla.txt")

# print m.nods.tipos

m.graficar()