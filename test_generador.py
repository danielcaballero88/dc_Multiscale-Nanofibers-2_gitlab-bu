from Malla_completa import Malla as Mc

pi = 3.1416

L = 1.0


m = Mc(L)

for i in range(10): # capas
    for j in range(5): # fibras
        m.make_fibra(0.05*L, pi*0.1, capa=i)
        m.trim_fibra_at_frontera(m.fibs.con[-1])

# m.guardar_en_archivo()

# m = Rve_generador2.Malla.leer_de_archivo()

m.intersectar_fibras()

m.guardar_en_archivo("Malla.txt")

# print m.nods.tipos

m.graficar()