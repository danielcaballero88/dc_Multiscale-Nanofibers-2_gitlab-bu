import Rve_generador2

pi = 3.1416

L = 1.0

m = Rve_generador2.Malla(L)

for i in range(5): # capas
    for j in range(3): # fibra por capa
        m.make_fibra(0.1*L, pi*0.1, i)
        m.trim_fibra_at_frontera(m.fibs.con[-1])


# for i in range(4):
#     m.make_fibra(0.1*L, pi*0.1)
#     m.trim_fibra_at_frontera(m.fibs.con[-1])

# m.guardar_en_archivo()

# m = Rve_generador2.Malla.leer_de_archivo()

m.intersectar_fibras()

m.guardar_en_archivo("Malla.txt")

# print m.nods.tipos

m.graficar()