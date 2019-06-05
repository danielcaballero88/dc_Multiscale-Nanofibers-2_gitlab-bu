import Rve_generador2

pi = 3.1416

m = Rve_generador2.Malla(1.0, 0.1, pi*0.1)
for i in range(25):
    m.make_fibra()
    m.trim_fibra_at_frontera(m.fibs.con[-1])

# m.guardar_en_archivo()

# m = Rve_generador2.Malla.leer_de_archivo()

m.intersectar_fibras()

m.guardar_en_archivo("malla_i.txt")

# print m.nods.tipos

m.graficar()