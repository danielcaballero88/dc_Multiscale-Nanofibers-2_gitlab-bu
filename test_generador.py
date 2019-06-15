from Malla_completa import Malla as Mc

pi = 3.1416

L = 1.0


m = Mc(L)
for i in range(50):
    print "making fibra: ", i
    m.make_fibra(0.01*L, pi*0.1, capa=0)
    m.trim_fibra_at_frontera(m.fibs.con[-1])

# m.guardar_en_archivo()

# m = Rve_generador2.Malla.leer_de_archivo()

m.intersectar_fibras()

m.guardar_en_archivo("Malla.txt")

# print m.nods.tipos

m.graficar()