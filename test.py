import Rve_generador2

pi = 3.1416

# m = Rve_generador2.Malla(1.0, 0.1, pi*0.1)
# for i in range(50):
#     m.make_fibra()

# m.guardar_en_archivo()

m = Rve_generador2.Malla.leer_de_archivo()

for i in range( len(m.fibs.con) ):
    if i==5:
        pass
    m.trim_fibra_at_frontera(m.fibs.con[i])

m.graficar()