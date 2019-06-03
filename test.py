import Rve_generador2 

pi = 3.1416

m = Rve_generador2.Malla(1.0, 0.1, pi*0.1)

for i in range(10):
    m.make_fibra()

m.graficar()