import numpy as np
from Malla_simplificada import Malla as Ms, Iterador
from Malla_completa import Malla as Mc
from matplotlib import pyplot as plt

parcon = [0.1e9, 0.1e6] # Et, Eb
eccon = 0 # lineal con reclutamiento
pseudovisc = 0.2e7

mc = Mc.leer_de_archivo("Malla.txt")

ms = Ms()
ms.simplificar_malla_completa(mc, parcon, eccon, pseudovisc)

ms.guardar_en_archivo("Malla_simplificada.txt")


# iterador
n_sis = ms.nodos.num # numero de variables a resolver (en este caso son arrays de len 2: x e y)
semilla = ms.nodos.x0
sistema = ms
ref_grande = ms.L * 1.0e-1
ref_divergente = 0.95
max_iters = 1000
tolerancia = ms.L*1.0e-16
ref_pequeno = tolerancia
ite = Iterador(n_sis, sistema, ref_pequeno, ref_grande, ref_divergente, max_iters, tolerancia)
# ---


# deformacion
F = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype=float
)


# ms.deformar_afin_frontera(F)

# x = ite.iterar()
# ms.set_x(x)

print "---"

F[1,1] = 1.5
ms.deformar_afin_frontera(F)

ms.psv = ms.psv / 100.
x = ite.iterar()
ms.set_x(x)

trac_sfs = ms.calcular_tracciones_de_subfibras()
trac_nods = ms.calcular_tracciones_sobre_nodos(trac_sfs)
print trac_nods

ms.graficar(F)
