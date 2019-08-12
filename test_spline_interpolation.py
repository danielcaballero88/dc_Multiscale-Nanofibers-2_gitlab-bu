from Malla_completa import Malla as Mc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate

L = 100.0
Dm = 1.0
devang = 30. * np.pi / 180.
dl = 0.04 * L
nfibs = 1

mc = Mc(L, Dm)
for i in range(1):
    mc.make_capa2(dl, Dm, devang, nfibs)

fig, ax = plt.subplots()
mc.pre_graficar_bordes(fig, ax, byn=True)
mc.pre_graficar_fibras(fig, ax, byn=True)
plt.show()