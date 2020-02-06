
"""
ARCHIVO INDEPENDIENTE
Se resuelve el caso de una sola fibra bajo deformacion elastoplastica
Ver Silberstein et. al.
"""

import numpy as np
from matplotlib import pyplot as plt

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

## seno hiperbolico
#x = np.linspace(0., 1., 100000)
#y = np.sinh(x)
#fig, ax = plt.subplots()
#ax.plot(x,y)
#i_nearest = find_nearest_index(y, 1.)
#print i_nearest, x[i_nearest], y[i_nearest]

# parametros
D0 = 1.e-6 #  [micron]
A0 = np.pi*D0**2/4.
doteps0 = 1.0e-4 #  [1/seg]
s0 = 14.0e6 # [Pa]
nhard = 1. # hardening coefficient
lamp0 = 1.0
Et = 2.9e9 # [Pa]
Kt = Et*A0 # [N]

# tiempo y tasa de deformacion
tiempo0 = 0.
dtiempo = 0.1
dotlam = .6/60.
lamf = 1.3
tiempof = tiempo0 + (lamf-1.)/dotlam

# esquema explicito
tiempo = tiempo0
rec_tiempo = list()
rec_lam = list()
rec_ten = list()
rec_dotlamp = list()
rec_lamp = list()
lam = 1.
lamp = 1.
rec_tiempo.append(0.)
rec_lam.append(lam)
rec_ten.append(0.)
rec_dotlamp.append(0.)
rec_lamp.append(lamp)
switch1 = 0
switch2 = 0
while lam < 1.3:
    tiempo += dtiempo
    lam += dotlam*dtiempo
    print tiempo, lam
    ten = Et*(lam/lamp - 1.)
    # # modelo de plasticidad sencillo armado por mi al boleo
    # if ten > 50.e6:
    #     dotlamp = (ten - 50.e6)/(lamp**5*1.e6) * .002
    # else:
    #     dotlamp = 0.
    # modelo de plasticidad de Silberstein, andan igual
    s = lamp**nhard * s0
    if ten > 0:
        dotlamp = doteps0 * np.sinh(ten/s)
    else:
        dotlamp = 0.
    lamp += dotlamp*dtiempo
    # if lamp>lam:
    #     raise ValueError
    rec_tiempo.append(tiempo)
    rec_lam.append(lam)
    rec_ten.append(ten)
    rec_dotlamp.append(dotlamp)
    rec_lamp.append(lamp)
    if switch1==0 and lam>1.1:
        dotlam = -dotlam
        switch1 = 1
    if switch1==1 and switch2==0 and ten < 1.e-6:
        dotlam = - dotlam
        switch2 = 1

rec_tiempo = np.array(rec_tiempo)
rec_lam = np.array(rec_lam)
rec_ten = np.array(rec_ten)
rec_fuerza = rec_ten * A0 *1.e5 # [1.e-5 N]
rec_dotlamp = np.array(rec_dotlamp)
rec_lamp = np.array(rec_lamp)
rec_ten = rec_ten * 1.e-6 # [MPa]



fig, ax = plt.subplots()
ax.plot(rec_tiempo, rec_dotlamp)
ax.set_title("dotlamp vs t")
fig, ax = plt.subplots()
ax.plot(rec_tiempo, rec_lam)
ax.set_title("lam vs t")
fig, ax = plt.subplots()
ax.plot(rec_tiempo, rec_ten)
ax.set_title("ten vs t")
fig, ax = plt.subplots()
ax.plot(rec_tiempo, rec_lamp)
ax.set_title("lamp vs t")
fig, ax = plt.subplots()
ax.plot(rec_lam, rec_lamp)
ax.set_title("lamp vs lam")
fig, ax = plt.subplots()
ax.plot(rec_ten, rec_lamp)
ax.set_title("lamp vs ten")
fig, ax = plt.subplots()
ax.plot(rec_lam, rec_ten)
ax.set_title("ten vs lam")
fig, ax = plt.subplots()
ax.plot(rec_lam, rec_fuerza)
ax.set_title("fuerza vs lam")

plt.show()