
"""
ARCHIVO INDEPENDIENTE
Se resuelve el caso de una sola fibra bajo deformacion elastoplastica
Ver Silberstein et. al.

sistema de unidades: micron, kg, segundo
luego: [tension] = MPa, [fuerza] = microNewton

"""

import numpy as np
from matplotlib import pyplot as plt

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def ten_elas_fibra(lam, Et, Eb, lamr, lamp, lamprot=1.5):
    lamrL = lamr*lamp
    if lam<=lamrL:
        ten = Eb*(lam/lamp-1.)
    else:
        ten = Eb*(lamrL-1.) + Et*(lam/lamrL-1.)
    if lamp>lamprot:
        ten = 0.
    return ten

## seno hiperbolico
#x = np.linspace(0., 1., 100000)
#y = np.sinh(x)
#fig, ax = plt.subplots()
#ax.plot(x,y)
#i_nearest = find_nearest_index(y, 1.)
#print i_nearest, x[i_nearest], y[i_nearest]

# parametros
D0 = 1. #  [micron]
A0 = np.pi*D0**2/4.
doteps0 = 1.0e-8 #  [1/seg]
s0 = 4.5 # [MPa]
nhard = 1. # hardening coefficient
Et = 2.9e3 # [MPa]
Kt = Et*A0 # [uN]
Eb = Et * 1.e-3
lamr = 1.1
lamp0 = 1.0
lamprot = 1.2

# tiempo y tasa de deformacion
tiempo0 = 0.
dtiempo = .01
dotlam = .01
lamf = 1.5
lamback = 10.12
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
while lam < lamf:
    tiempo += dtiempo
    lam += dotlam*dtiempo
    # ten = Et*(lam/lamp - 1.)
    ten = ten_elas_fibra(lam, Et, Eb, lamr, lamp, lamprot)
    lamef = lam/lamp/lamr
    print "{:10.4f}{:10.4f}{:20.8e}{:10.4f}".format(tiempo, lam, ten, lamef)
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
    if switch1==0 and lam>lamback:
        dotlam = -dotlam
        switch1 = 1
    if switch1==1 and switch2==0 and ten < 1.e-2:
        # ten = ten_elas_fibra(lam, Et, Eb, lamr, lamp)
        dotlam = - dotlam
        switch2 = 1

rec_tiempo = np.array(rec_tiempo)
rec_lam = np.array(rec_lam)
rec_ten = np.array(rec_ten)
rec_fuerza = rec_ten * A0  # [uN]
rec_dotlamp = np.array(rec_dotlamp)
rec_lamp = np.array(rec_lamp)
rec_ten = rec_ten # [MPa]

rec_lamef = rec_lam / rec_lamp / lamr







SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, ax = plt.subplots()
ax.plot(rec_lam, rec_ten, c="k", lw=1.5)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Tension ingenieril [MPa]")
fig.tight_layout()
# ax.set_title("ten vs lam")
# fig.savefig("curva_tension_fibra_plasticidad_2.pdf")
plt.show()



# fig, ax = plt.subplots()
# ax.plot(rec_tiempo, rec_dotlamp)
# ax.set_title("dotlamp vs t")

# fig, ax = plt.subplots()
# ax.plot(rec_tiempo, rec_lam)
# ax.set_title("lam vs t")

# fig, ax = plt.subplots()
# ax.plot(rec_tiempo, rec_ten)
# ax.set_title("ten vs t")

# fig, ax = plt.subplots()
# ax.plot(rec_tiempo, rec_lamp)
# ax.set_title("lamp vs t")

# fig, ax = plt.subplots()
# ax.plot(rec_lam, rec_lamp)
# ax.set_title("lamp vs lam")

# fig, ax = plt.subplots()
# ax.plot(rec_ten, rec_lamp)
# ax.set_title("lamp vs ten")



# fig, ax = plt.subplots()
# ax.plot(rec_lam, rec_fuerza)
# ax.set_title("fuerza vs lam")

# fig, ax = plt.subplots()
# ax.plot(rec_lamef, rec_ten, linewidth=0, marker=".")
# ax.set_title("ten vs lamef")

# fig, ax = plt.subplots()
# ax.plot(rec_lamef, rec_tiempo, linewidth=0, marker=".")
# ax.set_title("time vs lamef")

print np.max(rec_lamef)

plt.show()