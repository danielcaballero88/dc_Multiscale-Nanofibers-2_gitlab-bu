"""
ARCHIVO INDEPENDIENTE
Se resuelve el caso de una sola fibra bajo deformacion elastoplastica
Modificado respecto de test_singlefiber_plastic
En este caso modifico de manera constante en el tiempo a la carga que sufre la fibra
Y veo como aumenta la tension, deformacion, deformacion plastica, etc
No es tasa de deformacion constante, es tasa de carga constante (una cosa rara pero me sirve para ver que funciona)
En cada step calculo el equilibrio con el metodo rustico a emplear en la malla en Fortran:
Muevo el nodo libre segun el signo de la resultante (entre la carga y la resistencia de la fibra)
Segun valores preestablecidos y numero de iteraciones preestablecidos y con esa iteracion bruta encuentro el equilibrio (o casi, good enough for me)
"""

import numpy as np
from matplotlib import pyplot as plt

def ten_elas_fibra(lam, Et, Eb, lamr, lamp):
    if lam<=lamr:
        return Eb*(lam/lamp-1.)
    else:
        return Eb*(lamr-1.) + Et*(lam/lamr/lamp-1.)

def mover_x1(carga, x0, x1, lete0, Et, Eb, lamr, A0, dx, lamp):
    lete = x1-x0
    lam = lete/lete0
    ten = ten_elas_fibra(lam, Et, Eb, lamr, lamp)
    fza = ten*A0
    resultante = carga - fza
    if abs(resultante) < 1.e-7:
        return x1, lam, ten, fza, True
    elif resultante > 0.:
        return x1+dx, lam, ten, fza, False
    else:
        return x1-dx, lam, ten, fza, False

def encontrar_x1(carga, x0, x1, lete0, Et, Eb, lamr, A0, lamp):
    x1new = x1
    for k in range(2):
        x1new, lam, ten, fza, fin = mover_x1(carga, x0, x1new, lete0, Et, Eb, lamr, A0, 0.1, lamp)
        if fin:
            return x1new, lam, ten, fza, fin
    for k in range(10):
        x1new, lam, ten, fza, fin = mover_x1(carga, x0, x1new, lete0, Et, Eb, lamr, A0, 0.01, lamp)
        if fin:
            return x1new, lam, ten, fza, fin
    for k in range(10):
        x1new, lam, ten, fza, fin = mover_x1(carga, x0, x1new, lete0, Et, Eb, lamr, A0, 0.001, lamp)
        if fin:
            return x1new, lam, ten, fza, fin
    for k in range(10):
        x1new, lam, ten, fza, fin = mover_x1(carga, x0, x1new, lete0, Et, Eb, lamr, A0, 0.0001, lamp)
        if fin:
            return x1new, lam, ten, fza, fin
    return x1new, lam, ten, fza, fin

# parametros
D0 = 1.e-6 #  [micron]
A0 = np.pi*D0**2/4.
doteps0 = 1.0e-5 #  [1/seg]
s0 = 6.0e6 # [Pa]
nhard = 1. # hardening coefficient
lamp0 = 1.0
Et = 2.9e9 # [Pa]
Eb = Et*1.e-3
Kt = Et*A0 # [N]


# ==========
# pruebita
x0 = 0.
x1 = 1.
lete0 = x1 - x0
lamr = 1.1
loco0 = lamr*lete0
carga = 3.e-5
x1new, lam, ten, fza, fin = encontrar_x1(carga, x0, x1, lete0, Et, Eb, lamr, A0, 1.0)
print fza, carga
# ==========


# tiempo y tasa de deformacion
tiempo0 = 0.
dtiempo = 0.1
dotcarga = .1e-5
cargaf = 6.e-5

# esquema explicito con fuerza variando
tiempo = tiempo0
rec_tiempo = list()
rec_lam = list()
rec_ten = list()
rec_fza = list()
rec_carga = list()
rec_dotlamp = list()
rec_lamp = list()
lam = 1.
lamp = 1.
carga = 0.
rec_tiempo.append(0.)
rec_carga.append(0.)
rec_lam.append(lam)
rec_ten.append(0.)
rec_fza.append(0.)
rec_dotlamp.append(0.)
rec_lamp.append(lamp)
switch1 = 0
switch2 = 0
while carga < 6.e-5:
    tiempo += dtiempo
    carga += dotcarga*dtiempo
    newx1, lam, ten, fza, status = encontrar_x1(carga, x0, x1, lete0, Et, Eb, lamr, A0, lamp)
    print ("{:20.8f}"*7).format(tiempo, carga, newx1, lam, ten, fza, status)
    s = lamp**nhard * s0
    if ten > 0:
        dotlamp = doteps0 * np.sinh(ten/s)
    else:
        dotlamp = 0.
    lamp += dotlamp*dtiempo
    if lamp>lam:
        raise ValueError
    rec_tiempo.append(tiempo)
    rec_carga.append(carga)
    rec_lam.append(lam)
    rec_ten.append(ten)
    rec_fza.append(fza)
    rec_dotlamp.append(dotlamp)
    rec_lamp.append(lamp)
    if switch1==0 and carga>5.e-5:
        dotcarga = -dotcarga
        switch1 = 1
    if switch1==1 and switch2==0 and carga < 1.e-8:
        dotcarga = -dotcarga
        switch2 = 1
    x1 = newx1

rec_tiempo = np.array(rec_tiempo)
rec_carga = np.array(rec_carga)
rec_lam = np.array(rec_lam)
rec_ten = np.array(rec_ten) # [Pa]
rec_fza = np.array(rec_fza)
rec_dotlamp = np.array(rec_dotlamp)
rec_lamp = np.array(rec_lamp)
rec_ten = rec_ten * 1.e-6 # MPa
rec_carga = rec_carga * 1.e5 # [1.e-5 N]
rec_fza = rec_fza * 1.e5 # [1.e-5 N]

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
ax.plot(rec_lam, rec_carga)
ax.set_title("carga vs lam")

plt.show()