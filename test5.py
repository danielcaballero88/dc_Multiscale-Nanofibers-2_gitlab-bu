import numpy as np
import matplotlib.pyplot as plt


def fR(R, C, L):
    return L - 2.*R*np.arcsin(C/2./R)

def dfR(R, C, L):
    aux = np.arcsin(C/2./R)
    return 2*( np.tan(aux) - aux )

def fL(R, C):
    return 2.*R*np.arcsin(C/2./R)

def biseccion(f,a,b,N=None,err=1.e-12, fargs=()):
    fa = f(a, *fargs)
    fb = f(b, *fargs)
    if fa == 0.:
        return a
    elif fb == 0.:
        return b
    elif fa*fb >= 0.:
        raise ValueError
    if N is None:
        N = np.log2( (b-a)/err )
        N = int(np.ceil(N))
    for k in range(1,N+1):
        m = (a+b)/2.
        fm = f(m, *fargs)
        if fa*fm < 0.:
            b = m
            fb = fm
        elif fm*fb <0.:
            a = m
            fa = fm
        elif fm == 0.:
            return m
        else:
            raise ValueError
    return m

def resolver_R(C, L, N=None):
    R1 = 0.5*C # es el minimo valor posible
    L1 = 2.*R1*np.arcsin(0.5*C/R1)
    if abs(L-L1)<1.e-8:
        return R1
    if L1<L:
        raise ValueError
    # itero para encontrar los limites para biseccion
    continuo = True
    while continuo:
        R2 = 2.*R1
        L2 = 2.*R2*np.arcsin(0.5*C/R2)
        if L2<L:
            # encontre los limites
            continuo=False
        else:
            # sigo iterando
            R1 = R2
    # ya deberia tener R1 y R2: limites para biseccion
    # hago biseccion
    Rm = biseccion(fR, R1, R2, fargs=(C,L))
    return Rm

def graficar_arco(R, L):
    n = 100
    alpha = L/R
    beta0 = 0.5*(np.pi - alpha)
    rec_phi = np.linspace(0., alpha, n).tolist()
    rec_x = list()
    rec_y = list()
    for phi in rec_phi:
        beta = beta0 + phi
        x = R*(np.cos(beta0) - np.cos(beta))
        y = R*(np.sin(beta) - np.sin(beta0))
        rec_x.append(x)
        rec_y.append(y)
    fig, ax = plt.subplots()
    ax.plot(rec_x, rec_y)
    C = 2 * R * np.sin(0.5*alpha)
    ax.set_xlim(left=0., right=C)
    ax.set_ylim(bottom=0., top=C)


kt = 1000.
kf = 0.1

C0 = 1.
R0 = 0.6
L0 = fL(R0, C0)
print L0


C = 1.18
# C<L0, tengo que hallar el valor de lam que minimiza la energia
lam1 = 1.
lam2 = 1.05

def Energia(kt, kf, L0, R0, lam, R):
    Ef = 0.5*L0*kf*(1./R - 1./R0)**2
    Et = 0.5*L0*kt*(lam-1.)**2
    E = Ef+Et
    return E, Et, Ef

rec_lam = list()
rec_E = list()
for lam in np.linspace(lam1, lam2, 100):
    L = lam*L0
    # C<L0<L
    R = resolver_R(C, L)
    E, Et, Ef = Energia(kt, kf, L0, R0, lam, R)
    rec_lam.append(lam)
    rec_E.append(E)

fig, ax = plt.subplots()
ax.plot(rec_lam, rec_E, ".")
plt.show()

rec_C = np.linspace(C0, 2., 1000)
rec_Et = list()
rec_Ef = list()
rec_E = list()
rec_F = list()
rec_R = list()

def lam_L_R(C,L0,R0):
    if C<L0:
        lam = 1.
        L = lam*L0
        R = resolver_R(C,L)
    else:
        L = C
        lam = L/L0
        R = 1.e3*R0
    return lam, L, R

for C in rec_C:
    lam, L, R = lam_L_R(C, L0, R0)
    Et = 0.5*kt*L0*(lam-1)**2
    Ef = 0.5*kf*L0*(1/R - 1/R0)**2
    E = Et + Ef
    rec_Et.append(Et)
    rec_Ef.append(Ef)
    rec_E.append(E)
    rec_R.append(R)
    # ahora para la fuerza calculo variaciones
    Cm = C - 1.e-8
    lam, L, R = lam_L_R(Cm, L0, R0)
    Et = 0.5*kt*L0*(lam-1)**2
    Ef = 0.5*kf*L0*(1/R - 1/R0)**2
    Em = Et + Ef
    Cp = C + 1.e-8
    lam, L, R = lam_L_R(Cp, L0, R0)
    Et = 0.5*kt*L0*(lam-1)**2
    Ef = 0.5*kf*L0*(1/R - 1/R0)**2
    Ep = Et + Ef
    # derivada = fuerza
    F = (Ep-Em)/1.e-8
    rec_F.append(F)

fig, ax = plt.subplots()
ax.plot(rec_C, rec_Et)
ax.plot(rec_C, rec_Ef)
ax.plot(rec_C, rec_E)

fig, ax = plt.subplots()
ax.plot(rec_C, rec_F)

fig, ax = plt.subplots()
ax.plot(rec_C, rec_R)

plt.show()