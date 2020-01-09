import numpy as np
import matplotlib.pyplot as plt

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

class Arco(object):
    def __init__(self, C0, R0, kt, kf):
        self.C0 = C0
        self.R0 = R0
        self.kt = kt
        self.kf = kf
        self.L0 = self.calc_L(self.R0, self.C0)

    def calc_L(self, R, C):
        return 2.*R*np.arcsin(C/2./R)

    @staticmethod
    def fun_RCL(R, C, L):
        # funcion que se hace cero cuando R, C, y L estan correctos
        return L - 2.*R*np.arcsin(C/2./R)

    def calc_R(self, C, L, N=None, err=1.e-12):
        R1 = 0.5*C # menor valor posible
        L1 = self.calc_L(R1, C)
        if abs(L-L1)<err:
            return R1
        if L1<L:
            raise ValueError
        # itero para encontrar los limites para biseccion
        continuo = True
        while continuo:
            R2 = 2.*R1
            L2 = self.calc_L(R2, C)
            if L2<L:
                # encontre los limites
                continuo=False
            else:
                # sigo iterando
                R1 = R2
        # ya deberia tener R1 y R2: limites para biseccion
        # hago biseccion
        Rm = biseccion(self.fun_RCL, R1, R2, fargs=(C,L))
        return Rm

    def calc_E(self, lam, R):
        Ef = 0.5*self.L0*self.kf*(1./R - 1./self.R0)**2
        Et = 0.5*self.L0*self.kt*(lam-1.)**2
        E = Ef+Et
        return E, Et, Ef

    def calc_lam_L_R(self, C, dlam=1.e-4):
        if C<self.L0:
            # no esta recto
            # hay que encontrar el minimo de energia
            # el primer valor es con lam=1.
            lam1 = 1.0
            L1 = self.L0
            R1 = self.calc_R(C, L1)
            E1 = self.calc_E(lam1, R1)[0]
            lam = None
            n = int(1.1/dlam)
            for i in range(1,n+1):
                lam2 = 1. + float(i)*dlam
                L2 = lam2*self.L0
                R2 = self.calc_R(C, L2)
                E2 = self.calc_E(lam2, R2)[0]
                if E2>E1 and lam is None:
                    # entonces me quedo con el valor anterior de lam
                    lam = lam1
                    L = L1
                    R = R1
                    break
                lam1 = lam2
                E1 = E2
                R1 = R2
                L1 = L2
        else:
            L = C
            lam = L/self.L0
            R = 1.e1*self.R0
            R = 5.
        return lam, L, R

    def plot_lam_L_R(self, C, dlam=1.e-4):
        # no esta recto
        # hay que encontrar el minimo de energia
        # el primer valor es con lam=1.
        rec_lam=list()
        rec_E=list()
        lam1 = 1.0
        L1 = self.L0
        R1 = self.calc_R(C, L1)
        E1 = self.calc_E(lam1, R1)[0]
        lam = None
        n = int(0.1/dlam)
        rec_lam.append(lam1)
        rec_E.append(E1)
        for i in range(1,n+1):
            lam2 = 1. + float(i)*dlam
            L2 = lam2*self.L0
            R2 = self.calc_R(C, L2)
            E2 = self.calc_E(lam2, R2)[0]
            rec_lam.append(lam2)
            rec_E.append(E2)
            if E2>E1 and lam is None:
                # entonces me quedo con el valor anterior de lam
                lam = lam1
                L = L1
                R = R1
            lam1 = lam2
            E1 = E2
            R1 = R2
            L1 = L2
        fig, ax = plt.subplots()
        ax.plot(rec_lam, rec_E)
        plt.show()

    def calc_lam_L_R_2(self, C, dlam=1.e-6):
        if C<self.C0:
            # compresion
            # empiezo en lam=1 y voy hacia valores de compresion
            lam1 = 1.0
            L1 = self.L0
            R1 = self.calc_R(C, L1)
            E1 = self.calc_E(lam1, R1)[0]
            lam = None
            n = int(0.1/dlam)
            for i in range(1,n+1):
                lam2 = 1. - float(i)*dlam
                L2 = lam2*self.L0
                R2 = self.calc_R(C, L2)
                E2 = self.calc_E(lam2, R2)[0]
                if E2>E1 and lam is None:
                    lam = lam1
                    L = L1
                    R = R1
                    break
                lam1 = lam2
                E1 = E2
                R1 = R2
                L1 = L2
        elif C<self.L0:
            # no esta recto
            # hay que encontrar el minimo de energia
            # el primer valor es con lam=1.
            lam1 = 1.0
            L1 = self.L0
            R1 = self.calc_R(C, L1)
            E1 = self.calc_E(lam1, R1)[0]
            lam = None
            n = int(0.1/dlam)
            for i in range(1,n+1):
                lam2 = 1. + float(i)*dlam
                L2 = lam2*self.L0
                R2 = self.calc_R(C, L2)
                E2 = self.calc_E(lam2, R2)[0]
                if E2>E1 and lam is None:
                    # entonces me quedo con el valor anterior de lam
                    lam = lam1
                    L = L1
                    R = R1
                    break
                lam1 = lam2
                E1 = E2
                R1 = R2
                L1 = L2
        else:
            # arco recto
            # empiezo con le arco casi casi recto (radio muy grande)
            # y empiezo a curvarlo de a poco si es necesario
            R1 = 1.e2*self.R0
            L1 = self.calc_L(R1, C)
            lam1 = L1/self.L0
            E1 = self.calc_E(lam1, R1)[0]
            lam = None
            n = int(0.1/dlam)
            for i in range(1,n+1):
                lam2 = lam1 + float(i)*dlam
                L2 = lam2*self.L0
                R2 = self.calc_R(C, L2)
                E2 = self.calc_E(lam2, R2)[0]
                if E2>E1 and lam is None:
                    lam = lam1
                    L = L1
                    R = R1
                    break
                lam1 = lam2
                E1 = E2
                R1 = R2
                L1 = L2
        return lam, L, R

    def graficar_arco(self, R, L):
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
        ax.plot(rec_x, rec_y, linewidth=2, c="k")
        C = 2 * R * np.sin(0.5*alpha)
        ax.set_xlim(left=-0.2*self.C0, right=1.5*self.C0)
        ax.set_ylim(bottom=-0.2*self.C0, top=1.5*self.C0)
        return fig, ax

a1 = Arco(C0=1.0, R0=0.6, kt=1000., kf=100.)
L0 = 1.05 * a1.C0
R0 = a1.calc_R(a1.C0, L0)
print a1.calc_R(0.8,1.)
kt = 6.e-4
kf = kt * 1.e-14
a1 = Arco(C0=1.0, R0=R0, kt=kt, kf=kf)


# C = 0.95
# lam, L, R = a1.calc_lam_L_R_2(C)
# fig, ax = a1.graficar_arco(R, L)
# fname = "C_" + "{:4.2f}".format(C) +".pdf"
# fig.savefig(fname, bbox_inches='tight')

# C = 1.0
# lam, L, R = a1.calc_lam_L_R_2(C)
# fig, ax = a1.graficar_arco(R, L)
# fname = "C_" + "{:4.2f}".format(C) +".pdf"
# fig.savefig(fname, bbox_inches='tight')

# C = 1.05
# lam, L, R = a1.calc_lam_L_R_2(C)
# fig, ax = a1.graficar_arco(R, L)
# fname = "C_" + "{:4.2f}".format(C) +".pdf"
# fig.savefig(fname, bbox_inches='tight')

# C = 1.1
# lam, L, R = a1.calc_lam_L_R_2(C)
# fig, ax = a1.graficar_arco(R, L)
# fname = "C_" + "{:4.2f}".format(C) +".pdf"
# fig.savefig(fname, bbox_inches='tight')

# C = 1.15
# lam, L, R = a1.calc_lam_L_R_2(C)
# fig, ax = a1.graficar_arco(R, L)
# fname = "C_" + "{:4.2f}".format(C) +".pdf"
# fig.savefig(fname, bbox_inches='tight')

# plt.show()

rec_C = np.linspace(0.95, 1.15, 500)
rec_lam = list()
rec_Et = list()
rec_Ef = list()
rec_E = list()
rec_F = list()
rec_Ft = list()
rec_Ff = list()
rec_R = list()

for C in rec_C:
    lam, L, R = a1.calc_lam_L_R_2(C)
    E, Et, Ef = a1.calc_E(lam, R)
    rec_Et.append(Et)
    rec_Ef.append(Ef)
    rec_E.append(E)
    rec_R.append(R)
    rec_lam.append(lam)
    # ahora para la fuerza calculo variaciones
    Cm = C - 1.e-8
    lam, L, R = a1.calc_lam_L_R_2(Cm)
    Em, Emt, Emf = a1.calc_E(lam, R)
    Cp = C + 1.e-8
    lam, L, R = a1.calc_lam_L_R_2(Cp)
    Ep, Ept, Epf = a1.calc_E(lam, R)
    # derivada = fuerza
    F = (Ep-Em)/2.e-8
    Ft = (Ept-Emt)/2.e-8
    Ff = (Epf-Emf)/2.e-8
    rec_F.append(F)
    rec_Ft.append(Ft)
    rec_Ff.append(Ff)

fid = open("log_arco.txt", "w")

formato = "{:>30s}"*9
linea = formato.format("C", "lam", "R", "Et", "Ef", "E", "Ft", "Ff", "F") + "\n"
fid.write(linea)
for C, lam, R, Et, Ef, E, Ft, Ff, F in zip(rec_C, rec_lam, rec_R, rec_Et, rec_Ef, rec_E, rec_Ft, rec_Ff, rec_F):
    formato = "{:30.18e}"*9
    linea = formato.format(C, lam, R, Et, Ef, E, Ft, Ff, F) + "\n"
    fid.write(linea)

fid.close()


fig, ax = plt.subplots()
ax.plot(rec_C, rec_Et)
ax.plot(rec_C, rec_Ef)
ax.plot(rec_C, rec_E)

fig, ax = plt.subplots()
ax.plot(rec_C, rec_Ft, "--")
ax.plot(rec_C, rec_Ff, "-.")
ax.plot(rec_C, rec_F)

fig, ax = plt.subplots()
ax.plot(rec_C, rec_R)

fig, ax = plt.subplots()
ax.plot(rec_C, rec_lam)

plt.show()