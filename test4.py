import numpy as np
import matplotlib.pyplot as plt


def deltaVector(n,i):
    dvec = np.zeros(n, dtype=float)
    dvec[i] = 1.
    return dvec

class Fibra(object):
    def __init__(self, kt, kf, r0, rA, rB):
        self.kt = kt
        self.kf = kf
        self.r0 = r0
        self.rA = rA
        self.rB = rB
        self.r = r0.copy()
        self.dr0, self.dl0, self.phi0 = self.calcular_dr_dl_phi(self.r0)

    def calcular_dr_dl_phi(self, r):
        dr = r[1:] - r[:-1]
        dl = np.sqrt( np.sum(dr*dr, axis=1, keepdims=True) )
        cosphi = np.sum( dr[1:] * dr[:-1], axis=1, keepdims=True ) / dl[1:] / dl[:-1]
        phi = np.arccos(cosphi - 1.e-8*np.sign(cosphi))
        return dr, dl, phi

    def calcular_energia(self, r, verbose=False):
        dr, dl, phi = self.calcular_dr_dl_phi(r)
        ddl = dl - self.dl0
        # Energia de traccion
        Et = 0.5 * self.kt * np.sum(ddl*ddl)
        # print "Et: ", Et
        dphi = phi - self.phi0
        # Energia de flexion
        Ef = 0.5 * self.kf * np.sum(dphi*dphi)
        # print "Ef: ", Ef
        drA = r[0] - self.rA
        drB = r[-1] - self.rB
        # Energia de restricciones
        Er = 0.5 * 100.*self.kt * (np.sum(drA*drA) + np.sum(drB*drB))
        if verbose:
            print "mean lambda: ", np.mean(dl/self.dl0)
        return Et+Ef+Er

    def deformar_afin(self, F):
        self.r = np.matmul(self.r0, np.transpose(F))

    def calcular_E_d_a(self):
        # calculo de gradiente (dE) y Hessiano en direccion del gradiente (ddE)
        beta = 1.e-4
        #
        E = self.calcular_energia(self.r)
        # gradiente
        nr = self.r.shape[0]
        ndof = 2*nr
        rL = np.zeros( ndof, dtype=float )
        dE = np.zeros( ndof, dtype=float)
        rL[:] = self.r.reshape(ndof,1)[:,0]
        for idof in range(ndof):
            rLp = rL + beta*deltaVector(ndof,idof)
            rLm = rL - beta*deltaVector(ndof,idof)
            Ep = self.calcular_energia(rLp.reshape(nr,2))
            Em = self.calcular_energia(rLm.reshape(nr,2))
            dE[idof] = (Ep-Em)/(2*beta)
        # vector direccion de maximo descenso
        d = -dE / np.sqrt(np.sum(dE*dE))
        dEd = np.sum(dE*d)
        # hessiano
        rLp = rL + beta*d
        rLm = rL - beta*d
        Ep = self.calcular_energia(rLp.reshape(nr,2))
        Em = self.calcular_energia(rLm.reshape(nr,2))
        ddEd = (Ep - 2*E + Em) / beta**2
        # magnitud de desplazamiento
        a = - dEd/ddEd
        return E, d, a

    def graficar(self):
        fig, ax = plt.subplots()
        xx = list()
        yy = list()
        xx0 = list()
        yy0 = list()
        for r0, r in zip(f1.r0, f1.r):
            xx0.append(r0[0])
            yy0.append(r0[1])
            xx.append(r[0])
            yy.append(r[1])
        ax.plot(xx0,yy0,"--")
        ax.plot(xx,yy)

kt = 1000.
kf = 0.01
av_r0 = [
    [0.,0.],
    [1.,1],
    [2.,0.]
]
av_r0 = np.array(av_r0, dtype=float)
rA = av_r0[0].copy()
rB = av_r0[-1].copy()


nr = 5
xmax = 2*np.pi
Ampl = np.sqrt(0.69)*np.pi*0.5
av_r0 = np.zeros( (nr,2), dtype=float )
av_r0[:,0] = np.linspace(0, xmax, nr)
av_r0[:,1] = np.sin(av_r0[:,0])
rA = av_r0[0].copy()
rB = av_r0[-1].copy()

f1 = Fibra(kt, kf, av_r0, rA, rB)

print f1.calcular_energia(f1.r)

nE = 11
extensiones1 = np.linspace(1.01, 1.51, nE)
extensiones = list()
for iE in range(nE):
    em = extensiones1[iE] - 1.0e-2
    ep = extensiones1[iE] + 1.0e-2
    extensiones.append(em)
    extensiones.append(ep)
nE = 2*nE
rec_E = np.zeros(nE, dtype=float)

for iE in range(nE):

    extension = extensiones[iE]

    F = np.array(
        [
            [extension, 0.],
            [0., 1.]
        ]
    )

    f1.deformar_afin(F)
    if extension >= 0.3:
        f1.r[:,1] = 0.

    f1.rB = f1.r[-1].copy()

    f1.graficar()

    print "Energia afin: ", f1.calcular_energia(f1.r, verbose=True)


    for j in range(1000):
        E, d, a = f1.calcular_E_d_a()
        new_r = f1.r + a * d.reshape(-1,2)
        # print "E: ", E
        # print d
        # print a
        # print new_r
        f1.r = new_r

    E = f1.calcular_energia(f1.r, verbose=True)
    print "Energia final: ", E

    f1.graficar()

    rec_E[iE] = E

fig, ax = plt.subplots()
ax.plot(extensiones, rec_E)

print "resultados: extension, energia "
for iE in range(nE):
    print "{:20.8e}".format(extensiones[iE]), "{:20.8e}".format(rec_E[iE])

n = len(extensiones)
rec_F = list()
rec_extm = list()
for i in range(n/2):
    j = 2*i
    ext1 = extensiones[j]
    ext2 = extensiones[j+1]
    E1 = rec_E[j]
    E2 = rec_E[j+1]
    extm = (ext1+ext2)*0.5
    F = (E2-E1)/(ext2-ext1)
    rec_extm.append(extm)
    rec_F.append(F)

fig, ax = plt.subplots()
ax.plot(rec_extm, rec_F, marker=".")

plt.show()