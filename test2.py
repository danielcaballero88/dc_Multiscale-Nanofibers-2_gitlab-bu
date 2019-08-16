import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time

class Energy(object):
    def __init__(self, r0, k1, k2):
        self.r0 = np.array(r0, dtype=float)
        self.k1 = k1
        self.k2 = k2
        self.r_ini = self.r0[0].copy()
        self.r_fin = self.r0[-1].copy()

    def energy(self, r_in, r_ini, r_fin, r0, lam, lam_r):
        """ r_in es el vector de coordenadas de solamente los nodos interiores
        los nodos frontera no """
        r0L = self.r0.reshape(-1,2)
        r0L = r0
        r_inL = r_in.reshape(-1,2)
        rL = np.zeros( np.shape(self.r0), dtype=float )
        rL[0] = r_ini
        rL[-1] = r_fin
        rL[1:-1] = r_inL
        drs0 = r0L[1:] - r0L[:-1]
        drs = rL[1:] - rL[:-1]
        dls0 = np.sqrt(np.sum(drs0*drs0,axis=1,keepdims=True))
        dls = np.sqrt(np.sum(drs*drs,axis=1,keepdims=True))
        drs0drs0 = np.sum(drs0[1:]*drs0[:-1], axis=1, keepdims=True)
        cos_phis0 = drs0drs0 / dls0[1:] / dls0[:-1]
        phis0 = np.arccos(cos_phis0 - 1.e-8*np.sign(cos_phis0))
        drsdrs = np.sum(drs[1:]*drs[:-1], axis=1, keepdims=True)
        cos_phis = drsdrs / dls[1:] / dls[:-1]
        phis = np.arccos(cos_phis- 1.e-8*np.sign(cos_phis))
        # energia por traccion
        ddls = dls - dls0
        E = 1000. * np.sum( ddls*ddls )
        # energia por torsion
        dphis = phis - phis0
        if lam<lam_r:
            k2 = 100.
        else:
            k2 = 1.
        E += k2 * np.sum( dphis*dphis )
        # restricciones
        # dr0 = rL[0] - np.array([0.,0.])
        # E += 1000. * np.sum(dr0*dr0)
        # dr2 = rL[5] - np.array([10.0,0.])
        # E += 1000. * np.sum(dr2*dr2)
        return E


# r0 = [
#     [0., 0.],
#     [0.5, -0.15],
#     [1., -0.2],
#     [1.5, -0.05],
#     [2., 0.0],
#     [2.5, 0.05],
#     [3., 0.2],
#     [3.5, 0.25],
#     [4., 0.15],
#     [4.5, 0.05],
#     [5., 0.]
# ]

nr0 = 20
x0 = np.linspace(0., 10., nr0)
y0 = np.sin(x0)
r0 = np.zeros( (nr0,2), dtype=float)
r0[:,0] = x0
r0[:,1] = y0

drete0 = r0[-1] - r0[0]
lete0 = np.sqrt(np.sum(drete0*drete0))
drs0 = r0[1:] - r0[:-1]
dls0 = np.sqrt(np.sum(drs0*drs0,axis=1))
loco0 = np.sum(dls0)
lamr = loco0 / lete0

E = Energy(r0, 1.e3, 1.e0)

poten = E.energy(E.r0[1:-1], E.r0[0], E.r0[-1], E.r0, 1.0, lamr)
print "energia inicial: ", poten

# E.r_ini[:] = [25., 32.]
E.r_fin[:] = [5., 0.]

drete = E.r_fin - E.r_ini
lete = np.sqrt(np.sum(drete*drete))


start = time.time()
P1 = opt.minimize( E.energy, E.r0[1:-1], (E.r_ini, E.r_fin, E.r0, lete/lete0, lamr))
print "time: ", time.time() - start
r_in = P1.x.reshape(-1,2)

r = E.r0.copy()
r[1:-1] = r_in
r[0] = E.r_ini
r[-1] = E.r_fin

poten = E.energy(r[1:-1], r[0], r[-1], E.r0, lete/lete0, lamr)
print "energia minimizada: ",poten

drs0 = E.r0[1:] - E.r0[:-1]
dls0 = np.sqrt(np.sum(drs0*drs0,axis=1,keepdims=True))
drs = r[1:] - r[:-1]
dls = np.sqrt(np.sum(drs*drs,axis=1,keepdims=True))

print (dls - dls0)/dls0


fig, ax = plt.subplots()
xx = list()
yy = list()
xx0 = list()
yy0 = list()
for r0val, rval in zip(E.r0,r):
    xx0.append(r0val[0])
    yy0.append(r0val[1])
    xx.append(rval[0])
    yy.append(rval[1])
ax.plot(xx0,yy0,"--")
ax.plot(xx,yy, ls="-", marker="s")
# ax.set_xlim(left=-0.2, right=10.2)
# ax.set_ylim(bottom=-5.7, top=5.7)
plt.show()

# print P1