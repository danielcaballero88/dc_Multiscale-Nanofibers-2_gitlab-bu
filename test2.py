import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class Energy(object):
    def __init__(self, r0, k1, k2):
        self.r0 = np.array(r0, dtype=float)
        self.r = np.array(r0, dtype=float)
        self.k1 = k1
        self.k2 = k2
        self.r_ini = self.r0[0].copy()
        self.r_fin = self.r0[-1].copy()

    def energy(self, r_in):
        """ r_in es el vector de coordenadas de solamente los nodos interiores
        los nodos frontera no """
        r0L = self.r0.reshape(-1,2)
        r_inL = r_in.reshape(-1,2)
        rL = np.zeros( np.shape(self.r0), dtype=float )
        rL[0] = self.r_ini
        rL[-1] = self.r_fin
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
        E = 0.5 * np.sum( self.k1*ddls*ddls )
        # energia por torsion
        dphis = phis - phis0
        E += 0.5 * np.sum( self.k2*dphis*dphis )
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

# curva: medio circulo
# nr0 = 20
# Radio = 1.
# Long = np.pi*Radio
# a_s0 = np.linspace(0., Long, nr0)
# a_th0 = a_s0/Radio
# a_x0 = Radio - np.cos(a_th0)
# a_y0 = np.sin(a_th0)

# curva: sinusoidal
nr0 = 20
Long_x = 2.*np.pi
Amplitud = 0.5;
a_x0 = np.linspace(0., Long_x, nr0)
a_y0 = Amplitud * np.sin(a_x0)

av_r0 = np.zeros( (nr0,2), dtype=float)
av_r0[:,0] = a_x0
av_r0[:,1] = a_y0

av_drs0 = av_r0[1:] - av_r0[:-1]
a_dls0 = np.sqrt(np.sum(av_drs0*av_drs0,axis=1,keepdims=True))
Long = np.sum(a_dls0)

E = Energy(av_r0, 1.e6, 1.e3)


poten = E.energy(E.r0[1:-1])
print "energia inicial: ", poten

a_poten = list()
a_xfin = np.linspace(2., 2.2, 10) * np.pi
# a_xfin = np.linspace(2.*Radio, np.pi*Radio, 5)
# a_xfin = np.append(a_xfin, np.linspace(np.pi*Radio, 4.*Radio,5))
dx = 0.01
a_F = list()

for xfin in a_xfin:
    print "xfin: ", xfin

    E.r_fin[:] = [xfin, .0]
    P1 = opt.minimize( E.energy, E.r0[1:-1], options={"ftol":1.e-12, "gtol":1.e-12})
    av_r_in = P1.x.reshape(-1,2)
    av_r = E.r0.copy()
    av_r[1:-1] = av_r_in
    av_r[0] = E.r_ini
    av_r[-1] = E.r_fin
    # E.r = av_r.copy()

    poten = E.energy(av_r[1:-1])
    print "energia minimizada: ",poten
    a_poten.append(poten)

    # # ahora un valor aumentado en x para calcular la derivada numerica
    # E.r_fin[:] = [xfin+dx, .0]
    # P1 = opt.minimize( E.energy, E.r0[1:-1])
    # av_r_in = P1.x.reshape(-1,2)
    # av_r = E.r0.copy()
    # av_r[1:-1] = av_r_in
    # av_r[0] = E.r_ini
    # av_r[-1] = E.r_fin

    # potenPlus = E.energy(av_r[1:-1])

    # derPoten = (potenPlus-poten) / dx
    # a_F.append(derPoten)

    # av_drs0 = E.r0[1:] - E.r0[:-1]
    # a_dls0 = np.sqrt(np.sum(av_drs0*av_drs0,axis=1,keepdims=True))
    # av_drs = av_r[1:] - av_r[:-1]
    # a_dls = np.sqrt(np.sum(av_drs*av_drs,axis=1,keepdims=True))
    # print av_drs0
    # print a_dls0
    # print
    # print av_drs
    # print a_dls

    fig, ax = plt.subplots()
    xx = list()
    yy = list()
    xx0 = list()
    yy0 = list()
    for r0val, rval in zip(E.r0,av_r):
        xx0.append(r0val[0])
        yy0.append(r0val[1])
        xx.append(rval[0])
        yy.append(rval[1])
    ax.plot(xx0,yy0,"--")
    ax.plot(xx,yy)
    # ax.set_xlim(left=-0.2, right=5.2)
    # ax.set_ylim(bottom=-2.7, top=2.7)

fig, ax = plt.subplots()
ax.plot(a_xfin, a_poten)

# fig, ax = plt.subplots()
# ax.plot(a_xfin, a_F)

plt.show()