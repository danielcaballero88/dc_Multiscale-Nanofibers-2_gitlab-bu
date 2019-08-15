import numpy as np
import matplotlib.pyplot as plt

class Mallita(object):
    def __init__(self):
        # long de lado
        self.L = 1.
        # posiciones iniciales
        self.nnodos = 6
        self.r0 = np.array(
            [
                [0., 0.],
                [1., 0.],
                [1., 1.],
                [0., 1.],
                [.2, .5],
                [.8, .5]
            ],
            dtype=float
        )
        # tipos (1=frontera, 2=interseccion)
        self.tipos = np.transpose(np.array(
            [
            [1, 1, 1, 1, 2, 2]
            ],
            dtype=int
        ))
        # fibras: conectividad de nodos
        self.nfibras = 5
        self.fibras = np.array(
            [
                [0,4],
                [1,5],
                [2,5],
                [3,4],
                [4,5]
            ],
            dtype=int
        )
        # parametros constitutivos
        self.param = np.array(
            [
                [1.1, 10., .01],
                [1.1, 10., .01],
                [1.1, 10., .01],
                [1.1, 10., .01],
                [1.1, 10., .01]
            ],
            dtype=float
        )
        # ------------------------------
        # posiciones actuales
        self.r = self.r0.copy()
        # longitudes iniciales
        self.longs0 = self.calcular_longitudes_iniciales()
        # fuerzas (escalares)
        self.fuerzas = np.zeros( (self.nfibras,1), dtype=float)

    @staticmethod
    def fuerza_fibra(lam, lam_r=1.1, k1=10., k2=.1):
        """ fuerza de una fibra """
        if lam<=lam_r:
            return k2*(lam-1.)
        else:
            return k2*(lam_r-1.) + k1*(lam/lam_r - 1.)

    def calcular_longitudes_iniciales(self):
        fibras_dr0 = self.r0[ self.fibras[:,1] ] - self.r0[ self.fibras[:,0] ]
        fibras_dl0 = np.sqrt( np.sum( fibras_dr0*fibras_dr0, axis=1, keepdims=True ) )
        return fibras_dl0

    def setear_r(self, r):
        self.r = r

    def calcular_vectores_fibras(self):
        fibras_dr = self.r[ self.fibras[:,1] ]  - self.r[ self.fibras[:,0] ]
        return fibras_dr

    def calcular_fuerzas(self):
        drs = self.calcular_vectores_fibras()
        longs = np.sqrt(np.sum(drs*drs, axis=1, keepdims=True))
        lams = longs / self.longs0
        mr = np.greater(lams, self.param[:,0,None]) # mask rectas
        me = np.logical_not(mr) # mask enruladas
        lamsr = self.param[:,0,None] # el None es para tener un vector columna
        ks1 = self.param[:,1,None]
        ks2 = self.param[:,2,None]
        fuerzas_de_reclut = ks2*(lamsr-1.)
        self.fuerzas[mr] = fuerzas_de_reclut[mr] + ks1[mr]*(lams[mr]/lamsr[mr] - 1.)
        self.fuerzas[me] = ks2[me]*(lams[me] - 1.)
        a = self.fuerzas/longs * drs
        return a

    def calcular_fuerza_de_una_fibra(self, f, r_n0, r_n1):
        dr = r_n1 - r_n0
        long = np.sqrt(np.sum(dr*dr))
        lam = long / self.longs0[f]
        lamr, k1, k2 = self.param[f]
        k1 = self.param[f,1]
        fuerza_de_reclut = k2*(lamr-1.)
        if lam > lamr:
            fuerza = fuerza_de_reclut + k1*(lam/lamr - 1.)
        else:
            fuerza = k2*(lam-1.)
        return fuerza/long * dr[:,None]

    def calcular_matriz_tangente(self):
        nG = self.nnodos*2
        matG = np.zeros( (nG,nG), dtype=float  )
        vecG = np.zeros( (nG,1), dtype=float )
        nL = 2 # tengo que armar la matriz tangente respecto de un solo nodo (hay doble simetria)
        matL = np.zeros( (nL,nL), dtype=float )
        vecL = np.zeros( (nL,1), dtype=float )
        #
        delta = 1.e-4
        delta21 =  1. / (2.*delta)
        delta_x = delta * np.array( [1., 0.], dtype=float )
        delta_y = delta * np.array( [0., 1.], dtype=float )
        for f, (n0,n1) in enumerate(self.fibras):
            if f==2:
                pass
            r_n0 = self.r[n0]
            r_n1 = self.r[n1]
            r_n0_px = self.r[n0] + delta_x
            r_n0_mx = self.r[n0] - delta_x
            r_n0_py = self.r[n0] + delta_y
            r_n0_my = self.r[n0] - delta_y
            F_c = self.calcular_fuerza_de_una_fibra(f, r_n0, r_n1)
            F_mx = self.calcular_fuerza_de_una_fibra(f, r_n0_mx, r_n1)
            F_px = self.calcular_fuerza_de_una_fibra(f, r_n0_px, r_n1)
            F_my = self.calcular_fuerza_de_una_fibra(f, r_n0_my, r_n1)
            F_py = self.calcular_fuerza_de_una_fibra(f, r_n0_py, r_n1)
            dFdx = (F_px - F_mx) * delta21
            dFdy = (F_py - F_my) * delta21
            matL[:,0] = dFdx[:,0]
            matL[:,1] = dFdy[:,0]
            vecL = - F_c
            # ahora a ensamblar
            # primero el vector de cargas
            row = n0*2
            col = n1*2
            vecG[row:row+2] += vecL
            vecG[col:col+2] += -vecL
            # luego matriz local va a 4 submatrices de la global
            # primero en el nodo 0
            row = n0*2
            col = n0*2
            matG[row:row+2,col:col+2] += matL
            # luego lo mismo en el nodo 1
            row = n1*2
            col = n1*2
            matG[row:row+2,col:col+2] += matL
            # luego las cruzadas
            row = n0*2
            col = n1*2
            matG[row:row+2,col:col+2] += - matL
            row = n1*2
            col = n0*2
            matG[row:row+2,col:col+2] += - matL
        # ahora las condiciones de dirichlet
        for n, (x0,y0) in enumerate(self.r0):
            if self.tipos[n] == 1:
                ix = 2*n
                iy = 2*n+1
                matG[ix,:] = 0.
                matG[ix,ix] = 1.
                vecG[ix] = 0.
                matG[iy,:] = 0.
                matG[iy,iy] = 1.
                vecG[iy] = 0.
        # fin
        return matG, vecG



m = Mallita()

# m.r[2,:] = [1.1, 1.1]
# m.r[1,:] = [1.1, 0.1]

Fmacro = np.array(
    [
        [1.1, 0.0],
        [0.0, 1.0]
    ],
    dtype=float
)

m.r = np.matmul(m.r0, np.transpose(Fmacro))

F = m.calcular_fuerzas()
print "F"
print F
print "---"

A, b = m.calcular_matriz_tangente()

print "A // b"
msg = ""
for n in range(m.nnodos):
    for i in range(2):
        for o in range(m.nnodos):
            msg += "{:12.2e}{:12.2e}".format(A[2*n+i,2*o+0],A[2*n+i,2*o+1])
        msg += "  //  "
        msg += "{:12.2e}".format(b[2*n+i,0])
        msg += "\n"
print msg
print "---"



dr = np.linalg.solve(A,b)
print "dr"
print dr
print "---"

residuo = np.matmul(A,dr) - b
print "residuo"
print residuo
print "---"



print "ITER 2"
m.r = m.r + dr.reshape(-1,2)

F = m.calcular_fuerzas()
print "F"
print F
print "---"

A, b = m.calcular_matriz_tangente()

print "A // b"
msg = ""
for n in range(m.nnodos):
    for i in range(2):
        for o in range(m.nnodos):
            msg += "{:12.2e}{:12.2e}".format(A[2*n+i,2*o+0],A[2*n+i,2*o+1])
        msg += "  //  "
        msg += "{:12.2e}".format(b[2*n+i,0])
        msg += "\n"
print msg
print "---"



dr = np.linalg.solve(A,b)
print "dr"
print dr
print "---"


print "ITER 3"
m.r = m.r + dr.reshape(-1,2)

F = m.calcular_fuerzas()
print "F"
print F
print "---"

A, b = m.calcular_matriz_tangente()

print "A // b"
msg = ""
for n in range(m.nnodos):
    for i in range(2):
        for o in range(m.nnodos):
            msg += "{:12.2e}{:12.2e}".format(A[2*n+i,2*o+0],A[2*n+i,2*o+1])
        msg += "  //  "
        msg += "{:12.2e}".format(b[2*n+i,0])
        msg += "\n"
print msg
print "---"



dr = np.linalg.solve(A,b)
print "dr"
print dr
print "---"


print "r"
print m.r
print "---"

print m.longs0
drs = m.calcular_vectores_fibras()
print np.sqrt(np.sum(drs*drs,axis=1,keepdims=True))

m.r = m.r + dr.reshape(-1,2)

fig,ax = plt.subplots()
for f, (n0,n1) in enumerate(m.fibras):
    x0, y0 = m.r0[n0]
    dx, dy = m.r0[n1] - m.r0[n0]
    ax.arrow(x0,y0,dx,dy, linestyle=":")
for f, (n0,n1) in enumerate(m.fibras):
    x0, y0 = m.r[n0]
    dx, dy = m.r[n1] - m.r[n0]
    ax.arrow(x0,y0,dx,dy)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
plt.show()