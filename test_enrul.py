import Rve_generador2
import Malla_equilibrio_2
from matplotlib import pyplot as plt
import numpy as np
import os.path


def calcular_malla_completa(L, dl, dtheta, intersecciones=False):
    mc = Rve_generador2.Malla(L)
    for i in range(100):
        mc.make_fibra(dl, dtheta)
        mc.trim_fibra_at_frontera(mc.fibs.con[-1])
        if intersecciones:
            mc.intersectar_fibras()
    return mc

def calcular_malla_simplificada(mc):
    # malla simplificada
    parcon = [0.1e9, 0.1e6, 0.0] # Et, Eb y lamr (que va a ser calculado fibra a fibra)
    def eccon(lam, paramcon):
        Et = paramcon[0]
        Eb = paramcon[1]
        lamr = paramcon[2]
        if lam<=lamr:
            return Et*(lam-1.)
        else:
            return Eb*(lamr-1.) + Et*(lam/lamr - 1.)
    pseudovisc = 0.1e11
    ms = Malla_equilibrio_2.Malla.simplificar_malla_completa(mc, parcon, eccon, pseudovisc)
    return ms

def calcular_distr_de_enrul(ms):
    # distr de enrul
    lams_r = list()
    for i in range( ms.sfs.num ):
        lete = ms.sfs.lete0[i]
        loco = ms.sfs.loco0[i]
        lam_r = loco/lete
        lams_r.append( lam_r )
    lams_r = np.array(lams_r)
    meanv = np.mean(lams_r)
    stdev = np.std(lams_r)
    return meanv, stdev


def calcular_y_guardar():
    pi = 3.1416
    L = 1.0
    n = 10
    # crear carpeta
    subdir = "mallas"
    try:
        os.mkdir(subdir)
    except Exception as e:
        print e
    dths = np.linspace(0.1*pi, 0.2*pi, n)
    dls = np.linspace(0.1*L, 0.02*L, n)
    means = list()
    stdevs = list()
    for i in range(n):
        print i
        dl = dls[i]
        dth = dths[i]
        mc = calcular_malla_completa(L, dl, dth)
        dummystring = os.path.join(subdir, "mc_" + "{:05d}".format(i) + ".txt")
        mc.guardar_en_archivo(archivo=dummystring)
        ms = calcular_malla_simplificada(mc)
        dummystring = os.path.join(subdir, "ms_" + "{:05d}".format(i) + ".txt")
        ms.guardar_en_archivo(archivo=dummystring)
        mu, sigma = calcular_distr_de_enrul(ms)
        means.append(mu)
        stdevs.append(sigma)
        return n, means, stdevs


def leer_y_graficar(n):
    print i
    for i in range(n):
        fdir = os.path.join("mallas", "ms_" + "{:05d}".format(i) + ".txt")
        ms = Malla_equilibrio_2.Malla.leer_de_archivo(fdir)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(means, label="means")
ax.plot(stdevs, label="stdevs")
ax.legend(loc="upper left")
plt.show()