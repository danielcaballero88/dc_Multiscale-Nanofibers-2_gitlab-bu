from Malla_completa import Malla as Mc
from Malla_simplificada import Malla as Ms
import time
from matplotlib import pyplot as plt
import numpy as np

Dm = 1.0
nfibs = 0.3

ncapss = [10]
Ls = [50.]
devangs_deg = [5., 10., 15., 20., 25., 30., 35., 40., 45.]
dls_rel = [1.]

nmallas = 10

working_dictory = "/home/dancab/Documents/academia/doctorado/articulos/multiscale_nanofibers_randomRVE_2/Analisis_geometria/06_distr_de_recl_vs_Nf/"

def get_mean_stdev_lamr(mc):
    rec_lamsr = mc.calcular_enrulamientos_de_interfibras()
    mean_lamr = np.mean(rec_lamsr)
    stdev_lamr = np.std(rec_lamsr)
    return mean_lamr, stdev_lamr

rec_means_means = list()
rec_stdevs_means = list()
rec_means_stdevs = list()
rec_stdevs_stdevs = list()

for ncaps in ncapss:
    for L in Ls:
        for dl_rel in dls_rel:
            dl = dl_rel * Dm
            for devang_deg in devangs_deg:
                devang = devang_deg*np.pi/180.
                rec_means = list()
                rec_stdevs = list()
                for nm in range(1,nmallas+1):
                    print "ncaps={:05d}  L = {:08.2f}  devang = {:05.2f}  dl_rel = {:05.2f}  nm = {:07d}".format(ncaps, L, devang_deg, dl_rel, nm)
                    #
                    nombrearchivo = working_dictory + \
                                    "L_" + "{:08.1f}".format(L) + \
                                    "_dlrel_" + "{:05.2f}".format(dl_rel) + \
                                    "_devang_" + "{:05.2f}".format(devang_deg) + \
                                    "_ncaps_" + "{:07d}".format(ncaps) + \
                                    "_nm_" + "{:07d}".format(nm) + \
                                    "_i.txt"
                    #
                    mc = Mc.leer_de_archivo(archivo=nombrearchivo)
                    #
                    mean, stdev = get_mean_stdev_lamr(mc)
                    rec_means.append(mean)
                    rec_stdevs.append(stdev)
                mean_means = np.mean(rec_means)
                stdev_means = np.std(rec_means)
                mean_stdevs = np.mean(rec_stdevs)
                stdev_stdevs = np.std(rec_stdevs)
                rec_means_means.append(mean_means)
                rec_stdevs_means.append(stdev_means)
                rec_means_stdevs.append(mean_stdevs)
                rec_stdevs_stdevs.append(stdev_stdevs)

fid = open(working_dictory + "000_data_distr_de_recl_interf.txt", "w")
formato = "{:>20s}"*5
encabezado = formato.format("devangmax", "rec_means_means", "rec_stdevs_means", "rec_means_stdevs", "rec_stdevs_stdevs")
fid.write(encabezado + "\n")

formato = "{:20.8f}"*5
for devangmax, rmm, rsm, rms, rss in zip(devangs_deg, rec_means_means, rec_stdevs_means, rec_means_stdevs, rec_stdevs_stdevs):
    linea = formato.format(devangmax, rmm, rsm, rms, rss) + "\n"
    fid.write(linea)

fid.close()

# SMALL_SIZE = 8
# MEDIUM_SIZE = 16
# BIGGER_SIZE = 18
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# ax.errorbar(devangs_deg, rec_means_means, yerr=rec_stdevs_means, fmt="o", markersize=7, mfc="k", mec="k", capsize=10, ecolor="k")
# # ax.set_xticks(np.arange(50,201,50))
# # ax.set_yticks(np.array([0., 0.25, 0.5, 0.75, 1.])*np.pi)
# # ax.set_yticklabels(ax.get_yticks()/np.pi)
# # hline_y = 0.5*np.pi
# # hline_x0, hline_x1 = ax.get_xlim()
# # ax.plot((hline_x0, hline_x1), (hline_y,hline_y), color="k", linestyle=":")
# # ax.set_ylim(bottom=0.25*np.pi, top=0.75*np.pi)
# # ax.set_xlim(left=hline_x0, right=hline_x1)
# # ax.set_xlabel(r"Numero de fibras $N_f$")
# # ax.set_ylabel(r"Valor medio de orientaciones")
# # fig.savefig("calibracion_rve_orientdistr_mu_mean_vs_Nf.pdf", bbox="tight")


# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# ax.errorbar(devangs_deg, rec_means_stdevs, yerr=rec_stdevs_stdevs, fmt="o", markersize=7, mfc="k", mec="k", capsize=10, ecolor="k")


# plt.show()